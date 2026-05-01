"""
Multi-Block Score-Based Temporal Graph (SBTG) Model.

This module implements the multi-block score model for lag separation as described
in the theoretical derivation (Theorem 5.1).

Theory:
    For order-p Markov process: x_{t+1} = f(x_t, ..., x_{t-p+1}) + ε_t
    
    Using multi-block windows z_t = (x_{t-p_max+1}, ..., x_t, x_{t+1}):
        μ^(r) = 𝔼[s_p(z) s_{p-r}(z)ᵀ] = -𝔼[Σ_ε^{-1} J_r]
    
    where:
        s_k(z) = ∇_{z^(k)} log p(z)  is the score of block k
        J_r = ∇_{x_{t+1-r}} f(...)   is the direct lag-r Jacobian

Key difference from sbtg.py:
    - sbtg.py: Pair windows z_t = [x_t, x_{t+lag}] (2 blocks)
    - This: Multi-block windows z_t = (x_{t-p+1}, ..., x_t, x_{t+1}) (p+1 blocks)

References:
    - docs/SCRIPT_13_IMPLEMENTATION_PLAN.md
    - Theorem 5.1 in companion LaTeX document
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm


# =============================================================================
# DEFAULT HYPERPARAMETERS
# =============================================================================

DEFAULT_P_MAX = 20
DEFAULT_NOISE_STD = 0.1
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 3
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 128

DEFAULT_HAC_MAX_LAG = 5
DEFAULT_FDR_ALPHA = 0.1
DEFAULT_FDR_METHOD = 'bh'

# Structured model defaults (matching original SBTG)
DEFAULT_STRUCTURED_HIDDEN_DIM = 32
DEFAULT_STRUCTURED_NUM_LAYERS = 2
DEFAULT_L1_LAMBDA = 0.018
DEFAULT_FEATURE_DIM = 16


# =============================================================================
# HELPER MODULES FOR STRUCTURED MODEL
# =============================================================================

class ScalarMLP(nn.Module):
    """MLP that outputs a scalar energy value."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers = []
        dim_in = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.SiLU())
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class VectorMLP(nn.Module):
    """MLP mapping R^n -> R^r (feature vector)."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers = []
        dim_in = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.SiLU())
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# STRUCTURED MULTI-BLOCK SCORE NETWORK (THE KEY INNOVATION)
# =============================================================================

class MultiBlockStructuredScoreNet(nn.Module):
    """
    Structured score model for multi-block windows with EXPLICIT lag coupling.
    
    This is the multi-block generalization of the original StructuredScoreNet.
    
    Energy Function:
        U(z) = Σ_k g_k(z^(k)) + Σ_{r=1}^{p_max} z_future^T W_r z_{lag-r}
        
    Where:
        - z = (z^(0), ..., z^(p_max)) with z^(k) ∈ ℝ^n
        - z_future = z^(p_max) = x_{t+1}
        - z_{lag-r} = z^(p_max-r) = x_{t+1-r}
        - g_k: Per-block energy (captures marginal structure)
        - W_r: Lag-r coupling matrix (DIRECT connectivity at lag r)
        
    Score Function:
        s(z) = -∇U(z)
        
        For the future block:
            s_{p_max}(z) = -∂g_{p_max}/∂z_{p_max} - Σ_r W_r z_{lag-r}
            
        For a lag-r block:
            s_{p_max-r}(z) = -∂g_{p_max-r}/∂z_{p_max-r} - W_r^T z_future
            
    Key Property (Theorem 5.1):
        𝔼[s_{p_max}(z) s_{p_max-r}(z)^T] recovers information about W_r
        
    The W_r matrices DIRECTLY encode lag-r connectivity!
    Apply L1 regularization on each W_r for sparsity.
    
    Args:
        n_neurons: Number of neurons (dimension per block)
        p_max: Maximum lag order (total blocks = p_max + 1)
        hidden_dim: Hidden dimension for per-block MLPs
        num_layers: Number of layers in per-block MLPs
        init_scale: Initialization scale for W matrices
        model_type: "linear" (W directly in n×n) or "feature_bilinear" (W in r×r)
        feature_dim: Dimension of feature space for feature_bilinear
    """
    
    def __init__(
        self,
        n_neurons: int,
        p_max: int,
        hidden_dim: int = DEFAULT_STRUCTURED_HIDDEN_DIM,
        num_layers: int = DEFAULT_STRUCTURED_NUM_LAYERS,
        init_scale: float = 0.1,
        model_type: str = "linear",
        feature_dim: int = DEFAULT_FEATURE_DIM,
    ):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.p_max = p_max
        self.n_blocks = p_max + 1
        self.model_type = model_type
        self.feature_dim = feature_dim
        
        # Per-block energy functions g_k
        self.g = nn.ModuleList([
            ScalarMLP(n_neurons, hidden_dim, num_layers)
            for _ in range(self.n_blocks)
        ])
        
        # Lag-specific coupling matrices W_r for r = 1, ..., p_max
        if model_type == "linear":
            # Direct n×n coupling
            self.W = nn.ParameterList([
                nn.Parameter(torch.empty(n_neurons, n_neurons))
                for _ in range(p_max)
            ])
            for W_r in self.W:
                nn.init.uniform_(W_r, -init_scale, init_scale)
        else:
            # Feature bilinear: φ(x_future)^T W_r ψ(x_lag-r)
            # Shared feature extractors across lags
            self.phi = VectorMLP(n_neurons, feature_dim, hidden_dim, num_layers)  # future
            self.psi = VectorMLP(n_neurons, feature_dim, hidden_dim, num_layers)  # past
            
            # Per-lag coupling in feature space (r×r matrices)
            self.W = nn.ParameterList([
                nn.Parameter(torch.empty(feature_dim, feature_dim))
                for _ in range(p_max)
            ])
            for W_r in self.W:
                nn.init.uniform_(W_r, -init_scale, init_scale)
    
    def compute_energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute energy U(z).
        
        Args:
            z: (batch, (p_max+1)*n) input windows
            
        Returns:
            U: (batch,) energy values
        """
        batch_size = z.shape[0]
        
        # Split into blocks
        blocks = z.reshape(batch_size, self.n_blocks, self.n_neurons)
        
        # Per-block energy: Σ_k g_k(z^(k))
        energy = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
        for k in range(self.n_blocks):
            energy = energy + self.g[k](blocks[:, k, :])
        
        # Future block (last block)
        z_future = blocks[:, self.p_max, :]  # (batch, n)
        
        # Cross-block coupling: Σ_r z_future^T W_r z_{lag-r}
        for r in range(1, self.p_max + 1):
            lag_block_idx = self.p_max - r  # Block index for lag r
            z_lag_r = blocks[:, lag_block_idx, :]  # (batch, n)
            
            if self.model_type == "linear":
                # z_future^T W_r z_lag_r
                cross_r = torch.einsum('bi,ij,bj->b', z_future, self.W[r-1], z_lag_r)
            else:
                # φ(z_future)^T W_r ψ(z_lag_r)
                phi_future = self.phi(z_future)  # (batch, feature_dim)
                psi_lag = self.psi(z_lag_r)      # (batch, feature_dim)
                cross_r = torch.einsum('bi,ij,bj->b', phi_future, self.W[r-1], psi_lag)
            
            energy = energy + cross_r
        
        return energy
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute score s(z) = -∇_z U(z).
        
        Uses autograd for exact gradients.
        
        Args:
            z: (batch, (p_max+1)*n) input windows
            
        Returns:
            s: (batch, (p_max+1)*n) score estimates
        """
        z = z.clone().requires_grad_(True)
        
        U = self.compute_energy(z).sum()  # Sum for batch gradient
        
        need_graph = self.training
        grad_z, = torch.autograd.grad(U, z, create_graph=need_graph, retain_graph=need_graph)
        
        return -grad_z  # Score is negative gradient of energy
    
    def get_lag_coupling(self, lag: int) -> np.ndarray:
        """
        Get the coupling matrix W_r for a specific lag.
        
        Args:
            lag: Lag value (1 to p_max)
            
        Returns:
            W_r: (n, n) or (feature_dim, feature_dim) coupling matrix
        """
        if lag < 1 or lag > self.p_max:
            raise ValueError(f"lag must be 1..{self.p_max}, got {lag}")
        return self.W[lag-1].detach().cpu().numpy()
    
    def get_all_lag_couplings(self) -> Dict[int, np.ndarray]:
        """Get all lag coupling matrices."""
        return {r: self.get_lag_coupling(r) for r in range(1, self.p_max + 1)}
    
    def get_block_scores(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get scores reshaped to (batch, p_max+1, n) for block-wise analysis.
        
        Args:
            z: (batch, (p_max+1)*n) input windows
            
        Returns:
            s_blocks: (batch, p_max+1, n) score per block
        """
        s = self.forward(z)  # (batch, (p_max+1)*n)
        batch_size = s.shape[0]
        # Reshape: (batch, p_max+1, n_neurons)
        s_blocks = s.view(batch_size, self.p_max + 1, self.n_neurons)
        return s_blocks
    
    def l1_penalty(self) -> torch.Tensor:
        """Compute L1 penalty on all coupling matrices."""
        penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for W_r in self.W:
            penalty = penalty + torch.abs(W_r).sum()
        return penalty


# =============================================================================
# MULTI-BLOCK SCORE NETWORK (BLACK-BOX MLP - ORIGINAL)
# =============================================================================

class MultiBlockScoreNet(nn.Module):
    """
    Score model for multi-block windows z = (z^(0), ..., z^(p_max)).
    
    Uses a shared backbone that processes the full window and outputs
    block-structured scores. This is more parameter-efficient than
    independent networks per block.
    
    Architecture:
        Input: z ∈ ℝ^{(p_max+1)*n}
        → Hidden layers (shared)
        → Output: s(z) ∈ ℝ^{(p_max+1)*n}
        
    The output can be reshaped to (p_max+1, n) to get per-block scores.
    """
    
    def __init__(
        self,
        n_neurons: int,
        p_max: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
    ):
        """
        Args:
            n_neurons: Number of neurons (dimension per block)
            p_max: Maximum lag order (total blocks = p_max + 1)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.n_neurons = n_neurons
        self.p_max = p_max
        self.n_blocks = p_max + 1
        self.input_dim = self.n_blocks * n_neurons
        self.output_dim = self.n_blocks * n_neurons
        
        # Build network with LayerNorm for better gradient flow
        layers = []
        
        # First layer
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())  # GELU often works better than ReLU
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        
        # Output layer (no activation - score can be positive or negative)
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for score matching.
        
        Use standard Xavier initialization with gain=1 (ReLU compatible).
        The key insight: DSM targets have magnitude ~1/sigma, so outputs
        should also have reasonable magnitude.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Standard Xavier init with gain appropriate for ReLU
                nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: (batch, (p_max+1)*n) input windows
            
        Returns:
            s: (batch, (p_max+1)*n) score estimates
        """
        return self.network(z)
    
    def get_block_scores(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get scores reshaped to (batch, p_max+1, n).
        
        Args:
            z: (batch, (p_max+1)*n) input windows
            
        Returns:
            s_blocks: (batch, p_max+1, n) per-block scores
        """
        s = self.forward(z)
        batch_size = z.shape[0]
        return s.reshape(batch_size, self.n_blocks, self.n_neurons)


# =============================================================================
# DENOISING SCORE MATCHING
# =============================================================================

def dsm_loss(
    model: MultiBlockScoreNet,
    z_clean: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    """
    Compute Denoising Score Matching loss.
    
    Theory:
        z_noisy = z_clean + σ * ε,  where ε ~ N(0, I)
        s(z_noisy) ≈ ∇_z log p(z_noisy)
        
        DSM target: -ε / σ  (NOT -ε / σ²)
        
        Loss = 𝔼[‖s(z_noisy) + ε/σ‖²]
    
    Args:
        model: Multi-block score network
        z_clean: (batch, dim) clean windows
        noise_std: Noise standard deviation σ
        
    Returns:
        loss: Scalar DSM loss
    """
    # Add noise
    eps = torch.randn_like(z_clean)
    z_noisy = z_clean + noise_std * eps
    
    # Get score estimate
    s_pred = model(z_noisy)
    
    # DSM target: -ε / σ (CORRECTED scaling)
    s_target = -eps / noise_std
    
    # MSE loss (sum over dimensions, mean over batch)
    loss = ((s_pred - s_target) ** 2).sum(dim=1).mean()
    
    return loss


def compute_validation_loss(
    model: MultiBlockScoreNet,
    windows: np.ndarray,
    noise_std: float,
    device: str = 'cpu',
    batch_size: int = 1024,
) -> float:
    """
    Compute DSM loss on validation set.
    
    Args:
        model: Trained model
        windows: (N, dim) validation windows
        noise_std: Noise std used during training
        device: Device to use
        batch_size: Batch size for evaluation
        
    Returns:
        val_loss: Mean validation loss
    """
    model.eval()
    
    windows_tensor = torch.tensor(windows, dtype=torch.float32, device=device)
    dataset = TensorDataset(windows_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for (batch,) in loader:
            # For validation, we use the same noise level but don't backprop
            eps = torch.randn_like(batch)
            z_noisy = batch + noise_std * eps
            s_pred = model(z_noisy)
            s_target = -eps / noise_std
            loss = ((s_pred - s_target) ** 2).sum(dim=1).mean()
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else float('inf')


# =============================================================================
# TRAINING
# =============================================================================

def make_train_mask(
    stim_ids: np.ndarray,
    local_t: np.ndarray,
    split_type: str = 'odd_even',
    train_parity: str = 'even',
    train_frac: float = 0.7,
    random_state: int = 42,
) -> np.ndarray:
    """
    Create train/validation split mask.
    
    Args:
        stim_ids: (N,) stimulus ID per window
        local_t: (N,) local time index per window
        split_type: 'odd_even', 'random', 'prefix'
        train_parity: For odd_even: 'even' or 'odd'
        train_frac: Fraction for random/prefix splits
        random_state: Random seed
        
    Returns:
        train_mask: (N,) boolean array (True = train, False = val)
    """
    N = len(stim_ids)
    
    if split_type == 'odd_even':
        # Split by time parity (avoids adjacent windows in same split)
        if train_parity == 'even':
            train_mask = (local_t % 2 == 0)
        else:
            train_mask = (local_t % 2 == 1)
    
    elif split_type == 'random':
        rng = np.random.default_rng(random_state)
        n_train = int(train_frac * N)
        indices = rng.permutation(N)
        train_mask = np.zeros(N, dtype=bool)
        train_mask[indices[:n_train]] = True
    
    elif split_type == 'prefix':
        # First train_frac of each stimulus
        train_mask = np.zeros(N, dtype=bool)
        for stim_id in np.unique(stim_ids):
            stim_mask = (stim_ids == stim_id)
            n_stim = stim_mask.sum()
            n_train = int(train_frac * n_stim)
            stim_indices = np.where(stim_mask)[0]
            train_mask[stim_indices[:n_train]] = True
    
    else:
        raise ValueError(f"Unknown split_type: {split_type}")
    
    return train_mask


def train_multiblock_score_model(
    windows: np.ndarray,
    train_mask: np.ndarray,
    n_neurons: int,
    p_max: int,
    noise_std: float = DEFAULT_NOISE_STD,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    num_layers: int = DEFAULT_NUM_LAYERS,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = 'cpu',
    verbose: bool = True,
) -> Tuple[MultiBlockScoreNet, Dict]:
    """
    Train multi-block score model using DSM.
    
    Args:
        windows: (N, (p_max+1)*n) all windows
        train_mask: (N,) boolean mask for training
        n_neurons: Number of neurons
        p_max: Maximum lag order
        noise_std: DSM noise std
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        lr: Learning rate
        epochs: Number of epochs
        batch_size: Batch size
        device: 'cpu' or 'cuda'
        verbose: Print progress
        
    Returns:
        model: Trained MultiBlockScoreNet
        history: Dict with training metrics
    """
    # Standardize using TRAINING data only
    train_windows = windows[train_mask]
    val_windows = windows[~train_mask]
    
    mean = train_windows.mean(axis=0)
    std = train_windows.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
    
    train_windows_std = (train_windows - mean) / std
    val_windows_std = (val_windows - mean) / std
    
    # Clip extreme values to prevent NaN (robust to outliers)
    clip_val = 10.0
    train_windows_std = np.clip(train_windows_std, -clip_val, clip_val)
    val_windows_std = np.clip(val_windows_std, -clip_val, clip_val)
    
    # Check for NaN/Inf in data
    if not np.isfinite(train_windows_std).all():
        if verbose:
            print("  Warning: Non-finite values in training data, replacing with 0")
        train_windows_std = np.nan_to_num(train_windows_std, nan=0.0, posinf=clip_val, neginf=-clip_val)
    if not np.isfinite(val_windows_std).all():
        val_windows_std = np.nan_to_num(val_windows_std, nan=0.0, posinf=clip_val, neginf=-clip_val)
    
    # Create model
    model = MultiBlockScoreNet(n_neurons, p_max, hidden_dim, num_layers)
    model = model.to(device)
    
    # Create data loader
    train_tensor = torch.tensor(train_windows_std, dtype=torch.float32, device=device)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': [],
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for (batch,) in train_loader:
            optimizer.zero_grad()
            loss = dsm_loss(model, batch, noise_std)
            
            # Check for NaN loss
            if torch.isnan(loss):
                if verbose and n_batches == 0:
                    print(f"  Warning: NaN loss detected at epoch {epoch}, skipping batch")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        train_loss = epoch_loss / n_batches if n_batches > 0 else float('nan')
        
        # Validation loss
        val_loss = compute_validation_loss(
            model, val_windows_std, noise_std, device, batch_size
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Store standardization params
    history['mean'] = mean
    history['std'] = std
    
    return model, history


def train_structured_multiblock_model(
    windows: np.ndarray,
    train_mask: np.ndarray,
    n_neurons: int,
    p_max: int,
    noise_std: float = DEFAULT_NOISE_STD,
    hidden_dim: int = DEFAULT_STRUCTURED_HIDDEN_DIM,
    num_layers: int = DEFAULT_STRUCTURED_NUM_LAYERS,
    l1_lambda: float = DEFAULT_L1_LAMBDA,
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_type: str = "linear",
    feature_dim: int = DEFAULT_FEATURE_DIM,
    device: str = 'cpu',
    verbose: bool = True,
) -> Tuple[MultiBlockStructuredScoreNet, Dict]:
    """
    Train STRUCTURED multi-block score model using DSM with L1 regularization.
    
    This is the multi-block generalization of the original SBTG.
    The key difference from train_multiblock_score_model:
      - Uses structured energy with explicit W_r coupling matrices
      - Applies L1 regularization on W_r for sparsity
      - W_r matrices directly encode lag-r connectivity
    
    Args:
        windows: (N, (p_max+1)*n) all windows
        train_mask: (N,) boolean mask for training
        n_neurons: Number of neurons
        p_max: Maximum lag order (W_1, ..., W_{p_max})
        noise_std: DSM noise std
        hidden_dim: Hidden dimension for per-block MLPs
        num_layers: Number of layers in per-block MLPs
        l1_lambda: L1 regularization weight on W matrices
        lr: Learning rate
        epochs: Number of epochs
        batch_size: Batch size
        model_type: "linear" or "feature_bilinear"
        feature_dim: Feature dimension for bilinear model
        device: 'cpu' or 'cuda' or 'mps'
        verbose: Print progress
        
    Returns:
        model: Trained MultiBlockStructuredScoreNet
        history: Dict with training metrics and lag coupling matrices
    """
    # Standardize using TRAINING data only
    train_windows = windows[train_mask]
    val_windows = windows[~train_mask]
    
    mean = train_windows.mean(axis=0)
    std = train_windows.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    
    train_windows_std = (train_windows - mean) / std
    val_windows_std = (val_windows - mean) / std
    
    # Clip extreme values
    clip_val = 10.0
    train_windows_std = np.clip(train_windows_std, -clip_val, clip_val)
    val_windows_std = np.clip(val_windows_std, -clip_val, clip_val)
    
    # Handle NaN/Inf
    train_windows_std = np.nan_to_num(train_windows_std, nan=0.0, posinf=clip_val, neginf=-clip_val)
    val_windows_std = np.nan_to_num(val_windows_std, nan=0.0, posinf=clip_val, neginf=-clip_val)
    
    # Create STRUCTURED model
    model = MultiBlockStructuredScoreNet(
        n_neurons=n_neurons,
        p_max=p_max,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        model_type=model_type,
        feature_dim=feature_dim,
    )
    model = model.to(device)
    
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Structured model: {model_type}, p_max={p_max}, {n_params:,} params")
        print(f"  L1 lambda: {l1_lambda}")
    
    # Create data loader
    train_tensor = torch.tensor(train_windows_std, dtype=torch.float32, device=device)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'l1_penalty': [],
        'epoch': [],
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        n_batches = 0
        
        for (batch,) in train_loader:
            optimizer.zero_grad()
            
            # DSM loss
            loss_dsm = dsm_loss_structured(model, batch, noise_std)
            
            # L1 penalty on coupling matrices
            l1_penalty = model.l1_penalty()
            
            # Total loss
            loss = loss_dsm + l1_lambda * l1_penalty
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_dsm.item()
            epoch_l1 += l1_penalty.item()
            n_batches += 1
        
        train_loss = epoch_loss / n_batches if n_batches > 0 else float('nan')
        avg_l1 = epoch_l1 / n_batches if n_batches > 0 else float('nan')
        
        # Validation loss (DSM only, no L1)
        val_loss = compute_validation_loss_structured(
            model, val_windows_std, noise_std, device, batch_size
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['l1_penalty'].append(avg_l1)
        history['epoch'].append(epoch)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}/{epochs}: dsm={train_loss:.4f}, val={val_loss:.4f}, L1={avg_l1:.4f}")
    
    # Store standardization params and coupling matrices
    history['mean'] = mean
    history['std'] = std
    history['W'] = model.get_all_lag_couplings()  # Dict[lag, np.ndarray]
    
    return model, history


def dsm_loss_structured(
    model: MultiBlockStructuredScoreNet,
    z_clean: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    """DSM loss for structured model (same as regular DSM).
    
    Uses mean over all elements (batch and dimension) to match original SBTG.
    """
    eps = torch.randn_like(z_clean)
    z_noisy = z_clean + noise_std * eps
    
    s_pred = model(z_noisy)
    s_target = -eps / noise_std
    
    # Use mean over all elements to match original SBTG loss computation
    loss = ((s_pred - s_target) ** 2).mean()
    return loss


def compute_validation_loss_structured(
    model: MultiBlockStructuredScoreNet,
    val_windows: np.ndarray,
    noise_std: float,
    device: str,
    batch_size: int,
) -> float:
    """Compute validation DSM loss for structured model.
    
    Note: We cannot use torch.no_grad() because the structured model's forward()
    method uses torch.autograd.grad() to compute the score from the energy.
    We detach tensors after computing loss to avoid memory buildup.
    """
    model.eval()
    val_tensor = torch.tensor(val_windows, dtype=torch.float32, device=device)
    
    total_loss = 0.0
    n_batches = 0
    
    # Cannot use torch.no_grad() - model.forward() needs autograd for score computation
    for start in range(0, len(val_tensor), batch_size):
        batch = val_tensor[start:start + batch_size]
        
        eps = torch.randn_like(batch)
        z_noisy = batch + noise_std * eps
        z_noisy.requires_grad_(True)  # Needed for autograd in forward()
        
        s_pred = model(z_noisy)
        s_target = -eps / noise_std
        
        # Use mean over all elements to match original SBTG loss computation
        loss = ((s_pred - s_target) ** 2).mean()
        total_loss += loss.detach().item()  # Detach to free graph
        n_batches += 1
    
    return total_loss / n_batches if n_batches > 0 else float('nan')


def extract_structured_block_scores(
    model: MultiBlockStructuredScoreNet,
    windows: np.ndarray,
    n_neurons: int,
    p_max: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: str = 'cpu',
    batch_size: int = 256,
) -> np.ndarray:
    """
    Extract block-structured scores from structured model.
    
    Since the structured model uses autograd for score computation,
    we cannot use torch.no_grad(). We process in batches.
    
    Args:
        model: Trained MultiBlockStructuredScoreNet
        windows: (N, (p_max+1)*n) windows (NOT standardized)
        n_neurons: Neurons per block
        p_max: Maximum lag order
        mean: Standardization mean (from training)
        std: Standardization std (from training)
        device: Device
        batch_size: Batch size (smaller to save memory)
        
    Returns:
        scores: (N, p_max+1, n) block-structured scores
    """
    model.eval()
    
    # Standardize
    windows_std = (windows - mean) / std
    
    all_scores = []
    
    # Cannot use no_grad - model needs autograd for forward()
    for start in range(0, len(windows_std), batch_size):
        end = min(start + batch_size, len(windows_std))
        batch = torch.tensor(
            windows_std[start:end],
            dtype=torch.float32,
            device=device,
            requires_grad=True,  # Required for autograd in forward()
        )
        
        # Get block scores
        s_blocks = model.get_block_scores(batch)
        all_scores.append(s_blocks.detach().cpu().numpy())
    
    scores = np.concatenate(all_scores, axis=0)
    return scores


def compute_structured_score_crosscorr(
    scores: np.ndarray,
    p_max: int,
    lag_r: int,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute μ̂^(r) = mean(s_future_j × s_lag_r_i) for lag r.
    
    This is analogous to the original SBTG's mu_hat computation,
    but generalized for multi-block windows.
    
    Args:
        scores: (N, p_max+1, n) block-structured scores
        p_max: Maximum lag order
        lag_r: Lag to compute (1 to p_max)
        
    Returns:
        mu_hat: (n, n) mean cross-correlation matrix
        stats: Dict with additional statistics
    """
    if lag_r < 1 or lag_r > p_max:
        raise ValueError(f"lag_r must be 1..{p_max}, got {lag_r}")
    
    # Future block is at index p_max (the last block)
    # Lag-r block is at index p_max - lag_r
    future_idx = p_max
    lag_idx = p_max - lag_r
    
    s_future = scores[:, future_idx, :]  # (N, n)
    s_lag = scores[:, lag_idx, :]        # (N, n)
    
    # Compute Y(j,i) = s_future_j × s_lag_i for each time point
    # Result: (N, n, n) where Y[t,j,i] = s_future[t,j] * s_lag[t,i]
    Y = np.einsum('tj,ti->tji', s_future, s_lag)
    
    # Mean over time
    mu_hat = Y.mean(axis=0)  # (n, n)
    
    return mu_hat, {
        'lag_r': lag_r,
        'future_idx': future_idx,
        'lag_idx': lag_idx,
        'n_samples': len(scores),
        'mean_abs': float(np.abs(mu_hat).mean()),
        'max_abs': float(np.abs(mu_hat).max()),
    }


# =============================================================================
# SCORE EXTRACTION
# =============================================================================

def extract_block_scores(
    model: MultiBlockScoreNet,
    windows: np.ndarray,
    n_neurons: int,
    p_max: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: str = 'cpu',
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Extract block-structured scores for all windows.
    
    Args:
        model: Trained multi-block score model
        windows: (N, (p_max+1)*n) windows (NOT standardized)
        n_neurons: Neurons per block
        p_max: Maximum lag order
        mean: Standardization mean (from training)
        std: Standardization std (from training)
        device: Device
        batch_size: Batch size
        
    Returns:
        scores: (N, p_max+1, n) block-structured scores
    """
    model.eval()
    
    # Standardize
    windows_std = (windows - mean) / std
    
    all_scores = []
    
    with torch.no_grad():
        for start in range(0, len(windows_std), batch_size):
            end = min(start + batch_size, len(windows_std))
            batch = torch.tensor(
                windows_std[start:end],
                dtype=torch.float32,
                device=device
            )
            
            # Get block scores
            s_blocks = model.get_block_scores(batch)
            all_scores.append(s_blocks.cpu().numpy())
    
    scores = np.concatenate(all_scores, axis=0)
    return scores


# =============================================================================
# LAG-SPECIFIC MEAN TRANSFER
# =============================================================================

def compute_lag_r_mean_transfer(
    scores: np.ndarray,
    p_max: int,
    lag_r: int,
    stim_ids: np.ndarray,
    local_t: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute μ^(r) = 𝔼[s_p(z) s_{p-r}(z)ᵀ] for lag r.
    
    Theory (Theorem 5.1):
        μ^(r) = -𝔼[Σ_ε^{-1} J_r]
        
        where J_r is the direct lag-r Jacobian.
    
    Args:
        scores: (N, p_max+1, n) block scores from model
        p_max: Maximum lag order
        lag_r: Which lag to compute (1 ≤ lag_r ≤ p_max)
        stim_ids: (N,) stimulus ID per window
        local_t: (N,) time index per window
        
    Returns:
        mu_r: (n, n) lag-r Mean Transfer matrix
        metadata: Dict with computation details
    """
    if not (1 <= lag_r <= p_max):
        raise ValueError(f"lag_r must be in [1, {p_max}], got {lag_r}")
    
    N, n_blocks, n = scores.shape
    
    if n_blocks != p_max + 1:
        raise ValueError(f"Expected {p_max+1} blocks, got {n_blocks}")
    
    # Future block: index p_max (x_{t+1})
    s_future = scores[:, p_max, :]  # (N, n)
    
    # Lag-r block: index p_max - r (x_{t+1-r})
    lag_r_idx = p_max - lag_r
    s_lag_r = scores[:, lag_r_idx, :]  # (N, n)
    
    # Compute cross-moment: μ^(r) = 𝔼[s_future s_lag_r^T]
    # μ^(r)[j, i] = 𝔼[s_future_j * s_lag_r_i]
    mu_r = (s_future.T @ s_lag_r) / N  # (n, n)
    
    metadata = {
        'lag_r': lag_r,
        'p_max': p_max,
        'future_block_idx': p_max,
        'lag_r_block_idx': lag_r_idx,
        'n_windows': N,
        'n_neurons': n,
    }
    
    return mu_r, metadata


# =============================================================================
# STATISTICAL TESTING (HAC + FDR)
# =============================================================================

def hac_variance(
    Y: np.ndarray,
    max_lag: int = 5,
) -> float:
    """
    Compute HAC (Newey-West) variance estimate.
    
    Args:
        Y: (N,) time series
        max_lag: Maximum lag for autocovariance
        
    Returns:
        var_hac: HAC variance estimate
    """
    N = len(Y)
    Y_centered = Y - Y.mean()
    
    # Lag 0 (variance)
    gamma_0 = np.sum(Y_centered ** 2) / N
    
    # Sum of lagged autocovariances with Bartlett weights
    gamma_sum = 0.0
    for lag in range(1, min(max_lag + 1, N)):
        weight = 1.0 - lag / (max_lag + 1)
        gamma_lag = np.sum(Y_centered[lag:] * Y_centered[:-lag]) / N
        gamma_sum += 2 * weight * gamma_lag
    
    var_hac = gamma_0 + gamma_sum
    
    # Ensure positive
    return max(var_hac, 1e-10)


def hac_ttest(
    Y: np.ndarray,
    max_lag: int = 5,
) -> Tuple[float, float]:
    """
    Perform HAC t-test for H0: E[Y] = 0.
    
    Args:
        Y: (N,) time series
        max_lag: Maximum lag for HAC
        
    Returns:
        t_stat: t-statistic
        p_value: two-sided p-value
    """
    N = len(Y)
    mean_Y = Y.mean()
    var_hac = hac_variance(Y, max_lag)
    se = np.sqrt(var_hac / N)
    
    if se < 1e-10:
        return 0.0, 1.0
    
    t_stat = mean_Y / se
    p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))
    
    return t_stat, p_value


def apply_fdr(
    pvalues: np.ndarray,
    alpha: float = 0.1,
    method: str = 'bh',
) -> np.ndarray:
    """
    Apply FDR control to p-values.
    
    Args:
        pvalues: (n, n) matrix of p-values
        alpha: FDR level
        method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)
        
    Returns:
        significant: (n, n) boolean matrix of significant edges
    """
    # Flatten, excluding diagonal
    n = pvalues.shape[0]
    mask = ~np.eye(n, dtype=bool)
    pvals_flat = pvalues[mask]
    
    m = len(pvals_flat)
    if m == 0:
        return np.zeros_like(pvalues, dtype=bool)
    
    # Sort p-values
    sorted_idx = np.argsort(pvals_flat)
    sorted_pvals = pvals_flat[sorted_idx]
    
    # Compute thresholds
    if method == 'bh':
        # Benjamini-Hochberg
        thresholds = alpha * np.arange(1, m + 1) / m
    elif method == 'by':
        # Benjamini-Yekutieli (more conservative)
        c_m = np.sum(1.0 / np.arange(1, m + 1))
        thresholds = alpha * np.arange(1, m + 1) / (m * c_m)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Find largest k where p_(k) <= threshold_k
    significant_sorted = sorted_pvals <= thresholds
    
    if not significant_sorted.any():
        return np.zeros_like(pvalues, dtype=bool)
    
    # All p-values up to the largest significant one are significant
    k_max = np.where(significant_sorted)[0][-1]
    threshold = sorted_pvals[k_max]
    
    # Apply threshold to original matrix
    significant = (pvalues <= threshold) & mask
    
    return significant


def multiblock_mean_test_per_lag(
    scores: np.ndarray,
    p_max: int,
    lag_r: int,
    stim_ids: np.ndarray,
    local_t: np.ndarray,
    hac_max_lag: int = DEFAULT_HAC_MAX_LAG,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    fdr_method: str = DEFAULT_FDR_METHOD,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Perform Mean Test for lag r using HAC t-test + FDR.
    
    Theory:
        Y_{t,ji} = s_{p,j}(z_t) * s_{p-r,i}(z_t)
        
        H0: 𝔼[Y_{ji}] = 0 (no lag-r coupling i→j)
        HA: 𝔼[Y_{ji}] ≠ 0 (lag-r coupling exists)
    
    Args:
        scores: (N, p_max+1, n) block scores
        p_max: Maximum lag order
        lag_r: Which lag to test
        stim_ids: (N,) stimulus ID per window
        local_t: (N,) time index per window
        hac_max_lag: Maximum lag for HAC variance
        fdr_alpha: FDR level
        fdr_method: 'bh' or 'by'
        
    Returns:
        A_func: (n, n) binary adjacency matrix
        pvalues: (n, n) p-value matrix
        results: Dict with test details
    """
    N, n_blocks, n = scores.shape
    
    # Get relevant block scores
    s_future = scores[:, p_max, :]  # (N, n)
    s_lag_r = scores[:, p_max - lag_r, :]  # (N, n)
    
    # Compute Y_{ji} for all edges
    # Y_{t,ji} = s_future[t,j] * s_lag_r[t,i]
    # We need this for each (j, i) pair
    
    pvalues = np.ones((n, n))
    tstat = np.zeros((n, n))
    
    for j in range(n):
        for i in range(n):
            if i == j:
                continue  # Skip diagonal
            
            Y_ji = s_future[:, j] * s_lag_r[:, i]
            t, p = hac_ttest(Y_ji, max_lag=hac_max_lag)
            tstat[j, i] = t
            pvalues[j, i] = p
    
    # Apply FDR
    A_func = apply_fdr(pvalues, alpha=fdr_alpha, method=fdr_method).astype(np.float64)
    
    results = {
        'lag_r': lag_r,
        'n_edges': int(A_func.sum()),
        'fdr_alpha': fdr_alpha,
        'fdr_method': fdr_method,
        'hac_max_lag': hac_max_lag,
    }
    
    return A_func, pvalues, results


def multiblock_volatility_test_per_lag(
    scores: np.ndarray,
    p_max: int,
    lag_r: int,
    A_func: np.ndarray,
    stim_ids: np.ndarray,
    local_t: np.ndarray,
    hac_max_lag: int = DEFAULT_HAC_MAX_LAG,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    fdr_method: str = DEFAULT_FDR_METHOD,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Perform Volatility Transfer Test for lag r.
    
    Tests for covariance of squared scores, which detects
    sign-flipping effects that the mean test misses.
    
    Only tests edges NOT already in A_func.
    
    Theory:
        U_t = s_{p,j}(z_t)²
        V_t = s_{p-r,i}(z_t)²
        Z_t = (U_t - mean(U)) * (V_t - mean(V))
        
        Test: 𝔼[Z_t] = Cov(U, V) ≠ 0?
    
    Args:
        scores: (N, p_max+1, n) block scores
        p_max: Maximum lag order
        lag_r: Which lag to test
        A_func: (n, n) edges already found by mean test
        stim_ids: (N,) stimulus ID
        local_t: (N,) time index
        hac_max_lag: HAC max lag
        fdr_alpha: FDR level
        fdr_method: FDR method
        
    Returns:
        A_vol: (n, n) additional edges from volatility test
        pvalues: (n, n) p-values
        results: Dict with details
    """
    N, n_blocks, n = scores.shape
    
    s_future = scores[:, p_max, :]
    s_lag_r = scores[:, p_max - lag_r, :]
    
    # Squared scores (volatility)
    U = s_future ** 2  # (N, n)
    V = s_lag_r ** 2  # (N, n)
    
    # Center
    U_centered = U - U.mean(axis=0, keepdims=True)
    V_centered = V - V.mean(axis=0, keepdims=True)
    
    pvalues = np.ones((n, n))
    tstat = np.zeros((n, n))
    
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            if A_func[j, i] > 0:
                # Already found by mean test, skip
                continue
            
            Z_ji = U_centered[:, j] * V_centered[:, i]
            t, p = hac_ttest(Z_ji, max_lag=hac_max_lag)
            tstat[j, i] = t
            pvalues[j, i] = p
    
    # Apply FDR only to edges not in A_func
    mask_not_func = (A_func == 0)
    pvalues_masked = np.where(mask_not_func, pvalues, 1.0)
    A_vol = apply_fdr(pvalues_masked, alpha=fdr_alpha, method=fdr_method).astype(np.float64)
    
    results = {
        'lag_r': lag_r,
        'n_edges': int(A_vol.sum()),
        'fdr_alpha': fdr_alpha,
        'fdr_method': fdr_method,
    }
    
    return A_vol, pvalues, results


# =============================================================================
# SYNTHETIC DATA FOR TESTING
# =============================================================================

def simulate_var2(
    A_1: np.ndarray,
    A_2: np.ndarray,
    Sigma_eps: np.ndarray,
    T: int,
    burn_in: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate stationary VAR(2) process.
    
    Model: x_{t+1} = A_1 x_t + A_2 x_{t-1} + ε_t
    
    Args:
        A_1: (n, n) lag-1 coefficient matrix
        A_2: (n, n) lag-2 coefficient matrix
        Sigma_eps: (n, n) innovation covariance
        T: Number of timepoints to return
        burn_in: Burn-in period (discarded)
        seed: Random seed
        
    Returns:
        X: (T, n) simulated time series
    """
    rng = np.random.default_rng(seed)
    n = A_1.shape[0]
    
    # Cholesky for sampling
    L = np.linalg.cholesky(Sigma_eps)
    
    # Initialize
    X = np.zeros((T + burn_in, n))
    
    # Simulate
    for t in range(2, T + burn_in):
        eps = L @ rng.standard_normal(n)
        X[t] = A_1 @ X[t-1] + A_2 @ X[t-2] + eps
    
    return X[burn_in:]


def simulate_varp(
    A_list: List[np.ndarray],
    Sigma_eps: np.ndarray,
    T: int,
    burn_in: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate stationary VAR(p) process.
    
    Model: x_{t+1} = A_1 x_t + A_2 x_{t-1} + ... + A_p x_{t-p+1} + ε_t
    
    Args:
        A_list: List of (n, n) coefficient matrices [A_1, A_2, ..., A_p]
        Sigma_eps: (n, n) innovation covariance
        T: Number of timepoints
        burn_in: Burn-in period
        seed: Random seed
        
    Returns:
        X: (T, n) simulated time series
    """
    rng = np.random.default_rng(seed)
    p = len(A_list)
    n = A_list[0].shape[0]
    
    L = np.linalg.cholesky(Sigma_eps)
    X = np.zeros((T + burn_in, n))
    
    for t in range(p, T + burn_in):
        eps = L @ rng.standard_normal(n)
        x_new = eps.copy()
        for r, A_r in enumerate(A_list):
            x_new += A_r @ X[t - 1 - r]
        X[t] = x_new
    
    return X[burn_in:]


if __name__ == "__main__":
    # Quick test
    print("Testing MultiBlockScoreNet...")
    
    n = 10
    p_max = 3
    batch = 32
    
    model = MultiBlockScoreNet(n, p_max, hidden_dim=64, num_layers=2)
    z = torch.randn(batch, (p_max + 1) * n)
    
    s = model(z)
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {s.shape}")
    
    s_blocks = model.get_block_scores(z)
    print(f"Block scores shape: {s_blocks.shape}")
    
    # Test DSM loss
    loss = dsm_loss(model, z, noise_std=0.1)
    print(f"DSM loss: {loss.item():.4f}")
    
    # Test backward
    loss.backward()
    print("✅ Backward pass successful!")
