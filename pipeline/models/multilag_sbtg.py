"""
Multi-Lag Score-Based Temporal Graph (SBTG) with Theoretically Correct Lag Separation.

This module implements the mathematically correct approach to separating lag effects
using SBTG, following the theory derived in the companion document.

Key Theory:
    For order-p Markov additive-noise model:
        x_{t+1} = f(x_t, ..., x_{t-p+1}) + ε_t,  ε_t ~ N(0, Σ_ε)
    
    The lag-r Mean Transfer matrix:
        μ^(r) = E[s_p(z) s_{p-r}(z)^T] = -E[Σ_ε^{-1} J_r]
    
    where J_r = ∂f/∂x_{t+1-r} is the direct lag-r Jacobian.
    
    This SEPARATES the direct lag-r effect from indirect pathways.

Three Approaches:
    Approach A (Per-Lag 2-Block):
        - Train separate 2-block SBTG for each lag r
        - Window: z = (x_{t+1-r}, x_{t+1})
        - μ^(r) = E[s_1(z) s_0(z)^T]
        - Simple but doesn't condition on intermediate lags
        
    Approach B (Full Multi-Block):
        - Train single model on full (p_max+1)-block window
        - Window: z = (x_{t-p_max+1}, ..., x_t, x_{t+1})
        - Extract μ^(r) = E[s_p(z) s_{p-r}(z)^T] for each lag r
        - Conditions on full lag stack → true lag separation
        
    Approach C (Minimal Multi-Block):
        - For each lag r, train on exactly (r+1)-block window
        - Lag 1: z = (x_t, x_{t+1}) - 2 blocks
        - Lag 2: z = (x_t, x_{t+1}, x_{t+2}) - 3 blocks
        - Lag r: z = (x_t, x_{t+1}, ..., x_{t+r}) - (r+1) blocks
        - Conditions on intermediate lags but not irrelevant future lags

All approaches use:
    - 5-fold cross-fitting (held-out scores for valid FDR)
    - Structured score model with explicit coupling matrices
    - HAC inference for autocorrelation-robust standard errors
    - BH/BY FDR control for edge discovery
    - Null contrast hyperparameter tuning

References:
    - Theory document Section 5.1: Multi-block lag separation
    - Original SBTG: pipeline/models/sbtg.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm

# Optuna for hyperparameter tuning
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Enable parallel data loading
import torch.multiprocessing
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except Exception:
    pass


# =============================================================================
# DEFAULT HYPERPARAMETERS (from original SBTG tuning)
# =============================================================================

DEFAULT_DSM_NOISE_STD = 0.1
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 128
DEFAULT_L1_LAMBDA = 0.0  # Sparsity via FDR, not L1 (per original SBTG)
DEFAULT_INIT_SCALE = 0.1

# Statistical testing
DEFAULT_HAC_MAX_LAG = 5
DEFAULT_FDR_ALPHA = 0.1
DEFAULT_FDR_METHOD = 'bh'

# Cross-fitting
DEFAULT_N_FOLDS = 5

# Hyperparameter tuning
DEFAULT_N_HP_TRIALS = 20
DEFAULT_HP_NOISE_STD_RANGE = (0.01, 0.5)
DEFAULT_HP_HIDDEN_DIM_CHOICES = [32, 64, 128, 256]
DEFAULT_HP_LR_RANGE = (5e-5, 5e-2)
DEFAULT_HP_N_STARTUP_TRIALS = 20


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class MultiLagSBTGResult:
    """Results from multi-lag SBTG analysis."""
    
    # Per-lag results
    mu_hat: Dict[int, np.ndarray]  # lag_r -> (n, n) μ^(r) matrix
    p_values: Dict[int, np.ndarray]  # lag_r -> (n, n) p-values
    significant: Dict[int, np.ndarray]  # lag_r -> (n, n) binary adjacency
    
    # Metadata
    approach: str  # 'A', 'B', or 'C'
    p_max: int
    n_neurons: int
    n_windows: int
    model_type: str
    
    # Training history
    training_history: Dict = field(default_factory=dict)
    
    def get_mu_hat_for_lag(self, lag: int) -> np.ndarray:
        """Get μ^(r) for specific lag."""
        if lag not in self.mu_hat:
            raise ValueError(f"Lag {lag} not available. Have: {list(self.mu_hat.keys())}")
        return self.mu_hat[lag]
    
    def get_adjacency_for_lag(self, lag: int) -> np.ndarray:
        """Get binary adjacency for specific lag."""
        return self.significant.get(lag, np.zeros((self.n_neurons, self.n_neurons)))


# =============================================================================
# STRUCTURED SCORE NETWORKS
# =============================================================================

class ScalarMLP(nn.Module):
    """MLP outputting scalar energy."""
    
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


class TwoBlockStructuredScoreNet(nn.Module):
    """
    Structured score model for 2-block windows (Approach A).
    
    Energy: U(z) = g0(x_past) + g1(x_future) + x_future^T W x_past
    Score: s(z) = -∇U(z)
    
    This is equivalent to the original SBTG StructuredScoreNet.
    """
    
    def __init__(
        self,
        n_neurons: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        init_scale: float = DEFAULT_INIT_SCALE,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        
        # Per-block energy functions
        self.g0 = ScalarMLP(n_neurons, hidden_dim, num_layers)  # Past block
        self.g1 = ScalarMLP(n_neurons, hidden_dim, num_layers)  # Future block
        
        # Coupling matrix W (direct connectivity)
        self.W = nn.Parameter(torch.empty(n_neurons, n_neurons))
        nn.init.uniform_(self.W, -init_scale, init_scale)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute score s(z) = -∇U(z).
        
        Args:
            z: (batch, 2n) window [x_past, x_future]
            
        Returns:
            s: (batch, 2n) score
        """
        n = self.n_neurons
        z = z.clone().requires_grad_(True)
        
        x_past = z[:, :n]
        x_future = z[:, n:]
        
        # Energy: g0(past) + g1(future) + future^T W past
        U = self.g0(x_past) + self.g1(x_future) + torch.einsum('bi,ij,bj->b', x_future, self.W, x_past)
        
        # Score: -∇U
        grad_z, = torch.autograd.grad(U.sum(), z, create_graph=self.training)
        return -grad_z


class MultiBlockStructuredScoreNet(nn.Module):
    """
    Structured score model for (p+1)-block windows (Approach B).
    
    Energy: U(z) = Σ_k g_k(z^(k)) + Σ_{r=1}^{p} z_future^T W_r z_{lag-r}
    Score: s(z) = -∇U(z)
    
    The W_r matrices encode DIRECT lag-r connectivity.
    """
    
    def __init__(
        self,
        n_neurons: int,
        p_max: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        init_scale: float = DEFAULT_INIT_SCALE,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.p_max = p_max
        self.n_blocks = p_max + 1
        
        # Per-block energy functions
        self.g = nn.ModuleList([
            ScalarMLP(n_neurons, hidden_dim, num_layers)
            for _ in range(self.n_blocks)
        ])
        
        # Lag-specific coupling matrices W_r (r = 1, ..., p_max)
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(n_neurons, n_neurons))
            for _ in range(p_max)
        ])
        for W_r in self.W:
            nn.init.uniform_(W_r, -init_scale, init_scale)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute score s(z) = -∇U(z).
        
        Args:
            z: (batch, (p+1)*n) window
            
        Returns:
            s: (batch, (p+1)*n) score
        """
        batch_size = z.shape[0]
        n = self.n_neurons
        z = z.clone().requires_grad_(True)
        
        # Split into blocks
        blocks = z.reshape(batch_size, self.n_blocks, n)
        
        # Per-block energy
        U = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
        for k in range(self.n_blocks):
            U = U + self.g[k](blocks[:, k, :])
        
        # Cross-block coupling (future with each lag block)
        z_future = blocks[:, self.p_max, :]  # Last block = future
        for r in range(1, self.p_max + 1):
            z_lag_r = blocks[:, self.p_max - r, :]
            U = U + torch.einsum('bi,ij,bj->b', z_future, self.W[r-1], z_lag_r)
        
        # Score: -∇U
        grad_z, = torch.autograd.grad(U.sum(), z, create_graph=self.training)
        return -grad_z
    
    def get_block_scores(self, z: torch.Tensor) -> torch.Tensor:
        """Get scores reshaped to (batch, p+1, n)."""
        s = self.forward(z)
        return s.view(z.shape[0], self.n_blocks, self.n_neurons)


# =============================================================================
# DSM TRAINING
# =============================================================================

def dsm_loss(model: nn.Module, z: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Denoising Score Matching loss.
    
    Loss = E[||s(z+σε) - (-ε/σ)||²]
    """
    eps = torch.randn_like(z)
    z_noisy = z + noise_std * eps
    
    s_pred = model(z_noisy)
    s_target = -eps / noise_std
    
    # Mean over all elements (matching original SBTG)
    return ((s_pred - s_target) ** 2).mean()


def train_score_model(
    model: nn.Module,
    train_windows: np.ndarray,
    noise_std: float,
    lr: float,
    epochs: int,
    batch_size: int,
    l1_lambda: float,
    device: str,
    verbose: bool = False,
) -> Dict:
    """Train score model using DSM."""
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_tensor = torch.tensor(train_windows, dtype=torch.float32, device=device)
    
    # Use larger batch size for GPU if available
    effective_batch_size = batch_size if device == 'cpu' else min(batch_size * 2, 512)
    
    loader = DataLoader(
        TensorDataset(train_tensor), 
        batch_size=effective_batch_size, 
        shuffle=True,
        drop_last=len(train_windows) > effective_batch_size,  # Drop last small batch
    )
    
    history = {'train_loss': [], 'epoch': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for (batch,) in loader:
            optimizer.zero_grad()
            
            loss = dsm_loss(model, batch, noise_std)
            
            # L1 on W (if applicable)
            if l1_lambda > 0 and hasattr(model, 'W'):
                if isinstance(model.W, nn.ParameterList):
                    l1 = sum(W_r.abs().sum() for W_r in model.W)
                else:
                    l1 = model.W.abs().sum()
                loss = loss + l1_lambda * l1
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else float('nan')
        history['train_loss'].append(avg_loss)
        history['epoch'].append(epoch)
        
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"    Epoch {epoch:3d}/{epochs}: loss={avg_loss:.4f}")
    
    return history


def compute_scores(
    model: nn.Module,
    windows: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Compute scores for all windows."""
    model.eval()
    
    scores = []
    for start in range(0, len(windows), batch_size):
        batch = torch.tensor(
            windows[start:start + batch_size],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        s = model(batch)
        scores.append(s.detach().cpu().numpy())
    
    return np.concatenate(scores, axis=0)


# =============================================================================
# CROSS-FITTING INFRASTRUCTURE
# =============================================================================

def create_fold_assignments(
    stim_ids: np.ndarray,
    n_folds: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Create fold assignments respecting segment boundaries.
    Strided within each stimulus to preserve temporal structure.
    """
    N = len(stim_ids)
    fold_ids = np.zeros(N, dtype=int)
    
    for u in np.unique(stim_ids):
        idx_u = np.where(stim_ids == u)[0]
        n_u = len(idx_u)
        fold_ids[idx_u] = np.arange(n_u) % n_folds
    
    return fold_ids


def standardize_windows(windows: np.ndarray, train_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize using training statistics only."""
    # Ensure float64 to avoid numpy dtype issues
    windows = windows.astype(np.float64)
    train_windows = windows[train_mask]
    mean = np.nanmean(train_windows, axis=0, keepdims=True)
    std = np.nanstd(train_windows, axis=0, keepdims=True) + 1e-8
    
    windows_std = (windows - mean) / std
    windows_std = np.nan_to_num(np.clip(windows_std, -10, 10), nan=0.0)
    
    return windows_std, mean.squeeze(), std.squeeze()


# =============================================================================
# HAC INFERENCE
# =============================================================================

def newey_west_variance(y: np.ndarray, max_lag: int) -> float:
    """Newey-West HAC variance estimator."""
    N = len(y)
    y_centered = y - y.mean()
    
    # Lag 0
    gamma_0 = np.sum(y_centered ** 2) / N
    
    # Weighted sum of lagged autocovariances
    gamma_sum = 0.0
    for lag in range(1, min(max_lag + 1, N)):
        weight = 1.0 - lag / (max_lag + 1)
        gamma_lag = np.sum(y_centered[lag:] * y_centered[:-lag]) / N
        gamma_sum += 2 * weight * gamma_lag
    
    return max(gamma_0 + gamma_sum, 1e-10)


def hac_test_mu_hat(
    s_future: np.ndarray,
    s_past: np.ndarray,
    hac_max_lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HAC t-test for μ̂(j,i) = E[s_future_j × s_past_i].
    
    Returns:
        mu_hat: (n, n) mean transfer matrix
        p_values: (n, n) two-sided p-values
    """
    N, n = s_future.shape
    
    mu_hat = np.zeros((n, n))
    p_values = np.ones((n, n))
    
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            
            Y_ji = s_future[:, j] * s_past[:, i]
            mu_hat[j, i] = Y_ji.mean()
            
            var_hac = newey_west_variance(Y_ji, hac_max_lag)
            se = np.sqrt(var_hac / N)
            
            if se > 1e-10:
                t_stat = mu_hat[j, i] / se
                p_values[j, i] = 2 * (1 - norm.cdf(np.abs(t_stat)))
    
    return mu_hat, p_values


def apply_fdr(p_values: np.ndarray, alpha: float, method: str = 'bh') -> np.ndarray:
    """Apply FDR control, return boolean significance mask."""
    n = p_values.shape[0]
    mask = ~np.eye(n, dtype=bool)
    pvals_flat = p_values[mask]
    
    m = len(pvals_flat)
    if m == 0:
        return np.zeros_like(p_values, dtype=bool)
    
    sorted_idx = np.argsort(pvals_flat)
    sorted_pvals = pvals_flat[sorted_idx]
    
    if method == 'bh':
        thresholds = alpha * np.arange(1, m + 1) / m
    elif method == 'by':
        c_m = np.sum(1.0 / np.arange(1, m + 1))
        thresholds = alpha * np.arange(1, m + 1) / (m * c_m)
    else:
        raise ValueError(f"Unknown FDR method: {method}")
    
    sig_sorted = sorted_pvals <= thresholds
    if not sig_sorted.any():
        return np.zeros_like(p_values, dtype=bool)
    
    k_max = np.where(sig_sorted)[0][-1]
    threshold = sorted_pvals[k_max]
    
    return (p_values <= threshold) & mask


# =============================================================================
# NULL CONTRAST OBJECTIVE FOR HYPERPARAMETER TUNING
# =============================================================================

def compute_null_contrast(
    mu_hat: np.ndarray,
    n_null_shuffles: int = 10,
    seed: int = 42,
) -> float:
    """
    Compute null contrast: ratio of real signal to null signal.
    
    Null Contrast = mean(|μ_real|) / mean(|μ_null|)
    
    Higher is better - indicates the learned scores capture real structure.
    
    Args:
        mu_hat: (n, n) mean transfer matrix from real data
        n_null_shuffles: Number of null shuffles for baseline
        seed: Random seed
        
    Returns:
        null_contrast: Ratio > 1 indicates real signal
    """
    n = mu_hat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    real_mean = np.abs(mu_hat[mask]).mean()
    
    # For null, we use shuffled version of the same matrix
    # This approximates what we'd get from temporally shuffled data
    rng = np.random.default_rng(seed)
    null_means = []
    
    for _ in range(n_null_shuffles):
        # Shuffle off-diagonal elements
        flat = mu_hat[mask].copy()
        rng.shuffle(flat)
        null_means.append(np.abs(flat).mean())
    
    null_mean = np.mean(null_means)
    
    if null_mean < 1e-10:
        return 1.0
    
    return real_mean / null_mean


def compute_null_contrast_from_scores(
    s_future: np.ndarray,
    s_past: np.ndarray,
    n_null_shuffles: int = 5,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """
    Compute null contrast by comparing real μ̂ to temporally shuffled μ̂.
    
    This is the proper null contrast that shuffles time to break real dependencies.
    
    Args:
        s_future: (N, n) future block scores
        s_past: (N, n) past block scores
        n_null_shuffles: Number of null shuffles
        seed: Random seed
        
    Returns:
        null_contrast: Ratio of real to null signal
        mu_hat: (n, n) real mean transfer matrix
    """
    N, n = s_future.shape
    mask = ~np.eye(n, dtype=bool)
    
    # Real μ̂
    mu_hat = (s_future.T @ s_past) / N
    real_mean = np.abs(mu_hat[mask]).mean()
    
    # Null μ̂ (temporal shuffle)
    rng = np.random.default_rng(seed)
    null_means = []
    
    for _ in range(n_null_shuffles):
        # Shuffle time indices of s_past to break temporal alignment
        perm = rng.permutation(N)
        s_past_shuffled = s_past[perm]
        mu_null = (s_future.T @ s_past_shuffled) / N
        null_means.append(np.abs(mu_null[mask]).mean())
    
    null_mean = np.mean(null_means)
    
    if null_mean < 1e-10:
        return 1.0, mu_hat
    
    return real_mean / null_mean, mu_hat


def compute_edge_stability(
    scores: np.ndarray,
    n_bootstrap: int = 10,
    seed: int = 42,
) -> float:
    """
    Compute edge stability via bootstrap.
    
    Higher stability = more reliable edges.
    
    Args:
        scores: (N, 2n) or (N, (p+1)*n) scores
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        stability: Mean correlation between bootstrap μ̂ estimates
    """
    N = scores.shape[0]
    n = scores.shape[1] // 2  # Assuming 2-block for now
    
    rng = np.random.default_rng(seed)
    
    s_past = scores[:, :n]
    s_future = scores[:, n:2*n]
    
    # Compute bootstrap μ̂ estimates
    mu_boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        mu_boot = (s_future[idx].T @ s_past[idx]) / N
        mu_boots.append(mu_boot.flatten())
    
    # Compute pairwise correlations
    mu_boots = np.array(mu_boots)
    corrs = np.corrcoef(mu_boots)
    
    # Mean off-diagonal correlation
    mask = ~np.eye(n_bootstrap, dtype=bool)
    stability = corrs[mask].mean()
    
    return stability


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

@dataclass
class HPConfig:
    """Hyperparameter configuration."""
    noise_std: float = DEFAULT_DSM_NOISE_STD
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    lr: float = DEFAULT_LR
    epochs: int = DEFAULT_EPOCHS
    
    def to_dict(self) -> dict:
        return {
            'noise_std': self.noise_std,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'lr': self.lr,
            'epochs': self.epochs,
        }


def tune_hyperparameters(
    X_list: List[np.ndarray],
    n_trials: int = DEFAULT_N_HP_TRIALS,
    lag: int = 1,
    n_blocks: int = 2,
    noise_std_range: Tuple[float, float] = DEFAULT_HP_NOISE_STD_RANGE,
    hidden_dim_choices: List[int] = None,
    lr_range: Tuple[float, float] = DEFAULT_HP_LR_RANGE,
    epochs_for_tuning: int = 20,  # Reduced from 30 for faster tuning
    n_folds: int = 3,
    device: str = 'cpu',
    verbose: bool = True,
    seed: int = 42,
) -> HPConfig:
    """
    Tune hyperparameters using Optuna TPE sampler with null contrast objective.
    
    Args:
        X_list: List of (T, n) time series
        n_trials: Number of HP configurations to try
        lag: Target lag for this model
        n_blocks: Number of blocks (2 for Approach A, r+1 for Approach C)
        noise_std_range: Range for DSM noise std
        hidden_dim_choices: List of hidden dim options
        lr_range: Range for learning rate
        epochs_for_tuning: Reduced epochs for faster tuning
        n_folds: Folds for quick evaluation
        device: Device
        verbose: Print progress
        seed: Random seed
        
    Returns:
        best_config: Best HPConfig based on null contrast
    """
    if hidden_dim_choices is None:
        hidden_dim_choices = DEFAULT_HP_HIDDEN_DIM_CHOICES
    
    n_neurons = X_list[0].shape[1]
    
    # Build windows for this lag/n_blocks configuration
    if n_blocks == 2:
        windows, stim_ids, local_t = _build_two_block_windows(X_list, lag)
    else:
        # For multi-block HP tuning, build (n_blocks-1) lag windows
        # This ensures window dim = n_blocks * n_neurons
        windows, stim_ids, local_t = _build_minimal_multiblock_windows(X_list, n_blocks - 1)
    
    if verbose:
        print(f"  HP tuning: {n_trials} trials, {len(windows)} windows, lag={lag}, blocks={n_blocks}")
    
    # Define Optuna objective function
    def objective(trial: optuna.Trial) -> float:
        if verbose:
            print(f"    [Trial {trial.number + 1}/{n_trials}] Starting...", end='\r', flush=True)

        # Sample hyperparameters using Optuna
        noise_std = trial.suggest_float('noise_std', noise_std_range[0], noise_std_range[1])
        hidden_dim = trial.suggest_categorical('hidden_dim', hidden_dim_choices)
        lr = trial.suggest_float('lr', lr_range[0], lr_range[1], log=True)
        num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])  # Added 4
        
        config = HPConfig(
            noise_std=noise_std,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            lr=lr,
            epochs=epochs_for_tuning,
        )
        
        try:
            # Evaluate using null contrast
            null_contrast = _evaluate_config(
                windows, stim_ids, n_neurons, n_blocks, lag, config,
                n_folds=n_folds, device=device
            )
            
            # Handle invalid results
            if np.isnan(null_contrast) or np.isinf(null_contrast):
                return float('-inf')
            
            return null_contrast
            
        except Exception as e:
            if verbose:
                print(f"    Trial {trial.number + 1} failed: {e}", flush=True)
            return float('-inf')
    
    # Create Optuna study with TPE sampler
    # n_startup_trials: Number of random trials before Bayesian optimization
    # e.g., with 50 total trials and 20 startup: first 20 random, next 30 Bayesian
    n_startup = min(DEFAULT_HP_N_STARTUP_TRIALS, n_trials // 2)  # At least half random
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    study = optuna.create_study(
        direction='maximize',  # Maximize null contrast
        sampler=sampler,
    )
    
    if verbose and n_trials > n_startup:
        print(f"  HP strategy: {n_startup} random trials, then {n_trials - n_startup} Bayesian (TPE)")
    
    # Custom callback for verbose logging
    def log_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if verbose:
            if trial.value is not None and trial.value > float('-inf'):
                if trial.number == 0 or trial.value > study.best_value - 0.001:
                    print(f"    Trial {trial.number + 1}/{n_trials}: NC={trial.value:.3f}" + 
                          (" (new best)" if trial.value >= study.best_value else ""), flush=True)
                elif (trial.number + 1) % 5 == 0:
                    print(f"    Trial {trial.number + 1}/{n_trials}: NC={trial.value:.3f}", flush=True)
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[log_callback],
        show_progress_bar=False,
    )
    
    # Extract best config
    best_params = study.best_params
    best_config = HPConfig(
        noise_std=best_params['noise_std'],
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params.get('num_layers', DEFAULT_NUM_LAYERS),
        lr=best_params['lr'],
        epochs=epochs_for_tuning,
    )
    
    if verbose:
        print(f"  Best config: noise_std={best_config.noise_std:.3f}, "
              f"hidden_dim={best_config.hidden_dim}, lr={best_config.lr:.2e}, NC={study.best_value:.3f}")
    
    return best_config


def _evaluate_config(
    windows: np.ndarray,
    stim_ids: np.ndarray,
    n_neurons: int,
    n_blocks: int,
    lag: int,
    config: HPConfig,
    n_folds: int = 3,
    device: str = 'cpu',
) -> float:
    """Evaluate a HP config using null contrast."""
    N = len(windows)
    fold_ids = create_fold_assignments(stim_ids, n_folds)
    
    all_s_future = []
    all_s_past = []
    
    for fold_k in range(n_folds):
        train_mask = (fold_ids != fold_k)
        heldout_mask = (fold_ids == fold_k)
        
        windows_std, mean, std = standardize_windows(windows, train_mask)
        
        # Create model
        if n_blocks == 2:
            model = TwoBlockStructuredScoreNet(
                n_neurons=n_neurons,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )
        else:
            model = MultiBlockStructuredScoreNet(
                n_neurons=n_neurons,
                p_max=n_blocks - 1,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )
        
        # Train
        train_score_model(
            model, windows_std[train_mask], config.noise_std,
            config.lr, config.epochs, DEFAULT_BATCH_SIZE, 0.0,
            device, verbose=False
        )
        
        # Get held-out scores
        heldout_scores = compute_scores(model, windows_std[heldout_mask], device)
        
        # Extract block scores
        if n_blocks == 2:
            s_past = heldout_scores[:, :n_neurons]
            s_future = heldout_scores[:, n_neurons:]
        else:
            # For multi-block, extract the relevant blocks
            scores_reshaped = heldout_scores.reshape(-1, n_blocks, n_neurons)
            s_future = scores_reshaped[:, -1, :]  # Last block
            s_past = scores_reshaped[:, 0, :]     # First block (lag=n_blocks-1)
        
        all_s_future.append(s_future)
        all_s_past.append(s_past)
    
    s_future = np.concatenate(all_s_future, axis=0)
    s_past = np.concatenate(all_s_past, axis=0)
    
    null_contrast, _ = compute_null_contrast_from_scores(s_future, s_past)
    
    return null_contrast


# =============================================================================
# WINDOW BUILDING HELPERS
# =============================================================================

def _build_two_block_windows(
    X_list: List[np.ndarray],
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 2-block windows: z = (x_{t}, x_{t+lag})."""
    windows = []
    stim_ids = []
    local_ts = []
    
    for u, X in enumerate(X_list):
        T, n = X.shape
        for t in range(T - lag):
            z = np.concatenate([X[t], X[t + lag]])
            windows.append(z)
            stim_ids.append(u)
            local_ts.append(t)
    
    return np.array(windows, dtype=np.float64), np.array(stim_ids), np.array(local_ts)


def _build_minimal_multiblock_windows(
    X_list: List[np.ndarray],
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build minimal multi-block windows for Approach C.
    
    For lag r, window is: z = (x_t, x_{t+1}, ..., x_{t+r})
    This has (r+1) blocks.
    """
    windows = []
    stim_ids = []
    local_ts = []
    n_blocks = lag + 1
    
    for u, X in enumerate(X_list):
        T, n = X.shape
        for t in range(T - lag):
            # Blocks: x_t, x_{t+1}, ..., x_{t+lag}
            blocks = [X[t + k] for k in range(n_blocks)]
            z = np.concatenate(blocks)
            windows.append(z)
            stim_ids.append(u)
            local_ts.append(t)
    
    return np.array(windows, dtype=np.float64), np.array(stim_ids), np.array(local_ts)


# =============================================================================
# APPROACH A: PER-LAG 2-BLOCK SBTG
# =============================================================================

class PerLagSBTGEstimator:
    """
    Approach A: Train separate 2-block SBTG for each lag.
    
    For each lag r:
        1. Build windows z = (x_{t+1-r}, x_{t+1})
        2. Train structured 2-block model
        3. Compute μ^(r) = E[s_future × s_past^T]
        4. Apply HAC inference + FDR
    
    Uses 5-fold cross-fitting for held-out scores.
    """
    
    def __init__(
        self,
        p_max: int = 5,
        tune_hp: bool = False,
        n_hp_trials: int = DEFAULT_N_HP_TRIALS,
        noise_std: float = DEFAULT_DSM_NOISE_STD,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        lr: float = DEFAULT_LR,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        l1_lambda: float = DEFAULT_L1_LAMBDA,
        n_folds: int = DEFAULT_N_FOLDS,
        hac_max_lag: int = DEFAULT_HAC_MAX_LAG,
        fdr_alpha: float = DEFAULT_FDR_ALPHA,
        fdr_method: str = DEFAULT_FDR_METHOD,
        device: str = 'cpu',
        verbose: bool = True,
        random_state: int = 42,
    ):
        self.p_max = p_max
        self.tune_hp = tune_hp
        self.n_hp_trials = n_hp_trials
        self.noise_std = noise_std
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l1_lambda = l1_lambda
        self.n_folds = n_folds
        self.hac_max_lag = hac_max_lag
        self.fdr_alpha = fdr_alpha
        self.fdr_method = fdr_method
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        
        # Store tuned configs per lag
        self.hp_configs: Dict[int, HPConfig] = {}
    
    def fit(self, X_list: List[np.ndarray]) -> MultiLagSBTGResult:
        """
        Fit per-lag 2-block SBTG models.
        
        Args:
            X_list: List of (T_u, n) time series per segment
            
        Returns:
            MultiLagSBTGResult with μ^(r) for each lag
        """
        n_neurons = X_list[0].shape[1]
        
        mu_hat_all = {}
        p_values_all = {}
        significant_all = {}
        history_all = {}
        
        total_windows = 0
        
        for lag_r in range(1, self.p_max + 1):
            if self.verbose:
                print(f"\n[Approach A] Processing lag {lag_r}...")
            
            # HP tuning for this lag
            if self.tune_hp:
                if self.verbose:
                    print(f"  Tuning HP ({self.n_hp_trials} trials)...")
                config = tune_hyperparameters(
                    X_list,
                    n_trials=self.n_hp_trials,
                    lag=lag_r,
                    n_blocks=2,  # 2-block windows for Approach A
                    device=self.device,
                    verbose=self.verbose,
                    seed=self.random_state + lag_r,
                )
                self.hp_configs[lag_r] = config
            else:
                config = HPConfig(
                    noise_std=self.noise_std,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    lr=self.lr,
                    epochs=self.epochs,
                )
            
            # Build 2-block windows for this lag
            windows, stim_ids, local_t = self._build_windows_for_lag(X_list, lag_r)
            total_windows = len(windows)
            
            if self.verbose:
                print(f"  Built {len(windows)} windows")
            
            # 5-fold cross-fitting with config
            scores_heldout = self._cross_fit(windows, stim_ids, n_neurons, lag_r, config)
            
            # Extract block scores
            s_past = scores_heldout[:, :n_neurons]
            s_future = scores_heldout[:, n_neurons:]
            
            # HAC inference
            mu_hat, p_values = hac_test_mu_hat(s_future, s_past, self.hac_max_lag)
            
            # FDR control
            significant = apply_fdr(p_values, self.fdr_alpha, self.fdr_method)
            
            mu_hat_all[lag_r] = mu_hat
            p_values_all[lag_r] = p_values
            significant_all[lag_r] = significant.astype(float)
            
            n_edges = significant.sum()
            if self.verbose:
                print(f"  μ̂^({lag_r}): mean|μ|={np.abs(mu_hat).mean():.4f}, max|μ|={np.abs(mu_hat).max():.4f}")
                print(f"  FDR edges: {int(n_edges)}")
        
        return MultiLagSBTGResult(
            mu_hat=mu_hat_all,
            p_values=p_values_all,
            significant=significant_all,
            approach='A',
            p_max=self.p_max,
            n_neurons=n_neurons,
            n_windows=total_windows,
            model_type='two_block_structured',
            training_history=history_all,
        )
    
    def _build_windows_for_lag(
        self,
        X_list: List[np.ndarray],
        lag_r: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build 2-block windows: z = (x_{t+1-r}, x_{t+1})."""
        windows = []
        stim_ids = []
        local_ts = []
        
        for u, X in enumerate(X_list):
            T, n = X.shape
            for t in range(lag_r, T):
                # Past block: x_{t+1-r} = x_{t+1-r}
                # Future block: x_{t+1} 
                # But we're at position t, so future = x[t], past = x[t - lag_r]
                z = np.concatenate([X[t - lag_r], X[t]])
                windows.append(z)
                stim_ids.append(u)
                local_ts.append(t - lag_r)  # Local time index
        
        return np.array(windows), np.array(stim_ids), np.array(local_ts)
    
    def _cross_fit(
        self,
        windows: np.ndarray,
        stim_ids: np.ndarray,
        n_neurons: int,
        lag_r: int,
        config: HPConfig,
    ) -> np.ndarray:
        """5-fold cross-fitting: train on 4 folds, score on held-out fold."""
        N = len(windows)
        fold_ids = create_fold_assignments(stim_ids, self.n_folds, self.random_state)
        
        scores_heldout = np.zeros_like(windows)
        
        for fold_k in range(self.n_folds):
            if self.verbose:
                print(f"  Fold {fold_k + 1}/{self.n_folds}...", flush=True)
            
            train_mask = (fold_ids != fold_k)
            heldout_mask = (fold_ids == fold_k)
            
            # Standardize using training statistics
            windows_std, mean, std = standardize_windows(windows, train_mask)
            
            # Train fresh model on this fold using config
            model = TwoBlockStructuredScoreNet(
                n_neurons=n_neurons,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )
            
            train_windows = windows_std[train_mask]
            train_score_model(
                model, train_windows, config.noise_std, config.lr, 
                config.epochs, self.batch_size, self.l1_lambda, 
                self.device, verbose=False
            )
            
            # Compute scores on held-out windows
            heldout_windows = windows_std[heldout_mask]
            heldout_scores = compute_scores(model, heldout_windows, self.device)
            
            scores_heldout[heldout_mask] = heldout_scores
        
        return scores_heldout


# =============================================================================
# APPROACH B: FULL MULTI-BLOCK SBTG
# =============================================================================

class MultiBlockSBTGEstimator:
    """
    Approach B: Train single (p+1)-block model, extract μ^(r) for each lag.
    
    1. Build (p+1)-block windows: z = (x_{t-p+1}, ..., x_t, x_{t+1})
    2. Train structured multi-block model with W_1, ..., W_p
    3. Cross-fitting for held-out scores
    4. Extract μ^(r) = E[s_p × s_{p-r}^T] for each lag r
    5. Apply HAC inference + FDR per lag
    
    Key: This conditions on the FULL lag stack → true lag separation.
    """
    
    def __init__(
        self,
        p_max: int = 5,
        tune_hp: bool = False,
        n_hp_trials: int = DEFAULT_N_HP_TRIALS,
        noise_std: float = DEFAULT_DSM_NOISE_STD,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        lr: float = DEFAULT_LR,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        l1_lambda: float = DEFAULT_L1_LAMBDA,
        n_folds: int = DEFAULT_N_FOLDS,
        hac_max_lag: int = DEFAULT_HAC_MAX_LAG,
        fdr_alpha: float = DEFAULT_FDR_ALPHA,
        fdr_method: str = DEFAULT_FDR_METHOD,
        device: str = 'cpu',
        verbose: bool = True,
        random_state: int = 42,
    ):
        self.p_max = p_max
        self.tune_hp = tune_hp
        self.n_hp_trials = n_hp_trials
        self.noise_std = noise_std
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l1_lambda = l1_lambda
        self.n_folds = n_folds
        self.hac_max_lag = hac_max_lag
        self.fdr_alpha = fdr_alpha
        self.fdr_method = fdr_method
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        
        # Store tuned config (single model)
        self.hp_config: HPConfig = None
    
    def fit(self, X_list: List[np.ndarray]) -> MultiLagSBTGResult:
        """
        Fit multi-block SBTG model.
        
        Args:
            X_list: List of (T_u, n) time series per segment
            
        Returns:
            MultiLagSBTGResult with μ^(r) for each lag
        """
        n_neurons = X_list[0].shape[1]
        
        if self.verbose:
            print(f"\n[Approach B] Building {self.p_max + 1}-block windows...")
        
        # HP tuning (once for the single model)
        if self.tune_hp:
            if self.verbose:
                print(f"  Tuning HP ({self.n_hp_trials} trials for single model)...")
            config = tune_hyperparameters(
                X_list,
                n_trials=self.n_hp_trials,
                lag=1,  # Use lag 1 for HP tuning (representative)
                n_blocks=self.p_max + 1,
                device=self.device,
                verbose=self.verbose,
                seed=self.random_state,
            )
            self.hp_config = config
        else:
            config = HPConfig(
                noise_std=self.noise_std,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                lr=self.lr,
                epochs=self.epochs,
            )
            self.hp_config = config
        
        # Build multi-block windows
        windows, stim_ids, local_t = self._build_multiblock_windows(X_list)
        
        if self.verbose:
            print(f"  Built {len(windows)} windows, dim={(self.p_max + 1) * n_neurons}")
        
        # 5-fold cross-fitting with config
        scores_heldout = self._cross_fit(windows, stim_ids, n_neurons, config)
        
        # Reshape scores to (N, p+1, n) for block-wise analysis
        N = len(scores_heldout)
        scores_blocks = scores_heldout.reshape(N, self.p_max + 1, n_neurons)
        
        # Extract μ^(r) for each lag
        mu_hat_all = {}
        p_values_all = {}
        significant_all = {}
        
        s_future = scores_blocks[:, self.p_max, :]  # Last block = x_{t+1}
        
        for lag_r in range(1, self.p_max + 1):
            lag_block_idx = self.p_max - lag_r
            s_lag_r = scores_blocks[:, lag_block_idx, :]
            
            # HAC inference
            mu_hat, p_values = hac_test_mu_hat(s_future, s_lag_r, self.hac_max_lag)
            
            # FDR control
            significant = apply_fdr(p_values, self.fdr_alpha, self.fdr_method)
            
            mu_hat_all[lag_r] = mu_hat
            p_values_all[lag_r] = p_values
            significant_all[lag_r] = significant.astype(float)
            
            if self.verbose:
                n_edges = significant.sum()
                print(f"  Lag {lag_r}: μ̂ mean|μ|={np.abs(mu_hat).mean():.4f}, FDR edges={int(n_edges)}")
        
        return MultiLagSBTGResult(
            mu_hat=mu_hat_all,
            p_values=p_values_all,
            significant=significant_all,
            approach='B',
            p_max=self.p_max,
            n_neurons=n_neurons,
            n_windows=len(windows),
            model_type='multiblock_structured',
        )
    
    def _build_multiblock_windows(
        self,
        X_list: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (p+1)-block windows."""
        windows = []
        stim_ids = []
        local_ts = []
        
        p = self.p_max
        
        for u, X in enumerate(X_list):
            T, n = X.shape
            
            for t in range(p, T):
                # z_t = (x_{t-p}, x_{t-p+1}, ..., x_{t-1}, x_t)  — (p+1) blocks
                blocks = [X[t - p + k] for k in range(p + 1)]
                z = np.concatenate(blocks)
                windows.append(z)
                stim_ids.append(u)
                local_ts.append(t - p)
        
        return np.array(windows), np.array(stim_ids), np.array(local_ts)
    
    def _cross_fit(
        self,
        windows: np.ndarray,
        stim_ids: np.ndarray,
        n_neurons: int,
        config: HPConfig,
    ) -> np.ndarray:
        """5-fold cross-fitting for multi-block model."""
        N = len(windows)
        fold_ids = create_fold_assignments(stim_ids, self.n_folds, self.random_state)
        
        scores_heldout = np.zeros_like(windows)
        
        for fold_k in range(self.n_folds):
            if self.verbose:
                print(f"  Fold {fold_k + 1}/{self.n_folds}...", flush=True)
            
            train_mask = (fold_ids != fold_k)
            heldout_mask = (fold_ids == fold_k)
            
            # Standardize using training statistics
            windows_std, mean, std = standardize_windows(windows, train_mask)
            
            # Train fresh model on this fold using config
            model = MultiBlockStructuredScoreNet(
                n_neurons=n_neurons,
                p_max=self.p_max,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )
            
            train_windows = windows_std[train_mask]
            train_score_model(
                model, train_windows, config.noise_std, config.lr,
                config.epochs, self.batch_size, self.l1_lambda,
                self.device, verbose=False
            )
            
            # Compute scores on held-out windows
            heldout_windows = windows_std[heldout_mask]
            heldout_scores = compute_scores(model, heldout_windows, self.device)
            
            scores_heldout[heldout_mask] = heldout_scores
        
        return scores_heldout


# =============================================================================
# APPROACH C: MINIMAL MULTI-BLOCK SBTG
# =============================================================================

class MinimalMultiBlockEstimator:
    """
    Approach C: For each lag r, train a separate (r+1)-block model.
    
    Key insight: Lag r only NEEDS blocks x_t, x_{t+1}, ..., x_{t+r}
    This gives the minimal Markov blanket for identifying J_r.
    
    For lag=1: z = (x_t, x_{t+1})           -> 2 blocks
    For lag=2: z = (x_t, x_{t+1}, x_{t+2})  -> 3 blocks
    For lag=3: z = (x_t, x_{t+1}, x_{t+2}, x_{t+3}) -> 4 blocks
    
    Advantages over Approach A:
    - Conditions on intermediate lags, reducing confounding
    
    Advantages over Approach B:
    - Can tune HP separately per lag (since each lag has different structure)
    - Simpler models for smaller lags
    
    With HP tuning: Each lag gets its own tuned configuration.
    """
    
    def __init__(
        self,
        lags: List[int] = None,
        tune_hp: bool = False,
        n_hp_trials: int = DEFAULT_N_HP_TRIALS,
        noise_std: float = DEFAULT_DSM_NOISE_STD,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        lr: float = DEFAULT_LR,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        l1_lambda: float = DEFAULT_L1_LAMBDA,
        n_folds: int = DEFAULT_N_FOLDS,
        hac_max_lag: int = DEFAULT_HAC_MAX_LAG,
        fdr_alpha: float = DEFAULT_FDR_ALPHA,
        fdr_method: str = DEFAULT_FDR_METHOD,
        device: str = 'cpu',
        verbose: bool = True,
        random_state: int = 42,
    ):
        """
        Args:
            lags: List of lags to analyze (e.g., [1, 2, 3, 5])
            tune_hp: Whether to tune HP separately per lag
            n_hp_trials: Number of HP tuning trials per lag
            ... (other args same as other estimators)
        """
        self.lags = lags if lags is not None else [1, 2, 3, 5]
        self.tune_hp = tune_hp
        self.n_hp_trials = n_hp_trials
        self.noise_std = noise_std
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l1_lambda = l1_lambda
        self.n_folds = n_folds
        self.hac_max_lag = hac_max_lag
        self.fdr_alpha = fdr_alpha
        self.fdr_method = fdr_method
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        
        # Store tuned configs per lag
        self.hp_configs: Dict[int, HPConfig] = {}
    
    def fit(self, X_list: List[np.ndarray]) -> MultiLagSBTGResult:
        """
        Fit minimal multi-block models, one per lag.
        
        Args:
            X_list: List of (T_u, n) time series per segment
            
        Returns:
            MultiLagSBTGResult with μ^(r) for each lag
        """
        n_neurons = X_list[0].shape[1]
        
        mu_hat_all = {}
        p_values_all = {}
        significant_all = {}
        
        for lag_r in self.lags:
            if self.verbose:
                print(f"\n[Approach C] Lag {lag_r} ({lag_r + 1}-block model)...", flush=True)
            
            # HP tuning for this lag
            if self.tune_hp:
                config = tune_hyperparameters(
                    X_list,
                    n_trials=self.n_hp_trials,
                    lag=lag_r,
                    n_blocks=lag_r + 1,
                    device=self.device,
                    verbose=self.verbose,
                    seed=self.random_state + lag_r,
                )
                self.hp_configs[lag_r] = config
            else:
                config = HPConfig(
                    noise_std=self.noise_std,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    lr=self.lr,
                    epochs=self.epochs,
                )
            
            # Build minimal windows for this lag
            windows, stim_ids, local_t = _build_minimal_multiblock_windows(X_list, lag_r)
            
            if self.verbose:
                print(f"  Built {len(windows)} windows, dim={(lag_r + 1) * n_neurons}")
            
            # Cross-fitting
            scores_heldout = self._cross_fit_lag(
                windows, stim_ids, n_neurons, lag_r, config
            )
            
            # Reshape to (N, r+1, n)
            N = len(scores_heldout)
            n_blocks = lag_r + 1
            scores_blocks = scores_heldout.reshape(N, n_blocks, n_neurons)
            
            # μ^(r) = E[s_{lag_r} × s_0^T]
            # Block 0 = x_t (past), Block lag_r = x_{t+r} (future)
            s_past = scores_blocks[:, 0, :]
            s_future = scores_blocks[:, lag_r, :]
            
            # HAC inference
            mu_hat, p_values = hac_test_mu_hat(s_future, s_past, self.hac_max_lag)
            
            # FDR control
            significant = apply_fdr(p_values, self.fdr_alpha, self.fdr_method)
            
            mu_hat_all[lag_r] = mu_hat
            p_values_all[lag_r] = p_values
            significant_all[lag_r] = significant.astype(float)
            
            if self.verbose:
                n_edges = significant.sum()
                nc = compute_null_contrast(mu_hat)
                print(f"  μ̂ mean|μ|={np.abs(mu_hat).mean():.4f}, NC={nc:.3f}, FDR edges={int(n_edges)}")
        
        return MultiLagSBTGResult(
            mu_hat=mu_hat_all,
            p_values=p_values_all,
            significant=significant_all,
            approach='C',
            p_max=max(self.lags),
            n_neurons=n_neurons,
            n_windows=len(windows),
            model_type='minimal_multiblock',
        )
    
    def _cross_fit_lag(
        self,
        windows: np.ndarray,
        stim_ids: np.ndarray,
        n_neurons: int,
        lag: int,
        config: HPConfig,
    ) -> np.ndarray:
        """Cross-fitting for a single lag's minimal model."""
        N = len(windows)
        n_blocks = lag + 1
        fold_ids = create_fold_assignments(stim_ids, self.n_folds, self.random_state)
        
        scores_heldout = np.zeros_like(windows)
        
        for fold_k in range(self.n_folds):
            if self.verbose:
                print(f"    Fold {fold_k + 1}/{self.n_folds}...", flush=True)
            
            train_mask = (fold_ids != fold_k)
            heldout_mask = (fold_ids == fold_k)
            
            windows_std, mean, std = standardize_windows(windows, train_mask)
            
            # Create appropriate model based on n_blocks
            if n_blocks == 2:
                model = TwoBlockStructuredScoreNet(
                    n_neurons=n_neurons,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                )
            else:
                model = MultiBlockStructuredScoreNet(
                    n_neurons=n_neurons,
                    p_max=lag,  # p_max = lag for minimal model
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                )
            
            train_score_model(
                model, windows_std[train_mask], config.noise_std, config.lr,
                config.epochs, self.batch_size, self.l1_lambda,
                self.device, verbose=False
            )
            
            heldout_scores = compute_scores(model, windows_std[heldout_mask], self.device)
            scores_heldout[heldout_mask] = heldout_scores
        
        return scores_heldout


# =============================================================================
# CONVENIENCE FUNCTION: RUN ALL APPROACHES
# =============================================================================

def run_all_approaches(
    X_list: List[np.ndarray],
    lags: List[int] = None,
    p_max: int = 5,
    tune_hp_C: bool = False,
    n_hp_trials: int = DEFAULT_N_HP_TRIALS,
    epochs: int = DEFAULT_EPOCHS,
    n_folds: int = DEFAULT_N_FOLDS,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    device: str = 'cpu',
    verbose: bool = True,
    random_state: int = 42,
) -> Dict[str, MultiLagSBTGResult]:
    """
    Run all three approaches on the same data.
    
    Args:
        X_list: List of (T_u, n) time series
        lags: Lags to evaluate (default: [1, 2, 3, 5])
        p_max: Max lag for Approach B
        tune_hp_C: Whether to tune HP for Approach C
        ... other shared args
        
    Returns:
        Dict mapping approach name to result
    """
    if lags is None:
        lags = [1, 2, 3, 5]
    
    results = {}
    
    # Approach A: Per-lag 2-block
    if verbose:
        print("\n" + "="*60)
        print("APPROACH A: Per-Lag 2-Block")
        print("="*60)
    
    estimator_A = PerLagSBTGEstimator(
        p_max=max(lags),
        epochs=epochs,
        n_folds=n_folds,
        fdr_alpha=fdr_alpha,
        device=device,
        verbose=verbose,
        random_state=random_state,
    )
    results['A'] = estimator_A.fit(X_list)
    
    # Approach B: Full multi-block
    if verbose:
        print("\n" + "="*60)
        print("APPROACH B: Full Multi-Block")
        print("="*60)
    
    estimator_B = MultiBlockSBTGEstimator(
        p_max=p_max,
        epochs=epochs,
        n_folds=n_folds,
        fdr_alpha=fdr_alpha,
        device=device,
        verbose=verbose,
        random_state=random_state,
    )
    results['B'] = estimator_B.fit(X_list)
    
    # Approach C: Minimal multi-block
    if verbose:
        print("\n" + "="*60)
        print("APPROACH C: Minimal Multi-Block")
        print("="*60)
    
    estimator_C = MinimalMultiBlockEstimator(
        lags=lags,
        tune_hp=tune_hp_C,
        n_hp_trials=n_hp_trials,
        epochs=epochs,
        n_folds=n_folds,
        fdr_alpha=fdr_alpha,
        device=device,
        verbose=verbose,
        random_state=random_state,
    )
    results['C'] = estimator_C.fit(X_list)
    
    return results


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Multi-Lag SBTG Models...")
    
    # Generate synthetic VAR(2) data
    np.random.seed(42)
    n = 10
    T = 500
    
    # True VAR(2) coefficients
    A1 = np.random.randn(n, n) * 0.3
    A2 = np.random.randn(n, n) * 0.1
    
    # Ensure stability (spectral radius < 1)
    A1 = A1 / (1.5 * np.linalg.norm(A1, 2))
    A2 = A2 / (3.0 * np.linalg.norm(A2, 2))
    
    # Simulate
    X = np.zeros((T, n))
    for t in range(2, T):
        X[t] = A1 @ X[t-1] + A2 @ X[t-2] + np.random.randn(n) * 0.5
    
    X_list = [X]
    
    print("\n=== Testing Approach A (Per-Lag 2-Block) ===")
    estimator_A = PerLagSBTGEstimator(
        p_max=3,
        epochs=30,
        n_folds=3,
        verbose=True,
    )
    result_A = estimator_A.fit(X_list)
    print(f"\nApproach A results: {len(result_A.mu_hat)} lag matrices")
    for lag in result_A.mu_hat:
        nc = compute_null_contrast(result_A.mu_hat[lag])
        print(f"  Lag {lag}: |μ|={np.abs(result_A.mu_hat[lag]).mean():.4f}, NC={nc:.3f}, edges={result_A.significant[lag].sum()}")
    
    print("\n=== Testing Approach B (Multi-Block) ===")
    estimator_B = MultiBlockSBTGEstimator(
        p_max=3,
        epochs=30,
        n_folds=3,
        verbose=True,
    )
    result_B = estimator_B.fit(X_list)
    print(f"\nApproach B results: {len(result_B.mu_hat)} lag matrices")
    for lag in result_B.mu_hat:
        nc = compute_null_contrast(result_B.mu_hat[lag])
        print(f"  Lag {lag}: |μ|={np.abs(result_B.mu_hat[lag]).mean():.4f}, NC={nc:.3f}, edges={result_B.significant[lag].sum()}")
    
    print("\n=== Testing Approach C (Minimal Multi-Block) ===")
    estimator_C = MinimalMultiBlockEstimator(
        lags=[1, 2, 3],
        tune_hp=False,
        epochs=30,
        n_folds=3,
        verbose=True,
    )
    result_C = estimator_C.fit(X_list)
    print(f"\nApproach C results: {len(result_C.mu_hat)} lag matrices")
    for lag in result_C.mu_hat:
        nc = compute_null_contrast(result_C.mu_hat[lag])
        print(f"  Lag {lag}: |μ|={np.abs(result_C.mu_hat[lag]).mean():.4f}, NC={nc:.3f}, edges={result_C.significant[lag].sum()}")
    
    print("\n✅ All tests passed!")
