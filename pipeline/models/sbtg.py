"""
Score-based Temporal Graph (SBTG) – Structured Variant with Volatility Transfer Test

This module mirrors the structured SBTG implementation but replaces the legacy
energy test with the "Volatility Transfer Test". The goal is to detect edges
whose influence may flip sign over time by testing whether bursts of activity
("volatility") in neuron *i* precede bursts in neuron *j* regardless of the
instantaneous sign of the effect.

Workflow
========
1. Train a structured denoising score model (StructuredScoreNet) on windows
   ``z_t = [x_t, x_{t+lag}]`` of standardized data, where ``lag`` is configurable.
2. Mean Test (Dynamic Partial Correlation):
      ``Y_t = s_{1,j}(z_t) * s_{0,i}(z_t)``.
3. Volatility Transfer Test (replaces energy test):
      ``U_t = s_{1,j}(z_t)^2`` and ``V_t = s_{0,i}(z_t)^2``.
      Center each series and form ``Z_t = (U_t - \bar{U})(V_t - \bar{V})``.
      Apply a HAC (Newey–West) t-test to ``Z_t``. This estimates
      ``Cov(s_{1,j}^2, s_{0,i}^2)`` and is sensitive to sign-flipping effects.
4. Apply Benjamini–Hochberg / Benjamini–Yekutieli FDR control to the mean and
   volatility tests (volatility only on edges not already selected by the mean
   test).

Multi-Lag Analysis (NEW)
========================
The ``time_lag`` parameter controls the temporal offset between paired observations:
- ``time_lag=1`` (default): Adjacent pairs ``z_t = [x_t, x_{t+1}]`` (original behavior)
- ``time_lag=5``: 5-frame offset ``z_t = [x_t, x_{t+5}]`` (~1.25s at 4Hz)
- ``time_lag=20``: 20-frame offset ``z_t = [x_t, x_{t+20}]`` (~5s at 4Hz)

This allows analyzing connectivity at different temporal scales:
- Short lags capture fast/direct interactions
- Longer lags capture slower/indirect pathways and behavioral timescales

Implementation notes
====================
- DSM target for corruption ``y = z + sigma * eps`` uses ``target = -eps / sigma``.
- Train split options:
  - ``prefix``: first ``train_frac`` of pooled windows
  - ``per_stimulus_prefix``: first ``train_frac`` within each stimulus segment
  - ``random``: random subset of size ``floor(train_frac * N)``
  - ``odd_even``: per-stimulus alternating windows by local t parity
- Standardization uses training statistics only (mean/std from train windows).
- ``time_lag`` supports longer temporal offsets for multi-timescale analysis.

Dependencies: numpy, torch, scipy (for the normal CDF), and standard library modules.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm


# -------------------------------------------------------------------------
# Default hyperparameters (tuned for structured score version)
# -------------------------------------------------------------------------

DEFAULT_WINDOW_LENGTH: int = 2
DEFAULT_TIME_LAG: int = 1  # Time offset between x_t and x_{t+lag}; lag=1 for adjacent pairs
DEFAULT_SMOOTH_SIGMA: Optional[float] = None
DEFAULT_CAUSAL_SMOOTHING: bool = True  # Use causal smoothing to avoid future leakage

DEFAULT_DSM_HIDDEN_DIM: int = 128
DEFAULT_DSM_NUM_LAYERS: int = 3
DEFAULT_DSM_NOISE_STD: float = 0.1
DEFAULT_DSM_EPOCHS: int = 100
DEFAULT_DSM_BATCH_SIZE: int = 128
DEFAULT_DSM_LR: float = 1e-3
DEFAULT_TRAIN_FRAC: float = 0.7

# Statistical testing (HAC/FDR) hyperparameters
DEFAULT_HAC_MAX_LAG: int = 5
DEFAULT_FDR_ALPHA: float = 0.1
DEFAULT_FDR_METHOD: str = "bh"
DEFAULT_VOLATILITY_TEST: bool = True  # Toggle to disable the volatility test entirely

DEFAULT_STRUCTURED_HIDDEN_DIM: int = 64
DEFAULT_STRUCTURED_NUM_LAYERS: int = 2
DEFAULT_STRUCTURED_L1_LAMBDA: float = 0.0
DEFAULT_STRUCTURED_INIT_SCALE: float = 0.1

DEFAULT_COMPUTE_UNDIRECTED: bool = True

# NEW: training split defaults
DEFAULT_TRAIN_SPLIT: str = "prefix"       # {"prefix","per_stimulus_prefix","random","odd_even"}
DEFAULT_TRAIN_PARITY: str = "even"        # used only if train_split == "odd_even": {"even","odd"}

# Cross-fitting defaults (for held-out inference)
DEFAULT_INFERENCE_MODE: str = "cross_fit"  # {"in_sample", "cross_fit"}
DEFAULT_N_FOLDS: int = 5                   # Number of folds for cross-fitting


# -------------------------------------------------------------------------
# Utility: simple Gaussian smoothing for spike trains / counts
# -------------------------------------------------------------------------


def gaussian_smooth_1d(
    x: np.ndarray, 
    sigma: float, 
    truncate: float = 3.0,
    causal: bool = True
) -> np.ndarray:
    """
    Apply 1D Gaussian smoothing along axis 0.
    
    Args:
        x: Input array of shape (T, n)
        sigma: Standard deviation of Gaussian kernel
        truncate: Truncate kernel at this many sigmas
        causal: If True, use only past values (no future leakage).
                If False, use symmetric smoothing (can leak future info across splits).
    
    Returns:
        Smoothed array of same shape
    
    Note:
        When using train/test splits, causal=True should be used to avoid
        future information leaking into smoothed values near split boundaries.
    """
    if sigma is None or sigma <= 0.0:
        return x

    radius = int(truncate * sigma)
    if radius == 0:
        return x.copy()

    if causal:
        # Causal kernel: only past values (exponential-like decay)
        t = np.arange(0, radius + 1)
        kernel = np.exp(-0.5 * (t / sigma) ** 2)
        kernel /= kernel.sum()
        
        T, n = x.shape
        out = np.zeros_like(x, dtype=float)
        for j in range(n):
            # Causal convolution: output[t] depends only on x[t-k] for k >= 0
            # We need to flip the kernel and use 'full' mode, then slice
            conv_result = np.convolve(x[:, j], kernel[::-1], mode="full")
            # Take the first T elements (aligned with original time)
            out[:, j] = conv_result[:T]
        return out
    else:
        # Symmetric kernel (original behavior - can leak future info)
        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (t / sigma) ** 2)
        kernel /= kernel.sum()

        T, n = x.shape
        out = np.zeros_like(x, dtype=float)
        for j in range(n):
            out[:, j] = np.convolve(x[:, j], kernel, mode="same")
        return out


# -------------------------------------------------------------------------
# Structured score networks (energy-defined; score computed by autograd)
# -------------------------------------------------------------------------


class ScalarMLP(nn.Module):
    """MLP mapping R^n -> R (scalar energy)."""

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


class StructuredScoreNet(nn.Module):
    """Structured score derived from energy U(z) = g0(x_t) + g1(x_{t+1}) + x_{t+1}^T W x_t."""

    def __init__(
        self,
        n: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.g0 = ScalarMLP(n, hidden_dim, num_layers)
        self.g1 = ScalarMLP(n, hidden_dim, num_layers)
        self.W = nn.Parameter(torch.empty(n, n))
        nn.init.uniform_(self.W, -init_scale, init_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, d = z.shape
        if d != 2 * self.n:
            raise ValueError(f"StructuredScoreNet expects input_dim=2n, got d={d}, n={self.n}")

        x0, x1 = torch.split(z, self.n, dim=-1)
        x0 = x0.clone().requires_grad_(True)
        x1 = x1.clone().requires_grad_(True)

        e0 = self.g0(x0).sum()
        e1 = self.g1(x1).sum()

        cross = ((x1 @ self.W) * x0).sum()
        U = e0 + e1 + cross

        need_graph = self.training  # True during training so gradients flow to params
        grad_x0, grad_x1 = torch.autograd.grad(
            U,
            (x0, x1),
            create_graph=need_graph,
            retain_graph=need_graph,
        )

        s0 = -grad_x0
        s1 = -grad_x1
        return torch.cat([s0, s1], dim=-1)


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


class FeatureBilinearScoreNet(nn.Module):
    """Feature-bilinear coupling U(z) = g0(x0) + g1(x1) + psi(x1)^T W phi(x0)."""

    def __init__(
        self,
        n: int,
        r: int = 16,
        hidden_dim_g: int = 64,
        num_layers_g: int = 2,
        hidden_dim_feat: int = 64,
        num_layers_feat: int = 2,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.r = r
        self.g0 = ScalarMLP(n, hidden_dim_g, num_layers_g)
        self.g1 = ScalarMLP(n, hidden_dim_g, num_layers_g)
        self.phi = VectorMLP(n, r, hidden_dim_feat, num_layers_feat)
        self.psi = VectorMLP(n, r, hidden_dim_feat, num_layers_feat)
        self.W = nn.Parameter(torch.empty(r, r))
        nn.init.uniform_(self.W, -init_scale, init_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, d = z.shape
        if d != 2 * self.n:
            raise ValueError(f"FeatureBilinearScoreNet expects input_dim=2n, got d={d}")

        x0, x1 = torch.split(z, self.n, dim=-1)
        x0 = x0.clone().requires_grad_(True)
        x1 = x1.clone().requires_grad_(True)

        e0 = self.g0(x0).sum()
        e1 = self.g1(x1).sum()

        p = self.phi(x0)  # (B, r)
        q = self.psi(x1)  # (B, r)
        cross = ((q @ self.W) * p).sum()

        U = e0 + e1 + cross

        need_graph = self.training
        grad_x0, grad_x1 = torch.autograd.grad(
            U,
            (x0, x1),
            create_graph=need_graph,
            retain_graph=need_graph,
        )

        s0 = -grad_x0
        s1 = -grad_x1
        return torch.cat([s0, s1], dim=-1)


class GateNet(nn.Module):
    """MLP mapping R^n -> R^K (logits for gating)."""

    def __init__(self, input_dim: int, num_regimes: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        layers = []
        dim_in = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.SiLU())
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, num_regimes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegimeGatedScoreNet(nn.Module):
    """Regime-gated coupling U(z) = g0(x0) + g1(x1) + x1^T (sum_k alpha_k(x0) W_k) x0."""

    def __init__(
        self,
        n: int,
        num_regimes: int = 2,
        hidden_dim_g: int = 64,
        num_layers_g: int = 2,
        hidden_dim_gate: int = 64,
        num_layers_gate: int = 2,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.K = num_regimes
        self.g0 = ScalarMLP(n, hidden_dim_g, num_layers_g)
        self.g1 = ScalarMLP(n, hidden_dim_g, num_layers_g)
        self.gate = GateNet(n, num_regimes, hidden_dim_gate, num_layers_gate)

        self.W = nn.Parameter(torch.empty(num_regimes, n, n))  # (K, n, n)
        nn.init.uniform_(self.W, -init_scale, init_scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, d = z.shape
        if d != 2 * self.n:
            raise ValueError(f"RegimeGatedScoreNet expects input_dim=2n, got d={d}")

        x0, x1 = torch.split(z, self.n, dim=-1)
        x0 = x0.clone().requires_grad_(True)
        x1 = x1.clone().requires_grad_(True)

        e0 = self.g0(x0).sum()
        e1 = self.g1(x1).sum()

        logits = self.gate(x0)
        alpha = torch.softmax(logits, dim=-1)  # (B, K)

        W_eff = torch.einsum("bk,kij->bij", alpha, self.W)  # (B, n, n)
        cross = torch.einsum("bi,bij,bj->", x1, W_eff, x0)  # scalar

        U = e0 + e1 + cross

        need_graph = self.training
        grad_x0, grad_x1 = torch.autograd.grad(
            U,
            (x0, x1),
            create_graph=need_graph,
            retain_graph=need_graph,
        )

        s0 = -grad_x0
        s1 = -grad_x1
        return torch.cat([s0, s1], dim=-1)


# -------------------------------------------------------------------------
# HAC (Newey–West) variance estimation
# -------------------------------------------------------------------------


def newey_west_variance(y: np.ndarray, max_lag: int) -> float:
    """Compute HAC variance estimate with Bartlett weights."""
    y = np.asarray(y, dtype=float)
    N = len(y)
    if N < 2:
        return float(np.var(y) + 1e-12)

    y_centered = y - y.mean()
    gamma0 = np.mean(y_centered * y_centered)
    var = gamma0

    L = min(max_lag, N - 1)
    for ell in range(1, L + 1):
        cov = np.mean(y_centered[ell:] * y_centered[:-ell])
        weight = 1.0 - ell / (L + 1)
        var += 2.0 * weight * cov

    return float(max(var, 1e-12))


# -------------------------------------------------------------------------
# FDR control (BH / BY)
# -------------------------------------------------------------------------


def fdr_control(pvals: np.ndarray, alpha: float, method: str = "bh") -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    if m == 0:
        return np.zeros_like(pvals, dtype=bool)

    order = np.argsort(pvals)
    p_sorted = pvals[order]

    if method.lower() == "by":
        harmonic_m = np.sum(1.0 / np.arange(1, m + 1))
        alpha_adj = alpha / harmonic_m
    else:
        alpha_adj = alpha

    thresholds = alpha_adj * (np.arange(1, m + 1) / m)
    below = p_sorted <= thresholds
    reject_sorted = np.zeros_like(p_sorted, dtype=bool)
    if np.any(below):
        k_max = np.max(np.where(below)[0])
        reject_sorted[: k_max + 1] = True

    reject = np.zeros_like(pvals, dtype=bool)
    reject[order] = reject_sorted
    return reject


# -------------------------------------------------------------------------
# Result container
# -------------------------------------------------------------------------


@dataclass
class SBTGVolatilityResult:
    sign_adj: np.ndarray
    volatility_adj: np.ndarray
    undirected_adj: Optional[np.ndarray]
    p_mean: np.ndarray
    p_volatility: Optional[np.ndarray]
    mu_hat: np.ndarray
    volatility_stat: Optional[np.ndarray]
    model_type: str = "linear"
    W_param: Optional[np.ndarray] = None
    gate_alpha_eval: Optional[np.ndarray] = None
    W_eff_mean: Optional[np.ndarray] = None


# -------------------------------------------------------------------------
# Estimator
# -------------------------------------------------------------------------


class SBTGStructuredVolatilityEstimator:
    """Structured SBTG estimator that augments the mean test with a volatility transfer test.

    Minimal additions:
    - train_split selection (prefix / per_stimulus_prefix / random / odd_even)
    - training-stat-only standardization
    - corrected DSM target scaling
    """

    def __init__(
        self,
        window_length: int = DEFAULT_WINDOW_LENGTH,
        time_lag: int = DEFAULT_TIME_LAG,  # NEW: configurable temporal lag
        smooth_sigma: Optional[float] = DEFAULT_SMOOTH_SIGMA,
        causal_smoothing: bool = DEFAULT_CAUSAL_SMOOTHING,
        dsm_hidden_dim: int = DEFAULT_DSM_HIDDEN_DIM,
        dsm_num_layers: int = DEFAULT_DSM_NUM_LAYERS,
        dsm_noise_std: float = DEFAULT_DSM_NOISE_STD,
        dsm_epochs: int = DEFAULT_DSM_EPOCHS,
        dsm_batch_size: int = DEFAULT_DSM_BATCH_SIZE,
        dsm_lr: float = DEFAULT_DSM_LR,
        train_frac: float = DEFAULT_TRAIN_FRAC,
        hac_max_lag: int = DEFAULT_HAC_MAX_LAG,
        fdr_alpha: float = DEFAULT_FDR_ALPHA,
        fdr_method: str = DEFAULT_FDR_METHOD,
        volatility_test: bool = DEFAULT_VOLATILITY_TEST,
        structured_hidden_dim: int = DEFAULT_STRUCTURED_HIDDEN_DIM,
        structured_num_layers: int = DEFAULT_STRUCTURED_NUM_LAYERS,
        structured_l1_lambda: float = DEFAULT_STRUCTURED_L1_LAMBDA,
        structured_init_scale: float = DEFAULT_STRUCTURED_INIT_SCALE,
        compute_undirected: bool = DEFAULT_COMPUTE_UNDIRECTED,
        # Model options
        model_type: str = "linear",  # "linear", "feature_bilinear", "regime_gated"
        feature_dim: int = 16,
        feature_hidden_dim: int = 64,
        feature_num_layers: int = 2,
        num_regimes: int = 2,
        gate_hidden_dim: int = 64,
        gate_num_layers: int = 2,
        gate_entropy_lambda: float = 0.0,
        # NEW: split controls
        train_split: str = DEFAULT_TRAIN_SPLIT,
        train_parity: str = DEFAULT_TRAIN_PARITY,
        # Cross-fitting controls (critical for valid p-values)
        inference_mode: str = DEFAULT_INFERENCE_MODE,
        n_folds: int = DEFAULT_N_FOLDS,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        # Validate time_lag (must be positive integer)
        if not isinstance(time_lag, int) or time_lag < 1:
            raise ValueError(f"time_lag must be a positive integer, got {time_lag}")
        
        # Window length is always 2 (x_t and x_{t+lag}), but we allow any time_lag
        if window_length != 2:
            raise ValueError("Structured SBTG requires window_length=2 (pairs of timepoints).")
        if not (0.0 < train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1).")
        if dsm_noise_std <= 0:
            raise ValueError("dsm_noise_std must be > 0 for Gaussian corruption y=z+sigma*eps.")

        self.window_length = window_length
        self.time_lag = time_lag  # NEW: temporal offset between x_t and x_{t+lag}
        self.smooth_sigma = smooth_sigma
        self.causal_smoothing = causal_smoothing

        self.dsm_hidden_dim = dsm_hidden_dim
        self.dsm_num_layers = dsm_num_layers
        self.dsm_noise_std = dsm_noise_std
        self.dsm_epochs = dsm_epochs
        self.dsm_batch_size = dsm_batch_size
        self.dsm_lr = dsm_lr
        self.train_frac = train_frac

        self.hac_max_lag = hac_max_lag
        self.fdr_alpha = fdr_alpha
        self.fdr_method = fdr_method.lower()
        self.volatility_test = volatility_test

        self.structured_hidden_dim = structured_hidden_dim
        self.structured_num_layers = structured_num_layers
        self.structured_l1_lambda = structured_l1_lambda
        self.structured_init_scale = structured_init_scale

        self.compute_undirected = compute_undirected

        self.model_type = model_type
        self.feature_dim = feature_dim
        self.feature_hidden_dim = feature_hidden_dim
        self.feature_num_layers = feature_num_layers
        self.num_regimes = num_regimes
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_num_layers = gate_num_layers
        self.gate_entropy_lambda = gate_entropy_lambda

        self.train_split = train_split.lower()
        self.train_parity = train_parity.lower()
        if self.train_split not in {"prefix", "per_stimulus_prefix", "random", "odd_even"}:
            raise ValueError(
                "train_split must be one of: {'prefix','per_stimulus_prefix','random','odd_even'}"
            )
        if self.train_parity not in {"even", "odd"}:
            raise ValueError("train_parity must be one of: {'even','odd'}")
            
        # Strict logic check for FDR method (Audit Blocker B)
        # Prevent "fdr_by" string literal from falling through to BH
        valid_fdr = {"bh", "by", "benjamini-hochberg", "benjamini-yekutieli"}
        if self.fdr_method not in valid_fdr:
             raise ValueError(f"Invalid fdr_method: '{self.fdr_method}'. Must be one of {valid_fdr}")

        # Cross-fitting validation
        self.inference_mode = inference_mode.lower()
        if self.inference_mode not in {"in_sample", "cross_fit"}:
            raise ValueError("inference_mode must be one of: {'in_sample', 'cross_fit'}")
        self.n_folds = n_folds
        if n_folds < 2 and self.inference_mode == "cross_fit":
            raise ValueError("n_folds must be >= 2 for cross_fit inference mode")

        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        if device is None:
            # Auto-detect best available device: CUDA > MPS > CPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.verbose = verbose
        self.model: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None
        self.n_neurons: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], skip_edge_tests: bool = False) -> SBTGVolatilityResult:
        if isinstance(X, np.ndarray):
            X_list = [X]
        else:
            X_list = X

        # Build RAW windows (no standardization here)
        windows_raw, stim_ids, local_t = self._build_windows_raw(X_list)
        N_total, d = windows_raw.shape
        if self.verbose:
            print(f"[SBTG-Struct] Total windows: {N_total}, window dimension d={d}")

        self.input_dim = d
        self.n_neurons = d // 2

        if self.inference_mode == "cross_fit":
            # Cross-fitted inference: train on K-1 folds, compute scores on held-out fold
            # This ensures p-values are computed on truly held-out scores
            scores_heldout, windows_all = self._cross_fit_inference(
                windows_raw, stim_ids, local_t, N_total
            )
            if skip_edge_tests:
                 result = SBTGVolatilityResult(
                     scores=scores_heldout, 
                     p_values=np.zeros((self.n_neurons, self.n_neurons)), 
                     q_values=np.zeros((self.n_neurons, self.n_neurons)),
                     sign_adj=np.zeros((self.n_neurons, self.n_neurons)),
                     mu_hat=np.zeros(d),
                     W_param=np.zeros((self.n_neurons, self.n_neurons))
                 )
            else:
                result = self._edge_tests(scores_heldout)
        else:
            # Legacy in-sample mode (original behavior, p-values may be anti-conservative)
            if self.verbose:
                print(
                    "[SBTG-Struct] WARNING: Using 'in_sample' inference mode. "
                    "P-values may be anti-conservative due to train/test overlap in edge testing."
                )
            
            # Choose training mask according to split option
            train_mask = self._make_train_mask(stim_ids=stim_ids, local_t=local_t, N_total=N_total)
            n_train = int(train_mask.sum())
            if n_train < 2:
                raise ValueError("Training split produced too few windows; adjust train_frac/split.")
            if self.verbose:
                print(
                    f"[SBTG-Struct] train_split='{self.train_split}' produced "
                    f"{n_train}/{N_total} train windows."
                )

            # Standardize using TRAINING statistics only; apply to all windows
            windows_all = self._standardize_with_train_stats(windows_raw, train_mask)

            # Train DSM on standardized train subset; evaluate scores on all standardized windows
            self._train_dsm(windows_all, train_mask)
            scores_all = self._compute_scores(windows_all)

            if skip_edge_tests:
                 result = SBTGVolatilityResult(
                     scores=scores_all, 
                     p_values=np.zeros((self.n_neurons, self.n_neurons)), 
                     q_values=np.zeros((self.n_neurons, self.n_neurons)),
                     sign_adj=np.zeros((self.n_neurons, self.n_neurons)),
                     mu_hat=np.zeros(d),
                     W_param=np.zeros((self.n_neurons, self.n_neurons))
                 )
            else:
                result = self._edge_tests(scores_all)

        # Export learned coupling parameters (optional diagnostics)
        result.model_type = self.model_type
        if self.model is not None:
            if self.model_type in {"linear", "feature_bilinear"}:
                result.W_param = self.model.W.detach().cpu().numpy()
            elif self.model_type == "regime_gated":
                result.W_param = self.model.W.detach().cpu().numpy()

                # Compute alpha_eval and mean effective coupling W_eff_mean on all windows
                self.model.eval()
                z_tensor = torch.from_numpy(windows_all.astype(np.float32)).to(self.device)
                n = self.n_neurons
                alphas = []
                batch_size = self.dsm_batch_size
                N = z_tensor.size(0)

                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    z_batch = z_tensor[start:end]
                    x0_batch = z_batch[:, :n]
                    logits = self.model.gate(x0_batch)
                    alpha_batch = torch.softmax(logits, dim=-1)
                    alphas.append(alpha_batch.detach().cpu().numpy())

                alpha_eval = np.concatenate(alphas, axis=0)
                result.gate_alpha_eval = alpha_eval

                W_np = result.W_param  # (K, n, n)
                mean_alpha = alpha_eval.mean(axis=0)  # (K,)
                result.W_eff_mean = np.einsum("k,kij->ij", mean_alpha, W_np)

        return result

    # ------------------------------------------------------------------
    # Helpers: windowing / splits / standardization
    # ------------------------------------------------------------------

    def _build_windows_raw(self, X_list: List[np.ndarray]) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Construct raw windows z_t = [x_t, x_{t+lag}] pooled across stimuli.
        
        The time_lag parameter controls the temporal offset:
        - lag=1: Adjacent pairs (original behavior) → z_t = [x_t, x_{t+1}]
        - lag=5: 5-frame offset → z_t = [x_t, x_{t+5}]
        - lag=L: L-frame offset → z_t = [x_t, x_{t+L}]
        
        Returns:
          windows_raw: (N, 2n) where N = sum(T_u - lag) across stimuli
          stim_ids   : (N,) stimulus id per window
          local_t    : (N,) local time index t (0..T_u-lag-1) within each stimulus
        """
        lag = self.time_lag
        windows = []
        stim_ids = []
        local_ts = []

        for u, X in enumerate(X_list):
            X = np.asarray(X, dtype=float)
            if X.ndim != 2:
                raise ValueError("Each stimulus array must be 2D (T_u, n).")

            X_sm = gaussian_smooth_1d(X, self.smooth_sigma, causal=self.causal_smoothing) if self.smooth_sigma else X
            T_u, n = X_sm.shape
            
            # Need at least lag+1 timepoints to form one window
            if T_u < lag + 1:
                if self.verbose:
                    print(f"[SBTG-Struct] Skipping stimulus {u}: T_u={T_u} < lag+1={lag+1}")
                continue

            # z_t for t=0..T_u-lag-1: pairs (x_t, x_{t+lag})
            z_u = np.stack(
                [np.concatenate([X_sm[t], X_sm[t + lag]]) for t in range(T_u - lag)],
                axis=0,
            )
            windows.append(z_u)
            stim_ids.append(np.full(z_u.shape[0], u, dtype=int))
            local_ts.append(np.arange(z_u.shape[0], dtype=int))

        if not windows:
            raise ValueError(
                f"No valid windows constructed with time_lag={lag}. "
                f"Check that input sequences have length > {lag}."
            )

        windows_all = np.concatenate(windows, axis=0)
        stim_ids_all = np.concatenate(stim_ids, axis=0)
        local_t_all = np.concatenate(local_ts, axis=0)
        
        if self.verbose and lag > 1:
            print(f"[SBTG-Struct] Built {len(windows_all)} windows with time_lag={lag}")
        
        return windows_all, stim_ids_all, local_t_all

        windows_all = np.concatenate(windows, axis=0)
        stim_ids_all = np.concatenate(stim_ids, axis=0)
        local_t_all = np.concatenate(local_ts, axis=0)
        return windows_all, stim_ids_all, local_t_all

    def _make_train_mask(self, stim_ids: np.ndarray, local_t: np.ndarray, N_total: int) -> np.ndarray:
        """
        Produce a boolean mask of length N_total indicating which windows are used for training.
        The evaluation series (HAC/FDR) is still computed on all windows unless you later
        choose to restrict it; here we only control training selection and standardization stats.
        """
        rng = np.random.default_rng(self.random_state)
        train_mask = np.zeros(N_total, dtype=bool)

        if self.train_split == "prefix":
            N_train = int(self.train_frac * N_total)
            train_mask[:N_train] = True

        elif self.train_split == "per_stimulus_prefix":
            # First floor(train_frac * N_u) windows within each stimulus segment
            for u in np.unique(stim_ids):
                idx_u = np.where(stim_ids == u)[0]
                N_u = idx_u.size
                N_train_u = int(self.train_frac * N_u)
                train_mask[idx_u[:N_train_u]] = True

        elif self.train_split == "random":
            N_train = int(self.train_frac * N_total)
            chosen = rng.choice(N_total, size=N_train, replace=False)
            train_mask[chosen] = True

        elif self.train_split == "odd_even":
            # Alternating windows per stimulus by local_t parity (non-overlapping windows within split)
            want_even = (self.train_parity == "even")
            parity_mask = (local_t % 2 == 0) if want_even else (local_t % 2 == 1)
            train_mask[parity_mask] = True

        else:
            raise ValueError(f"Unknown train_split: {self.train_split}")

        return train_mask

    def _standardize_with_train_stats(self, windows_raw: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
        """
        Standardize all windows using mean/std computed on training windows only.
        """
        train_windows = windows_raw[train_mask]
        mean = np.nanmean(train_windows, axis=0, keepdims=True)
        std = np.nanstd(train_windows, axis=0, keepdims=True) + 1e-8
        windows_std = (windows_raw - mean) / std
        return np.nan_to_num(windows_std, nan=0.0)

    def _cross_fit_inference(
        self,
        windows_raw: np.ndarray,
        stim_ids: np.ndarray,
        local_t: np.ndarray,
        N_total: int
    ) -> (np.ndarray, np.ndarray):
        """
        Cross-fitted inference: for each fold, train on other folds and compute
        scores on held-out fold. This ensures p-values are computed on truly
        held-out scores, making FDR control valid.
        
        Returns:
            scores_heldout: (N_total, 2n) held-out scores for all windows
            windows_std: (N_total, 2n) standardized windows (using full-data stats for reference)
        """
        rng = np.random.default_rng(self.random_state)
        n = self.n_neurons
        d = windows_raw.shape[1]
        
        # Create fold assignments respecting segment boundaries
        fold_ids = self._create_fold_assignments(stim_ids, N_total, rng)
        
        if self.verbose:
            fold_sizes = [np.sum(fold_ids == k) for k in range(self.n_folds)]
            print(f"[SBTG-Struct] Cross-fitting with {self.n_folds} folds: {fold_sizes}")
        
        # Store held-out scores
        scores_heldout = np.zeros((N_total, d), dtype=np.float32)
        
        for fold_k in range(self.n_folds):
            if self.verbose:
                print(f"[SBTG-Struct] Training fold {fold_k + 1}/{self.n_folds}...")
            
            # Training mask: all folds except fold_k
            train_mask_k = (fold_ids != fold_k)
            heldout_mask_k = (fold_ids == fold_k)
            
            n_train_k = int(train_mask_k.sum())
            n_heldout_k = int(heldout_mask_k.sum())
            
            if n_train_k < 2:
                raise ValueError(f"Fold {fold_k} has insufficient training windows ({n_train_k})")
            if n_heldout_k == 0:
                continue  # Skip empty folds
            
            # Standardize using THIS fold's training statistics
            windows_k = self._standardize_with_train_stats(windows_raw, train_mask_k)
            
            # Train a fresh model on this fold's training data
            self._train_dsm(windows_k, train_mask_k)
            
            # Compute scores ONLY on held-out windows
            heldout_windows_k = windows_k[heldout_mask_k]
            scores_k = self._compute_scores(heldout_windows_k)
            
            # Store held-out scores at the appropriate indices
            scores_heldout[heldout_mask_k] = scores_k
        
        # For reference, also compute full-data standardized windows
        # (using all data for mean/std, just for diagnostics)
        mean_all = np.nanmean(windows_raw, axis=0, keepdims=True)
        std_all = np.nanstd(windows_raw, axis=0, keepdims=True) + 1e-8
        windows_std = np.nan_to_num((windows_raw - mean_all) / std_all, nan=0.0)
        
        if self.verbose:
            print(f"[SBTG-Struct] Cross-fitting complete. Held-out scores: {scores_heldout.shape}")
        
        return scores_heldout, windows_std

    def _create_fold_assignments(
        self,
        stim_ids: np.ndarray,
        N_total: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Create fold assignments that respect segment (stimulus) boundaries.
        
        We assign folds in a strided manner within each stimulus segment to ensure
        temporal locality is preserved and each fold has windows from all stimuli.
        
        Returns:
            fold_ids: (N_total,) array with fold assignment (0 to n_folds-1) for each window
        """
        fold_ids = np.zeros(N_total, dtype=int)
        
        for u in np.unique(stim_ids):
            idx_u = np.where(stim_ids == u)[0]
            N_u = idx_u.size
            
            # Strided assignment within this segment: 0,1,2,...,K-1,0,1,2,...
            local_folds = np.arange(N_u) % self.n_folds
            fold_ids[idx_u] = local_folds
        
        return fold_ids

    # ------------------------------------------------------------------
    # DSM training / scoring
    # ------------------------------------------------------------------

    def _train_dsm(self, windows_all: np.ndarray, train_mask: np.ndarray) -> None:
        """
        Train DSM using only the training subset indicated by train_mask.
        """
        N_total, d = windows_all.shape
        z_train = windows_all[train_mask]
        N_train = z_train.shape[0]
        if N_train < 1:
            raise ValueError("Not enough windows for DSM training.")

        if self.verbose:
            print(f"[SBTG-Struct] Training DSM on {N_train} windows (d={d})")

        dataset = TensorDataset(torch.from_numpy(z_train.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.dsm_batch_size, shuffle=True)

        n = d // 2

        if self.model_type == "linear":
            self.model = StructuredScoreNet(
                n=n,
                hidden_dim=self.structured_hidden_dim,
                num_layers=self.structured_num_layers,
                init_scale=self.structured_init_scale,
            ).to(self.device)
        elif self.model_type == "feature_bilinear":
            self.model = FeatureBilinearScoreNet(
                n=n,
                r=self.feature_dim,
                hidden_dim_g=self.structured_hidden_dim,
                num_layers_g=self.structured_num_layers,
                hidden_dim_feat=self.feature_hidden_dim,
                num_layers_feat=self.feature_num_layers,
                init_scale=self.structured_init_scale,
            ).to(self.device)
        elif self.model_type == "regime_gated":
            self.model = RegimeGatedScoreNet(
                n=n,
                num_regimes=self.num_regimes,
                hidden_dim_g=self.structured_hidden_dim,
                num_layers_g=self.structured_num_layers,
                hidden_dim_gate=self.gate_hidden_dim,
                num_layers_gate=self.gate_num_layers,
                init_scale=self.structured_init_scale,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.dsm_lr)
        sigma = float(self.dsm_noise_std)
        l1_lambda = float(self.structured_l1_lambda)

        self.model.train()
        for epoch in range(self.dsm_epochs):
            epoch_loss = 0.0
            total_seen = 0
            for (z_batch,) in loader:
                z_batch = z_batch.to(self.device)

                eps = torch.randn_like(z_batch)
                y_noisy = z_batch + sigma * eps

                # ------------------------------------------------------------------
                # FIX (minimal): correct conditional score for y = z + sigma * eps
                #   ∇_y log p(y|z) = -(y-z)/sigma^2 = -(sigma*eps)/sigma^2 = -eps/sigma
                # ------------------------------------------------------------------
                target = -eps / sigma

                pred = self.model(y_noisy)
                loss_dsm = ((pred - target) ** 2).mean()
                loss = loss_dsm
                
                if torch.isnan(loss):
                    raise ValueError(f"DSM loss is NaN at epoch {epoch}")

                if l1_lambda > 0.0:
                    # Penalize coupling parameter W (including regime-gated W which is (K,n,n))
                    if hasattr(self.model, "W"):
                        loss = loss + l1_lambda * self.model.W.abs().sum()

                if self.model_type == "regime_gated" and self.gate_entropy_lambda != 0.0:
                    x0_batch = z_batch[:, :n]
                    logits = self.model.gate(x0_batch)
                    alpha = torch.softmax(logits, dim=-1)
                    entropy_term = (alpha * torch.log(alpha + 1e-9)).sum(dim=-1).mean()
                    loss = loss + self.gate_entropy_lambda * entropy_term

                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                bsz = z_batch.size(0)
                epoch_loss += loss_dsm.detach().item() * bsz
                total_seen += bsz

            if total_seen > 0:
                epoch_loss /= total_seen
            if self.verbose and ((epoch + 1) % max(1, self.dsm_epochs // 10) == 0):
                print(f"[SBTG-Struct] DSM epoch {epoch + 1}/{self.dsm_epochs}, loss={epoch_loss:.6f}")

    def _compute_scores(self, windows_all: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DSM model not trained. Call fit() first.")

        self.model.eval()
        z_tensor = torch.from_numpy(windows_all.astype(np.float32)).to(self.device)
        scores = []
        N = z_tensor.size(0)
        batch_size = self.dsm_batch_size
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            z_batch = z_tensor[start:end]
            s_batch = self.model(z_batch)
            scores.append(s_batch.detach().cpu().numpy())
        return np.concatenate(scores, axis=0)

    # ------------------------------------------------------------------
    # Edge-wise testing
    # ------------------------------------------------------------------

    def _edge_tests(self, scores_all: np.ndarray) -> SBTGVolatilityResult:
        N, d = scores_all.shape
        if d % 2 != 0:
            raise ValueError("Scores dimension must be 2 * n for window_length=2.")
        n = d // 2

        s0 = scores_all[:, :n]
        s1 = scores_all[:, n:]

        # Mean series Y_t(j,i) = s1_j * s0_i
        Y = np.einsum("tj,ti->tji", s1, s0)
        mu_hat = Y.mean(axis=0)

        if self.volatility_test:
            U = s1 ** 2
            V = s0 ** 2
            U_mean = U.mean(axis=0)
            V_mean = V.mean(axis=0)
        else:
            U = V = U_mean = V_mean = None

        p_mean = np.ones((n, n), dtype=float)
        p_vol = np.ones((n, n), dtype=float) if self.volatility_test else None
        vol_stat = np.zeros((n, n), dtype=float) if self.volatility_test else None

        for j in range(n):
            for i in range(n):
                if i == j:
                    p_mean[j, i] = 1.0
                    if self.volatility_test:
                        p_vol[j, i] = 1.0
                    continue

                y_series = Y[:, j, i]
                sigma2 = newey_west_variance(y_series, self.hac_max_lag)
                t_stat = np.sqrt(N) * mu_hat[j, i] / np.sqrt(sigma2)
                p_mean[j, i] = 2.0 * (1.0 - norm.cdf(abs(t_stat)))

                if self.volatility_test:
                    z_series = (U[:, j] - U_mean[j]) * (V[:, i] - V_mean[i])
                    mean_z = z_series.mean()
                    vol_stat[j, i] = mean_z
                    sigma2_z = newey_west_variance(z_series, self.hac_max_lag)
                    t_z = np.sqrt(N) * mean_z / np.sqrt(sigma2_z)
                    p_vol[j, i] = 2.0 * (1.0 - norm.cdf(abs(t_z)))

        mask_offdiag = ~np.eye(n, dtype=bool)

        # Mean FDR selection
        reject_mean_vec = fdr_control(p_mean[mask_offdiag], self.fdr_alpha, self.fdr_method)
        reject_mean = np.zeros_like(p_mean, dtype=bool)
        reject_mean[mask_offdiag] = reject_mean_vec

        sign_adj = np.zeros((n, n), dtype=int)
        sign_adj[reject_mean] = np.sign(mu_hat[reject_mean]).astype(int)

        # Volatility FDR selection on eligible edges only (not already selected by mean)
        if self.volatility_test:
            eligible = mask_offdiag & (~reject_mean)
            reject_vol_vec = fdr_control(p_vol[eligible], self.fdr_alpha, self.fdr_method)
            volatility_adj = np.zeros((n, n), dtype=bool)
            volatility_adj[eligible] = reject_vol_vec
        else:
            volatility_adj = np.zeros((n, n), dtype=bool)

        if self.compute_undirected:
            presence = (sign_adj != 0) | volatility_adj
            undirected_adj = np.logical_or(presence, presence.T)
        else:
            undirected_adj = None

        return SBTGVolatilityResult(
            sign_adj=sign_adj,
            volatility_adj=volatility_adj,
            undirected_adj=undirected_adj,
            p_mean=p_mean,
            p_volatility=p_vol,
            mu_hat=mu_hat,
            volatility_stat=vol_stat,
        )


# End of sbtg_main.py
