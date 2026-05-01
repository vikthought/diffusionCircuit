"""
SyntheticTestingUtils – data generators, baseline wrappers, and evaluation helpers
for the SBTG synthetic benchmark suite.

This module provides the building blocks used by SyntheticTesting.py:

1. **Synthetic Data Generators** – create multivariate time series with known
   directed graphs at lag 1 (and optionally lag 2).
2. **Baseline Wrappers** – unified interface for classical causal discovery
   methods (VAR-LASSO, VAR-Ridge, LiNGAM, Poisson-GLM, PCMCI+, NOTEARS,
   DYNOTEARS).
3. **Evaluation Helpers** – edge-level metrics (F1, AUROC, PR-AUC) for both
   weighted score matrices and binary adjacencies.
4. **SBTG HP Search** – Optuna-based hyperparameter tuning with null contrast
   objective.

Synthetic Data Families
========================
Each generator returns ``(X_list, truth_dict)`` where ``X_list`` is a list of
``(T, n)`` arrays (one per stimulus) and ``truth_dict`` maps lag indices to
boolean ``(n, n)`` ground-truth adjacency matrices.

* **Linear VAR(2)** (family="var") – ``x_t = A1 x_{t-1} + A2 x_{t-2} + eps``.
* **Poisson GLM** (family="poisson") – count data with lag-1 and lag-2 coupling.
* **Hawkes-like** (family="hawkes") – softplus intensity with lag-1 and lag-2.
* **Nonlinear tanh VAR** (family="tanh") – ``x_t = tanh(W1 x_{t-1} + W2 x_{t-2}) + eps``.

Baseline Learners
=================
* **VAR-LASSO / VAR-Ridge** – lag-1 linear VAR via scikit-learn.
* **VAR-LiNGAM** – continuous optimization LiNGAM (requires ``lingam``).
* **Poisson-GLM** – statsmodels GLM per neuron with canonical log link.
* **PCMCI+** – conditional independence testing via ``tigramite``.
* **NOTEARS** – two-slice DAG learning from ``xunzheng/notears``.
* **DYNOTEARS** – dynamic NOTEARS from ``causalnex``.

Each baseline exposes an ``enabled`` flag derived from optional imports so the
benchmark degrades gracefully when a dependency is missing.
"""

import os
import signal
import time
import functools
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

import statsmodels.api as sm

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.models.sbtg import (
    SBTGStructuredVolatilityEstimator,
    SBTGVolatilityResult,
)
from pipeline.models.multilag_sbtg import (
    MinimalMultiBlockEstimator,
    MultiLagSBTGResult,
    HPConfig as MultiLagHPConfig,
    tune_hyperparameters as tune_hp_null_contrast,
    compute_null_contrast,
)

import torch
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = 'cuda'
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_COUNT = torch.cuda.device_count()
else:
    DEVICE = 'cpu'
    GPU_NAME = None
    GPU_COUNT = 0

CPU_COUNT = os.cpu_count() or 1

# ---------------------------------------------------------------------------
# Optional baseline imports
# ---------------------------------------------------------------------------

try:
    import notears
    from notears import linear as notears_linear
    HAS_NOTEARS = True
    NOTEARS_IMPORT_ERROR = None
except ImportError as e:
    HAS_NOTEARS = False
    NOTEARS_IMPORT_ERROR = str(e)
except Exception as e:
    HAS_NOTEARS = False
    NOTEARS_IMPORT_ERROR = f"Unexpected error: {str(e)}"

try:
    from causalnex.structure.dynotears import from_pandas_dynamic
    HAS_DYNOTEARS = True
    DYNOTEARS_IMPORT_ERROR = None
except ImportError as e:
    HAS_DYNOTEARS = False
    DYNOTEARS_IMPORT_ERROR = str(e)
except Exception as e:
    HAS_DYNOTEARS = False
    DYNOTEARS_IMPORT_ERROR = f"Unexpected error: {str(e)}"

try:
    import lingam
    HAS_LINGAM = True
    LINGAM_IMPORT_ERROR = None
except ImportError as e:
    HAS_LINGAM = False
    LINGAM_IMPORT_ERROR = str(e)
except Exception as e:
    HAS_LINGAM = False
    LINGAM_IMPORT_ERROR = f"Unexpected error: {str(e)}"

try:
    from tigramite import data_processing as pp
    from tigramite import independence_tests, pcmci
    from tigramite.independence_tests.parcorr import ParCorr
    HAS_TIGRAMITE = True
    TIGRAMITE_IMPORT_ERROR = None
except ImportError as e:
    HAS_TIGRAMITE = False
    TIGRAMITE_IMPORT_ERROR = str(e)
    ParCorr = None
except Exception as e:
    HAS_TIGRAMITE = False
    TIGRAMITE_IMPORT_ERROR = f"Unexpected error: {str(e)}"
    ParCorr = None


# ---------------------------------------------------------------------------
# Method name constants
# ---------------------------------------------------------------------------

SBTG_LIN_NAME = "SBTG-Linear"
SBTG_A_NAME = "SBTG-FeatureBilinear"
SBTG_MINIMAL_NAME = "SBTG-Minimal"
SBTG_METHODS = [SBTG_LIN_NAME, SBTG_A_NAME, SBTG_MINIMAL_NAME]
BASELINE_DEFAULT_STAT_NAME = "default"


# ---------------------------------------------------------------------------
# Timeout decorator
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def timeout(seconds):
    """Decorator to add timeout to functions (Unix-based systems only)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")

            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def check_dependencies() -> int:
    """Print summary of available baseline packages and return count of available."""
    print("=" * 70)
    print("[DEPENDENCY CHECK] Baseline Package Status")
    print("=" * 70)

    deps = [
        ("VAR-LASSO", True, "sklearn (built-in)", None, None),
        ("VAR-Ridge", True, "sklearn (built-in)", None, None),
        ("VAR-LiNGAM", HAS_LINGAM, "lingam", "pip install lingam", LINGAM_IMPORT_ERROR),
        ("Poisson-GLM", True, "statsmodels (built-in)", None, None),
        (
            "NOTEARS",
            HAS_NOTEARS,
            "notears",
            "pip install git+https://github.com/xunzheng/notears.git",
            NOTEARS_IMPORT_ERROR,
        ),
        ("PCMCI+", HAS_TIGRAMITE, "tigramite", "pip install tigramite", TIGRAMITE_IMPORT_ERROR),
        ("DYNOTEARS", HAS_DYNOTEARS, "causalnex", "pip install causalnex", DYNOTEARS_IMPORT_ERROR),
    ]

    available = 0
    for name, has_pkg, pkg_name, install_cmd, import_error in deps:
        if has_pkg:
            print(f"  ✓ {name:<15} ({pkg_name})")
            available += 1
        else:
            print(f"  ✗ {name:<15} → {install_cmd}")
            if import_error:
                print(f"      reason: {import_error}")

    print("-" * 70)
    print(f"  {available}/{len(deps)} baselines available")
    print("=" * 70)
    return available


def print_stage(stage_num: int, total_stages: int, title: str) -> None:
    """Print a stage banner."""
    print("\n" + "=" * 70)
    print(f"[STAGE {stage_num}/{total_stages}] {title}")
    print("=" * 70 + "\n", flush=True)


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name: str):
        self.name = name
        self.start = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        print(f"[TIMER] {self.name} started...", flush=True)
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"[TIMER] {self.name} completed in {self.elapsed:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# SBTG fitting wrappers with timeout protection
# ---------------------------------------------------------------------------

@timeout(600)
def fit_sbtg_with_timeout(
    est_kwargs: Dict[str, Any],
    X_list: List[np.ndarray],
) -> SBTGVolatilityResult:
    """Fit SBTG with timeout protection to prevent hanging."""
    est = SBTGStructuredVolatilityEstimator(**est_kwargs)
    return est.fit(X_list)


@timeout(900)
def fit_minimal_with_timeout(
    est_kwargs: Dict[str, Any],
    X_list: List[np.ndarray],
) -> MultiLagSBTGResult:
    """Fit Minimal Multi-Block SBTG with timeout."""
    lags = est_kwargs.pop('lags', [1, 2])
    est = MinimalMultiBlockEstimator(lags=lags, **est_kwargs)
    return est.fit(X_list)


# ---------------------------------------------------------------------------
# HP configuration
# ---------------------------------------------------------------------------

N_HP_TRIALS = 20

SBTG_TRAINING_GRID_MINIMAL: List[Dict[str, Any]] = [
    {
        "name": "minimal_tuned",
        "tune_hp": True,
        "n_hp_trials": N_HP_TRIALS,
        "lags": [1, 2],
        "dsm_epochs": 100,
    }
]

SBTG_STAT_PARAM_GRID: List[Dict[str, Any]] = [
    {
        "name": "hac5_alpha010_by",
        "hac_max_lag": 5,
        "fdr_alpha": 0.10,
        "fdr_method": "by",
        "energy_test": True,
    },
    {
        "name": "hac7_alpha010_by",
        "hac_max_lag": 7,
        "fdr_alpha": 0.10,
        "fdr_method": "by",
        "energy_test": True,
    },
]


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _make_sparse_matrix(
    n: int, sparsity: float, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Create a sparse random matrix with given sparsity and spectral scaling.

    Parameters
    ----------
    n : int
        Matrix dimension.
    sparsity : float
        Fraction of nonzero entries.
    scale : float
        Scaling factor for nonzero entries (before spectral normalization).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        (n, n) sparse matrix with spectral radius < 1.
    """
    A = np.zeros((n, n), dtype=float)
    mask = rng.uniform(size=(n, n)) < sparsity
    A[mask] = rng.normal(loc=0.0, scale=scale, size=mask.sum())
    np.fill_diagonal(A, 0.0)

    eigvals = np.linalg.eigvals(A)
    rho = max(abs(eigvals)) if eigvals.size > 0 else 0.0
    if rho > 0:
        A = A / (1.2 * rho)
    return A


def generate_var_data(
    n: int, T: int, m_stim: int, noise_level: str, seed: int,
) -> Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
    """
    Generate VAR(2) data with ground-truth adjacency.

    x_t = A1 x_{t-1} + A2 x_{t-2} + eps_t,  eps_t ~ N(0, sigma^2 I)

    Returns
    -------
    X_list : list of (T, n) arrays, one per stimulus.
    truth_dict : {1: support(A1), 2: support(A2)}.
    """
    rng = np.random.default_rng(seed)
    sigma = 0.1 if noise_level == "low" else 0.5

    A1 = _make_sparse_matrix(n, sparsity=0.1, scale=0.8, rng=rng)
    A2 = _make_sparse_matrix(n, sparsity=0.1, scale=0.5, rng=rng)

    X_list = []
    for _ in range(m_stim):
        X = np.zeros((T, n), dtype=float)
        X[0] = rng.normal(scale=1.0, size=n)
        X[1] = rng.normal(scale=1.0, size=n)
        for t in range(2, T):
            eps = rng.normal(scale=sigma, size=n)
            X[t] = A1 @ X[t - 1] + A2 @ X[t - 2] + eps
        X_list.append(X)

    truth_dict = {1: (np.abs(A1) > 1e-8), 2: (np.abs(A2) > 1e-8)}
    return X_list, truth_dict


def generate_poisson_glm_data(
    n: int, T: int, m_stim: int, noise_level: str, seed: int,
) -> Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
    """
    Generate Poisson-GLM spiking data with lag-1 and lag-2 dependencies.

    lambda_t = exp(alpha + A1 x_{t-1} + A2 x_{t-2} + s^{(u)}_t),
    x_{t} ~ Poisson(lambda_t).

    Returns
    -------
    X_list : list of (T, n) arrays with nonnegative integer counts.
    truth_dict : {1: support(A1), 2: support(A2)}.
    """
    rng = np.random.default_rng(seed)
    A1 = _make_sparse_matrix(n, sparsity=0.1, scale=0.5, rng=rng)
    A2 = _make_sparse_matrix(n, sparsity=0.1, scale=0.25, rng=rng)

    base_rate = 0.1 if noise_level == "low" else 0.5
    alpha = rng.normal(loc=np.log(base_rate + 1e-3), scale=0.1, size=n)

    X_list = []
    time_arr = np.arange(T)
    for u in range(m_stim):
        stim_drive = 0.2 * np.sin(2 * np.pi * (u + 1) * time_arr / T)
        X = np.zeros((T, n), dtype=float)
        X[0] = rng.poisson(lam=np.exp(alpha), size=n)
        X[1] = rng.poisson(lam=np.exp(alpha), size=n)
        for t in range(2, T):
            linear = alpha + A1 @ X[t - 1] + A2 @ X[t - 2] + stim_drive[t]
            lam = np.exp(np.clip(linear, -5, 5))
            X[t] = rng.poisson(lam=lam, size=n)
        X_list.append(X)

    truth_dict = {1: (np.abs(A1) > 1e-8), 2: (np.abs(A2) > 1e-8)}
    return X_list, truth_dict


def generate_hawkes_like_data(
    n: int, T: int, m_stim: int, noise_level: str, seed: int,
) -> Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
    """
    Generate a discrete-time Hawkes-like process.

    lambda_t = softplus(alpha + A1 x_{t-1} + A2 x_{t-2}),
    x_t ~ Poisson(lambda_t).

    Returns
    -------
    X_list : list of (T, n) arrays.
    truth_dict : {1: support(A1), 2: support(A2)}.
    """
    rng = np.random.default_rng(seed)
    A1 = _make_sparse_matrix(n, sparsity=0.1, scale=0.4, rng=rng)
    A2 = _make_sparse_matrix(n, sparsity=0.1, scale=0.3, rng=rng)

    base_rate = 0.05 if noise_level == "low" else 0.2
    alpha = rng.normal(loc=np.log(base_rate + 1e-3), scale=0.1, size=n)

    def softplus(x):
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    X_list = []
    for _ in range(m_stim):
        X = np.zeros((T, n), dtype=float)
        X[0] = rng.poisson(lam=np.exp(alpha), size=n)
        X[1] = rng.poisson(lam=np.exp(alpha), size=n)
        for t in range(2, T):
            linear = alpha + A1 @ X[t - 1] + A2 @ X[t - 2]
            lam = softplus(np.clip(linear, -5, 5))
            X[t] = rng.poisson(lam=lam, size=n)
        X_list.append(X)

    truth_dict = {1: (np.abs(A1) > 1e-8), 2: (np.abs(A2) > 1e-8)}
    return X_list, truth_dict


def generate_tanh_var_data(
    n: int, T: int, m_stim: int, noise_level: str, seed: int,
) -> Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
    """
    Generate nonlinear tanh-VAR(2) data.

    x_t = tanh(W1 x_{t-1} + W2 x_{t-2}) + eps_t,  eps_t ~ N(0, sigma^2 I).

    Returns
    -------
    X_list : list of (T, n) arrays.
    truth_dict : {1: support(W1), 2: support(W2)}.
    """
    rng = np.random.default_rng(seed)
    W1 = _make_sparse_matrix(n, sparsity=0.1, scale=0.8, rng=rng)
    W2 = _make_sparse_matrix(n, sparsity=0.1, scale=0.4, rng=rng)
    sigma = 0.1 if noise_level == "low" else 0.5

    X_list = []
    for _ in range(m_stim):
        X = np.zeros((T, n), dtype=float)
        X[0] = rng.normal(scale=1.0, size=n)
        X[1] = rng.normal(scale=1.0, size=n)
        for t in range(2, T):
            eps = rng.normal(scale=sigma, size=n)
            X[t] = np.tanh(W1 @ X[t - 1] + W2 @ X[t - 2]) + eps
        X_list.append(X)

    truth_dict = {1: (np.abs(W1) > 1e-8), 2: (np.abs(W2) > 1e-8)}
    return X_list, truth_dict


def _generate_dataset(
    family: str, n: int, T: int, m_stim: int, noise_level: str, seed: int,
) -> Tuple[List[np.ndarray], Dict[int, np.ndarray]]:
    """Dispatch to the appropriate data generator by family name."""
    generators = {
        "var": generate_var_data,
        "poisson": generate_poisson_glm_data,
        "hawkes": generate_hawkes_like_data,
        "tanh": generate_tanh_var_data,
    }
    if family not in generators:
        raise ValueError(f"Unknown dataset family: {family}")
    return generators[family](n=n, T=T, m_stim=m_stim, noise_level=noise_level, seed=seed)


# ---------------------------------------------------------------------------
# Baseline wrappers
# ---------------------------------------------------------------------------

def var_lasso_baseline(X_list: List[np.ndarray], alpha: float = 0.1) -> np.ndarray:
    """
    VAR-LASSO baseline: fit lag-1 VAR via L1 regression, pooled across stimuli.

    Returns
    -------
    np.ndarray
        Weight matrix A_hat of shape (n, n).
    """
    X_all = np.concatenate(X_list, axis=0)
    T_all, n = X_all.shape
    if T_all < 2:
        raise ValueError("Not enough samples for VAR-LASSO.")

    X_t_minus_1 = X_all[:-1]
    X_t = X_all[1:]

    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X_t_minus_1)
    feature_scale = feature_scaler.scale_ + 1e-12

    A_hat = np.zeros((n, n), dtype=float)
    for j in range(n):
        y = X_t[:, j]
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y[:, None]).ravel()
        y_scale = float(target_scaler.scale_ if target_scaler.scale_ is not None else 1.0)
        if y_scale < 1e-12:
            continue

        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000, tol=1e-4)
        model.fit(X_scaled, y_scaled)
        beta_original = model.coef_ * (y_scale / feature_scale)
        A_hat[:, j] = beta_original
    return A_hat


def var_ridge_baseline(X_list: List[np.ndarray], alpha: float = 1.0) -> np.ndarray:
    """
    VAR-Ridge baseline: fit lag-1 VAR via L2 regression, pooled across stimuli.

    Returns
    -------
    np.ndarray
        Weight matrix A_hat of shape (n, n).
    """
    X_all = np.concatenate(X_list, axis=0)
    T_all, n = X_all.shape
    if T_all < 2:
        raise ValueError("Not enough samples for VAR-Ridge.")

    X_t_minus_1 = X_all[:-1]
    X_t = X_all[1:]

    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X_t_minus_1)
    feature_scale = feature_scaler.scale_ + 1e-12

    A_hat = np.zeros((n, n), dtype=float)
    for j in range(n):
        y = X_t[:, j]
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y[:, None]).ravel()
        y_scale = float(target_scaler.scale_ if target_scaler.scale_ is not None else 1.0)
        if y_scale < 1e-12:
            continue

        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X_scaled, y_scaled)
        beta_original = model.coef_ * (y_scale / feature_scale)
        A_hat[:, j] = beta_original
    return A_hat


def var_lingam_baseline(X_list: List[np.ndarray]) -> np.ndarray:
    """
    VAR-LiNGAM baseline using the ``lingam`` package.

    Returns
    -------
    np.ndarray
        Estimated lag-1 adjacency matrix A_hat (n, n).
    """
    if not HAS_LINGAM:
        raise RuntimeError("lingam package not available.")

    X_all = np.concatenate(X_list, axis=0)
    T_all, n = X_all.shape
    if T_all < 2:
        raise ValueError("Not enough samples for VAR-LiNGAM.")

    X_centered = X_all - X_all.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(0)
    X_proc = X_centered + 1e-6 * rng.standard_normal(size=X_centered.shape)

    model = lingam.VARLiNGAM(lags=1)
    try:
        model.fit(X_proc)
    except Exception as exc:
        raise RuntimeError(
            "VAR-LiNGAM failed to converge; try increasing sample size or "
            "checking data scaling"
        ) from exc

    B_matrices = model.adjacency_matrices_
    A_hat = B_matrices[0].T
    return A_hat


def poisson_glm_baseline(X_list: List[np.ndarray]) -> np.ndarray:
    """
    Poisson-GLM baseline for count data (Poisson / Hawkes-like families).

    Fits per neuron: x_{t,j} ~ Poisson(exp(beta_0 + beta^T x_{t-1})).

    Returns
    -------
    np.ndarray
        Weight matrix A_hat of shape (n, n).
    """
    X_all = np.concatenate(X_list, axis=0)
    T_all, n = X_all.shape
    if T_all < 2:
        raise ValueError("Not enough samples for Poisson-GLM.")

    X_t_minus_1 = X_all[:-1]
    X_t = X_all[1:]

    A_hat = np.zeros((n, n), dtype=float)
    for j in range(n):
        y = X_t[:, j]
        X_design = sm.add_constant(X_t_minus_1)
        model = sm.GLM(y, X_design, family=sm.families.Poisson())
        res = model.fit()
        A_hat[:, j] = res.params[1:]
    return A_hat


@timeout(300)
def notears_baseline(X_list: List[np.ndarray], lambda1: float = 0.1, max_iter: int = 100) -> np.ndarray:
    """NOTEARS baseline using the xunzheng/notears two-slice representation.
    Limited to 5 minutes execution time.
    """
    if not HAS_NOTEARS:
        raise RuntimeError("NOTEARS baseline not available.")
    if not X_list:
        raise ValueError("Empty dataset list provided to NOTEARS baseline.")

    X_all = np.concatenate(X_list, axis=0)
    if X_all.shape[0] < 2:
        raise ValueError("Not enough samples for NOTEARS baseline.")

    n = X_all.shape[1]
    X_tm1 = X_all[:-1]
    X_t = X_all[1:]
    X_concat = np.hstack([X_tm1, X_t])

    W_est = notears_linear.notears_linear(
        X_concat,
        lambda1=lambda1,
        loss_type='l2',
        max_iter=max_iter,
        h_tol=1e-8,
        rho_max=1e+16,
        w_threshold=0.3,
    )

    A_lag1 = W_est[:n, n:]
    return A_lag1


@timeout(300)
def pcmci_plus_baseline(
    X_list: List[np.ndarray], tau_max: int = 1, alpha_level: float = 0.05,
) -> np.ndarray:
    """
    PCMCI+ baseline using tigramite with ParCorr test.
    Limited to 5 minutes execution time.

    Returns
    -------
    np.ndarray
        Boolean adjacency matrix for lag-1 edges (n, n).
    """
    if not HAS_TIGRAMITE:
        raise RuntimeError("tigramite package not available.")

    X_all = np.concatenate(X_list, axis=0)
    dataframe = pp.DataFrame(X_all)

    if ParCorr is None:
        raise RuntimeError("ParCorr independence test unavailable despite tigramite import.")

    parcorr = ParCorr(significance="analytic")
    pcmci_obj = pcmci.PCMCI(dataframe=dataframe, cond_ind_test=parcorr)

    results = pcmci_obj.run_pcmci(
        tau_max=tau_max,
        pc_alpha=0.2,
        alpha_level=alpha_level,
    )
    p_matrix = results["p_matrix"]
    adj = p_matrix[:, :, 1] < alpha_level
    np.fill_diagonal(adj, False)
    return adj


@timeout(300)
def dynotears_baseline(
    X_list: List[np.ndarray],
    lambda_w: float = 0.01,
    lambda_a: float = 0.01,
    max_iter: int = 50,
    w_threshold: float = 0.2,
) -> np.ndarray:
    """DYNOTEARS baseline using causalnex. Limited to 5 minutes execution time."""
    if not HAS_DYNOTEARS:
        raise RuntimeError("DYNOTEARS baseline not available.")

    X_all = np.concatenate(X_list, axis=0)
    n = X_all.shape[1]

    column_names = [f"X{i}" for i in range(n)]
    df = pd.DataFrame(X_all, columns=column_names)

    try:
        import re

        def _fit_and_extract(threshold: float, n_iter: int) -> np.ndarray:
            sm_model = from_pandas_dynamic(
                df, p=1,
                lambda_w=lambda_w, lambda_a=lambda_a,
                max_iter=n_iter, h_tol=1e-6, w_threshold=threshold,
            )
            A_lag1_local = np.zeros((n, n))
            for edge in sm_model.edges():
                src, dst = edge[0], edge[1]
                lag_match = re.search(r"_lag(\d+)", src)
                if lag_match and lag_match.group(1) == "1":
                    src_clean = re.sub(r"_lag\d+", "", src).replace("X", "")
                    dst_clean = re.sub(r"_lag\d+", "", dst).replace("X", "")
                    try:
                        from_idx = int(src_clean)
                        to_idx = int(dst_clean)
                        if 0 <= from_idx < n and 0 <= to_idx < n:
                            weight = sm_model.edges[edge].get("weight", 1.0)
                            A_lag1_local[from_idx, to_idx] = weight
                    except (ValueError, IndexError):
                        continue
            return A_lag1_local

        # First pass: moderate pruning for cleaner graphs.
        A_lag1 = _fit_and_extract(threshold=w_threshold, n_iter=max_iter)

        # Fallback: if over-pruned to zero edges, retry with lighter pruning.
        if np.count_nonzero(A_lag1) == 0:
            for retry_threshold in (0.1, 0.05, 0.0):
                A_retry = _fit_and_extract(threshold=retry_threshold, n_iter=max(max_iter, 100))
                if np.count_nonzero(A_retry) > 0:
                    A_lag1 = A_retry
                    break

        return A_lag1
    except Exception as e:
        raise RuntimeError(f"DYNOTEARS fitting failed: {e}") from e


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_weighted(
    truth_graph: np.ndarray,
    scored_matrix: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate a method given a ground-truth adjacency and a real-valued score matrix.

    Parameters
    ----------
    truth_graph : (n, n) boolean true adjacency.
    scored_matrix : (n, n) real-valued scores.
    threshold : binarization threshold (edge if |score| > threshold).

    Returns
    -------
    Dict with precision, recall, f1, roc_auc, pr_auc, y_true, y_score.
    """
    n = truth_graph.shape[0]
    mask_offdiag = ~np.eye(n, dtype=bool)

    y_true = truth_graph[mask_offdiag].astype(int)
    y_score = np.abs(scored_matrix)[mask_offdiag]
    y_pred = (y_score > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = np.nan

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "y_true": y_true, "y_score": y_score,
        "pred_edges": int(y_pred.sum()),
        "true_edges": int(y_true.sum()),
        "pred_density": float(y_pred.mean()),
        "true_density": float(y_true.mean()),
    }


def evaluate_multilag_auroc(
    truth_dict: Dict[int, np.ndarray],
    pred_dict: Dict[int, np.ndarray],
) -> Dict[str, float]:
    """Compute AUROC for multiple lags."""
    results = {}
    for lag, truth in truth_dict.items():
        if lag in pred_dict:
            pred = pred_dict[lag]
            n = truth.shape[0]
            mask = ~np.eye(n, dtype=bool)
            y_true = truth[mask].astype(int)
            y_score = np.abs(pred)[mask]
            try:
                auroc = roc_auc_score(y_true, y_score)
            except ValueError:
                auroc = np.nan
            results[f"auroc_lag{lag}"] = auroc
        else:
            results[f"auroc_lag{lag}"] = np.nan
    return results


def evaluate_binary(
    truth_graph: np.ndarray,
    adj_matrix: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate a binary adjacency matrix against ground truth.

    Returns
    -------
    Dict with precision, recall, f1, roc_auc, pr_auc, y_true, y_score.
    """
    n = truth_graph.shape[0]
    mask_offdiag = ~np.eye(n, dtype=bool)

    y_true = truth_graph[mask_offdiag].astype(int)
    y_pred = adj_matrix[mask_offdiag].astype(int)
    y_score = y_pred.astype(float)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = np.nan

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "y_true": y_true, "y_score": y_score,
        "pred_edges": int(y_pred.sum()),
        "true_edges": int(y_true.sum()),
        "pred_density": float(y_pred.mean()),
        "true_density": float(y_true.mean()),
    }


def evaluate_sbtg(
    truth_graph: np.ndarray,
    sbtg_result: Any,
    use_complex_edges: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate SBTG using its statistical-test adjacency.

    Final adjacency = adj_mean OR adj_complex (if use_complex_edges).
    """
    sign_adj = sbtg_result.sign_adj
    n = sign_adj.shape[0]
    mask_offdiag = ~np.eye(n, dtype=bool)

    adj_mean = (sign_adj != 0)
    if use_complex_edges and sbtg_result.complex_adj is not None:
        complex_adj = sbtg_result.complex_adj.astype(bool)
        complex_only = complex_adj & (~adj_mean)
        adj = np.logical_or(adj_mean, complex_adj)
    else:
        complex_adj = None
        complex_only = np.zeros_like(adj_mean, dtype=bool)
        adj = adj_mean

    np.fill_diagonal(adj, False)

    metrics = evaluate_binary(truth_graph, adj.astype(int))
    metrics["edges_mean"] = int(adj_mean[mask_offdiag].sum())
    metrics["edges_energy_only"] = int(complex_only[mask_offdiag].sum()) if complex_adj is not None else 0
    metrics["edges_volatility_only"] = 0
    metrics["edges_union"] = int(adj[mask_offdiag].sum())
    return metrics


def evaluate_structured_volatility(
    truth_graph: np.ndarray,
    sbtg_result: SBTGVolatilityResult,
    include_volatility_edges: bool = True,
) -> Dict[str, Any]:
    """Evaluate structured-volatility SBTG by combining mean and volatility edges."""
    sign_adj = sbtg_result.sign_adj != 0
    n = sign_adj.shape[0]
    mask_offdiag = ~np.eye(n, dtype=bool)

    if include_volatility_edges and sbtg_result.volatility_adj is not None:
        volatility_adj = sbtg_result.volatility_adj.astype(bool)
    else:
        volatility_adj = np.zeros_like(sign_adj, dtype=bool)

    adj = np.logical_or(sign_adj, volatility_adj).astype(int)
    np.fill_diagonal(adj, 0)

    metrics = evaluate_binary(truth_graph, adj)
    metrics["edges_mean"] = int(sign_adj[mask_offdiag].sum())
    metrics["edges_energy_only"] = 0
    metrics["edges_volatility_only"] = int(volatility_adj[mask_offdiag].sum())
    metrics["edges_union"] = int(adj[mask_offdiag].sum())
    return metrics


# ---------------------------------------------------------------------------
# SBTG hyperparameter search (Optuna + null contrast)
# ---------------------------------------------------------------------------

def hyperparam_search_sbtg_null_contrast(
    X_list_hp: List[np.ndarray],
    smooth_sigma: float,
    model_type: str,
    method_name: str,
    family: str,
    noise: str,
    length_type: str,
    n_trials: int,
    seed: int,
    sbtg_epochs: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Hyperparameter search for SBTG variants using Optuna + null contrast objective.

    The null contrast objective measures real signal vs shuffled baseline, which
    correlates better with biological ground truth than F1 on synthetic data.

    Returns
    -------
    best_cfg : dict with best hyperparameters in the format expected by the
        SBTG training code.
    """
    if verbose:
        print(
            f"[SBTG-HP] {family}/{noise}/{length_type} {method_name}: "
            f"tuning with {n_trials} Optuna trials (null contrast, device={DEVICE})",
            flush=True,
        )

    best_hp_config = tune_hp_null_contrast(
        X_list_hp,
        n_trials=n_trials,
        lag=1,
        n_blocks=2,
        epochs_for_tuning=sbtg_epochs,
        n_folds=3,
        device=DEVICE,
        verbose=verbose,
        seed=seed,
    )

    best_cfg = {
        "name": f"{model_type}_optuna_best",
        "model_type": model_type,
        "dsm_hidden_dim": best_hp_config.hidden_dim,
        "dsm_num_layers": best_hp_config.num_layers,
        "dsm_noise_std": best_hp_config.noise_std,
        "dsm_epochs": sbtg_epochs,
        "dsm_lr": best_hp_config.lr,
        "dsm_batch_size": 128,
        "train_frac": 0.8,
        "window_length": 2,
        "smooth_sigma": smooth_sigma,
    }

    if model_type == "feature_bilinear":
        best_cfg["feature_dim"] = 16
        best_cfg["structured_l1_lambda"] = 0.01

    if verbose:
        print(
            f"[SBTG-HP] {family}/{noise}/{length_type} {method_name}: "
            f"best: hidden_dim={best_hp_config.hidden_dim}, "
            f"num_layers={best_hp_config.num_layers}, "
            f"noise_std={best_hp_config.noise_std:.3f}, "
            f"lr={best_hp_config.lr:.6f}",
            flush=True,
        )

    return best_cfg
