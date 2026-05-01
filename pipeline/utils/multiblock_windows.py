"""
Multi-block window construction for lag separation.

This module implements the window construction for multi-block SBTG as described
in the theoretical derivation (Theorem 5.1).

Theory:
    For order-p Markov process: x_{t+1} = f(x_t, ..., x_{t-p+1}) + ε_t
    
    Multi-block windows contain the full lag stack:
        z_t = (x_{t-p_max+1}, x_{t-p_max+2}, ..., x_t, x_{t+1}) ∈ ℝ^{(p_max+1)n}
    
    Block indexing:
        z^(k) = x_{t-p_max+1+k}  for k = 0, 1, ..., p_max
        z^(0) = oldest block (x_{t-p_max+1})
        z^(p_max) = future block (x_{t+1})
    
    For lag r analysis:
        Future block: index p_max
        Lag-r block: index p_max - r

References:
    - docs/SCRIPT_13_IMPLEMENTATION_PLAN.md
    - docs/MULTILAG_THEORY_MISMATCH.md
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def gaussian_smooth_1d(
    X: np.ndarray,
    sigma: Optional[float],
    causal: bool = True,
) -> np.ndarray:
    """
    Apply 1D Gaussian smoothing along time axis.
    
    Args:
        X: (T, n) array
        sigma: Smoothing width in frames (None = no smoothing)
        causal: If True, only use past/present frames
        
    Returns:
        X_smooth: (T, n) smoothed array
    """
    if sigma is None or sigma <= 0:
        return X
    
    T, n = X.shape
    
    # Create kernel
    kernel_size = int(6 * sigma) + 1
    if causal:
        # Causal: only past and present
        t_kernel = np.arange(kernel_size)
        kernel = np.exp(-0.5 * (t_kernel / sigma) ** 2)
    else:
        # Acausal: symmetric
        t_kernel = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-0.5 * (t_kernel / sigma) ** 2)
    
    kernel = kernel / kernel.sum()
    
    # Convolve each neuron
    X_smooth = np.zeros_like(X)
    for i in range(n):
        if causal:
            # Causal convolution: pad at start
            padded = np.concatenate([np.full(kernel_size - 1, X[0, i]), X[:, i]])
            X_smooth[:, i] = np.convolve(padded, kernel, mode='valid')
        else:
            X_smooth[:, i] = np.convolve(X[:, i], kernel, mode='same')
    
    return X_smooth


def build_multiblock_windows(
    X_list: List[np.ndarray],
    p_max: int,
    smooth_sigma: Optional[float] = None,
    causal_smoothing: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Construct multi-block windows z_t = (x_{t-p_max+1}, ..., x_t, x_{t+1}).
    
    Args:
        X_list: List of (T_u, n) arrays (one per stimulus/segment)
        p_max: Maximum lag order (window will have p_max+1 blocks)
        smooth_sigma: Optional Gaussian smoothing width in frames
        causal_smoothing: If True, only smooth using past/present
        verbose: Print construction details
        
    Returns:
        windows: (N_total, (p_max+1)*n) array of stacked blocks
        stim_ids: (N_total,) stimulus index per window
        local_t: (N_total,) time index within stimulus
        metadata: Dict with construction details
        
    Window structure (flat concatenation):
        z_t[0:n]                    = x_{t-p_max+1}  (block 0, oldest)
        z_t[n:2n]                   = x_{t-p_max+2}  (block 1)
        ...
        z_t[(p_max-1)*n:p_max*n]    = x_t            (block p_max-1, present)
        z_t[p_max*n:(p_max+1)*n]    = x_{t+1}        (block p_max, future)
        
    For a window built at time index t (0-indexed from start of stimulus):
        The window contains x[t], x[t+1], ..., x[t+p_max]
        So block k contains x[t+k]
        And:
            block 0 = x[t] corresponds to x_{t-p_max+1} in the notation
            block p_max = x[t+p_max] corresponds to x_{t+1} in the notation
        
        In "logical time" notation, if we index the target step by τ:
            z_τ = (x_{τ-p_max+1}, ..., x_τ, x_{τ+1})
            
        If our array indexing starts at 0, and we want to build a window:
            - Earliest valid τ is when τ-p_max+1 = 0, i.e., τ = p_max-1
            - At τ = p_max-1: window contains x[0], x[1], ..., x[p_max]
            - At τ = T-2: window contains x[T-p_max-1], ..., x[T-2], x[T-1]
            
        So for array index i (0-indexed window number):
            τ = p_max - 1 + i
            Window i contains: x[i], x[i+1], ..., x[i+p_max]
    """
    if not X_list:
        raise ValueError("X_list is empty")
    
    if p_max < 1:
        raise ValueError(f"p_max must be >= 1, got {p_max}")
    
    # Check neuron count consistency
    n_neurons = X_list[0].shape[1]
    for i, X in enumerate(X_list):
        if X.shape[1] != n_neurons:
            raise ValueError(
                f"Neuron count mismatch: X_list[0] has {n_neurons}, "
                f"X_list[{i}] has {X.shape[1]}"
            )
    
    windows_list = []
    stim_ids_list = []
    local_t_list = []
    
    n_stimuli_used = 0
    n_stimuli_skipped = 0
    
    for stim_idx, X in enumerate(X_list):
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(f"X_list[{stim_idx}] must be 2D, got {X.ndim}D")
        
        T_u, n = X.shape
        
        # Need at least p_max + 1 timepoints to form one window
        # (window contains p_max+1 blocks)
        if T_u < p_max + 1:
            if verbose:
                print(f"  Skipping stimulus {stim_idx}: T={T_u} < p_max+1={p_max+1}")
            n_stimuli_skipped += 1
            continue
        
        # Apply smoothing if requested
        if smooth_sigma is not None and smooth_sigma > 0:
            X = gaussian_smooth_1d(X, smooth_sigma, causal=causal_smoothing)
        
        # Number of valid windows from this stimulus
        # Window i uses indices [i, i+1, ..., i+p_max]
        # Last valid i is T_u - p_max - 1
        n_windows = T_u - p_max
        
        # Build windows
        # Each window is (p_max+1)*n dimensional
        windows_u = np.zeros((n_windows, (p_max + 1) * n), dtype=np.float64)
        
        for i in range(n_windows):
            # Window i contains x[i], x[i+1], ..., x[i+p_max]
            for k in range(p_max + 1):
                windows_u[i, k*n:(k+1)*n] = X[i + k]
        
        windows_list.append(windows_u)
        stim_ids_list.append(np.full(n_windows, stim_idx, dtype=np.int32))
        local_t_list.append(np.arange(n_windows, dtype=np.int32))
        
        n_stimuli_used += 1
        
        if verbose:
            print(f"  Stimulus {stim_idx}: T={T_u}, {n_windows} windows")
    
    if not windows_list:
        raise ValueError(
            f"No valid windows constructed with p_max={p_max}. "
            f"All {len(X_list)} stimuli have T < {p_max + 1}."
        )
    
    # Concatenate
    windows = np.concatenate(windows_list, axis=0)
    stim_ids = np.concatenate(stim_ids_list, axis=0)
    local_t = np.concatenate(local_t_list, axis=0)
    
    metadata = {
        'p_max': p_max,
        'n_neurons': n_neurons,
        'n_blocks': p_max + 1,
        'window_dim': (p_max + 1) * n_neurons,
        'n_windows': len(windows),
        'n_stimuli_total': len(X_list),
        'n_stimuli_used': n_stimuli_used,
        'n_stimuli_skipped': n_stimuli_skipped,
        'smooth_sigma': smooth_sigma,
        'causal_smoothing': causal_smoothing,
    }
    
    if verbose:
        print(f"  Total: {len(windows)} windows from {n_stimuli_used} stimuli")
        print(f"  Window dimension: {windows.shape[1]} = (p_max+1)*n = {p_max+1}*{n_neurons}")
    
    return windows, stim_ids, local_t, metadata


def get_block_from_window(
    window: np.ndarray,
    block_idx: int,
    n_neurons: int,
    p_max: int,
) -> np.ndarray:
    """
    Extract a specific block from a window.
    
    Args:
        window: (window_dim,) or (batch, window_dim) array
        block_idx: Which block to extract (0 to p_max)
        n_neurons: Number of neurons per block
        p_max: Maximum lag order
        
    Returns:
        block: (n_neurons,) or (batch, n_neurons) array
    """
    if block_idx < 0 or block_idx > p_max:
        raise ValueError(f"block_idx must be in [0, {p_max}], got {block_idx}")
    
    start = block_idx * n_neurons
    end = (block_idx + 1) * n_neurons
    
    if window.ndim == 1:
        return window[start:end]
    else:
        return window[:, start:end]


def get_block_time_offset(block_idx: int, p_max: int) -> int:
    """
    Get the time offset of a block relative to x_t (the "present").
    
    In the window z_τ = (x_{τ-p_max+1}, ..., x_τ, x_{τ+1}):
        - Block 0 contains x_{τ-p_max+1}, offset = -p_max+1 = 1-p_max
        - Block k contains x_{τ-p_max+1+k}, offset = -p_max+1+k = k+1-p_max
        - Block p_max-1 contains x_τ, offset = 0
        - Block p_max contains x_{τ+1}, offset = +1
    
    Args:
        block_idx: Block index (0 to p_max)
        p_max: Maximum lag order
        
    Returns:
        offset: Time offset relative to x_t (negative = past, 0 = present, +1 = future)
    """
    # offset = block_idx - (p_max - 1) = block_idx + 1 - p_max
    return block_idx + 1 - p_max


def get_block_for_lag(lag_r: int, p_max: int) -> int:
    """
    Get the block index for a specific lag.
    
    For lag r, we want x_{t+1-r}:
        - r=1: x_t (present), offset=0, block = p_max - 1
        - r=2: x_{t-1}, offset=-1, block = p_max - 2
        - r=p_max: x_{t-p_max+1} (oldest), offset=1-p_max, block = 0
    
    General: block_idx = p_max - r
    
    Args:
        lag_r: Lag order (1 to p_max)
        p_max: Maximum lag order
        
    Returns:
        block_idx: Block index containing x_{t+1-r}
    """
    if lag_r < 1 or lag_r > p_max:
        raise ValueError(f"lag_r must be in [1, {p_max}], got {lag_r}")
    
    return p_max - lag_r


def verify_block_indexing(p_max: int, verbose: bool = True) -> bool:
    """
    Verify block indexing is correct for all lags.
    
    Args:
        p_max: Maximum lag order
        verbose: Print verification details
        
    Returns:
        True if all checks pass
    """
    all_passed = True
    
    for r in range(1, p_max + 1):
        block_idx = get_block_for_lag(r, p_max)
        offset = get_block_time_offset(block_idx, p_max)
        expected_offset = 1 - r  # x_{t+1-r} has offset 1-r from x_t
        
        if offset != expected_offset:
            if verbose:
                print(f"FAIL: lag={r}, block={block_idx}, offset={offset}, expected={expected_offset}")
            all_passed = False
        elif verbose:
            print(f"OK: lag={r} → block {block_idx} → x_{{t+1-{r}}} (offset={offset})")
    
    # Also verify future block
    future_idx = p_max
    future_offset = get_block_time_offset(future_idx, p_max)
    if future_offset != 1:
        if verbose:
            print(f"FAIL: future block {future_idx} has offset {future_offset}, expected 1")
        all_passed = False
    elif verbose:
        print(f"OK: future block {future_idx} → x_{{t+1}} (offset={future_offset})")
    
    return all_passed


# =============================================================================
# TEMPORAL SHUFFLING (for Null Contrast objective)
# =============================================================================

def shuffle_temporal(X: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Shuffle time indices within a trace (destroys temporal structure).
    
    This preserves marginal distributions but destroys autocorrelation,
    creating a null baseline for edge detection.
    
    Args:
        X: (T, n) neural activity matrix
        seed: Random seed for reproducibility
        
    Returns:
        X_shuffled: (T, n) with rows shuffled
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    return X[perm].copy()


def shuffle_temporal_per_neuron(X: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Shuffle time indices independently per neuron.
    
    This preserves marginal distributions per neuron but destroys
    both temporal and cross-neuron correlations.
    
    Args:
        X: (T, n) neural activity matrix
        seed: Random seed for reproducibility
        
    Returns:
        X_shuffled: (T, n) with each column independently shuffled
    """
    rng = np.random.default_rng(seed)
    X_shuffled = X.copy()
    
    for i in range(X.shape[1]):
        perm = rng.permutation(X.shape[0])
        X_shuffled[:, i] = X[perm, i]
    
    return X_shuffled


if __name__ == "__main__":
    # Quick test
    print("Testing block indexing for p_max=5:")
    verify_block_indexing(5)
    
    print("\nTesting window construction:")
    X = np.arange(100).reshape(20, 5).astype(float)
    windows, stim_ids, local_t, meta = build_multiblock_windows([X], p_max=3, verbose=True)
    
    print(f"\nFirst window (t=0): {windows[0]}")
    print(f"Expected: x[0], x[1], x[2], x[3] = {X[0]}, {X[1]}, {X[2]}, {X[3]}")
    
    # Verify
    assert np.allclose(windows[0, :5], X[0]), "Block 0 mismatch"
    assert np.allclose(windows[0, 5:10], X[1]), "Block 1 mismatch"
    assert np.allclose(windows[0, 10:15], X[2]), "Block 2 mismatch"
    assert np.allclose(windows[0, 15:20], X[3]), "Block 3 mismatch"
    print("✅ Window construction verified!")
