#!/usr/bin/env python3
"""
05_temporal_analysis.py
=======================

Temporal and stimulus-specific connectivity analysis with SBTG models.

Phase Detection Strategies:
- UNIFORM - Divide recording into equal-length segments
- STIMULUS - Use actual stimulus timing from metadata  
- DERIVATIVE - Data-driven detection using activity transitions

Training Methods:
1. CORRELATION - Simple correlation-based methods (pearson, crosscorr, partial)
2. SBTG_DIRECT - Train SBTG from scratch on each phase
3. SBTG_TRANSFER - Pre-train on baseline, fine-tune on stimulus phases

Training Strategies (for each method):
1. GLOBAL - Train on all data → single graph
2. STIMULUS_ONLY - Train on stimulus windows → stimulus-specific graphs
3. GLOBAL_FINETUNE - Pre-train global, fine-tune on stimulus
4. PHASE_SPECIFIC - Train on each phase separately

Key Outputs:
- Connectivity matrices per condition/phase (both correlation and SBTG)
- Direct vs Transfer comparison metrics
- Graph difference analysis
- E:I ratio tracking across phases

Usage:
    python pipeline/05_temporal_analysis.py                    # Correlation methods
    python pipeline/05_temporal_analysis.py --sbtg             # SBTG direct training
    python pipeline/05_temporal_analysis.py --sbtg --transfer  # SBTG with transfer learning
    python pipeline/05_temporal_analysis.py --sbtg --hp-search # With Optuna HP search
    python pipeline/05_temporal_analysis.py --quick            # Fast mode (fewer epochs)
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.io import loadmat
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import seaborn as sns
from tqdm import tqdm
import warnings
import torch
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import SBTG model
from pipeline.models.sbtg import SBTGStructuredVolatilityEstimator
from pipeline.config import OPTIMIZED_HYPERPARAMS, DEFAULT_HYPERPARAMS

# Try to import Optuna for HP search
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Try to import phase optimal params
try:
    from pipeline.configs.phase_optimal_params import PHASE_OPTIMAL_PARAMS
except ImportError:
    PHASE_OPTIMAL_PARAMS = {}

from pipeline.utils.plotting import (
    create_network_graph,
    create_phase_grid_figure,
    create_difference_heatmap,
    create_strategy_comparison_plot
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "results" / "intermediate" / "datasets"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"
OUTPUT_DIR = PROJECT_ROOT / "results" / "stimulus_specific"
from pipeline.config import STIMULI

# Time phase definitions are derived from metadata by default.
# Set LOCK_PHASE_DEFINITIONS=True only for protocol-locked reproductions,
# and document the protocol assumption in the run notes.
LOCK_PHASE_DEFINITIONS = False  # Set True to use hard-coded phases
HARD_CODED_PHASES = {
    # Only used if LOCK_PHASE_DEFINITIONS = True
    # Protocol: OH16230 head imaging, 240s recording
    "baseline": (0, 60),
    "butanone_window": (60, 80),
    "inter1": (80, 120),
    "pentanedione_window": (120, 140),
    "inter2": (140, 180),
    "nacl_window": (180, 200),
    "post": (200, 240),
}

# This will be populated from metadata
PHASE_DEFINITIONS = {}

# Connectivity method parameters
DEFAULT_EDGE_DENSITY = 0.15  # 15% of possible edges

# Visualization settings
EXCITATORY_COLOR = "#E63946"  # Red
INHIBITORY_COLOR = "#457B9D"  # Blue
NEUTRAL_COLOR = "#A8DADC"     # Light teal
EDGE_ALPHA = 0.7

# =============================================================================
# PHASE DEFINITION FROM METADATA
# =============================================================================

def derive_phase_definitions_from_metadata(
    stim_names: List[str],
    stim_times: np.ndarray,
    total_duration: float = 240.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Derive phase window definitions from stimulus metadata.
    
    This is the correct approach: use the actual stimulus timing from the data
    rather than hard-coded values that may not match different datasets.
    
    Args:
        stim_names: List of stimulus names from dataset
        stim_times: Array of (start, end) times for each stimulus (in seconds)
        total_duration: Total recording duration in seconds
        
    Returns:
        Dict mapping phase name to (start, end) tuple
    """
    global PHASE_DEFINITIONS
    
    if LOCK_PHASE_DEFINITIONS:
        print("  ⚠ Using LOCKED (hard-coded) phase definitions")
        print("    This assumes OH16230 head imaging protocol with 240s recording")
        PHASE_DEFINITIONS = HARD_CODED_PHASES.copy()
        return PHASE_DEFINITIONS
    
    phases = {}
    
    # Sort stimuli by start time
    stim_order = np.argsort(stim_times[:, 0])
    
    # Baseline: from start until first stimulus
    first_stim_start = stim_times[stim_order[0], 0]
    phases["baseline"] = (0.0, first_stim_start)
    
    # Stimulus windows and inter-stimulus intervals
    prev_end = first_stim_start
    for idx, stim_idx in enumerate(stim_order):
        stim_name = stim_names[stim_idx].lower()
        stim_start, stim_end = stim_times[stim_idx]
        
        # If there's a gap since last phase, create inter-stimulus period
        if idx > 0 and stim_start > prev_end + 1:
            phases[f"inter{idx}"] = (prev_end, stim_start)
        
        # Stimulus window
        phases[f"{stim_name}_window"] = (stim_start, stim_end)
        prev_end = stim_end
    
    # Post-stimulus period: from last stimulus to end
    if prev_end < total_duration:
        phases["post"] = (prev_end, total_duration)
    
    # Log derived phases
    print("  Phase definitions (derived from metadata):")
    for name, (start, end) in sorted(phases.items(), key=lambda x: x[1][0]):
        print(f"    {name}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s duration)")
    
    PHASE_DEFINITIONS = phases
    return phases


# Minimum timepoints per phase for reliable estimation
MIN_TIMEPOINTS_PER_PHASE = 20


def create_uniform_phases(
    n_frames: int,
    n_phases: int,
    fps: float = 4.0
) -> Dict[str, Tuple[float, float]]:
    """
    Create uniform-duration temporal phases.
    
    Consolidated from script 10: Simple approach that divides the recording 
    into n equal segments. Useful as a baseline or when stimulus timing is unknown.
    
    Args:
        n_frames: Total number of frames in recording
        n_phases: Number of phases to create
        fps: Frames per second
    
    Returns:
        Dict mapping phase name to (start_sec, end_sec) tuple
    """
    frames_per_phase = n_frames // n_phases
    phases = {}
    
    for i in range(n_phases):
        start_frame = i * frames_per_phase
        end_frame = start_frame + frames_per_phase if i < n_phases - 1 else n_frames
        
        start_sec = start_frame / fps
        end_sec = end_frame / fps
        
        phases[f"phase_{i+1}"] = (start_sec, end_sec)
    
    return phases


def create_derivative_phases(
    X: np.ndarray,
    fps: float = 4.0,
    smoothing_window: int = 10,
    n_phases: int = 5
) -> Dict[str, Tuple[float, float]]:
    """
    Create phases based on aggregate activity derivative.
    
    Consolidated from script 10: Data-driven phase detection that finds transition
    points where population activity changes rapidly.
    
    Method:
    1. Compute mean activity across all neurons
    2. Smooth the signal
    3. Compute derivative (rate of change)
    4. Find peaks in absolute derivative (state transitions)
    5. Use peaks to define phase boundaries
    
    References:
        Allen et al. (2014). Cerebral Cortex - similar approach for fMRI
    
    Args:
        X: (T, n_neurons) timeseries matrix
        fps: Frames per second
        smoothing_window: Frames for smoothing (reduces noise)
        n_phases: Target number of phases
    
    Returns:
        Dict mapping phase name to (start_sec, end_sec) tuple
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d
    
    T = X.shape[0]
    
    # Compute population activity (mean across neurons)
    pop_activity = np.nanmean(X, axis=1)
    
    # Smooth
    pop_smooth = uniform_filter1d(pop_activity, size=smoothing_window)
    
    # Compute derivative
    derivative = np.gradient(pop_smooth)
    abs_derivative = np.abs(derivative)
    
    # Smooth the derivative too
    abs_deriv_smooth = uniform_filter1d(abs_derivative, size=smoothing_window)
    
    # Find peaks (state transitions)
    # Height threshold: above mean + 1 std
    threshold = np.mean(abs_deriv_smooth) + np.std(abs_deriv_smooth)
    peaks, properties = find_peaks(abs_deriv_smooth, height=threshold, distance=MIN_TIMEPOINTS_PER_PHASE)
    
    # If we found too few peaks, use uniform phases as fallback
    if len(peaks) < n_phases - 1:
        print(f"  Only found {len(peaks)} transitions, falling back to uniform phases")
        return create_uniform_phases(T, n_phases, fps)
    
    # Select top n_phases-1 peaks
    if len(peaks) > n_phases - 1:
        peak_heights = properties['peak_heights']
        top_indices = np.argsort(peak_heights)[::-1][:n_phases-1]
        peaks = np.sort(peaks[top_indices])
    
    # Create phases from peaks
    phases = {}
    boundaries = [0] + list(peaks) + [T]
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        if end - start >= MIN_TIMEPOINTS_PER_PHASE:
            start_sec = start / fps
            end_sec = end / fps
            phases[f"state_{i+1}"] = (start_sec, end_sec)
    
    return phases


# =============================================================================
# DATA LOADING
# =============================================================================

def load_neuropal_data() -> Dict:
    """Load raw NeuroPAL data."""
    mat_path = DATA_DIR / "Head_Activity_OH16230.mat"
    mat = loadmat(mat_path, simplify_cells=True)
    
    data = {
        "neuron_names": [str(n).strip().upper() for n in mat["neurons"]],
        "norm_traces": mat["norm_traces"],
        "fps": float(mat["fps"]),
        "stim_names": [str(s).lower() for s in mat["stim_names"]],
        "stim_times": np.asarray(mat["stim_times"], dtype=float),
        "n_worms": len(mat["files"]),
    }
    
    # Derive phase definitions from the loaded stimulus metadata
    derive_phase_definitions_from_metadata(
        stim_names=data["stim_names"],
        stim_times=data["stim_times"],
        total_duration=240.0,  # Could also infer from trace length
    )
    
    return data


def load_structural_connectome() -> Tuple[np.ndarray, List[str]]:
    """Load structural connectome."""
    A_struct = np.load(CONNECTOME_DIR / "A_struct.npy")
    with open(CONNECTOME_DIR / "nodes.json", 'r') as f:
        nodes = json.load(f)
    return A_struct, nodes


def build_worm_timeseries(
    neuropal_data: Dict,
    worm_idx: int,
    neuron_subset: List[str]
) -> Optional[np.ndarray]:
    """Build full timeseries matrix for one worm."""
    traces = neuropal_data["norm_traces"]
    names = neuropal_data["neuron_names"]
    n_worms = neuropal_data["n_worms"]
    
    name_to_idx = {n: i for i, n in enumerate(names)}
    
    columns = []
    for neuron_name in neuron_subset:
        if neuron_name not in name_to_idx:
            columns.append(None)
            continue
            
        idx = name_to_idx[neuron_name]
        neuron_traces = traces[idx]
        
        if not hasattr(neuron_traces, '__len__') or len(neuron_traces) == 0:
            columns.append(None)
            continue
        
        # Get trace for this worm
        trace_list = []
        for offset in [0, n_worms]:
            if worm_idx + offset < len(neuron_traces):
                t = neuron_traces[worm_idx + offset]
                if t is not None and hasattr(t, '__len__') and len(t) > 10:
                    trace_list.append(np.asarray(t, dtype=float))
        
        if trace_list:
            min_len = min(len(t) for t in trace_list)
            stacked = np.stack([t[:min_len] for t in trace_list])
            columns.append(stacked.mean(axis=0))
        else:
            columns.append(None)
    
    valid = [c is not None for c in columns]
    if not all(valid):
        return None
    
    min_len = min(len(c) for c in columns)
    X = np.stack([c[:min_len] for c in columns], axis=1)
    
    return X


def extract_phase_data(X: np.ndarray, phase_name: str, fps: float = 4.0) -> np.ndarray:
    """Extract data for a specific phase."""
    if phase_name not in PHASE_DEFINITIONS:
        raise ValueError(f"Unknown phase: {phase_name}")
    
    start_sec, end_sec = PHASE_DEFINITIONS[phase_name]
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), X.shape[0])
    
    return X[start_frame:end_frame, :]


# =============================================================================
# CONNECTIVITY ESTIMATION METHODS
# =============================================================================

def compute_connectivity_matrix(
    X_input: Union[np.ndarray, List[np.ndarray]],
    method: str = "pearson"
) -> np.ndarray:
    """
    Compute weighted connectivity matrix.
    
    Handles both single array (T, n) and list of arrays List[(T_w, n)].
    
    Returns signed weights: positive = excitatory-like, negative = inhibitory-like
    """
    # Standardize input to list of arrays
    if isinstance(X_input, np.ndarray):
        X_list = [X_input]
    else:
        # Filter None
        X_list = [x for x in X_input if x is not None]
    
    if not X_list:
        return np.array([])
        
    n = X_list[0].shape[1]
    
    if method == "pearson":
        # Pearson: concatenate all worms (no temporal structure assumed)
        X_all = np.vstack(X_list)
        # Remove NaNs
        X_clean = X_all[~np.isnan(X_all).any(axis=1)]
        
        adj = np.corrcoef(X_clean.T)
        adj = np.nan_to_num(adj, nan=0.0)
        
    elif method == "crosscorr":
        # Lag-1 cross-correlation (directed)
        # MUST compute per worm to avoid boundary artifacts
        
        cross_corr_matrices = []
        
        for X in X_list:
            if X.shape[0] < 5:
                continue
                
            nan_mask = np.isnan(X).any(axis=1)
            X_clean = X[~nan_mask]
            if X_clean.shape[0] < 5:
                continue
                
            X_t = X_clean[:-1, :]
            X_tp1 = X_clean[1:, :]
            
            # Vectorized lag-1 correlation
            # Standardize
            X_t_z = (X_t - X_t.mean(axis=0)) / (X_t.std(axis=0) + 1e-8)
            X_tp1_z = (X_tp1 - X_tp1.mean(axis=0)) / (X_tp1.std(axis=0) + 1e-8)
            
            # C = (1/T) * X_t.T @ X_tp1
            cc = (X_t_z.T @ X_tp1_z) / (X_t.shape[0] - 1)
            cross_corr_matrices.append(cc)
            
        if cross_corr_matrices:
            adj = np.mean(cross_corr_matrices, axis=0)
        else:
            adj = np.zeros((n, n))
            
    elif method == "partial":
        # For partial corr (covariance based), concatenation is generally safe
        X_all = np.vstack(X_list)
        X_clean = X_all[~np.isnan(X_all).any(axis=1)]
        
        from sklearn.covariance import LedoitWolf
        try:
            lw = LedoitWolf()
            lw.fit(X_clean)
            precision = lw.precision_
            d = np.sqrt(np.diag(precision))
            adj = -precision / np.outer(d, d)
            np.fill_diagonal(adj, 0)
        except Exception:
            adj = np.corrcoef(X_clean.T)
            adj = np.nan_to_num(adj, nan=0.0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    np.fill_diagonal(adj, 0)
    return adj


def threshold_to_binary(
    adj: np.ndarray, 
    density: float = DEFAULT_EDGE_DENSITY,
    keep_sign: bool = True
) -> np.ndarray:
    """
    Threshold adjacency matrix to target density.
    
    If keep_sign=True, returns {-1, 0, +1} based on original sign.
    If keep_sign=False, returns {0, 1} binary.
    """
    n = adj.shape[0]
    n_possible = n * (n - 1)
    n_target = int(density * n_possible)
    
    abs_adj = np.abs(adj.copy())
    np.fill_diagonal(abs_adj, 0)
    
    # Find threshold
    flat = abs_adj.flatten()
    if n_target >= len(flat):
        threshold = 0
    else:
        threshold = np.sort(flat)[::-1][max(0, n_target-1)]
    
    # Create binary mask
    mask = abs_adj >= threshold
    np.fill_diagonal(mask, False)
    
    if keep_sign:
        binary_adj = np.zeros_like(adj)
        binary_adj[mask & (adj > 0)] = 1
        binary_adj[mask & (adj < 0)] = -1
    else:
        binary_adj = mask.astype(int)
    
    return binary_adj


# =============================================================================
# TRAINING STRATEGIES
# =============================================================================

def train_global_model(
    all_worm_data: List[np.ndarray],
    method: str = "crosscorr"
) -> np.ndarray:
    """
    Strategy 1: Train on ALL data from all worms.
    
    Returns weighted adjacency matrix.
    """
    # Pass proper list of arrays to compute_connectivity_matrix
    # It will handle vstack internally for methods where it's safe (pearson)
    # and iterate properly for lag-methods (crosscorr)
    
    X_clean_list = [x for x in all_worm_data if x is not None]
    
    total_frames = sum(x.shape[0] for x in X_clean_list)
    print(f"  Global training: {total_frames} timepoints across {len(X_clean_list)} worms")
    
    return compute_connectivity_matrix(X_clean_list, method)


def train_stimulus_only(
    all_worm_data: List[np.ndarray],
    stimulus: str,
    fps: float = 4.0,
    method: str = "crosscorr"
) -> np.ndarray:
    """
    Strategy 2: Train ONLY on stimulus-specific windows.
    
    Uses the 10s window during stimulus presentation.
    """
    phase_name = f"{stimulus}_window"
    
    # Extract stimulus windows from each worm
    stimulus_data = []
    for X in all_worm_data:
        if X is None:
            continue
        X_phase = extract_phase_data(X, phase_name, fps)
        if X_phase.shape[0] > 5:
            stimulus_data.append(X_phase)
    
    if not stimulus_data:
        return None
    
    X_stim = np.vstack(stimulus_data)
    nan_mask = np.isnan(X_stim).any(axis=1)
    X_clean = X_stim[~nan_mask]
    
    print(f"  {stimulus} only: {X_clean.shape[0]} timepoints")
    
    return compute_connectivity_matrix(X_clean, method)


def train_global_then_finetune(
    all_worm_data: List[np.ndarray],
    stimulus: str,
    fps: float = 4.0,
    method: str = "crosscorr",
    finetune_weight: float = 0.3
) -> np.ndarray:
    """
    Strategy 3: Pre-train on global, fine-tune on stimulus.
    
    Simple implementation: weighted average of global and stimulus-specific.
    finetune_weight controls how much the stimulus-specific data influences.
    """
    # Get global connectivity
    adj_global = train_global_model(all_worm_data, method)
    
    # Get stimulus-specific connectivity
    adj_stimulus = train_stimulus_only(all_worm_data, stimulus, fps, method)
    
    if adj_stimulus is None:
        return adj_global
    
    # Weighted combination
    adj_finetuned = (1 - finetune_weight) * adj_global + finetune_weight * adj_stimulus
    
    return adj_finetuned


def train_phase_specific(
    all_worm_data: List[np.ndarray],
    phase_name: str,
    fps: float = 4.0,
    method: str = "crosscorr"
) -> np.ndarray:
    """
    Strategy 4: Train on specific time phase only.
    """
    phase_data = []
    for X in all_worm_data:
        if X is None:
            continue
        try:
            X_phase = extract_phase_data(X, phase_name, fps)
            if X_phase.shape[0] > 5:
                phase_data.append(X_phase)
        except Exception:
            continue
    
    if not phase_data:
        return None
    
    X_phase = np.vstack(phase_data)
    nan_mask = np.isnan(X_phase).any(axis=1)
    X_clean = X_phase[~nan_mask]
    
    print(f"  {phase_name}: {X_clean.shape[0]} timepoints")
    
    return compute_connectivity_matrix(X_clean, method)


# =============================================================================
# SBTG TRAINING FUNCTIONS
# =============================================================================

# Phase definitions for SBTG training (in seconds)
SBTG_PHASES = {
    'baseline': {'start_s': 0, 'end_s': 60, 'label': 'Baseline (0-60s)'},
    'butanone': {'start_s': 60.5, 'end_s': 70.5, 'label': 'Butanone (60.5-70.5s)'},
    'pentanedione': {'start_s': 120.5, 'end_s': 130.5, 'label': 'Pentanedione (120.5-130.5s)'},
    'nacl': {'start_s': 180.5, 'end_s': 190.5, 'label': 'NaCl (180.5-190.5s)'},
}


def load_imputed_phase_data(phase_name: str, fps: float = 4.0) -> List[np.ndarray]:
    """Load data for a specific phase from imputed full traces."""
    imputed_dir = DATASETS_DIR / 'full_traces_imputed'
    
    if not imputed_dir.exists():
        raise FileNotFoundError(f"Imputed data not found at {imputed_dir}. Run 01_prepare_data.py --impute-missing first.")
    
    X_segments = np.load(imputed_dir / 'X_segments.npy', allow_pickle=True)
    
    phase = SBTG_PHASES[phase_name]
    start_frame = int(phase['start_s'] * fps)
    end_frame = int(phase['end_s'] * fps)
    
    phase_segments = []
    for worm_data in X_segments:
        if worm_data is None:
            continue
        if end_frame <= worm_data.shape[0]:
            segment = worm_data[start_frame:end_frame, :]
            # Remove NaN rows
            nan_rows = np.any(np.isnan(segment), axis=1)
            segment_clean = segment[~nan_rows]
            if segment_clean.shape[0] >= 5:
                phase_segments.append(segment_clean)
    
    return phase_segments


def get_sbtg_hyperparams(phase_name: str, quick: bool = False) -> Dict:
    """Get hyperparameters for SBTG training."""
    # Start with phase-specific optimal params if available
    if phase_name in PHASE_OPTIMAL_PARAMS:
        params = PHASE_OPTIMAL_PARAMS[phase_name].copy()
    else:
        params = OPTIMIZED_HYPERPARAMS.copy()
    
    # Add required defaults
    params.update({
        'window_length': 2,
        'dsm_batch_size': 128,
        'dsm_num_layers': 3,
        'structured_num_layers': 2,
        'structured_init_scale': 0.1,
        'hac_max_lag': 5,
        'fdr_method': 'by',
        'train_frac': 0.7,
        'train_split': 'odd_even',
        'verbose': True,
        'inference_mode': 'in_sample',
    })
    
    if quick:
        params['dsm_epochs'] = min(params.get('dsm_epochs', 100), 20)
    
    return params


def train_sbtg_direct(
    phase_segments: List[np.ndarray],
    phase_name: str,
    hyperparams: Optional[Dict] = None,
    quick: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Train SBTG model directly on phase data (from scratch).
    
    Args:
        phase_segments: List of (T, n_neurons) arrays for this phase
        phase_name: Name of the phase
        hyperparams: Optional hyperparameters (uses defaults if None)
        quick: If True, use reduced epochs
    
    Returns:
        sign_adj: Signed adjacency matrix
        mu_hat: Coupling strength matrix
        metrics: Training metrics dict
    """
    if hyperparams is None:
        hyperparams = get_sbtg_hyperparams(phase_name, quick)
    
    total_frames = sum(seg.shape[0] for seg in phase_segments)
    print(f"    Training SBTG direct on {phase_name}: {len(phase_segments)} worms, {total_frames} frames")
    
    try:
        estimator = SBTGStructuredVolatilityEstimator(**hyperparams)
        result = estimator.fit(phase_segments)
        
        # Negate to match Leifer dFF polarity convention
        sign_adj = -result.sign_adj
        mu_hat = -result.mu_hat
        
        n_exc = int((sign_adj > 0).sum())
        n_inh = int((sign_adj < 0).sum())
        n_total = n_exc + n_inh
        ei_ratio = n_exc / n_inh if n_inh > 0 else float('inf')
        
        metrics = {
            'phase': phase_name,
            'method': 'sbtg_direct',
            'n_worms': len(phase_segments),
            'n_frames': total_frames,
            'n_excitatory': n_exc,
            'n_inhibitory': n_inh,
            'n_total': n_total,
            'ei_ratio': ei_ratio,
            'model_type': hyperparams.get('model_type', 'regime_gated'),
        }
        
        print(f"    → {n_total} edges (E:{n_exc}, I:{n_inh}), E:I = {ei_ratio:.3f}")
        
        return sign_adj, mu_hat, metrics
        
    except Exception as e:
        print(f"    ERROR training SBTG: {e}")
        n = phase_segments[0].shape[1] if phase_segments else 80
        return np.zeros((n, n)), np.zeros((n, n)), {'phase': phase_name, 'method': 'sbtg_direct', 'error': str(e)}


def train_sbtg_transfer(
    baseline_result: Tuple[np.ndarray, np.ndarray],
    phase_segments: List[np.ndarray],
    phase_name: str,
    hyperparams: Optional[Dict] = None,
    quick: bool = False,
    transfer_weight: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Train SBTG with transfer learning from baseline.
    
    Transfer approach: Train on phase data, then blend with baseline prior.
    This helps when phase-specific data is limited.
    
    Args:
        baseline_result: (sign_adj, mu_hat) from baseline training
        phase_segments: List of (T, n_neurons) arrays for this phase
        phase_name: Name of the phase
        hyperparams: Optional hyperparameters
        quick: If True, use reduced epochs
        transfer_weight: How much to weight the phase-specific model (0-1)
    
    Returns:
        sign_adj: Signed adjacency matrix (blended)
        mu_hat: Coupling strength matrix (blended)
        metrics: Training metrics dict
    """
    baseline_sign_adj, baseline_mu_hat = baseline_result
    
    if hyperparams is None:
        hyperparams = get_sbtg_hyperparams(phase_name, quick)
    
    # Halve DSM epochs for transfer learning (blended with baseline prior)
    if not quick:
        hyperparams = hyperparams.copy()
        hyperparams['dsm_epochs'] = max(50, hyperparams.get('dsm_epochs', 100) // 2)
    
    total_frames = sum(seg.shape[0] for seg in phase_segments)
    print(f"    Training SBTG transfer on {phase_name}: {len(phase_segments)} worms, {total_frames} frames")
    print(f"    Transfer weight: {transfer_weight:.2f} (phase) vs {1-transfer_weight:.2f} (baseline)")
    
    try:
        estimator = SBTGStructuredVolatilityEstimator(**hyperparams)
        result = estimator.fit(phase_segments)
        
        # Negate to match Leifer dFF polarity convention
        phase_sign_adj = -result.sign_adj
        phase_mu_hat = -result.mu_hat
        
        # Blend with baseline
        # For sign_adj: take phase-specific where confident, else baseline
        # For mu_hat: weighted average
        blended_mu_hat = transfer_weight * phase_mu_hat + (1 - transfer_weight) * baseline_mu_hat
        
        # For sign_adj: use phase-specific signs where phase found edges,
        # otherwise fallback to baseline
        phase_has_edge = phase_sign_adj != 0
        blended_sign_adj = np.where(phase_has_edge, phase_sign_adj, baseline_sign_adj)
        
        # Apply threshold based on blended mu_hat
        abs_mu = np.abs(blended_mu_hat)
        threshold = np.percentile(abs_mu[abs_mu > 0], 85) if (abs_mu > 0).sum() > 0 else 0
        final_sign_adj = np.where(abs_mu >= threshold, np.sign(blended_mu_hat), 0).astype(int)
        
        n_exc = int((final_sign_adj > 0).sum())
        n_inh = int((final_sign_adj < 0).sum())
        n_total = n_exc + n_inh
        ei_ratio = n_exc / n_inh if n_inh > 0 else float('inf')
        
        # Track how many edges came from each source
        from_phase = int((phase_has_edge & (final_sign_adj != 0)).sum())
        from_baseline = int(((~phase_has_edge) & (final_sign_adj != 0)).sum())
        
        metrics = {
            'phase': phase_name,
            'method': 'sbtg_transfer',
            'n_worms': len(phase_segments),
            'n_frames': total_frames,
            'n_excitatory': n_exc,
            'n_inhibitory': n_inh,
            'n_total': n_total,
            'ei_ratio': ei_ratio,
            'edges_from_phase': from_phase,
            'edges_from_baseline': from_baseline,
            'transfer_weight': transfer_weight,
            'model_type': hyperparams.get('model_type', 'regime_gated'),
        }
        
        print(f"    → {n_total} edges (E:{n_exc}, I:{n_inh}), E:I = {ei_ratio:.3f}")
        print(f"    → {from_phase} from phase, {from_baseline} from baseline")
        
        return final_sign_adj, blended_mu_hat, metrics
        
    except Exception as e:
        print(f"    ERROR training SBTG transfer: {e}")
        return baseline_sign_adj.copy(), baseline_mu_hat.copy(), {
            'phase': phase_name, 'method': 'sbtg_transfer', 'error': str(e)
        }


def create_sbtg_hp_objective(X_train: List[np.ndarray], X_val: List[np.ndarray]):
    """
    Create Optuna objective for SBTG hyperparameter search.
    
    Uses null_contrast as the objective (validated in Script 12 to correlate with
    biological AUROC better than DSM loss or edge_stability).
    
    Null contrast = (real_signal - null_mean) / null_std
    where null is computed from shuffled data.
    Higher null_contrast → stronger signal above noise → better AUROC.
    """
    
    def objective(trial):
        params = {
            'dsm_lr': trial.suggest_float('dsm_lr', 1e-5, 5e-3, log=True),
            'dsm_epochs': trial.suggest_int('dsm_epochs', 50, 300, step=25),
            'dsm_noise_std': trial.suggest_float('dsm_noise_std', 0.1, 0.8, step=0.05),
            'dsm_hidden_dim': trial.suggest_categorical('dsm_hidden_dim', [64, 128, 256]),
            'structured_hidden_dim': trial.suggest_categorical('structured_hidden_dim', [32, 64, 128]),
            'structured_l1_lambda': trial.suggest_float('structured_l1_lambda', 1e-5, 0.1, log=True),
            'fdr_alpha': trial.suggest_categorical('fdr_alpha', [0.1, 0.15, 0.2, 0.25]),
            'model_type': trial.suggest_categorical('model_type', ['linear', 'feature_bilinear', 'regime_gated']),
            'train_split': 'odd_even',
        }
        
        # Model-specific params
        if params['model_type'] == 'feature_bilinear':
            params['feature_dim'] = trial.suggest_categorical('feature_dim', [16, 32])
        elif params['model_type'] == 'regime_gated':
            params['num_regimes'] = trial.suggest_int('num_regimes', 2, 3)
            params['gate_hidden_dim'] = 64
        
        # Fixed params
        params.update({
            'window_length': 2,
            'dsm_batch_size': 128,
            'dsm_num_layers': 3,
            'structured_num_layers': 2,
            'structured_init_scale': 0.1,
            'hac_max_lag': 5,
            'fdr_method': 'by',
            'train_frac': 0.7,
            'verbose': False,
            'inference_mode': 'in_sample',
        })
        
        try:
            estimator = SBTGStructuredVolatilityEstimator(**params)
            result = estimator.fit(X_train)
            
            if result is None or result.mu_hat is None:
                return float('inf')
            
            # Compute NULL CONTRAST (this is the objective per Script 12 validation)
            # Higher null_contrast = stronger signal above shuffled null = better AUROC
            mu_hat = result.mu_hat
            n = mu_hat.shape[0]
            mask = ~np.eye(n, dtype=bool)
            real_signal = np.abs(mu_hat[mask]).mean()
            
            # Compute null from shuffled validation data (quick - 3 shuffles)
            null_signals = []
            for _ in range(3):
                X_val_shuffled = []
                for x in X_val:
                    perm = np.random.permutation(len(x))
                    X_val_shuffled.append(x[perm])
                
                params_null = params.copy()
                params_null['dsm_epochs'] = min(params['dsm_epochs'] // 3, 30)
                params_null['verbose'] = False
                estimator_null = SBTGStructuredVolatilityEstimator(**params_null)
                result_null = estimator_null.fit(X_val_shuffled)
                
                if result_null is not None and result_null.mu_hat is not None:
                    null_signal = np.abs(result_null.mu_hat[mask]).mean()
                    null_signals.append(null_signal)
            
            if len(null_signals) == 0:
                return float('inf')
            
            null_mean = np.mean(null_signals)
            null_std = np.std(null_signals) + 1e-8
            null_contrast = (real_signal - null_mean) / null_std
            
            if np.isnan(null_contrast) or np.isinf(null_contrast):
                return float('inf')
            
            trial.set_user_attr('null_contrast', null_contrast)
            trial.set_user_attr('model_type', params['model_type'])
            trial.set_user_attr('n_edges', int((result.sign_adj != 0).sum()))
            trial.set_user_attr('real_signal', real_signal)
            
            # Return negative null_contrast (Optuna minimizes, we want to maximize contrast)
            return -null_contrast
            
        except Exception as e:
            return float('inf')
    
    return objective


def run_sbtg_hp_search(
    phase_segments: List[np.ndarray],
    phase_name: str,
    n_trials: int = 30
) -> Tuple[Dict, float]:
    """
    Run Optuna hyperparameter search for a phase using null_contrast objective.
    
    The null_contrast objective was validated in Script 12 to correlate with
    biological AUROC better than DSM loss or edge_stability.
    
    Returns:
        best_params: Best hyperparameters found
        best_null_contrast: Best null contrast (higher = better)
    """
    if not OPTUNA_AVAILABLE:
        print("    Optuna not available, using default hyperparams")
        return get_sbtg_hyperparams(phase_name), float('nan')
    
    n_worms = len(phase_segments)
    if n_worms < 5:
        print(f"    Only {n_worms} worms, using default hyperparams")
        return get_sbtg_hyperparams(phase_name), float('nan')
    
    # Split into train/val
    n_val = max(3, n_worms // 5)
    np.random.seed(42)
    indices = np.random.permutation(n_worms)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train = [phase_segments[i] for i in train_indices]
    X_val = [phase_segments[i] for i in val_indices]
    
    print(f"    HP search (null_contrast objective): {len(X_train)} train, {len(X_val)} val worms, {n_trials} trials")
    
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=min(10, n_trials // 3))
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    objective = create_sbtg_hp_objective(X_train, X_val)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    
    best_params = study.best_trial.params
    best_null_contrast = -study.best_trial.value  # Convert back to positive (we minimized -null_contrast)
    
    # Add model-specific defaults
    if best_params.get('model_type') == 'regime_gated' and 'num_regimes' not in best_params:
        best_params['num_regimes'] = 2
    
    print(f"    Best: {best_params.get('model_type', 'unknown')} with null_contrast={best_null_contrast:.4f}")
    
    return best_params, best_null_contrast


# =============================================================================
# GRAPH DIFFERENCE ANALYSIS
# =============================================================================

def compute_graph_difference(
    adj1: np.ndarray,
    adj2: np.ndarray,
    name1: str = "graph1",
    name2: str = "graph2"
) -> Dict:
    """
    Compute differences between two adjacency matrices.
    
    Returns:
        - edges_only_in_1: edges present in adj1 but not adj2
        - edges_only_in_2: edges present in adj2 but not adj1
        - sign_changes: edges where sign flipped
        - correlation: overall correlation between matrices
    """
    # Binarize for edge comparison
    binary1 = (np.abs(adj1) > 0).astype(int)
    binary2 = (np.abs(adj2) > 0).astype(int)
    
    # Edge-level comparison
    only_in_1 = ((binary1 == 1) & (binary2 == 0)).sum()
    only_in_2 = ((binary1 == 0) & (binary2 == 1)).sum()
    shared = ((binary1 == 1) & (binary2 == 1)).sum()
    
    # Sign changes (among shared edges)
    sign1 = np.sign(adj1)
    sign2 = np.sign(adj2)
    shared_mask = (binary1 == 1) & (binary2 == 1)
    sign_changes = ((sign1 != sign2) & shared_mask).sum()
    
    # Correlation of weights
    mask = ~np.eye(adj1.shape[0], dtype=bool)
    corr = np.corrcoef(adj1[mask].flatten(), adj2[mask].flatten())[0, 1]
    
    return {
        f"edges_only_in_{name1}": int(only_in_1),
        f"edges_only_in_{name2}": int(only_in_2),
        "shared_edges": int(shared),
        "sign_changes": int(sign_changes),
        "weight_correlation": float(corr) if not np.isnan(corr) else 0.0,
        "jaccard_similarity": shared / (only_in_1 + only_in_2 + shared) if (only_in_1 + only_in_2 + shared) > 0 else 0.0,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================




# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_vs_connectome(
    pred_adj: np.ndarray,
    true_adj: np.ndarray
) -> Dict:
    """Evaluate predicted adjacency against structural connectome."""
    n = pred_adj.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    y_pred = (np.abs(pred_adj[mask]) > 0).astype(int).flatten()
    y_true = (true_adj[mask] > 0).astype(int).flatten()
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Random baseline
    n_possible = n * (n - 1)
    n_true = y_true.sum()
    random_precision = n_true / n_possible
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "n_edges": int(y_pred.sum()),
        "random_precision": random_precision,
        "f1_vs_random": f1 / (2 * random_precision) if random_precision > 0 else 0,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_sbtg_analysis(args, output_dir: Path) -> pd.DataFrame:
    """
    Run SBTG-based temporal analysis with direct and/or transfer training.
    
    Returns DataFrame with metrics for all phases and methods.
    """
    print("\n" + "=" * 80)
    print("SBTG TEMPORAL ANALYSIS")
    print("=" * 80)
    
    sbtg_output_dir = output_dir / "sbtg"
    sbtg_output_dir.mkdir(parents=True, exist_ok=True)
    (sbtg_output_dir / "adjacencies").mkdir(exist_ok=True)
    
    # Load neuron names for reference
    imputed_dir = DATASETS_DIR / 'full_traces_imputed'
    if not imputed_dir.exists():
        print("ERROR: Imputed data not found. Run: python pipeline/01_prepare_data.py --impute-missing")
        return pd.DataFrame()
    
    with open(imputed_dir / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    print(f"\nConfiguration:")
    print(f"  Quick mode: {args.quick}")
    print(f"  Transfer learning: {args.transfer}")
    print(f"  HP search: {args.hp_search} (n_trials={args.n_trials})")
    print(f"  Phases: {list(SBTG_PHASES.keys())}")
    
    all_results = []
    phase_results = {}
    
    # =========================================================================
    # STEP 1: Train baseline (always needed for transfer)
    # =========================================================================
    print("\n" + "-" * 60)
    print("PHASE: baseline (0-60s)")
    print("-" * 60)
    
    baseline_segments = load_imputed_phase_data('baseline')
    print(f"  Loaded {len(baseline_segments)} worm segments")
    
    # HP search for baseline if requested
    if args.hp_search:
        print("\n  [HP Search] Baseline...")
        baseline_params, baseline_loss = run_sbtg_hp_search(
            baseline_segments, 'baseline', n_trials=args.n_trials
        )
    else:
        baseline_params = None
    
    # Train baseline model (always direct for baseline)
    baseline_sign_adj, baseline_mu_hat, baseline_metrics = train_sbtg_direct(
        baseline_segments, 'baseline', hyperparams=baseline_params, quick=args.quick
    )
    
    baseline_metrics['training_type'] = 'direct'
    all_results.append(baseline_metrics)
    phase_results['baseline'] = {
        'direct': {'sign_adj': baseline_sign_adj, 'mu_hat': baseline_mu_hat, 'metrics': baseline_metrics}
    }
    
    # Save baseline
    np.save(sbtg_output_dir / "adjacencies" / "baseline_direct_sign_adj.npy", baseline_sign_adj)
    np.save(sbtg_output_dir / "adjacencies" / "baseline_direct_mu_hat.npy", baseline_mu_hat)
    
    # =========================================================================
    # STEP 2: Train stimulus phases (direct and/or transfer)
    # =========================================================================
    stimulus_phases = ['butanone', 'pentanedione', 'nacl']
    
    for phase_name in stimulus_phases:
        print("\n" + "-" * 60)
        print(f"PHASE: {phase_name} ({SBTG_PHASES[phase_name]['label']})")
        print("-" * 60)
        
        phase_segments = load_imputed_phase_data(phase_name)
        print(f"  Loaded {len(phase_segments)} worm segments")
        
        phase_results[phase_name] = {}
        
        # HP search if requested
        if args.hp_search:
            print(f"\n  [HP Search] {phase_name}...")
            phase_params, phase_loss = run_sbtg_hp_search(
                phase_segments, phase_name, n_trials=args.n_trials
            )
        else:
            phase_params = None
        
        # DIRECT training
        print(f"\n  [DIRECT] Training from scratch...")
        direct_sign_adj, direct_mu_hat, direct_metrics = train_sbtg_direct(
            phase_segments, phase_name, hyperparams=phase_params, quick=args.quick
        )
        direct_metrics['training_type'] = 'direct'
        all_results.append(direct_metrics)
        phase_results[phase_name]['direct'] = {
            'sign_adj': direct_sign_adj, 'mu_hat': direct_mu_hat, 'metrics': direct_metrics
        }
        
        # Save direct results
        np.save(sbtg_output_dir / "adjacencies" / f"{phase_name}_direct_sign_adj.npy", direct_sign_adj)
        np.save(sbtg_output_dir / "adjacencies" / f"{phase_name}_direct_mu_hat.npy", direct_mu_hat)
        
        # TRANSFER training (if enabled)
        if args.transfer:
            print(f"\n  [TRANSFER] Training with baseline prior...")
            transfer_sign_adj, transfer_mu_hat, transfer_metrics = train_sbtg_transfer(
                baseline_result=(baseline_sign_adj, baseline_mu_hat),
                phase_segments=phase_segments,
                phase_name=phase_name,
                hyperparams=phase_params,
                quick=args.quick,
                transfer_weight=args.transfer_weight
            )
            transfer_metrics['training_type'] = 'transfer'
            all_results.append(transfer_metrics)
            phase_results[phase_name]['transfer'] = {
                'sign_adj': transfer_sign_adj, 'mu_hat': transfer_mu_hat, 'metrics': transfer_metrics
            }
            
            # Save transfer results
            np.save(sbtg_output_dir / "adjacencies" / f"{phase_name}_transfer_sign_adj.npy", transfer_sign_adj)
            np.save(sbtg_output_dir / "adjacencies" / f"{phase_name}_transfer_mu_hat.npy", transfer_mu_hat)
    
    # =========================================================================
    # STEP 3: Compare direct vs transfer
    # =========================================================================
    if args.transfer:
        print("\n" + "=" * 60)
        print("DIRECT vs TRANSFER COMPARISON")
        print("=" * 60)
        
        comparison_results = []
        for phase_name in stimulus_phases:
            if 'direct' in phase_results[phase_name] and 'transfer' in phase_results[phase_name]:
                direct = phase_results[phase_name]['direct']['metrics']
                transfer = phase_results[phase_name]['transfer']['metrics']
                
                comparison = {
                    'phase': phase_name,
                    'direct_edges': direct.get('n_total', 0),
                    'transfer_edges': transfer.get('n_total', 0),
                    'direct_ei_ratio': direct.get('ei_ratio', 0),
                    'transfer_ei_ratio': transfer.get('ei_ratio', 0),
                    'edges_from_baseline': transfer.get('edges_from_baseline', 0),
                }
                comparison_results.append(comparison)
                
                print(f"\n  {phase_name}:")
                print(f"    Direct:   {direct.get('n_total', 0)} edges, E:I = {direct.get('ei_ratio', 0):.3f}")
                print(f"    Transfer: {transfer.get('n_total', 0)} edges, E:I = {transfer.get('ei_ratio', 0):.3f}")
                print(f"              ({transfer.get('edges_from_baseline', 0)} edges from baseline)")
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv(sbtg_output_dir / "direct_vs_transfer_comparison.csv", index=False)
    
    # =========================================================================
    # STEP 4: Save summary results
    # =========================================================================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(sbtg_output_dir / "sbtg_phase_results.csv", index=False)
    
    # Save phase results JSON (for figure generation)
    phase_summary = {}
    for phase_name, methods in phase_results.items():
        phase_summary[phase_name] = {}
        for method_name, data in methods.items():
            metrics = data['metrics']
            phase_summary[phase_name][method_name] = {
                'n_positive': metrics.get('n_excitatory', 0),
                'n_negative': metrics.get('n_inhibitory', 0),
                'n_total': metrics.get('n_total', 0),
                'ei_ratio': metrics.get('ei_ratio', 0),
                'method': metrics.get('method', 'sbtg'),
                'model_type': metrics.get('model_type', 'regime_gated'),
            }
    
    with open(sbtg_output_dir / "phase_results.json", 'w') as f:
        json.dump(phase_summary, f, indent=2)
    
    # Also save to sbtg_temporal for compatibility with existing figure scripts
    temporal_dir = PROJECT_ROOT / "results" / "sbtg_temporal"
    temporal_dir.mkdir(parents=True, exist_ok=True)
    
    # Use direct results for the standard output (or transfer if only transfer was run)
    for phase_name in SBTG_PHASES.keys():
        if phase_name in phase_results:
            method = 'direct' if 'direct' in phase_results[phase_name] else 'transfer'
            np.save(temporal_dir / f"sign_adj_{phase_name}.npy", 
                   phase_results[phase_name][method]['sign_adj'])
    
    # Save simplified phase_results.json for Figure 7/9
    simplified_results = {}
    for phase_name in SBTG_PHASES.keys():
        if phase_name in phase_results:
            method = 'direct' if 'direct' in phase_results[phase_name] else 'transfer'
            m = phase_results[phase_name][method]['metrics']
            simplified_results[phase_name] = {
                'n_positive': m.get('n_excitatory', 0),
                'n_negative': m.get('n_inhibitory', 0),
                'n_total': m.get('n_total', 0),
                'ei_ratio': m.get('ei_ratio', 0),
                'method': 'sbtg',
            }
    
    with open(temporal_dir / "phase_results.json", 'w') as f:
        json.dump(simplified_results, f, indent=2)
    
    print(f"\n  ✓ SBTG results saved to: {sbtg_output_dir}")
    print(f"  ✓ Phase data saved to: {temporal_dir}")
    
    return results_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Temporal and stimulus-specific connectivity analysis")
    
    # Mode selection
    parser.add_argument("--sbtg", action="store_true", 
                        help="Use SBTG models instead of correlation methods")
    parser.add_argument("--transfer", action="store_true",
                        help="Enable transfer learning (pre-train on baseline)")
    parser.add_argument("--transfer-weight", type=float, default=0.3,
                        help="Weight for phase-specific model in transfer (0-1)")
    
    # HP search
    parser.add_argument("--hp-search", action="store_true",
                        help="Run Optuna hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of HP search trials per phase")
    
    # General options
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer epochs)")
    parser.add_argument("--method", default="crosscorr", 
                        choices=["pearson", "crosscorr", "partial"],
                        help="Correlation method (when not using --sbtg)")
    parser.add_argument("--density", type=float, default=DEFAULT_EDGE_DENSITY,
                        help="Edge density for thresholding")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEMPORAL CONNECTIVITY ANALYSIS")
    print("=" * 80)
    print(f"\nMode: {'SBTG' if args.sbtg else 'Correlation-based'}")
    if args.sbtg:
        print(f"Transfer learning: {args.transfer}")
        print(f"HP search: {args.hp_search}")
    else:
        print(f"Method: {args.method}")
    print(f"Quick mode: {args.quick}")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "adjacencies").mkdir(exist_ok=True)
    
    # =========================================================================
    # SBTG MODE
    # =========================================================================
    if args.sbtg:
        sbtg_results = run_sbtg_analysis(args, OUTPUT_DIR)
        
        print("\n" + "=" * 80)
        print("SBTG ANALYSIS COMPLETE")
        print("=" * 80)
        
        if not sbtg_results.empty:
            print("\nResults Summary:")
            print(sbtg_results.to_string(index=False))
        
        return
    
    # =========================================================================
    # CORRELATION MODE (original behavior)
    # =========================================================================
    print(f"\nMethod: {args.method}")
    print(f"Edge density: {args.density:.1%}")
    
    # Load data
    print("\n[1/6] Loading data...")
    neuropal_data = load_neuropal_data()
    A_struct, struct_neurons = load_structural_connectome()
    
    print(f"  NeuroPAL: {len(neuropal_data['neuron_names'])} neurons, {neuropal_data['n_worms']} worms")
    print(f"  Connectome: {len(struct_neurons)} neurons")
    
    # Save phase definitions for auditability
    phase_defs_file = OUTPUT_DIR / "phase_definitions.json"
    with open(phase_defs_file, 'w') as f:
        json.dump({
            "source": "derived_from_metadata" if not LOCK_PHASE_DEFINITIONS else "hard_coded",
            "phases": {k: list(v) for k, v in PHASE_DEFINITIONS.items()},
            "stim_names": neuropal_data["stim_names"],
            "stim_times": neuropal_data["stim_times"].tolist(),
        }, f, indent=2)
    print(f"  ✓ Saved phase definitions to {phase_defs_file}")
    
    # Find overlapping neurons
    overlap_neurons = [n for n in struct_neurons if n in neuropal_data["neuron_names"]]
    struct_idx = [struct_neurons.index(n) for n in overlap_neurons]
    A_struct_aligned = A_struct[np.ix_(struct_idx, struct_idx)]
    
    print(f"  Overlap: {len(overlap_neurons)} neurons")
    
    # Build timeseries for all worms
    print("\n[2/6] Building timeseries for all worms...")
    all_worm_data = []
    fps = neuropal_data["fps"]
    
    for worm_idx in range(neuropal_data["n_worms"]):
        X = build_worm_timeseries(neuropal_data, worm_idx, overlap_neurons)
        if X is not None and X.shape[0] > 100:
            all_worm_data.append(X)
            print(f"  Worm {worm_idx}: {X.shape[0]} frames × {X.shape[1]} neurons")
    
    print(f"  Total: {len(all_worm_data)} complete worms")
    
    if len(all_worm_data) == 0:
        print("ERROR: No complete worm data found")
        return
    
    # ==========================================================================
    # TRAINING STRATEGY 1: GLOBAL
    # ==========================================================================
    print("\n[3/6] Training strategies...")
    print("\n  === STRATEGY 1: GLOBAL (all data) ===")
    
    adj_global = train_global_model(all_worm_data, args.method)
    adj_global_binary = threshold_to_binary(adj_global, args.density, keep_sign=True)
    
    metrics_global = evaluate_vs_connectome(adj_global_binary, A_struct_aligned)
    print(f"  Global: F1={metrics_global['f1_score']:.3f}, "
          f"P={metrics_global['precision']:.3f}, R={metrics_global['recall']:.3f}")
    
    # ==========================================================================
    # TRAINING STRATEGY 2: STIMULUS-ONLY
    # ==========================================================================
    print("\n  === STRATEGY 2: STIMULUS-ONLY ===")
    
    adj_stimulus_only = {}
    for stim in STIMULI:
        adj = train_stimulus_only(all_worm_data, stim, fps, args.method)
        if adj is not None:
            adj_stimulus_only[stim] = threshold_to_binary(adj, args.density, keep_sign=True)
            metrics = evaluate_vs_connectome(adj_stimulus_only[stim], A_struct_aligned)
            print(f"  {stim}: F1={metrics['f1_score']:.3f}, "
                  f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    # ==========================================================================
    # TRAINING STRATEGY 3: GLOBAL + FINE-TUNE
    # ==========================================================================
    print("\n  === STRATEGY 3: GLOBAL + FINE-TUNE ===")
    
    adj_finetuned = {}
    for stim in STIMULI:
        adj = train_global_then_finetune(all_worm_data, stim, fps, args.method)
        adj_finetuned[stim] = threshold_to_binary(adj, args.density, keep_sign=True)
        metrics = evaluate_vs_connectome(adj_finetuned[stim], A_struct_aligned)
        print(f"  {stim} (finetuned): F1={metrics['f1_score']:.3f}, "
              f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    # ==========================================================================
    # TRAINING STRATEGY 4: PHASE-SPECIFIC
    # ==========================================================================
    print("\n  === STRATEGY 4: PHASE-SPECIFIC ===")
    
    phase_adjacencies = {}
    phase_metrics = []
    
    for phase_name in PHASE_DEFINITIONS.keys():
        adj = train_phase_specific(all_worm_data, phase_name, fps, args.method)
        if adj is not None:
            adj_binary = threshold_to_binary(adj, args.density, keep_sign=True)
            phase_adjacencies[phase_name] = adj_binary
            
            metrics = evaluate_vs_connectome(adj_binary, A_struct_aligned)
            metrics["phase"] = phase_name
            phase_metrics.append(metrics)
            
            print(f"  {phase_name}: F1={metrics['f1_score']:.3f}, "
                  f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
    
    # ==========================================================================
    # GRAPH DIFFERENCE ANALYSIS
    # ==========================================================================
    print("\n[4/6] Computing graph differences...")
    
    differences = []
    
    # Compare global vs each stimulus-only
    for stim, adj_stim in adj_stimulus_only.items():
        diff = compute_graph_difference(adj_global_binary, adj_stim, "global", f"{stim}_only")
        diff["comparison"] = f"global_vs_{stim}_only"
        differences.append(diff)
        print(f"  Global vs {stim}-only: Jaccard={diff['jaccard_similarity']:.3f}, "
              f"r={diff['weight_correlation']:.3f}")
    
    # Compare global vs finetuned
    for stim, adj_ft in adj_finetuned.items():
        diff = compute_graph_difference(adj_global_binary, adj_ft, "global", f"{stim}_finetuned")
        diff["comparison"] = f"global_vs_{stim}_finetuned"
        differences.append(diff)
    
    # Compare adjacent phases
    phase_names = list(phase_adjacencies.keys())
    for i in range(len(phase_names) - 1):
        p1, p2 = phase_names[i], phase_names[i + 1]
        diff = compute_graph_difference(phase_adjacencies[p1], phase_adjacencies[p2], p1, p2)
        diff["comparison"] = f"{p1}_vs_{p2}"
        differences.append(diff)
    
    # ==========================================================================
    # VISUALIZATIONS
    # ==========================================================================
    print("\n[5/6] Generating visualizations...")
    
    # Grid of phase-specific graphs
    print("  Creating phase grid figure...")
    create_phase_grid_figure(
        phase_adjacencies,
        overlap_neurons,
        OUTPUT_DIR / "figures" / "phase_connectivity_grid.png"
    )
    
    # Difference heatmaps
    print("  Creating difference heatmaps...")
    for stim in STIMULI:
        if stim in adj_stimulus_only:
            create_difference_heatmap(
                adj_global_binary, adj_stimulus_only[stim],
                "Global", f"{stim.capitalize()} Only",
                OUTPUT_DIR / "figures" / f"diff_global_vs_{stim}_only.png"
            )
    
    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n[6/6] Saving results...")
    
    # Save adjacencies
    np.save(OUTPUT_DIR / "adjacencies" / "global.npy", adj_global_binary)
    for stim, adj in adj_stimulus_only.items():
        np.save(OUTPUT_DIR / "adjacencies" / f"stimulus_only_{stim}.npy", adj)
    for stim, adj in adj_finetuned.items():
        np.save(OUTPUT_DIR / "adjacencies" / f"finetuned_{stim}.npy", adj)
    for phase, adj in phase_adjacencies.items():
        np.save(OUTPUT_DIR / "adjacencies" / f"phase_{phase}.npy", adj)
    
    # Save metrics
    all_metrics = []
    
    # Global metrics
    metrics_global["strategy"] = "global"
    metrics_global["condition"] = "all_data"
    all_metrics.append(metrics_global)
    
    # Stimulus-only metrics
    for stim in STIMULI:
        if stim in adj_stimulus_only:
            m = evaluate_vs_connectome(adj_stimulus_only[stim], A_struct_aligned)
            m["strategy"] = "stimulus_only"
            m["condition"] = stim
            all_metrics.append(m)
    
    # Finetuned metrics
    for stim in STIMULI:
        m = evaluate_vs_connectome(adj_finetuned[stim], A_struct_aligned)
        m["strategy"] = "global_finetuned"
        m["condition"] = stim
        all_metrics.append(m)
    
    # Phase metrics
    for m in phase_metrics:
        m["strategy"] = "phase_specific"
        m["condition"] = m["phase"]
        all_metrics.append(m)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "strategy_comparison_metrics.csv", index=False)
    
    # Save differences
    diff_df = pd.DataFrame(differences)
    diff_df.to_csv(OUTPUT_DIR / "graph_differences.csv", index=False)
    
    # Save node order
    with open(OUTPUT_DIR / "node_order.json", "w") as f:
        json.dump(overlap_neurons, f, indent=2)
    
    print(f"\n  ✓ Results saved to: {OUTPUT_DIR}")
    
    # Generating summary plot using saved metrics
    print("  Creating summary comparison...")
    create_strategy_comparison_plot(metrics_df, OUTPUT_DIR / "figures")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Training Strategy Comparison")
    print("=" * 80)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ Strategy              │ Condition        │ F1    │ Precision │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    for _, row in metrics_df.iterrows():
        print(f"│ {row['strategy']:<21} │ {row['condition']:<16} │ {row['f1_score']:.3f} │ {row['precision']:.3f}     │")
    
    print("└─────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

