#!/usr/bin/env python3
"""
Script 15: Multi-Lag SBTG Analysis with Theoretically Correct Lag Separation

This script implements the full pipeline for multi-lag connectome inference using
the mathematically correct SBTG approach derived in the theory document.

Three Approaches:
    A. Per-Lag 2-Block: Train separate 2-block SBTG for each lag r
       - Simple, fast, but doesn't condition on full lag stack
       - Windows: z = (x_{t+1-r}, x_{t+1}) for each lag r
       
    B. Full Multi-Block: Train single (p+1)-block model
       - Conditions on full lag stack → true lag separation
       - Windows: z = (x_{t-p+1}, ..., x_t, x_{t+1})
       - Extract μ^(r) for each lag from held-out scores
    
    C. Minimal Multi-Block: For each lag r, train (r+1)-block model
       - Middle ground: conditions on intermediate lags
       - Per-lag HP tuning available
       - Windows: z = (x_t, x_{t+1}, ..., x_{t+r})

All use 5-fold cross-fitting and HAC inference + FDR control.

Outputs:
    - Per-lag μ^(r) matrices
    - AUROC vs lag plots for Cook_Synapses_2019 and Randi_Optogenetics_2023
    - Neuron-type interaction matrices (sensory/inter/motor)
    - Summary statistics

Usage:
    # Run all approaches on full dataset (default)
    python pipeline/15_multilag_analysis.py

    # Run Approach A only with custom lags
    python pipeline/15_multilag_analysis.py --approach A --p-max 10
    
    # Run Approach C with HP tuning
    python pipeline/15_multilag_analysis.py --approach C --tune-hp
    
    # Run Approach B with GPU
    python pipeline/15_multilag_analysis.py --approach B --device cuda
    
    # Quick test
    python pipeline/15_multilag_analysis.py --quick-test
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import torch

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from models.multilag_sbtg import (
    PerLagSBTGEstimator,
    MultiBlockSBTGEstimator,
    MinimalMultiBlockEstimator,
    MultiLagSBTGResult,
    run_all_approaches,
    compute_null_contrast,
)
from utils.neuron_types import get_neuron_type, get_neuron_types_for_list
from utils.io import load_structural_connectome as _load_structural_connectome
from utils.stimulus_periods import (
    get_stimulus_mask, segment_traces_by_stimulus, summarize_stimulus_periods,
    segment_trace_4periods, summarize_4period_segmentation, StimulusPeriod
)
from utils.display_names import FUNCTIONAL_LABEL, STRUCTURAL_LABEL

# =============================================================================
# BASELINE METHODS
# =============================================================================

def compute_pearson_baseline(X_list: List[np.ndarray], lag: int) -> np.ndarray:
    """
    Compute Pearson correlation baseline between x_t and x_{t+lag}.
    
    Returns:
        (n, n) correlation matrix.
    """
    if not X_list:
        return np.array([])
    
    n = X_list[0].shape[1]
    all_x_past = []
    all_x_future = []
    
    for X in X_list:
        T = X.shape[0]
        if T > lag:
            all_x_past.append(X[:-lag])
            all_x_future.append(X[lag:])
            
    if not all_x_past:
        return np.zeros((n, n))
        
    X_past = np.nan_to_num(np.concatenate(all_x_past, axis=0), nan=0.0)
    X_future = np.nan_to_num(np.concatenate(all_x_future, axis=0), nan=0.0)
    
    # Compute cross-correlation matrix
    # We want corr(x_{t+lag, j}, x_{t, i}) for each pair (j, i)
    # This is equivalent to corr(X_future[:, j], X_past[:, i])
    
    corr_matrix = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            if np.std(X_future[:, j]) < 1e-9 or np.std(X_past[:, i]) < 1e-9:
                corr_matrix[j, i] = 0.0
            else:
                # Manual Pearson correlation to avoid numpy internal errors
                x = X_future[:, j]
                y = X_past[:, i]
                x_c = x - x.mean()
                y_c = y - y.mean()
                corr_matrix[j, i] = (x_c @ y_c) / (np.linalg.norm(x_c) * np.linalg.norm(y_c) + 1e-10)
                
    return corr_matrix


def compute_var_baseline(X_list: List[np.ndarray], lag: int) -> np.ndarray:
    """
    Fit lag-specific Ridge regression: x_{t+lag} = A·x_t + ε.
    
    This computes the MARGINAL lag-r effect without conditioning on intermediate lags,
    making it directly comparable to Pearson correlation and interpretable as
    "how well does x_t predict x_{t+lag}?"
    
    Uses Ridge regression (alpha=1.0) for numerical stability with pooled data across worms.
    
    IMPORTANT: This was changed from VAR(lag) model which extracted CONDITIONAL effects.
    The old approach fit x_{t+1} = A₁·x_t + ... + Aᵣ·x_{t-r+1} and returned Aᵣ,
    which is the conditional effect of lag-r given all shorter lags in the model.
    The new approach fits separate models per lag for fair comparison with SBTG.
    
    Args:
        X_list: List of (T, n) arrays per segment/worm
        lag: Time lag to test
        
    Returns:
        (n, n) coefficient matrix where [j, i] = effect of neuron i at time t 
        on neuron j at time t+lag
    """
    from sklearn.linear_model import Ridge
    
    if not X_list:
        return np.array([])
    
    n = X_list[0].shape[1]
    
    # Collect all lag-r windows across worms
    all_X_past = []
    all_X_future = []
    
    for X in X_list:
        if X.shape[0] > lag:
            all_X_past.append(X[:-lag])      # t = 0, 1, ..., T-lag-1
            all_X_future.append(X[lag:])     # t = lag, lag+1, ..., T-1
    
    if not all_X_past:
        return np.zeros((n, n))
    
    X_past = np.nan_to_num(np.concatenate(all_X_past, axis=0), nan=0.0)      # (N_total, n)
    X_future = np.nan_to_num(np.concatenate(all_X_future, axis=0), nan=0.0)  # (N_total, n)
    
    coef_matrix = np.zeros((n, n))
    
    # Fit x_{t+lag,j} ~ x_{t,i} for all i (Ridge regression per target neuron)
    for j in range(n):  # Target neuron
        y = X_future[:, j]
        
        # Ridge regression for stability
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(X_past, y)
        
        # coef_matrix[j, :] = effects of all source neurons on target j
        coef_matrix[j, :] = ridge.coef_
    
    return coef_matrix


def compute_granger_baseline(X_list: List[np.ndarray], lag: int, max_lag: int = 5) -> np.ndarray:
    """
    Compute Granger Causality F-statistic for each edge.
    
    NOTE: For lag > 1, this function now only tests LAG-1 effects due to conceptual
    issues with lag-specific Granger tests. The standard grangercausalitytests with
    maxlag=r tests whether lags 1-r JOINTLY improve prediction, not whether lag-r
    specifically matters. For multi-lag analysis, use Pearson or VAR (Ridge) baselines.
    
    Args:
        X_list: List of (T, n) arrays per segment/worm
        lag: Requested lag (but only lag=1 is computed)
        max_lag: Ignored (kept for backward compatibility)
        
    Returns:
        (n, n) F-statistic matrix. Returns zeros for lag > 1 with a warning.
    """
    if not X_list:
        return np.array([])
    
    n = X_list[0].shape[1]
    
    # Only compute Granger for lag-1 (conceptually sound)
    if lag > 1:
        print(f"  Warning: Granger causality only computed for lag-1. Returning zeros for lag {lag}.")
        print(f"           (Standard Granger tests are not lag-specific for lag > 1)")
        return np.zeros((n, n))
    
    f_stats = np.zeros((n, n))
    
    print(f"  Computing Granger causality for {n}x{n} edges (lag {lag}, {len(X_list)} worms)...")
    
    for j in range(n): # Target
        for i in range(n): # Source
            if i == j: 
                continue
            
            f_stats_per_worm = []
            
            # Compute Granger F-test for EACH worm separately
            for X in X_list:
                # Need sufficient length for lags
                if X.shape[0] < (3 * lag + 5):
                    continue
                
                X_clean = np.nan_to_num(X, nan=0.0)
                data_pair = X_clean[:, [j, i]]  # [target, source]
                
                try:
                    # We care if 'i' Granger-causes 'j'
                    test_result = grangercausalitytests(data_pair, maxlag=[lag], verbose=False)
                    # Extract F-test statistic
                    f_stat = test_result[lag][0]['ssr_ftest'][0]
                    f_stats_per_worm.append(f_stat)
                except Exception:
                    continue
            
            if f_stats_per_worm:
                # Average F-stat across worms (robust to boundaries)
                f_stats[j, i] = np.mean(f_stats_per_worm)
            else:
                f_stats[j, i] = 0.0
                
    return f_stats


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS_DIR = RESULTS_DIR / "intermediate" / "datasets"
CONNECTOME_DIR = RESULTS_DIR / "intermediate" / "connectome"
LEIFER_DIR = RESULTS_DIR / "leifer_evaluation"
OUTPUT_DIR = RESULTS_DIR / "multilag_separation"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_neural_data(
    dataset: str = "nacl",
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load neural time series data.
    
    Args:
        dataset: Dataset name ('nacl', 'full_traces_imputed', etc.)
        verbose: Print loading info
        
    Returns:
        X_list: List of (T_u, n) arrays per segment
        neuron_names: List of neuron names
    """
    dataset_dir = DATASETS_DIR / dataset
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    
    # Load standardization info (contains neuron order)
    std_file = dataset_dir / "standardization.json"
    if std_file.exists():
        with open(std_file) as f:
            std_info = json.load(f)
            neuron_names = std_info.get('node_order', [])
    else:
        neuron_names = []
    
    # Load segments
    X_list = []
    
    # Try X_segments.npy first (list of segments)
    x_segments_file = dataset_dir / "X_segments.npy"
    if x_segments_file.exists():
        X_segments = np.load(x_segments_file, allow_pickle=True)
        X_list = list(X_segments)
    else:
        # Try segment_*.npy files
        segment_files = sorted(dataset_dir.glob("segment_*.npy"))
        if segment_files:
            for f in segment_files:
                X = np.load(f)
                X_list.append(X)
        else:
            # Try single file
            single_file = dataset_dir / "traces.npy"
            if single_file.exists():
                X = np.load(single_file)
                X_list = [X]
            else:
                raise FileNotFoundError(f"No data files found in {dataset_dir}")
    
    if verbose:
        total_T = sum(X.shape[0] for X in X_list)
        n_neurons = X_list[0].shape[1]
        print(f"Loaded dataset '{dataset}': {len(X_list)} segments, {n_neurons} neurons, {total_T} total frames")
    
    return X_list, neuron_names


def load_cook_connectome(
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load Cook_Synapses_2019 structural connectome with separate gap/chemical matrices.
    
    Returns:
        A_struct: Combined synaptic adjacency matrix (gap + chemical)
        A_gap: Gap junction adjacency matrix
        A_chem: Chemical synapse adjacency matrix
        neuron_names: List of neuron names
    """
    A_struct, neuron_names, metadata = _load_structural_connectome(CONNECTOME_DIR)
    
    # Load separate gap and chemical matrices
    connectome_dir = Path(CONNECTOME_DIR)
    A_gap_path = connectome_dir / "A_gap.npy"
    A_chem_path = connectome_dir / "A_chem.npy"
    
    if A_gap_path.exists() and A_chem_path.exists():
        A_gap = np.load(A_gap_path)
        A_chem = np.load(A_chem_path)
    else:
        # Fallback: use combined matrix for both
        A_gap = A_struct
        A_chem = A_struct
    
    if verbose:
        n = A_struct.shape[0]
        n_edges = (A_struct > 0).sum()
        n_gap = (A_gap > 0).sum()
        n_chem = (A_chem > 0).sum()
        print(f"{STRUCTURAL_LABEL} connectome: {n} neurons, {n_edges} structural edges ({n_gap} gap, {n_chem} chemical)")
    
    return A_struct, A_gap, A_chem, neuron_names


def load_leifer_atlas(
    verbose: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Load Randi_Optogenetics_2023 functional atlas.
    
    Returns:
        q: (n, n) q-values for functional connections
        dff: (n, n) delta-F/F amplitudes
        neuron_names: List of neuron names
    """
    wt_file = LEIFER_DIR / "aligned_atlas_wild-type.npz"
    
    if not wt_file.exists():
        if verbose:
            print(f"{FUNCTIONAL_LABEL} atlas not found: {wt_file}")
        return None, None, None
    
    data = np.load(wt_file, allow_pickle=True)
    q = data.get('q', None)
    dff = data.get('dff', None)
    neuron_names = list(data.get('neuron_order', []))
    
    if verbose and q is not None:
        n = q.shape[0]
        n_edges = (q < 0.05).sum()
        print(f"{FUNCTIONAL_LABEL} atlas: {n} neurons, {n_edges} functional edges (q<0.05)")
    
    return q, dff, neuron_names


# =============================================================================
# ALIGNMENT
# =============================================================================

def align_matrices(
    pred: np.ndarray,
    pred_neurons: List[str],
    gt: np.ndarray,
    gt_neurons: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Align predicted and ground truth matrices to common neurons.
    
    Returns:
        pred_aligned, gt_aligned, common_neurons
    """
    # Ensure neuron lists are proper Python lists
    if isinstance(pred_neurons, np.ndarray):
        pred_neurons = list(pred_neurons)
    if isinstance(gt_neurons, np.ndarray):
        gt_neurons = list(gt_neurons)
    
    common = [n for n in pred_neurons if n in gt_neurons]
    
    if len(common) == 0:
        return np.array([]), np.array([]), []
    
    pred_idx = [pred_neurons.index(n) for n in common]
    gt_idx = [gt_neurons.index(n) for n in common]
    
    pred_aligned = pred[np.ix_(pred_idx, pred_idx)]
    gt_aligned = gt[np.ix_(gt_idx, gt_idx)]
    
    return pred_aligned, gt_aligned, common


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_auroc(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    binarize_gt: bool = True,
) -> float:
    """
    Compute AUROC for connectivity prediction.
    
    Args:
        scores: (n, n) predicted scores (continuous)
        ground_truth: (n, n) ground truth (binary or continuous)
        binarize_gt: If True, binarize GT as (GT > 0)
    """
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    y_score = np.abs(scores[mask])
    y_true = ground_truth[mask]
    
    if binarize_gt:
        y_true = (y_true > 0).astype(float)
    
    # Handle NaN
    valid = np.isfinite(y_score) & np.isfinite(y_true)
    y_score = y_score[valid]
    y_true = y_true[valid]
    
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    
    return roc_auc_score(y_true, y_score)


def compute_auprc(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    binarize_gt: bool = True,
) -> float:
    """Compute AUPRC for connectivity prediction."""
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    y_score = np.abs(scores[mask])
    y_true = ground_truth[mask]
    
    if binarize_gt:
        y_true = (y_true > 0).astype(float)
    
    valid = np.isfinite(y_score) & np.isfinite(y_true)
    y_score = y_score[valid]
    y_true = y_true[valid]
    
    if len(y_true) == 0 or y_true.sum() == 0:
        return 0.0
    
    return average_precision_score(y_true, y_score)


def compute_spearman(
    scores: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute Spearman correlation between predicted scores and ground truth weights.
    
    This measures how well the ranking of predicted weights matches the ranking
    of actual synapse counts, capturing both direction and magnitude relationships.
    
    Args:
        scores: (n, n) predicted scores (continuous)
        ground_truth: (n, n) ground truth weights (continuous, e.g., synapse counts)
        
    Returns:
        Spearman correlation coefficient (higher = better ranking agreement)
    """
    from scipy.stats import spearmanr
    
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    y_score = np.abs(scores[mask])
    y_true = ground_truth[mask].astype(float)
    
    valid = np.isfinite(y_score) & np.isfinite(y_true)
    y_score = y_score[valid]
    y_true = y_true[valid]
    
    if len(y_score) < 3:
        return 0.0
    
    # Handle constant arrays
    if np.std(y_score) < 1e-10 or np.std(y_true) < 1e-10:
        return 0.0
    
    rho, _ = spearmanr(y_score, y_true)
    return float(rho) if np.isfinite(rho) else 0.0


def evaluate_vs_cook(
    result: MultiLagSBTGResult,
    pred_neurons: List[str],
    A_struct: np.ndarray,
    cook_neurons: List[str],
) -> pd.DataFrame:
    """
    Evaluate multi-lag results against Cook_Synapses_2019.
    
    Returns DataFrame with AUROC/AUPRC/Spearman for each lag.
    """
    rows = []
    
    for lag_r in result.mu_hat:
        mu_hat = result.mu_hat[lag_r]
        
        # Align with Cook
        mu_aligned, struct_aligned, common = align_matrices(
            mu_hat, pred_neurons, A_struct, cook_neurons
        )
        
        if len(common) == 0:
            continue
        
        # Compute all metrics
        auroc_struct = compute_auroc(mu_aligned, struct_aligned)
        auprc_struct = compute_auprc(mu_aligned, struct_aligned)
        spearman_struct = compute_spearman(mu_aligned, struct_aligned)
        
        rows.append({
            'lag': lag_r,
            'auroc_struct': auroc_struct,
            'auprc_struct': auprc_struct,
            'spearman_struct': spearman_struct,
            'n_common': len(common),
            'n_edges': int(result.significant[lag_r].sum()),
        })
    
    return pd.DataFrame(rows)


def evaluate_vs_cook_detailed(
    result: MultiLagSBTGResult,
    pred_neurons: List[str],
    A_struct: np.ndarray,
    A_gap: np.ndarray,
    A_chem: np.ndarray,
    cook_neurons: List[str],
) -> pd.DataFrame:
    """
    Evaluate multi-lag results against Cook_Synapses_2019 with gap/chemical breakdown.
    
    Returns DataFrame with AUROC/AUPRC for combined, gap junctions, and chemical synapses.
    """
    rows = []
    
    for lag_r in result.mu_hat:
        mu_hat = result.mu_hat[lag_r]
        
        # Align with Cook
        mu_aligned, struct_aligned, common = align_matrices(
            mu_hat, pred_neurons, A_struct, cook_neurons
        )
        _, gap_aligned, _ = align_matrices(
            mu_hat, pred_neurons, A_gap, cook_neurons
        )
        _, chem_aligned, _ = align_matrices(
            mu_hat, pred_neurons, A_chem, cook_neurons
        )
        
        if len(common) == 0:
            continue
        
        # Compute metrics for each connectome type
        auroc_combined = compute_auroc(mu_aligned, struct_aligned)
        auroc_gap = compute_auroc(mu_aligned, gap_aligned)
        auroc_chem = compute_auroc(mu_aligned, chem_aligned)
        
        auprc_combined = compute_auprc(mu_aligned, struct_aligned)
        auprc_gap = compute_auprc(mu_aligned, gap_aligned)
        auprc_chem = compute_auprc(mu_aligned, chem_aligned)
        
        spearman_combined = compute_spearman(mu_aligned, struct_aligned)
        spearman_gap = compute_spearman(mu_aligned, gap_aligned)
        spearman_chem = compute_spearman(mu_aligned, chem_aligned)
        
        rows.append({
            'lag': lag_r,
            'auroc_combined': auroc_combined,
            'auroc_gap': auroc_gap,
            'auroc_chem': auroc_chem,
            'auprc_combined': auprc_combined,
            'auprc_gap': auprc_gap,
            'auprc_chem': auprc_chem,
            'spearman_combined': spearman_combined,
            'spearman_gap': spearman_gap,
            'spearman_chem': spearman_chem,
            'n_common': len(common),
            'n_edges': int(result.significant[lag_r].sum()),
        })
    
    return pd.DataFrame(rows)


def evaluate_vs_leifer(
    result: MultiLagSBTGResult,
    pred_neurons: List[str],
    q: np.ndarray,
    leifer_neurons: List[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Evaluate multi-lag results against Randi_Optogenetics_2023.
    """
    rows = []
    
    # Create binary ground truth from q-values
    gt = (q < alpha).astype(float)
    
    for lag_r in result.mu_hat:
        mu_hat = result.mu_hat[lag_r]
        
        mu_aligned, gt_aligned, common = align_matrices(
            mu_hat, pred_neurons, gt, leifer_neurons
        )
        
        if len(common) == 0:
            continue
        
        auroc = compute_auroc(mu_aligned, gt_aligned, binarize_gt=False)
        auprc = compute_auprc(mu_aligned, gt_aligned, binarize_gt=False)
        
        rows.append({
            'lag': lag_r,
            'auroc': auroc,
            'auprc': auprc,
            'n_common': len(common),
            'n_edges': int(result.significant[lag_r].sum()),
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# NEURON TYPE ANALYSIS
# =============================================================================

def compute_type_interaction_matrix(
    mu_hat: np.ndarray,
    neuron_names: List[str],
) -> pd.DataFrame:
    """
    Compute interaction matrix between neuron types.
    
    Returns DataFrame with mean |μ| for sensory/inter/motor pairs.
    """
    types = get_neuron_types_for_list(neuron_names)
    n = len(neuron_names)
    
    categories = ['sensory', 'interneuron', 'motor']
    
    # Get indices for each type
    type_indices = {
        cat: [i for i, name in enumerate(neuron_names) if types[name] == cat]
        for cat in categories
    }
    
    # Compute mean |μ| for each pair
    interaction = np.zeros((3, 3))
    counts = np.zeros((3, 3))
    
    for i_cat, source in enumerate(categories):
        for j_cat, target in enumerate(categories):
            source_idx = type_indices[source]
            target_idx = type_indices[target]
            
            if len(source_idx) == 0 or len(target_idx) == 0:
                continue
            
            # Target row, source column (i → j)
            for j in target_idx:
                for i in source_idx:
                    if i != j:
                        interaction[j_cat, i_cat] += np.abs(mu_hat[j, i])
                        counts[j_cat, i_cat] += 1
    
    # Average
    with np.errstate(invalid='ignore', divide='ignore'):
        interaction = np.where(counts > 0, interaction / counts, 0)
    
    df = pd.DataFrame(
        interaction,
        index=[f"→{c[:3]}" for c in categories],
        columns=[f"{c[:3]}→" for c in categories],
    )
    
    return df


# =============================================================================
# PLOTTING - DETAILED FIGURES MATCHING SCRIPT 13 STYLE
# =============================================================================

SAMPLING_RATE = 4.0  # Hz, for converting frames to seconds


def plot_auroc_vs_lag(
    eval_cook: pd.DataFrame,
    eval_leifer: Optional[pd.DataFrame],
    approach: str,
    output_path: Path,
):
    """
    Plot AUROC vs lag for Cook_Synapses_2019 and Randi_Optogenetics_2023.
    
    Shows both AUROC and AUPRC with clear annotations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: AUROC
    ax = axes[0]
    lags = eval_cook['lag'].values
    lag_seconds = lags / SAMPLING_RATE
    
    ax.plot(lag_seconds, eval_cook['auroc_struct'], 'o-', 
            label=STRUCTURAL_LABEL, lw=2.5, markersize=10, color='steelblue')
    
    # Add value annotations
    for x, y in zip(lag_seconds, eval_cook['auroc_struct']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
    
    if eval_leifer is not None and len(eval_leifer) > 0:
        leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
        ax.plot(leifer_lag_sec, eval_leifer['auroc'], 's--', 
                label=FUNCTIONAL_LABEL, lw=2, markersize=8, color='forestgreen')
        for x, y in zip(leifer_lag_sec, eval_leifer['auroc']):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0, -12), ha='center', fontsize=9, color='forestgreen')
    
    ax.axhline(0.5, color='gray', linestyle=':', lw=1.5, label='Random (0.5)')
    ax.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(f'Approach {approach}: AUROC vs Time Lag\n(higher = better prediction of ground truth)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(0.4, 0.85)
    ax.grid(True, alpha=0.3)
    
    # Add secondary x-axis for frames
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lag_seconds)
    ax2.set_xticklabels([str(int(l)) for l in lags])
    ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=10)
    
    # Panel 2: Edge counts and density
    ax = axes[1]
    if 'n_edges' in eval_cook.columns:
        edges = eval_cook['n_edges'].values
        bars = ax.bar(range(len(lags)), edges, color='coral', edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels([f'{l/SAMPLING_RATE:.2f}s\n(lag {int(l)})' for l in lags])
        ax.set_xlabel('Time Lag', fontsize=12)
        ax.set_ylabel('Number of FDR-Significant Edges', fontsize=12)
        ax.set_title(f'Approach {approach}: Edge Count by Lag\n(FDR α=0.1)', fontsize=12)
        
        # Add value labels on bars
        for bar, val in zip(bars, edges):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        # If no edge counts, show AUPRC instead
        ax.plot(lag_seconds, eval_cook['auprc_struct'], 'o-', 
                label=f'{STRUCTURAL_LABEL} AUPRC', lw=2.5, markersize=10, color='coral')
        if eval_leifer is not None and 'auprc' in eval_leifer.columns:
            leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
            ax.plot(leifer_lag_sec, eval_leifer['auprc'], 's--', 
                    label=f'{FUNCTIONAL_LABEL} AUPRC', lw=2, markersize=8, color='darkgreen')
        ax.set_xlabel('Time Lag (seconds)', fontsize=12)
        ax.set_ylabel('AUPRC (Area Under Precision-Recall)', fontsize=12)
        ax.set_title(f'Approach {approach}: AUPRC vs Time Lag', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_separate_metrics(
    eval_cook: pd.DataFrame,
    eval_leifer: Optional[pd.DataFrame],
    approach: str,
    output_dir: Path,
):
    """
    Create separate figures for AUROC, AUPRC, and Correlation metrics.
    
    Generates three publication-ready figures:
    - fig_auroc_{approach}.png: AUROC for Cook_Synapses_2019 and Randi_Optogenetics_2023
    - fig_auprc_{approach}.png: AUPRC for Cook_Synapses_2019 and Randi_Optogenetics_2023
    - fig_correlation_{approach}.png: Spearman correlation for Cook_Synapses_2019 and Randi_Optogenetics_2023
    """
    lags = eval_cook['lag'].values
    lag_seconds = lags / SAMPLING_RATE
    
    # Figure 1: AUROC only
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lag_seconds, eval_cook['auroc_struct'], 'o-', 
            label=STRUCTURAL_LABEL, lw=2.5, markersize=12, color='steelblue')
    
    # Annotations
    for x, y in zip(lag_seconds, eval_cook['auroc_struct']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    if eval_leifer is not None and len(eval_leifer) > 0:
        leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
        ax.plot(leifer_lag_sec, eval_leifer['auroc'], 's--', 
                label=FUNCTIONAL_LABEL, lw=2.5, markersize=10, color='forestgreen')
        for x, y in zip(leifer_lag_sec, eval_leifer['auroc']):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0, -15), ha='center', fontsize=10, color='forestgreen', fontweight='bold')
    
    ax.axhline(0.5, color='gray', linestyle=':', lw=2, label='Random (0.5)', alpha=0.7)
    ax.set_xlabel('Time Lag (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=14, fontweight='bold')
    ax.set_title(f'SBTG Approach {approach}: AUROC vs Time Lag', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.set_ylim(0.4, 0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lag_seconds)
    ax2.set_xticklabels([str(int(l)) for l in lags])
    ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=11)
    ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'fig_auroc_{approach}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / f'fig_auroc_{approach}.png'}")
    
    # Figure 2: AUPRC only
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lag_seconds, eval_cook['auprc_struct'], 'o-', 
            label=STRUCTURAL_LABEL, lw=2.5, markersize=12, color='coral')
    
    # Annotations
    for x, y in zip(lag_seconds, eval_cook['auprc_struct']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    if eval_leifer is not None and 'auprc' in eval_leifer.columns:
        leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
        ax.plot(leifer_lag_sec, eval_leifer['auprc'], 's--', 
                label=FUNCTIONAL_LABEL, lw=2.5, markersize=10, color='darkgreen')
        for x, y in zip(leifer_lag_sec, eval_leifer['auprc']):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0, -15), ha='center', fontsize=10, color='darkgreen', fontweight='bold')
    
    ax.set_xlabel('Time Lag (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUPRC (Area Under Precision-Recall)', fontsize=14, fontweight='bold')
    ax.set_title(f'SBTG Approach {approach}: AUPRC vs Time Lag', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lag_seconds)
    ax2.set_xticklabels([str(int(l)) for l in lags])
    ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=11)
    ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'fig_auprc_{approach}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / f'fig_auprc_{approach}.png'}")
    
    # Figure 3: Spearman Correlation only
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lag_seconds, eval_cook['spearman_struct'], 'o-', 
            label=STRUCTURAL_LABEL, lw=2.5, markersize=12, color='purple')
    
    # Annotations
    for x, y in zip(lag_seconds, eval_cook['spearman_struct']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    if eval_leifer is not None and 'spearman' in eval_leifer.columns:
        leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
        ax.plot(leifer_lag_sec, eval_leifer['spearman'], 's--', 
                label=FUNCTIONAL_LABEL, lw=2.5, markersize=10, color='darkgoldenrod')
        for x, y in zip(leifer_lag_sec, eval_leifer['spearman']):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0, -15), ha='center', fontsize=10, color='darkgoldenrod', fontweight='bold')
    
    ax.axhline(0.0, color='gray', linestyle=':', lw=2, alpha=0.7)
    ax.set_xlabel('Time Lag (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel("Spearman's ρ", fontsize=14, fontweight='bold')
    ax.set_title(f'SBTG Approach {approach}: Weight Correlation vs Time Lag', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lag_seconds)
    ax2.set_xticklabels([str(int(l)) for l in lags])
    ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=11)
    ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'fig_correlation_{approach}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / f'fig_correlation_{approach}.png'}")


def plot_type_interactions(
    eval_cook: pd.DataFrame,
    eval_leifer: Optional[pd.DataFrame],
    approach: str,
    output_path: Path,
):
    """
    Plot AUROC vs lag for Cook_Synapses_2019 and Randi_Optogenetics_2023.
    
    Shows both AUROC and AUPRC with clear annotations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: AUROC
    ax = axes[0]
    lags = eval_cook['lag'].values
    lag_seconds = lags / SAMPLING_RATE
    
    ax.plot(lag_seconds, eval_cook['auroc_struct'], 'o-', 
            label=STRUCTURAL_LABEL, lw=2.5, markersize=10, color='steelblue')
    
    # Add value annotations
    for x, y in zip(lag_seconds, eval_cook['auroc_struct']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
    
    if eval_leifer is not None and len(eval_leifer) > 0:
        leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
        ax.plot(leifer_lag_sec, eval_leifer['auroc'], 's--', 
                label=FUNCTIONAL_LABEL, lw=2, markersize=8, color='forestgreen')
        for x, y in zip(leifer_lag_sec, eval_leifer['auroc']):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0, -12), ha='center', fontsize=9, color='forestgreen')
    
    ax.axhline(0.5, color='gray', linestyle=':', lw=1.5, label='Random (0.5)')
    ax.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(f'Approach {approach}: AUROC vs Time Lag\n(higher = better prediction of ground truth)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(0.4, 0.85)
    ax.grid(True, alpha=0.3)
    
    # Add secondary x-axis for frames
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lag_seconds)
    ax2.set_xticklabels([str(int(l)) for l in lags])
    ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=10)
    
    # Panel 2: Edge counts and density
    ax = axes[1]
    if 'n_edges' in eval_cook.columns:
        edges = eval_cook['n_edges'].values
        bars = ax.bar(range(len(lags)), edges, color='coral', edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels([f'{l/SAMPLING_RATE:.2f}s\n(lag {int(l)})' for l in lags])
        ax.set_xlabel('Time Lag', fontsize=12)
        ax.set_ylabel('Number of FDR-Significant Edges', fontsize=12)
        ax.set_title(f'Approach {approach}: Edge Count by Lag\n(FDR α=0.1)', fontsize=12)
        
        # Add value labels on bars
        for bar, val in zip(bars, edges):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        # If no edge counts, show AUPRC instead
        ax.plot(lag_seconds, eval_cook['auprc_struct'], 'o-', 
                label=f'{STRUCTURAL_LABEL} AUPRC', lw=2.5, markersize=10, color='coral')
        if eval_leifer is not None and 'auprc' in eval_leifer.columns:
            leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
            ax.plot(leifer_lag_sec, eval_leifer['auprc'], 's--', 
                    label=f'{FUNCTIONAL_LABEL} AUPRC', lw=2, markersize=8, color='darkgreen')
        ax.set_xlabel('Time Lag (seconds)', fontsize=12)
        ax.set_ylabel('AUPRC (Area Under Precision-Recall)', fontsize=12)
        ax.set_title(f'Approach {approach}: AUPRC vs Time Lag', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_type_interactions(
    result: MultiLagSBTGResult,
    neuron_names: List[str],
    approach: str,
    output_path: Path,
    lags_to_plot: Optional[List[int]] = None,
):
    """
    Plot neuron type interaction matrices for selected lags.
    
    Shows mean |μ| between sensory/inter/motor pairs, matching script 13 style.
    """
    if lags_to_plot is None:
        lags_to_plot = [1, 2, 5] if result.p_max >= 5 else list(range(1, result.p_max + 1))
    
    lags_to_plot = [lag for lag in lags_to_plot if lag in result.mu_hat]
    n_lags = len(lags_to_plot)
    
    if n_lags == 0:
        return
    
    fig, axes = plt.subplots(1, n_lags + 1, figsize=(4 * (n_lags + 1), 5))
    
    # Collect data for all lags
    all_data = []
    
    for idx, lag in enumerate(lags_to_plot):
        ax = axes[idx]
        df = compute_type_interaction_matrix(result.mu_hat[lag], neuron_names)
        all_data.append((lag, df))
        
        # Determine colormap scale
        vmax = max(0.05, df.values.max())
        
        sns.heatmap(
            df, ax=ax, annot=True, fmt='.4f', cmap='YlOrRd',
            vmin=0, vmax=vmax,
            cbar_kws={'label': 'Mean |μ̂|', 'shrink': 0.8},
            annot_kws={'fontsize': 10, 'fontweight': 'bold'},
        )
        lag_sec = lag / SAMPLING_RATE
        ax.set_title(f'Lag {lag} ({lag_sec:.2f}s)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Source Class →', fontsize=10)
        ax.set_ylabel('→ Target Class', fontsize=10)
    
    # Final panel: Summary bar chart of lag decay by class pair
    ax = axes[-1]
    pairs = ['S→S', 'S→I', 'S→M', 'I→S', 'I→I', 'I→M', 'M→S', 'M→I', 'M→M']
    x = np.arange(len(pairs))
    width = 0.8 / len(lags_to_plot)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(lags_to_plot)))
    
    for i, (lag, df) in enumerate(all_data):
        values = [df.iloc[j // 3, j % 3] for j in range(9)]
        ax.bar(x + i * width - 0.4 + width/2, values, width, 
               label=f'Lag {lag}', color=colors[i], edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('Class Pair (Source → Target)', fontsize=10)
    ax.set_ylabel('Mean |μ̂|', fontsize=10)
    ax.set_title('Class Interactions\nAcross Lags', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Approach {approach}: Neuron Type Interactions\n[S=Sensory, I=Interneuron, M=Motor]', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mu_hat_comparison(
    result_A: Optional[MultiLagSBTGResult],
    result_B: Optional[MultiLagSBTGResult],
    output_path: Path,
    lag: int = 1,
):
    """
    Compare μ̂^(r) matrices from different approaches.
    
    Shows heatmaps and correlation between approaches.
    """
    results = []
    labels = []
    
    if result_A is not None and lag in result_A.mu_hat:
        results.append(result_A.mu_hat[lag])
        labels.append('A: Per-Lag 2-Block')
    if result_B is not None and lag in result_B.mu_hat:
        results.append(result_B.mu_hat[lag])
        labels.append('B: Full Multi-Block')
    
    n_plots = len(results)
    if n_plots == 0:
        return
    
    # Create figure with heatmaps and scatter comparison
    fig = plt.figure(figsize=(6 * n_plots + 4, 6))
    
    # Heatmaps
    for idx, (mu, label) in enumerate(zip(results, labels)):
        ax = fig.add_subplot(1, n_plots + 1, idx + 1)
        vmax = np.percentile(np.abs(mu), 95)
        im = ax.imshow(mu, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(f'{label}\nμ̂^({lag})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Source Neuron', fontsize=10)
        ax.set_ylabel('Target Neuron', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8, label='μ̂ value')
        
        # Add stats
        n_neurons = mu.shape[0]
        mask = ~np.eye(n_neurons, dtype=bool)
        mean_abs = np.abs(mu[mask]).mean()
        ax.text(0.02, 0.98, f'Mean |μ̂|={mean_abs:.4f}', transform=ax.transAxes,
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter comparison if two approaches
    if n_plots == 2:
        ax = fig.add_subplot(1, n_plots + 1, n_plots + 1)
        mu_A = results[0]
        mu_B = results[1]
        n = mu_A.shape[0]
        mask = ~np.eye(n, dtype=bool)
        
        x = mu_A[mask].flatten()
        y = mu_B[mask].flatten()
        
        ax.scatter(x, y, alpha=0.3, s=10, c='steelblue')
        
        # Add correlation
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
               fontsize=11, fontweight='bold', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Add diagonal
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, lw=1.5, label='y=x')
        
        ax.set_xlabel(f'{labels[0]} μ̂', fontsize=10)
        ax.set_ylabel(f'{labels[1]} μ̂', fontsize=10)
        ax.set_title(f'Approach Comparison\nLag {lag}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_edge_density_by_lag(
    results: Dict[str, MultiLagSBTGResult],
    output_path: Path,
):
    """
    Plot edge density and E:I ratio across lags for all approaches.
    
    Matches the style of script 13's edge density figures.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    colors = {'A': 'steelblue', 'B': 'coral', 'C': 'forestgreen'}
    markers = {'A': 'o', 'B': 's', 'C': '^'}
    labels = {'A': 'Per-Lag 2-Block', 'B': 'Full Multi-Block', 'C': 'Minimal Multi-Block'}
    
    # Panel 1: Edge density by lag
    ax = axes[0]
    for approach, result in results.items():
        lags = sorted(result.significant.keys())
        densities = []
        for lag in lags:
            sig = result.significant[lag]
            n = sig.shape[0]
            n_edges = sig.sum() - np.trace(sig)  # Exclude diagonal
            n_possible = n * (n - 1)
            densities.append(n_edges / max(1, n_possible))
        
        lag_seconds = [l / SAMPLING_RATE for l in lags]
        ax.plot(lag_seconds, densities, f'{markers[approach]}-', 
                color=colors[approach], label=f'{approach}: {labels[approach]}',
                lw=2, markersize=8)
        
        # Annotate values
        for x, y in zip(lag_seconds, densities):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                       xytext=(0, 5), ha='center', fontsize=8)
    
    ax.set_xlabel('Time Lag (seconds)', fontsize=11)
    ax.set_ylabel('Edge Density\n(fraction of possible edges)', fontsize=11)
    ax.set_title('Edge Density vs Time Lag', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Edge count by lag
    ax = axes[1]
    for approach, result in results.items():
        lags = sorted(result.significant.keys())
        edge_counts = []
        for lag in lags:
            sig = result.significant[lag]
            n_edges = int(sig.sum() - np.trace(sig))
            edge_counts.append(n_edges)
        
        lag_seconds = [l / SAMPLING_RATE for l in lags]
        ax.plot(lag_seconds, edge_counts, f'{markers[approach]}-', 
                color=colors[approach], label=f'{approach}: {labels[approach]}',
                lw=2, markersize=8)
    
    ax.set_xlabel('Time Lag (seconds)', fontsize=11)
    ax.set_ylabel('Number of FDR-Significant Edges', fontsize=11)
    ax.set_title('Edge Count vs Time Lag\n(FDR α=0.1)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: E:I ratio by lag
    ax = axes[2]
    for approach, result in results.items():
        lags = sorted(result.mu_hat.keys())
        ei_ratios = []
        for lag in lags:
            mu = result.mu_hat[lag]
            sig = result.significant[lag]
            
            # Count excitatory (positive μ̂) and inhibitory (negative μ̂) among significant
            excit = ((mu > 0) & (sig > 0)).sum()
            inhib = ((mu < 0) & (sig > 0)).sum()
            ei_ratios.append(excit / max(1, inhib))
        
        lag_seconds = [l / SAMPLING_RATE for l in lags]
        ax.plot(lag_seconds, ei_ratios, f'{markers[approach]}-', 
                color=colors[approach], label=f'{approach}: {labels[approach]}',
                lw=2, markersize=8)
    
    ax.axhline(1.0, color='gray', linestyle='--', lw=1.5, label='Balanced (E=I)')
    ax.set_xlabel('Time Lag (seconds)', fontsize=11)
    ax.set_ylabel('E:I Ratio\n(Excitatory / Inhibitory)', fontsize=11)
    ax.set_title('Excitatory:Inhibitory Balance\nby Time Lag', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_figure(
    results: Dict[str, MultiLagSBTGResult],
    eval_results: Dict[str, Dict],
    output_path: Path,
):
    """
    Create a comprehensive summary figure with key results.
    
    4-panel figure: AUROC comparison, Null Contrast, Edge Density, Best lag summary.
    """
    fig = plt.figure(figsize=(16, 12))
    
    colors = {'A': 'steelblue', 'B': 'coral', 'C': 'forestgreen'}
    markers = {'A': 'o', 'B': 's', 'C': '^'}
    labels = {'A': 'A: Per-Lag 2-Block', 'B': 'B: Full Multi-Block', 'C': 'C: Minimal Multi-Block'}
    
    # Panel 1: AUROC vs structural benchmark (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    for approach in ['A', 'B', 'C']:
        if approach in eval_results:
            df = eval_results[approach]['cook']
            lag_sec = df['lag'].values / SAMPLING_RATE
            ax1.plot(lag_sec, df['auroc_struct'], f'{markers[approach]}-',
                    color=colors[approach], label=labels[approach], lw=2.5, markersize=10)
            
            # Find and mark best
            best_idx = df['auroc_struct'].idxmax()
            best_lag = df.loc[best_idx, 'lag'] / SAMPLING_RATE
            best_auroc = df.loc[best_idx, 'auroc_struct']
            ax1.scatter([best_lag], [best_auroc], s=200, c=colors[approach], 
                       marker='*', edgecolors='black', linewidths=1.5, zorder=10)
    
    ax1.axhline(0.5, color='gray', linestyle=':', lw=1.5, label='Random')
    ax1.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax1.set_ylabel(f'AUROC vs {STRUCTURAL_LABEL}', fontsize=12)
    ax1.set_title('Connectome Prediction Accuracy\n(★ = best lag per approach)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.set_ylim(0.4, 0.85)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Null Contrast (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    for approach in ['A', 'B', 'C']:
        if approach in results:
            result = results[approach]
            lags = sorted(result.mu_hat.keys())
            ncs = [compute_null_contrast(result.mu_hat[lag]) for lag in lags]
            lag_sec = [l / SAMPLING_RATE for l in lags]
            ax2.plot(lag_sec, ncs, f'{markers[approach]}-',
                    color=colors[approach], label=labels[approach], lw=2.5, markersize=10)
    
    ax2.axhline(1.0, color='gray', linestyle=':', lw=1.5, label='No Signal (NC=1)')
    ax2.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax2.set_ylabel('Null Contrast\n(higher = more biological signal)', fontsize=12)
    ax2.set_title('Null Contrast: Real vs Shuffled\n(measures signal vs noise)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Edge counts (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    all_lags = set()
    for approach, result in results.items():
        all_lags.update(result.significant.keys())
    all_lags = sorted(all_lags)
    
    x = np.arange(len(all_lags))
    width = 0.25
    
    for i, approach in enumerate(['A', 'B', 'C']):
        if approach in results:
            result = results[approach]
            edge_counts = []
            for lag in all_lags:
                if lag in result.significant:
                    sig = result.significant[lag]
                    edge_counts.append(int(sig.sum() - np.trace(sig)))
                else:
                    edge_counts.append(0)
            ax3.bar(x + i * width - width, edge_counts, width,
                   color=colors[approach], label=labels[approach], edgecolor='black', alpha=0.8)
    
    ax3.set_xlabel('Time Lag (frames)', fontsize=12)
    ax3.set_ylabel('Number of FDR-Significant Edges', fontsize=12)
    ax3.set_title('Detected Edges by Approach & Lag\n(FDR α=0.1)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{l}\n({l/SAMPLING_RATE:.1f}s)' for l in all_lags])
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary table (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Build summary table
    table_data = []
    headers = ['Approach', 'Best Lag', 'Best AUROC', 'Edges @Best', 'NC @Best']
    
    for approach in ['A', 'B', 'C']:
        if approach in eval_results and approach in results:
            df = eval_results[approach]['cook']
            result = results[approach]
            
            best_idx = df['auroc_struct'].idxmax()
            best_lag = int(df.loc[best_idx, 'lag'])
            best_auroc = df.loc[best_idx, 'auroc_struct']
            
            if best_lag in result.significant:
                sig = result.significant[best_lag]
                n_edges = int(sig.sum() - np.trace(sig))
            else:
                n_edges = 0
            
            nc = compute_null_contrast(result.mu_hat[best_lag]) if best_lag in result.mu_hat else 1.0
            
            row = [
                labels[approach].split(':')[0] + ': ' + labels[approach].split(':')[1][:15],
                f'{best_lag} ({best_lag/SAMPLING_RATE:.2f}s)',
                f'{best_auroc:.4f}',
                f'{n_edges}',
                f'{nc:.3f}',
            ]
            table_data.append(row)
    
    if table_data:
        table = ax4.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=['lightblue'] * len(headers),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        ax4.set_title('Summary: Best Results per Approach', fontsize=13, fontweight='bold', pad=20)
    
    fig.suptitle('Multi-Lag SBTG Analysis: Comprehensive Results', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    plt.close()
    print(f"Saved: {output_path}")


def plot_baseline_comparison(
    sbtg_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    output_path: Path,
):
    """
    Plot comprehensive comparison between SBTG and baselines.
    
    Creates a 2x3 figure showing three metrics:
    - AUROC: Area under ROC curve (binary classification)
    - AUPRC: Area under Precision-Recall curve (handles class imbalance)
    - Spearman: Rank correlation with synapse counts (weight agreement)
    
    Top row: Line plots showing metric vs lag for all methods
    Bottom row: Bar charts comparing best metric per method
    """
    if sbtg_results.empty or baseline_results.empty:
        print(f"  Warning: Empty results - SBTG: {len(sbtg_results)}, Baselines: {len(baseline_results)}")
        return
    
    # Combine results
    sbtg_results = sbtg_results.copy()
    if 'method' not in sbtg_results.columns:
        sbtg_results['method'] = 'SBTG'
    
    combined = pd.concat([sbtg_results, baseline_results], ignore_index=True)
    
    # Define metrics to plot
    metrics = [
        ('auroc_struct', 'AUROC', 0.5, 'Binary Classification'),
        ('auprc_struct', 'AUPRC', None, 'Precision-Recall'),
        ('spearman_struct', 'Spearman ρ', 0.0, 'Weight Correlation'),
    ]
    
    # Filter to available metrics
    available_metrics = [(m, n, r, d) for m, n, r, d in metrics if m in combined.columns]
    
    if len(available_metrics) == 0:
        print(f"  Warning: No metrics found in results")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(5 * n_metrics, 10))
    
    # If only one metric, axes needs to be 2D
    if n_metrics == 1:
        axes = axes.reshape(-1, 1)
    
    # Define colors and markers for methods
    method_styles = {
        'SBTG': {'color': 'steelblue', 'marker': 'o', 'linestyle': '-'},
        'SBTG-A': {'color': 'steelblue', 'marker': 'o', 'linestyle': '-'},
        'SBTG-B': {'color': 'coral', 'marker': 's', 'linestyle': '-'},
        'SBTG-C': {'color': 'forestgreen', 'marker': '^', 'linestyle': '-'},
        'Pearson': {'color': 'gray', 'marker': 'v', 'linestyle': '--'},
        'VAR': {'color': 'purple', 'marker': 'D', 'linestyle': '--'},
        'Granger': {'color': 'brown', 'marker': 'p', 'linestyle': '--'},
    }
    
    methods = combined['method'].unique()
    
    for col_idx, (metric_col, metric_name, random_baseline, metric_desc) in enumerate(available_metrics):
        # Top row: Line plot of metric vs lag
        ax = axes[0, col_idx]
        
        for method in methods:
            method_data = combined[combined['method'] == method].sort_values('lag')
            style = method_styles.get(method, {'color': 'black', 'marker': 'x', 'linestyle': '-'})
            
            if metric_col not in method_data.columns:
                continue
            
            lags = method_data['lag'].values
            values = method_data[metric_col].values
            lag_seconds = lags / SAMPLING_RATE
            
            ax.plot(lag_seconds, values, 
                    color=style['color'], marker=style['marker'], linestyle=style['linestyle'],
                    label=method, linewidth=2.5, markersize=8)
            
            # Annotate best point for SBTG methods
            if 'SBTG' in method and len(values) > 0:
                best_idx = np.argmax(values)
                ax.annotate(f'{values[best_idx]:.3f}', 
                           (lag_seconds[best_idx], values[best_idx]),
                           textcoords="offset points", xytext=(0, 8), ha='center',
                           fontsize=8, fontweight='bold', color=style['color'])
        
        # Add random baseline if applicable
        if random_baseline is not None:
            ax.axhline(random_baseline, color='gray', linestyle=':', lw=1.5, alpha=0.7, label='Random')
        
        ax.set_xlabel('Time Lag (seconds)', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} vs Lag\n({metric_desc})', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Set sensible y-limits
        valid_data = combined[metric_col].dropna()
        if len(valid_data) > 0:
            ymin = min(0.4 if metric_name == 'AUROC' else -0.1, valid_data.min() - 0.05)
            ymax = max(0.7, valid_data.max() + 0.05)
            ax.set_ylim(ymin, ymax)
        
        # Add secondary x-axis for frames
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        lag_ticks = sorted(combined['lag'].unique())
        ax2.set_xticks([l / SAMPLING_RATE for l in lag_ticks])
        ax2.set_xticklabels([str(int(l)) for l in lag_ticks], fontsize=9)
        ax2.set_xlabel('Lag (frames)', fontsize=9)
        
        # Bottom row: Bar chart comparing best metric per method
        ax = axes[1, col_idx]
        
        # Get best value per method (ignoring NaNs)
        valid_combined = combined.dropna(subset=[metric_col])
        if len(valid_combined) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            continue
            
        best_per_method = valid_combined.groupby('method')[metric_col].max().sort_values(ascending=False)
        best_lags = valid_combined.loc[valid_combined.groupby('method')[metric_col].idxmax(), ['method', 'lag']].set_index('method')['lag']
        
        colors = [method_styles.get(m, {'color': 'gray'})['color'] for m in best_per_method.index]
        bars = ax.barh(range(len(best_per_method)), best_per_method.values, color=colors, edgecolor='black', alpha=0.8)
        
        ax.set_yticks(range(len(best_per_method)))
        ax.set_yticklabels(best_per_method.index, fontsize=10)
        ax.set_xlabel(f'Best {metric_name}', fontsize=11)
        ax.set_title(f'Best {metric_name} per Method', fontsize=12, fontweight='bold')
        
        if random_baseline is not None:
            ax.axvline(random_baseline, color='gray', linestyle=':', lw=1.5, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value annotations on bars
        for i, (method, val) in enumerate(best_per_method.items()):
            lag = best_lags.get(method, 0)
            ax.text(val + 0.01, i, f'{val:.3f} (lag {int(lag)})', 
                    va='center', fontsize=9, fontweight='bold')
        
        # Set x-limits
        if len(best_per_method) > 0:
            xmin = min(0.4 if metric_name == 'AUROC' else -0.1, best_per_method.min() - 0.05)
            xmax = max(0.7, best_per_method.max() + 0.1)
            ax.set_xlim(xmin, xmax)
    
    fig.suptitle(f'SBTG vs Baseline Methods: Comprehensive Comparison\n({STRUCTURAL_LABEL})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def segment_data_4periods(
    X_list: List[np.ndarray], 
    fps: float = SAMPLING_RATE,
    min_segment_frames: int = 8,
) -> dict:
    """
    Segment time series data into 4 stimulus period types.
    
    Args:
        X_list: List of (T, n_neurons) arrays per segment/worm
        fps: Sampling rate
        min_segment_frames: Minimum frames for a valid segment
        
    Returns:
        Dictionary mapping period names to lists of segment arrays:
        {'NOTHING': [...], 'ON': [...], 'SHOWING': [...], 'OFF': [...]}
    """
    all_segments = {
        'NOTHING': [],
        'ON': [],
        'SHOWING': [],
        'OFF': [],
    }
    
    for X in X_list:
        # Get 4-period segments for this trace
        segments = segment_trace_4periods(
            X, fps=fps, min_segment_frames=min_segment_frames
        )
        
        # Append to global lists
        for period_name in all_segments.keys():
            all_segments[period_name].extend(segments[period_name])
    
    return all_segments


# Keep old function for backward compatibility
def segment_data_by_stimulus(X_list: List[np.ndarray], fps: float = SAMPLING_RATE) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Segment time series data into stimulus ON and OFF periods.
    (Legacy function - use segment_data_4periods instead)
    """
    segments = segment_data_4periods(X_list, fps)
    # ON = ON + SHOWING, OFF = OFF + NOTHING (for backward compatibility)
    X_on_list = segments['ON'] + segments['SHOWING']
    X_off_list = segments['OFF'] + segments['NOTHING']
    return X_on_list, X_off_list


def run_4period_analysis(
    period_segments: dict,
    neuron_names: List[str],
    A_struct: np.ndarray,
    A_gap: np.ndarray,
    A_chem: np.ndarray,
    cook_neurons: List[str],
    q_leifer: Optional[np.ndarray],
    leifer_neurons: Optional[List[str]],
    lags: List[int],
    run_dir: Path,
    device: str,
    args,
) -> dict:
    """
    Run multi-lag analysis separately for all 4 stimulus period types.
    
    Args:
        period_segments: Dict from segment_data_4periods()
        neuron_names: List of neuron names
        A_struct, A_gap, A_chem: structural benchmark matrices
        cook_neurons: structural benchmark neuron names
        q_leifer, leifer_neurons: functional benchmark atlas (optional)
        lags: Requested lag values
        run_dir: Output directory
        device: 'cuda' or 'cpu'
        args: Argparse namespace with hyperparameters
    
    Returns:
        Dictionary with results for each period type
    """
    import time
    
    PERIOD_NAMES = ['NOTHING', 'ON', 'SHOWING', 'OFF']
    PERIOD_COLORS = {'NOTHING': 'gray', 'ON': 'coral', 'SHOWING': 'green', 'OFF': 'steelblue'}
    
    results = {name: {} for name in PERIOD_NAMES}
    
    # Create output directory
    analysis_dir = run_dir / '4period_analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    for period_name in PERIOD_NAMES:
        X_list = period_segments[period_name]
        period_dir = analysis_dir / period_name
        period_dir.mkdir(parents=True, exist_ok=True)
        (period_dir / 'figures').mkdir(exist_ok=True)
        
        if len(X_list) < 3:
            print(f"\n  {period_name}: ✗ Insufficient segments ({len(X_list)} < 3)")
            continue
        
        # Determine feasible lags based on segment lengths
        lengths = [X.shape[0] for X in X_list]
        min_len = min(lengths)
        max_lag = max(1, min_len // 3)
        feasible_lags = [l for l in lags if l <= max_lag]
        
        if len(feasible_lags) == 0:
            print(f"\n  {period_name}: ✗ No feasible lags (min segment = {min_len} frames)")
            continue
        
        print(f"\n  {period_name}: {len(X_list)} segments, min={min_len} frames")
        print(f"    Using lags: {feasible_lags} (max feasible: {max_lag})")
        
        start = time.time()
        
        try:
            estimator = MinimalMultiBlockEstimator(
                lags=feasible_lags,
                tune_hp=False,  # Skip HP tuning for per-period analysis
                noise_std=args.noise_std,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                n_folds=min(args.n_folds, len(X_list)),
                fdr_alpha=args.fdr_alpha,
                device=device,
                random_state=args.seed,
                verbose=False,
            )
            
            result = estimator.fit(X_list)
            results[period_name]['result'] = result
            results[period_name]['lags'] = feasible_lags
            
            # Evaluate
            eval_cook = evaluate_vs_cook(result, neuron_names, A_struct, cook_neurons)
            eval_cook_detailed = evaluate_vs_cook_detailed(
                result, neuron_names, A_struct, A_gap, A_chem, cook_neurons
            )
            results[period_name]['eval_cook'] = eval_cook
            
            # Save results
            np.savez(
                period_dir / 'result.npz',
                **{f'mu_hat_lag{k}': v for k, v in result.mu_hat.items()},
                **{f'pval_lag{k}': v for k, v in result.p_values.items()},
                **{f'sig_lag{k}': v for k, v in result.significant.items()},
                neuron_names=neuron_names,
                lags=feasible_lags,
            )
            eval_cook.to_csv(period_dir / 'eval_cook.csv', index=False)
            
            best_auroc = eval_cook['auroc_struct'].max()
            print(f"    ✓ SBTG Complete ({time.time()-start:.1f}s), Best AUROC: {best_auroc:.4f}")

            # Run Baselines on this period's data
            print(f"    Running baselines for {period_name} period...")
            period_baselines = {}
            
            # Pearson
            for lag in feasible_lags:
                try:
                    p = compute_pearson_baseline(X_list, lag)
                    period_baselines[f'Pearson_lag{lag}'] = p
                except Exception as e:
                    print(f"      ! Pearson lag{lag} failed: {e}")
            
            # VAR
            for lag in feasible_lags:
                try:
                    v = compute_var_baseline(X_list, lag)
                    period_baselines[f'VAR_lag{lag}'] = v
                except Exception as e:
                    print(f"      ! VAR lag{lag} failed: {e}")

            # Granger (restrict to max lag 3 for computational feasibility)
            granger_lags = [l for l in feasible_lags if l <= 3]
            for lag in granger_lags:
                try:
                    g = compute_granger_baseline(X_list, lag)
                    period_baselines[f'Granger_lag{lag}'] = g
                except Exception as e:
                     print(f"      ! Granger lag{lag} failed: {e}")
                    
            if period_baselines:
                 np.savez(period_dir / 'baselines.npz', **period_baselines)
                 print(f"    ✓ Baselines (Pearson, VAR, Granger) saved to {period_dir / 'baselines.npz'}")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            continue
    
    return results


# Keep old function for backward compatibility
def run_stimulus_period_analysis(
    X_on_list: List[np.ndarray],
    X_off_list: List[np.ndarray],
    neuron_names: List[str],
    A_struct: np.ndarray,
    A_gap: np.ndarray,
    A_chem: np.ndarray,
    cook_neurons: List[str],
    q_leifer: Optional[np.ndarray],
    leifer_neurons: Optional[List[str]],
    lags: List[int],
    run_dir: Path,
    device: str,
    args,
) -> dict:
    """
    Legacy function - use run_4period_analysis instead.
    """
    # Convert to 4-period format
    period_segments = {
        'NOTHING': [],
        'ON': X_on_list,
        'SHOWING': [],
        'OFF': X_off_list,
    }
    return run_4period_analysis(
        period_segments, neuron_names, A_struct, A_gap, A_chem,
        cook_neurons, q_leifer, leifer_neurons, lags, run_dir, device, args
    )


def plot_on_off_comparison(on_off_results: dict, run_dir: Path):
    """
    Generate comparison figures for ON vs OFF analysis.
    """
    comparison_dir = run_dir / 'on_off_analysis' / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    has_on = 'result' in on_off_results.get('ON', {})
    has_off = 'result' in on_off_results.get('OFF', {})
    
    if not (has_on and has_off):
        print("  Cannot generate ON/OFF comparison (missing data)")
        return
    
    on_eval = on_off_results['ON']['eval_cook']
    off_eval = on_off_results['OFF']['eval_cook']
    on_lags = on_off_results['ON']['lags']
    off_lags = on_off_results['OFF']['lags']
    
    # Figure 1: AUROC comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    on_lag_sec = [l / SAMPLING_RATE for l in on_eval['lag']]
    off_lag_sec = [l / SAMPLING_RATE for l in off_eval['lag']]
    
    ax.plot(on_lag_sec, on_eval['auroc_struct'], 'o-', 
            label='Stimulus ON', color='coral', lw=2.5, markersize=10)
    ax.plot(off_lag_sec, off_eval['auroc_struct'], 's-', 
            label='Stimulus OFF', color='steelblue', lw=2.5, markersize=10)
    
    ax.axhline(0.5, color='gray', linestyle=':', lw=1.5, label='Random')
    ax.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax.set_ylabel(f'AUROC vs {STRUCTURAL_LABEL}', fontsize=12)
    ax.set_title('Connectivity Prediction: Stimulus ON vs OFF\n(AUROC by Time Lag)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.set_ylim(0.4, 0.85)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'fig_auroc_on_vs_off.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comparison_dir / 'fig_auroc_on_vs_off.png'}")
    
    # Figure 2: Edge density comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    on_result = on_off_results['ON']['result']
    off_result = on_off_results['OFF']['result']
    
    on_edges = [int(on_result.significant[l].sum()) for l in on_lags]
    off_edges = [int(off_result.significant[l].sum()) for l in off_lags]
    
    x_on = np.arange(len(on_lags))
    x_off = np.arange(len(off_lags))
    width = 0.35
    
    ax.bar(x_on - width/2, on_edges, width, label='Stimulus ON', color='coral', edgecolor='black')
    ax.bar(x_off + width/2, off_edges, width, label='Stimulus OFF', color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('FDR-Significant Edges', fontsize=12)
    ax.set_title('Edge Count: Stimulus ON vs OFF', fontsize=14, fontweight='bold')
    ax.set_xticks(range(max(len(on_lags), len(off_lags))))
    ax.set_xticklabels([f'{l}\n({l/SAMPLING_RATE:.2f}s)' for l in sorted(set(on_lags) | set(off_lags))])
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'fig_edge_density_on_vs_off.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comparison_dir / 'fig_edge_density_on_vs_off.png'}")


def plot_4period_comparison(period_results: dict, run_dir: Path, X_list: List[np.ndarray] = None):
    """
    Generate combined comparison figures for all 4 stimulus period types.
    
    Args:
        period_results: Dict with results for each period
        run_dir: Output directory
        X_list: Optional list of traces for visualization (if provided, generates trace figure)
    """
    PERIOD_NAMES = ['NOTHING', 'ON', 'SHOWING', 'OFF']
    PERIOD_COLORS = {'NOTHING': '#7f7f7f', 'ON': '#ff7f0e', 'SHOWING': '#2ca02c', 'OFF': '#1f77b4'}
    PERIOD_MARKERS = {'NOTHING': 'o', 'ON': '^', 'SHOWING': 's', 'OFF': 'v'}
    
    comparison_dir = run_dir / '4period_analysis' / 'comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Check which periods have results
    available_periods = [p for p in PERIOD_NAMES if 'eval_cook' in period_results.get(p, {})]
    
    if len(available_periods) < 2:
        print(f"  Cannot generate 4-period comparison (only {len(available_periods)} periods available)")
        return
    
    print(f"\n  Generating 4-period comparison figures for: {available_periods}")
    
    # Figure 0: Trace visualization with 4-period annotations (if X_list provided)
    if X_list is not None and len(X_list) > 0:
        from utils.stimulus_periods import get_4period_mask, StimulusPeriod, STIM_TIMES_SEC
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        # Use first trace (longest if they vary)
        X = X_list[0]
        T = X.shape[0]
        n_neurons = X.shape[1]
        time_sec = np.arange(T) / SAMPLING_RATE
        
        # Get 4-period mask
        mask = get_4period_mask(T, fps=SAMPLING_RATE)
        
        # Pick 3 representative neurons to show
        neuron_indices = [0, n_neurons // 2, n_neurons - 1]
        
        for ax_idx, neuron_idx in enumerate(neuron_indices):
            ax = axes[ax_idx]
            
            # Plot the trace
            ax.plot(time_sec, X[:, neuron_idx], 'k-', lw=0.8, alpha=0.8)
            
            # Shade each period with its color
            for period_name, period_val in [
                ('NOTHING', StimulusPeriod.NOTHING),
                ('ON', StimulusPeriod.ON),
                ('SHOWING', StimulusPeriod.SHOWING),
                ('OFF', StimulusPeriod.OFF),
            ]:
                period_mask = (mask == period_val)
                if period_mask.any():
                    # Find contiguous regions
                    diff = np.diff(period_mask.astype(int))
                    starts = np.where(diff == 1)[0] + 1
                    ends = np.where(diff == -1)[0] + 1
                    if period_mask[0]:
                        starts = np.concatenate([[0], starts])
                    if period_mask[-1]:
                        ends = np.concatenate([ends, [T]])
                    
                    for s, e in zip(starts, ends):
                        ax.axvspan(s/SAMPLING_RATE, e/SAMPLING_RATE, 
                                   alpha=0.2, color=PERIOD_COLORS[period_name],
                                   label=period_name if ax_idx == 0 and s == starts[0] else None)
            
            ax.set_ylabel(f'Neuron {neuron_idx+1}', fontsize=11)
            ax.grid(True, alpha=0.2)
            if ax_idx == 0:
                ax.set_title('Example Traces with 4-Period Segmentation', fontsize=14, fontweight='bold')
        
        # Add stimulus timing markers
        for ax in axes:
            for start, end in STIM_TIMES_SEC:
                ax.axvline(start, color='red', linestyle='--', alpha=0.5, lw=1)
                ax.axvline(end, color='red', linestyle='--', alpha=0.5, lw=1)
        
        axes[-1].set_xlabel('Time (seconds)', fontsize=12)
        
        # Create legend
        handles = [plt.Rectangle((0,0), 1, 1, fc=PERIOD_COLORS[p], alpha=0.3) for p in PERIOD_NAMES]
        handles.append(plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.5))
        labels = PERIOD_NAMES + ['Stimulus onset/offset']
        axes[0].legend(handles, labels, loc='upper right', fontsize=10, ncol=5)
        
        # Add text annotation explaining the periods
        fig.text(0.01, 0.01, 
                 'NOTHING: Baseline | ON: Stimulus onset (±2s) | SHOWING: Stimulus active | OFF: Stimulus offset (±2s)',
                 fontsize=10, ha='left', style='italic')
        
        plt.tight_layout()
        plt.savefig(comparison_dir / 'fig_trace_with_periods.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {comparison_dir / 'fig_trace_with_periods.png'}")
    
    # Figure 1: AUROC vs Lag for all periods
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for period_name in available_periods:
        eval_df = period_results[period_name]['eval_cook']
        lags_sec = [l / SAMPLING_RATE for l in eval_df['lag']]
        
        ax.plot(lags_sec, eval_df['auroc_struct'], 
                marker=PERIOD_MARKERS[period_name],
                color=PERIOD_COLORS[period_name],
                label=period_name,
                lw=2.5, markersize=10)
    
    ax.axhline(0.5, color='gray', linestyle=':', lw=1.5, alpha=0.7)
    ax.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax.set_ylabel(f'AUROC vs {STRUCTURAL_LABEL}', fontsize=12)
    ax.set_title('Connectivity Prediction by Stimulus Period\n(AUROC by Time Lag)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, title='Period')
    ax.set_ylim(0.4, 0.75)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'fig_auroc_4periods.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comparison_dir / 'fig_auroc_4periods.png'}")
    
    # Figure 2: Edge count comparison (bar chart)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Find common lags across all available periods
    all_lags = set()
    for p in available_periods:
        all_lags.update(period_results[p]['lags'])
    all_lags = sorted(all_lags)
    
    n_periods = len(available_periods)
    n_lags = len(all_lags)
    width = 0.8 / n_periods
    
    for i, period_name in enumerate(available_periods):
        result = period_results[period_name]['result']
        period_lags = period_results[period_name]['lags']
        
        edges = []
        x_positions = []
        for j, lag in enumerate(all_lags):
            if lag in period_lags:
                edges.append(int(result.significant[lag].sum()))
                x_positions.append(j + (i - n_periods/2 + 0.5) * width)
        
        ax.bar(x_positions, edges, width * 0.9, 
               label=period_name, 
               color=PERIOD_COLORS[period_name],
               edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Time Lag', fontsize=12)
    ax.set_ylabel('FDR-Significant Edges', fontsize=12)
    ax.set_title('Edge Count by Stimulus Period', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_lags))
    ax.set_xticklabels([f'{l}\n({l/SAMPLING_RATE:.1f}s)' for l in all_lags])
    ax.legend(loc='best', fontsize=11, title='Period')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'fig_edges_4periods.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comparison_dir / 'fig_edges_4periods.png'}")
    
    # Figure 3: Summary table as figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Collect summary data
    table_data = []
    for period_name in available_periods:
        eval_df = period_results[period_name]['eval_cook']
        result = period_results[period_name]['result']
        lags = period_results[period_name]['lags']
        
        best_idx = eval_df['auroc_struct'].idxmax()
        best_lag = eval_df.loc[best_idx, 'lag']
        best_auroc = eval_df.loc[best_idx, 'auroc_struct']
        total_edges = sum(int(result.significant[l].sum()) for l in lags)
        
        table_data.append([
            period_name,
            f"{len(lags)}",
            f"{best_lag} ({best_lag/SAMPLING_RATE:.1f}s)",
            f"{best_auroc:.4f}",
            f"{total_edges:,}",
        ])
    
    headers = ['Period', 'N Lags', 'Best Lag', 'Best AUROC', 'Total Edges']
    
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['lightblue'] * len(headers),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    ax.set_title('4-Period Analysis Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'fig_summary_4periods.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comparison_dir / 'fig_summary_4periods.png'}")
    
    # Figure 4: Edge Analysis (4-panel) - Edge density, E:I ratio, mean coupling
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for period_name in available_periods:
        result = period_results[period_name]['result']
        period_lags = period_results[period_name]['lags']
        lags_sec = [l / SAMPLING_RATE for l in period_lags]
        
        # Panel A: Edge density (edges / possible edges)
        n_neurons = list(result.mu_hat.values())[0].shape[0]
        max_edges = n_neurons * (n_neurons - 1)  # Excluding self-loops
        edge_density = [result.significant[l].sum() / max_edges for l in period_lags]
        axes[0, 0].plot(lags_sec, edge_density, 
                        marker=PERIOD_MARKERS[period_name],
                        color=PERIOD_COLORS[period_name],
                        label=period_name, lw=2, markersize=8)
        
        # Panel B: E:I ratio
        ei_ratios = []
        for l in period_lags:
            mu = result.mu_hat[l]
            sig = result.significant[l].astype(bool)
            exc = ((mu > 0) & sig).sum()
            inh = ((mu < 0) & sig).sum()
            ei_ratio = exc / inh if inh > 0 else exc
            ei_ratios.append(ei_ratio)
        axes[0, 1].plot(lags_sec, ei_ratios,
                        marker=PERIOD_MARKERS[period_name],
                        color=PERIOD_COLORS[period_name],
                        label=period_name, lw=2, markersize=8)
        
        # Panel C: Mean absolute coupling strength
        mean_coupling = []
        for l in period_lags:
            mu = result.mu_hat[l]
            sig = result.significant[l].astype(bool)
            if sig.sum() > 0:
                mean_coupling.append(np.abs(mu[sig]).mean())
            else:
                mean_coupling.append(0)
        axes[1, 0].plot(lags_sec, mean_coupling,
                        marker=PERIOD_MARKERS[period_name],
                        color=PERIOD_COLORS[period_name],
                        label=period_name, lw=2, markersize=8)
        
        # Panel D: AUROC (same as Fig 1 but smaller)
        eval_df = period_results[period_name]['eval_cook']
        axes[1, 1].plot(lags_sec, eval_df['auroc_struct'],
                        marker=PERIOD_MARKERS[period_name],
                        color=PERIOD_COLORS[period_name],
                        label=period_name, lw=2, markersize=8)
    
    # Panel A labels
    axes[0, 0].set_xlabel('Time Lag (s)', fontsize=11)
    axes[0, 0].set_ylabel('Edge Density', fontsize=11)
    axes[0, 0].set_title('A. Edge Density by Period', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Panel B labels
    axes[0, 1].axhline(1.0, color='gray', linestyle=':', alpha=0.7)
    axes[0, 1].set_xlabel('Time Lag (s)', fontsize=11)
    axes[0, 1].set_ylabel('E:I Ratio', fontsize=11)
    axes[0, 1].set_title('B. Excitatory:Inhibitory Ratio', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Panel C labels
    axes[1, 0].set_xlabel('Time Lag (s)', fontsize=11)
    axes[1, 0].set_ylabel('Mean |μ̂|', fontsize=11)
    axes[1, 0].set_title('C. Mean Coupling Strength', fontsize=12, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Panel D labels
    axes[1, 1].axhline(0.5, color='gray', linestyle=':', alpha=0.7)
    axes[1, 1].set_xlabel('Time Lag (s)', fontsize=11)
    axes[1, 1].set_ylabel('AUROC', fontsize=11)
    axes[1, 1].set_title(f'D. AUROC vs {STRUCTURAL_LABEL}', fontsize=12, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=9)
    axes[1, 1].set_ylim(0.45, 0.65)
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle('4-Period Edge Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(comparison_dir / 'fig_edge_analysis_4periods.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comparison_dir / 'fig_edge_analysis_4periods.png'}")
    
    # Figure 5: Structural connectome breakdown (Gap vs Chemical)
    # Check if we have detailed eval data
    has_detailed = False
    for period_name in available_periods:
        period_dir = run_dir / '4period_analysis' / period_name
        if (period_dir / 'eval_cook.csv').exists():
            has_detailed = True
            break
    
    if has_detailed:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for period_name in available_periods:
            eval_df = period_results[period_name]['eval_cook']
            lags_sec = [l / SAMPLING_RATE for l in eval_df['lag']]
            
            # Combined structural AUROC
            axes[0].plot(lags_sec, eval_df['auroc_struct'],
                        marker=PERIOD_MARKERS[period_name],
                        color=PERIOD_COLORS[period_name],
                        label=period_name, lw=2.5, markersize=10)
        
        axes[0].axhline(0.5, color='gray', linestyle=':', lw=1.5, alpha=0.7)
        axes[0].set_xlabel('Time Lag (seconds)', fontsize=12)
        axes[0].set_ylabel('AUROC', fontsize=12)
        axes[0].set_title(f'AUROC vs {STRUCTURAL_LABEL}\n(by Stimulus Period)', fontsize=13, fontweight='bold')
        axes[0].legend(loc='best', fontsize=10, title='Period')
        axes[0].set_ylim(0.45, 0.65)
        axes[0].grid(True, alpha=0.3)
        
        # AUPRC comparison
        for period_name in available_periods:
            eval_df = period_results[period_name]['eval_cook']
            lags_sec = [l / SAMPLING_RATE for l in eval_df['lag']]
            
            axes[1].plot(lags_sec, eval_df['auprc_struct'],
                        marker=PERIOD_MARKERS[period_name],
                        color=PERIOD_COLORS[period_name],
                        label=period_name, lw=2.5, markersize=10)
        
        axes[1].set_xlabel('Time Lag (seconds)', fontsize=12)
        axes[1].set_ylabel('AUPRC', fontsize=12)
        axes[1].set_title(f'AUPRC vs {STRUCTURAL_LABEL}\n(by Stimulus Period)', fontsize=13, fontweight='bold')
        axes[1].legend(loc='best', fontsize=10, title='Period')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(f'{STRUCTURAL_LABEL} Prediction by Stimulus Period', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(comparison_dir / 'fig_structural_auroc_4periods.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {comparison_dir / 'fig_structural_auroc_4periods.png'}")
    
    # Figure 6: Cell-type interaction heatmaps (lag x cell-type pair format)
    # Get neuron names from one of the results
    first_period = available_periods[0]
    first_result = period_results[first_period]['result']
    n_neurons = list(first_result.mu_hat.values())[0].shape[0]
    
    # Try to get neuron names from saved data
    neuron_names_local = None
    for period_name in available_periods:
        period_dir = run_dir / '4period_analysis' / period_name
        result_file = period_dir / 'result.npz'
        if result_file.exists():
            data = np.load(result_file, allow_pickle=True)
            if 'neuron_names' in data:
                neuron_names_local = list(data['neuron_names'])
                break
    
    if neuron_names_local is not None:
        from scipy.stats import mannwhitneyu
        
        pair_order = ['S→S', 'S→I', 'S→M', 'I→S', 'I→I', 'I→M', 'M→S', 'M→I', 'M→M']
        type_map = {'S': 'sensory', 'I': 'interneuron', 'M': 'motor'}
        types = get_neuron_types_for_list(neuron_names_local)
        
        # Get type indices
        type_indices = {
            t: [i for i, name in enumerate(neuron_names_local) if types[name] == t]
            for t in ['sensory', 'interneuron', 'motor']
        }
        
        # Generate per-period lag x celltype heatmaps
        for period_name in available_periods:
            result = period_results[period_name]['result']
            period_lags = period_results[period_name]['lags']
            
            n_lags = len(period_lags)
            n_pairs = len(pair_order)
            
            # Build data matrix and weights for statistical tests
            data_matrix = np.zeros((n_pairs, n_lags))
            all_weights = {pair: {lag: [] for lag in period_lags} for pair in pair_order}
            
            for lag_idx, lag in enumerate(period_lags):
                mu_hat = result.mu_hat[lag]
                sig_mask = result.significant[lag].astype(bool)
                
                for pair_idx, pair in enumerate(pair_order):
                    src_type = type_map[pair[0]]
                    tgt_type = type_map[pair[-1]]
                    
                    src_idx = type_indices[src_type]
                    tgt_idx = type_indices[tgt_type]
                    
                    if len(src_idx) == 0 or len(tgt_idx) == 0:
                        continue
                    
                    sub_mu = mu_hat[np.ix_(tgt_idx, src_idx)]
                    sub_sig = sig_mask[np.ix_(tgt_idx, src_idx)]
                    
                    weights = np.abs(sub_mu[sub_sig])
                    all_weights[pair][lag] = weights.tolist()
                    data_matrix[pair_idx, lag_idx] = np.mean(weights) if len(weights) > 0 else 0
            
            # Mann-Whitney tests: each pair vs all other pairs at same lag
            sig_markers = {}
            for lag_idx, lag in enumerate(period_lags):
                for pair_idx, pair in enumerate(pair_order):
                    pair_weights = all_weights[pair][lag]
                    if len(pair_weights) < 5:
                        continue
                    
                    # Pool all other pairs' weights at this lag
                    other_weights = []
                    for other_pair in pair_order:
                        if other_pair != pair:
                            other_weights.extend(all_weights[other_pair][lag])
                    
                    if len(other_weights) < 5:
                        continue
                    
                    try:
                        stat, pval = mannwhitneyu(pair_weights, other_weights, alternative='greater')
                        if pval < 0.001:
                            sig_markers[(pair_idx, lag_idx)] = '***'
                        elif pval < 0.01:
                            sig_markers[(pair_idx, lag_idx)] = '**'
                        elif pval < 0.05:
                            sig_markers[(pair_idx, lag_idx)] = '*'
                    except Exception:
                        pass
            
            # Create figure (normal and log scale side by side)
            for log_scale in [False, True]:
                fig, ax = plt.subplots(figsize=(max(8, n_lags * 1.5), max(6, n_pairs * 0.7)))
                
                plot_data = np.log10(data_matrix + 1e-6) if log_scale else data_matrix
                vmin = plot_data.min() if log_scale else 0
                vmax = plot_data.max()
                
                im = ax.imshow(plot_data, cmap='RdYlBu_r', aspect='auto', vmin=vmin, vmax=vmax)
                
                # Annotations
                for i in range(n_pairs):
                    for j in range(n_lags):
                        val = data_matrix[i, j]
                        scaled = plot_data[i, j]
                        threshold = (vmax + vmin) / 2
                        color = 'white' if scaled > threshold else 'black'
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                               fontsize=9, fontweight='bold', color=color)
                
                # Significance markers
                for (row, col), marker in sig_markers.items():
                    ax.text(col, row + 0.35, marker, ha='center', va='top',
                           fontsize=10, fontweight='bold', color='gold')
                
                ax.set_xticks(range(n_lags))
                ax.set_xticklabels([f'Lag {l}\n({l/SAMPLING_RATE:.1f}s)' for l in period_lags], fontsize=9)
                ax.set_yticks(range(n_pairs))
                ax.set_yticklabels(pair_order, fontsize=10, fontweight='bold')
                ax.set_xlabel('Time Lag', fontsize=12)
                ax.set_ylabel('Cell-Type Pair (Source → Target)', fontsize=12)
                
                scale_label = " (Log₁₀)" if log_scale else ""
                ax.set_title(f'{period_name}: Cell-Type Coupling Strength{scale_label}\n(Mean |μ̂| for FDR-Significant Edges)', 
                            fontsize=13, fontweight='bold')
                
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('log₁₀(Mean |μ̂|)' if log_scale else 'Mean |μ̂|', fontsize=11)
                
                # Legend for significance
                ax.text(1.15, 0.95, '*** p<0.001\n** p<0.01\n* p<0.05', transform=ax.transAxes,
                       fontsize=9, va='top', ha='left', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                suffix = '_log' if log_scale else ''
                plt.savefig(comparison_dir / f'fig_celltype_by_lag_{period_name}{suffix}.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            # Save CSV
            df = pd.DataFrame(data_matrix, index=pair_order, columns=[f'lag{l}' for l in period_lags])
            df.to_csv(comparison_dir / f'celltype_by_lag_{period_name}.csv')
        
        print(f"  Saved: Cell-type by lag heatmaps for each period (normal + log)")
        
        # Combined figure: all 4 periods in one figure (normal scale only)
        fig, axes = plt.subplots(1, len(available_periods), figsize=(4*len(available_periods), 7))
        if len(available_periods) == 1:
            axes = [axes]
        
        for ax_idx, period_name in enumerate(available_periods):
            result = period_results[period_name]['result']
            period_lags = period_results[period_name]['lags']
            
            n_lags = len(period_lags)
            data_matrix = np.zeros((n_pairs, n_lags))
            
            for lag_idx, lag in enumerate(period_lags):
                mu_hat = result.mu_hat[lag]
                sig_mask = result.significant[lag].astype(bool)
                
                for pair_idx, pair in enumerate(pair_order):
                    src_type = type_map[pair[0]]
                    tgt_type = type_map[pair[-1]]
                    src_idx = type_indices[src_type]
                    tgt_idx = type_indices[tgt_type]
                    
                    if len(src_idx) == 0 or len(tgt_idx) == 0:
                        continue
                    
                    sub_mu = mu_hat[np.ix_(tgt_idx, src_idx)]
                    sub_sig = sig_mask[np.ix_(tgt_idx, src_idx)]
                    weights = np.abs(sub_mu[sub_sig])
                    data_matrix[pair_idx, lag_idx] = np.mean(weights) if len(weights) > 0 else 0
            
            im = axes[ax_idx].imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.08)
            
            for i in range(n_pairs):
                for j in range(n_lags):
                    axes[ax_idx].text(j, i, f'{data_matrix[i,j]:.2f}', ha='center', va='center',
                                     fontsize=7, color='white' if data_matrix[i,j] > 0.04 else 'black')
            
            axes[ax_idx].set_xticks(range(n_lags))
            axes[ax_idx].set_xticklabels([str(l) for l in period_lags], fontsize=8)
            axes[ax_idx].set_yticks(range(n_pairs))
            axes[ax_idx].set_yticklabels(pair_order if ax_idx == 0 else [], fontsize=9)
            axes[ax_idx].set_xlabel('Lag', fontsize=10)
            if ax_idx == 0:
                axes[ax_idx].set_ylabel('Cell-Type Pair', fontsize=10)
            axes[ax_idx].set_title(period_name, fontsize=11, fontweight='bold')
        
        fig.suptitle('Cell-Type Coupling by Lag Across Stimulus Periods', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=axes, shrink=0.6, label='Mean |μ̂|')
        plt.tight_layout()
        plt.savefig(comparison_dir / 'fig_celltype_by_lag_combined.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {comparison_dir / 'fig_celltype_by_lag_combined.png'}")
    else:
        print("  Skipping cell-type heatmaps (neuron names not available)")
    
    # Save combined CSV
    combined_data = []
    for period_name in available_periods:
        eval_df = period_results[period_name]['eval_cook'].copy()
        eval_df['period'] = period_name
        combined_data.append(eval_df)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.to_csv(comparison_dir / 'eval_all_periods.csv', index=False)
        print(f"  Saved: {comparison_dir / 'eval_all_periods.csv'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Lag SBTG Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Data options
    parser.add_argument('--dataset', default='full_traces_imputed', help='Dataset name')
    parser.add_argument('--p-max', type=int, default=5, help='Maximum lag order')
    parser.add_argument('--lags', type=int, nargs='+', default=[1, 2, 3, 5],
                        help='Specific lags to analyze')
    
    # Approach selection
    parser.add_argument('--approach', choices=['A', 'B', 'C', 'all'], default='all',
                        help='Which approach to run')
    
    # Hyperparameters
    parser.add_argument('--noise-std', type=float, default=0.1, help='DSM noise std')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--n-folds', type=int, default=5, help='Cross-fitting folds')
    parser.add_argument('--fdr-alpha', type=float, default=0.1, help='FDR level')
    
    # HP tuning (for Approach C)
    parser.add_argument('--tune-hp', action='store_true', help='Tune HP for Approach C')
    parser.add_argument('--n-hp-trials', type=int, default=20, help='HP tuning trials')
    
    # Compute options
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output
    parser.add_argument('--no-plots', action='store_true', help='Skip plotting')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output dir')
    
    # Stimulus periods analysis
    parser.add_argument('--stimulus-periods', action='store_true',
                        help='Run additional ON/OFF analysis for stimulus periods')
    
    # Testing
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    
    if args.quick_test:
        args.p_max = 3
        args.epochs = 20
        args.n_folds = 3
        args.lags = [1, 2, 3]
        args.lags = [1, 2, 3]
        # args.dataset = 'nacl'  <-- Removed to allow phase analysis on full dataset
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SCRIPT 15: Multi-Lag SBTG Analysis")
    print("="*60)
    print(f"Timestamp: {timestamp}")
    print(f"\n[CONFIGURATION]")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max lag: {args.p_max} (at {SAMPLING_RATE}Hz = {args.p_max/SAMPLING_RATE:.2f}s)")
    print(f"  Lags to analyze: {args.lags}")
    print(f"  Approach(es): {args.approach}")
    print(f"  HP tuning: {args.tune_hp} ({args.n_hp_trials} trials if enabled)")
    print(f"  FDR alpha: {args.fdr_alpha}")
    print(f"  Cross-fitting folds: {args.n_folds}")
    print(f"  Output: {run_dir}")
    
    # Track total time
    import time
    start_time = time.time()
    
    # Step 1: Load data
    print("\n" + "-"*60)
    print("[1/5] LOADING DATA")
    print("-"*60)
    X_list, neuron_names = load_neural_data(args.dataset)
    n_neurons = len(neuron_names)
    n_worms = len(X_list)
    total_frames = sum(len(x) for x in X_list)
    print(f"  ✓ Loaded dataset '{args.dataset}': {n_worms} segments, {n_neurons} neurons, {total_frames} total frames", flush=True)

    if args.quick_test:
        print("  [Quick Test] Slicing dataset to first 3 worms for speed...")
        X_list = X_list[:3]
        n_worms = len(X_list)
        total_frames = sum(len(x) for x in X_list)
        print(f"  ✓ Sliced dataset: {n_worms} segments, {total_frames} total frames")
    
    # Step 2: Load ground truth
    print("\n" + "-"*60)
    print("[2/5] LOADING GROUND TRUTH")
    print("-"*60)
    A_struct, A_gap, A_chem, cook_neurons = load_cook_connectome()
    n_cook_edges = int((A_struct > 0).sum())
    n_gap_edges = int((A_gap > 0).sum())
    n_chem_edges = int((A_chem > 0).sum())
    print(f"  ✓ {STRUCTURAL_LABEL} connectome: {len(cook_neurons)} neurons, {n_cook_edges} structural edges")
    print(f"    ({n_gap_edges} gap junctions, {n_chem_edges} chemical synapses)")
    
    q_leifer, _, leifer_neurons = load_leifer_atlas()
    if q_leifer is not None:
        n_leifer_edges = int((q_leifer < 0.05).sum())
        print(f"  ✓ {FUNCTIONAL_LABEL} atlas: {len(leifer_neurons)} neurons, {n_leifer_edges} functional edges (q<0.05)")
    else:
        print(f"  ! {FUNCTIONAL_LABEL} atlas not available")
    
    # Determine device - auto-detect best available if 'auto' or fallback if requested device unavailable
    device = args.device
    
    def get_best_device():
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        try:
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except Exception:
            pass
        return 'cpu'
    
    if device == 'auto':
        device = get_best_device()
        print(f"Auto-detected device: {device}")
    elif device == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to auto-detect")
            device = get_best_device()
    elif device == 'mps':
        try:
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print("Warning: MPS requested but not available, falling back to auto-detect")
                device = get_best_device()
        except Exception:
            print("Warning: MPS check failed, falling back to auto-detect")
            device = get_best_device()
    
    print(f"Using device: {device}")
    
    # Step 3: Run approaches
    print("\n" + "-"*60)
    print("[3/5] RUNNING MULTI-LAG SBTG ANALYSIS")
    print("-"*60, flush=True)
    
    # Results storage
    results = {}
    eval_results = {}
    approach_times = {}
    
    # Run Approach A
    if args.approach in ['A', 'all']:
        approach_start = time.time()
        print("\n" + "="*40)
        print("APPROACH A: Per-Lag 2-Block Windows")
        print("="*40)
        print("  Theory: Separate 2-block SBTG per lag (reduced-form VAR)")
        print(f"  Training {len(args.lags)} models (lags {args.lags})...", flush=True)
        if args.tune_hp:
            print(f"  HP tuning: {args.n_hp_trials} trials per lag using null_contrast objective")
        
        estimator_A = PerLagSBTGEstimator(
            p_max=args.p_max,
            tune_hp=args.tune_hp,
            n_hp_trials=args.n_hp_trials,
            noise_std=args.noise_std,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            n_folds=args.n_folds,
            fdr_alpha=args.fdr_alpha,
            device=device,
            random_state=args.seed,
            verbose=True,
        )
        
        result_A = estimator_A.fit(X_list)
        results['A'] = result_A
        
        # Evaluate
        print(f"\n[Approach A] Evaluating vs {STRUCTURAL_LABEL}...")
        eval_cook_A = evaluate_vs_cook(result_A, neuron_names, A_struct, cook_neurons)
        print(eval_cook_A.to_string(index=False))
        
        # Detailed gap/chemical evaluation
        print("\n[Approach A] Gap vs Chemical breakdown...")
        eval_cook_detailed_A = evaluate_vs_cook_detailed(
            result_A, neuron_names, A_struct, A_gap, A_chem, cook_neurons
        )
        print(eval_cook_detailed_A[['lag', 'auroc_combined', 'auroc_gap', 'auroc_chem']].to_string(index=False))
        
        eval_leifer_A = None
        if q_leifer is not None:
            print(f"\n[Approach A] Evaluating vs {FUNCTIONAL_LABEL}...")
            eval_leifer_A = evaluate_vs_leifer(result_A, neuron_names, q_leifer, leifer_neurons)
            print(eval_leifer_A.to_string(index=False))
        
        eval_results['A'] = {'cook': eval_cook_A, 'cook_detailed': eval_cook_detailed_A, 'leifer': eval_leifer_A}
        approach_times['A'] = time.time() - approach_start
        print(f"  ✓ Approach A completed in {approach_times['A']:.1f}s")
        
        # Save results
        np.savez(
            run_dir / 'result_A.npz',
            **{f'mu_hat_lag{k}': v for k, v in result_A.mu_hat.items()},
            **{f'pval_lag{k}': v for k, v in result_A.p_values.items()},
            **{f'sig_lag{k}': v for k, v in result_A.significant.items()},
            neuron_names=neuron_names,
            p_max=result_A.p_max,
        )
        eval_cook_A.to_csv(run_dir / 'eval_cook_A.csv', index=False)
        eval_cook_detailed_A.to_csv(run_dir / 'eval_cook_detailed_A.csv', index=False)
        if eval_leifer_A is not None:
            eval_leifer_A.to_csv(run_dir / 'eval_leifer_A.csv', index=False)
    
    # Run Approach B
    if args.approach in ['B', 'all']:
        approach_start = time.time()
        print("\n" + "="*40)
        print("APPROACH B: Full Multi-Block Window")
        print("="*40)
        dim_z = (args.p_max + 1) * n_neurons
        print(f"  Theory: Single (p+1)-block model extracts all lags (Theorem 5.1)")
        print(f"  Window dimension: {dim_z} ({args.p_max+1} blocks × {n_neurons} neurons)")
        print(f"  Training 1 model, extracting lags {args.lags}...", flush=True)
        if args.tune_hp:
            print(f"  HP tuning: {args.n_hp_trials} trials using null_contrast objective")
        
        estimator_B = MultiBlockSBTGEstimator(
            p_max=args.p_max,
            tune_hp=args.tune_hp,
            n_hp_trials=args.n_hp_trials,
            noise_std=args.noise_std,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            n_folds=args.n_folds,
            fdr_alpha=args.fdr_alpha,
            device=device,
            random_state=args.seed,
            verbose=True,
        )
        
        result_B = estimator_B.fit(X_list)
        results['B'] = result_B
        
        # Evaluate
        print(f"\n[Approach B] Evaluating vs {STRUCTURAL_LABEL}...")
        eval_cook_B = evaluate_vs_cook(result_B, neuron_names, A_struct, cook_neurons)
        print(eval_cook_B.to_string(index=False))
        
        # Detailed gap/chemical evaluation
        print("\n[Approach B] Gap vs Chemical breakdown...")
        eval_cook_detailed_B = evaluate_vs_cook_detailed(
            result_B, neuron_names, A_struct, A_gap, A_chem, cook_neurons
        )
        print(eval_cook_detailed_B[['lag', 'auroc_combined', 'auroc_gap', 'auroc_chem']].to_string(index=False))
        
        eval_leifer_B = None
        if q_leifer is not None:
            print(f"\n[Approach B] Evaluating vs {FUNCTIONAL_LABEL}...")
            eval_leifer_B = evaluate_vs_leifer(result_B, neuron_names, q_leifer, leifer_neurons)
            print(eval_leifer_B.to_string(index=False))
        
        eval_results['B'] = {'cook': eval_cook_B, 'cook_detailed': eval_cook_detailed_B, 'leifer': eval_leifer_B}
        approach_times['B'] = time.time() - approach_start
        print(f"  ✓ Approach B completed in {approach_times['B']:.1f}s")
        
        # Save results
        np.savez(
            run_dir / 'result_B.npz',
            **{f'mu_hat_lag{k}': v for k, v in result_B.mu_hat.items()},
            **{f'pval_lag{k}': v for k, v in result_B.p_values.items()},
            **{f'sig_lag{k}': v for k, v in result_B.significant.items()},
            neuron_names=neuron_names,
            p_max=result_B.p_max,
        )
        eval_cook_B.to_csv(run_dir / 'eval_cook_B.csv', index=False)
        eval_cook_detailed_B.to_csv(run_dir / 'eval_cook_detailed_B.csv', index=False)
        if eval_leifer_B is not None:
            eval_leifer_B.to_csv(run_dir / 'eval_leifer_B.csv', index=False)
    
    # Run Approach C
    if args.approach in ['C', 'all']:
        approach_start = time.time()
        print("\n" + "="*40)
        print("APPROACH C: Minimal Multi-Block (Per-Lag)")
        print("="*40)
        print(f"  Theory: Per-lag model with minimal conditioning on intermediate lags")
        print(f"  Training {len(args.lags)} models with increasing block sizes...", flush=True)
        if args.tune_hp:
            print(f"  HP tuning: {args.n_hp_trials} trials per lag using null_contrast objective")
        
        estimator_C = MinimalMultiBlockEstimator(
            lags=args.lags,
            tune_hp=args.tune_hp,
            n_hp_trials=args.n_hp_trials,
            noise_std=args.noise_std,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            n_folds=args.n_folds,
            fdr_alpha=args.fdr_alpha,
            device=device,
            random_state=args.seed,
            verbose=True,
        )
        
        result_C = estimator_C.fit(X_list)
        results['C'] = result_C
        
        # Evaluate
        print(f"\n[Approach C] Evaluating vs {STRUCTURAL_LABEL}...")
        eval_cook_C = evaluate_vs_cook(result_C, neuron_names, A_struct, cook_neurons)
        print(eval_cook_C.to_string(index=False))
        
        # Detailed gap/chemical evaluation
        print("\n[Approach C] Gap vs Chemical breakdown...")
        eval_cook_detailed_C = evaluate_vs_cook_detailed(
            result_C, neuron_names, A_struct, A_gap, A_chem, cook_neurons
        )
        print(eval_cook_detailed_C[['lag', 'auroc_combined', 'auroc_gap', 'auroc_chem']].to_string(index=False))
        
        eval_leifer_C = None
        if q_leifer is not None:
            print(f"\n[Approach C] Evaluating vs {FUNCTIONAL_LABEL}...")
            eval_leifer_C = evaluate_vs_leifer(result_C, neuron_names, q_leifer, leifer_neurons)
            print(eval_leifer_C.to_string(index=False))
        
        eval_results['C'] = {'cook': eval_cook_C, 'cook_detailed': eval_cook_detailed_C, 'leifer': eval_leifer_C}
        approach_times['C'] = time.time() - approach_start
        print(f"  ✓ Approach C completed in {approach_times['C']:.1f}s")
        
        # Save results (including tuned HP if applicable)
        save_dict = {
            **{f'mu_hat_lag{k}': v for k, v in result_C.mu_hat.items()},
            **{f'pval_lag{k}': v for k, v in result_C.p_values.items()},
            **{f'sig_lag{k}': v for k, v in result_C.significant.items()},
            'neuron_names': neuron_names,
            'p_max': result_C.p_max,
            'lags': args.lags,
        }
        
        # Save tuned HP configs if applicable
        if args.tune_hp and hasattr(estimator_C, 'hp_configs'):
            for lag, config in estimator_C.hp_configs.items():
                save_dict[f'hp_config_lag{lag}'] = config.to_dict()
        
        np.savez(run_dir / 'result_C.npz', **save_dict)
        eval_cook_C.to_csv(run_dir / 'eval_cook_C.csv', index=False)
        eval_cook_detailed_C.to_csv(run_dir / 'eval_cook_detailed_C.csv', index=False)
        if eval_leifer_C is not None:
            eval_leifer_C.to_csv(run_dir / 'eval_leifer_C.csv', index=False)
    
    # Step 3.5: Run Baselines
    print("\n" + "-"*60)
    print("[3.5/5] RUNNING BASELINES")
    print("-"*60)
    
    baseline_rows_cook = []
    baseline_rows_leifer = []
    baseline_matrices = {}
    # Use the same lags as SBTG for fair comparison
    lags_to_test = args.lags
    print(f"  Evaluating baselines at lags: {lags_to_test}")
    
    # Prepare Randi_Optogenetics_2023 ground truth if available
    leifer_gt = None
    if q_leifer is not None:
        leifer_gt = (q_leifer < 0.05).astype(float)
        print(f"  {FUNCTIONAL_LABEL} atlas available: {len(leifer_neurons)} neurons")
    
    # Pearson
    print("  Evaluating Pearson Correlation...")
    for lag in lags_to_test:
        corr = compute_pearson_baseline(X_list, lag)
        baseline_matrices[f'Pearson_lag{lag}'] = corr
        
        # Evaluate against structural benchmark
        corr_aligned, struct_aligned, common = align_matrices(corr, neuron_names, A_struct, cook_neurons)
        if len(common) > 0:
            auroc = compute_auroc(corr_aligned, struct_aligned)
            auprc = compute_auprc(corr_aligned, struct_aligned)
            spearman = compute_spearman(corr_aligned, struct_aligned)
            baseline_rows_cook.append({
                'method': 'Pearson', 'lag': lag, 
                'auroc_struct': auroc, 'auprc_struct': auprc, 'spearman_struct': spearman
            })
        
        # Evaluate against functional benchmark
        if leifer_gt is not None:
            corr_leifer, gt_leifer, common_leifer = align_matrices(corr, neuron_names, leifer_gt, leifer_neurons)
            if len(common_leifer) > 0:
                auroc_leifer = compute_auroc(corr_leifer, gt_leifer)
                auprc_leifer = compute_auprc(corr_leifer, gt_leifer)
                spearman_leifer = compute_spearman(corr_leifer, gt_leifer)
                baseline_rows_leifer.append({
                    'method': 'Pearson', 'lag': lag,
                    'auroc': auroc_leifer, 'auprc': auprc_leifer, 'spearman': spearman_leifer
                })
            
    # VAR
    print("  Evaluating VAR...")
    for lag in lags_to_test:
         coef = compute_var_baseline(X_list, lag)
         baseline_matrices[f'VAR_lag{lag}'] = coef
         
         # Evaluate against structural benchmark
         coef_aligned, struct_aligned, common = align_matrices(coef, neuron_names, A_struct, cook_neurons)
         if len(common) > 0:
             auroc = compute_auroc(coef_aligned, struct_aligned)
             auprc = compute_auprc(coef_aligned, struct_aligned)
             spearman = compute_spearman(coef_aligned, struct_aligned)
             baseline_rows_cook.append({
                 'method': 'VAR', 'lag': lag, 
                 'auroc_struct': auroc, 'auprc_struct': auprc, 'spearman_struct': spearman
             })
         
         # Evaluate against functional benchmark
         if leifer_gt is not None:
             coef_leifer, gt_leifer, common_leifer = align_matrices(coef, neuron_names, leifer_gt, leifer_neurons)
             if len(common_leifer) > 0:
                 auroc_leifer = compute_auroc(coef_leifer, gt_leifer)
                 auprc_leifer = compute_auprc(coef_leifer, gt_leifer)
                 spearman_leifer = compute_spearman(coef_leifer, gt_leifer)
                 baseline_rows_leifer.append({
                     'method': 'VAR', 'lag': lag,
                     'auroc': auroc_leifer, 'auprc': auprc_leifer, 'spearman': spearman_leifer
                 })

    if not args.quick_test:
        # Granger (slow) - compute for all requested lags for fair comparison
        print("  Evaluating Granger Causality (this may take a while)...")
        # Limit to lags <= 3 for Granger since higher lags need more data per worm
        granger_lags = [l for l in lags_to_test if l <= 3]
        if len(granger_lags) < len(lags_to_test):
            print(f"    Note: Granger limited to lags {granger_lags} (higher lags need more data)")
        for lag in granger_lags:
             f_stats = compute_granger_baseline(X_list, lag)
             baseline_matrices[f'Granger_lag{lag}'] = f_stats
             
             # Evaluate against structural benchmark
             f_aligned, struct_aligned, common = align_matrices(f_stats, neuron_names, A_struct, cook_neurons)
             if len(common) > 0:
                 auroc = compute_auroc(f_aligned, struct_aligned)
                 auprc = compute_auprc(f_aligned, struct_aligned)
                 spearman = compute_spearman(f_aligned, struct_aligned)
                 baseline_rows_cook.append({
                     'method': 'Granger', 'lag': lag, 
                     'auroc_struct': auroc, 'auprc_struct': auprc, 'spearman_struct': spearman
                 })
             
             # Evaluate against functional benchmark
             if leifer_gt is not None:
                 f_leifer, gt_leifer, common_leifer = align_matrices(f_stats, neuron_names, leifer_gt, leifer_neurons)
                 if len(common_leifer) > 0:
                     auroc_leifer = compute_auroc(f_leifer, gt_leifer)
                     auprc_leifer = compute_auprc(f_leifer, gt_leifer)
                     spearman_leifer = compute_spearman(f_leifer, gt_leifer)
                     baseline_rows_leifer.append({
                         'method': 'Granger', 'lag': lag,
                         'auroc': auroc_leifer, 'auprc': auprc_leifer, 'spearman': spearman_leifer
                     })
                 
    # Save baseline metrics
    baseline_df_cook = pd.DataFrame(baseline_rows_cook)
    baseline_df_leifer = pd.DataFrame(baseline_rows_leifer)
    
    if not baseline_df_cook.empty:
        baseline_df_cook.to_csv(run_dir / "baseline_metrics_cook.csv", index=False)
        print(f"  ✓ Saved {STRUCTURAL_LABEL} baseline metrics to {run_dir / 'baseline_metrics_cook.csv'}")
    
    if not baseline_df_leifer.empty:
        baseline_df_leifer.to_csv(run_dir / "baseline_metrics_leifer.csv", index=False)
        print(f"  ✓ Saved {FUNCTIONAL_LABEL} baseline metrics to {run_dir / 'baseline_metrics_leifer.csv'}")
    
    # Also save combined for backward compatibility
    baseline_df_combined = baseline_df_cook.copy()
    if not baseline_df_leifer.empty:
        # Merge on method and lag
        baseline_df_combined = baseline_df_combined.merge(
            baseline_df_leifer, on=['method', 'lag'], how='outer', suffixes=('_cook', '_leifer')
        )
    baseline_df_combined.to_csv(run_dir / "baseline_metrics.csv", index=False)
    print(f"  ✓ Saved combined baseline metrics to {run_dir / 'baseline_metrics.csv'}")
        
    if baseline_matrices:
        np.savez(run_dir / 'baselines.npz', **baseline_matrices)
        print(f"  ✓ Saved baseline matrices to {run_dir / 'baselines.npz'}")

    # Step 4: Plotting
    if not args.no_plots:
        print("\n" + "-"*60)
        print("[4/5] GENERATING FIGURES")
        print("-"*60)
        fig_dir = run_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)
        
        # Per-approach plots
        for approach, result in results.items():
            eval_cook = eval_results[approach]['cook']
            eval_leifer = eval_results[approach]['leifer']
            
            # Original combined AUROC plot
            plot_auroc_vs_lag(
                eval_cook, eval_leifer, approach,
                fig_dir / f'fig_auroc_vs_lag_{approach}.png'
            )
            
            # NEW: Separate metric plots
            plot_separate_metrics(
                eval_cook, eval_leifer, approach, fig_dir
            )
            
            plot_type_interactions(
                result, neuron_names, approach,
                fig_dir / f'fig_type_interactions_{approach}.png',
                lags_to_plot=args.lags[:3] if len(args.lags) >= 3 else args.lags,
            )
        
        # Multi-approach comparison plots
        if len(results) > 1:
            # μ̂ comparison between approaches
            for lag in [1, 2] if args.p_max >= 2 else [1]:
                if 'A' in results and 'B' in results:
                    plot_mu_hat_comparison(
                        results.get('A'), results.get('B'),
                        fig_dir / f'fig_mu_hat_comparison_AB_lag{lag}.png',
                        lag=lag,
                    )
                if 'B' in results and 'C' in results:
                    plot_mu_hat_comparison(
                        results.get('B'), results.get('C'),
                        fig_dir / f'fig_mu_hat_comparison_BC_lag{lag}.png',
                        lag=lag,
                    )
            
            # Edge density comparison
            plot_edge_density_by_lag(
                results,
                fig_dir / 'fig_edge_density_by_lag.png'
            )
            
            # Comprehensive summary figure
            plot_summary_figure(
                results,
                eval_results,
                fig_dir / 'fig_summary.png'
            )
            
    # Baseline Comparison
    if 'baseline_df_cook' in locals() and not baseline_df_cook.empty:
        # Compare against the best SBTG approach found
        best_approach = 'C' if 'C' in results else ('B' if 'B' in results else ('A' if 'A' in results else None))
        
        if best_approach and best_approach in eval_results:
            sbtg_df = eval_results[best_approach]['cook'].copy()
            sbtg_df['method'] = f'SBTG-{best_approach}'
            
            plot_baseline_comparison(
                sbtg_df, baseline_df_cook,
                fig_dir / 'fig_baseline_comparison_cook.png'
            )

    # Stimulus periods analysis (if requested)
    if args.stimulus_periods:
        print("\n" + "-"*60)
        print("[EXTRA] 4-PERIOD STIMULUS ANALYSIS")
        print("-"*60)
        
        # Segment data into 4 periods
        print("Segmenting data into 4 periods (NOTHING, ON, SHOWING, OFF)...")
        period_segments = segment_data_4periods(X_list, fps=SAMPLING_RATE)
        
        # Show summary
        summary = summarize_4period_segmentation(X_list[0].shape[0], fps=SAMPLING_RATE)
        print(f"  Skip initial: {summary['skip_initial_sec']:.1f}s")
        print(f"  Transition window: ±{summary['transition_window_sec']:.1f}s")
        for period in ['NOTHING', 'ON', 'SHOWING', 'OFF']:
            dur = summary[f'{period.lower()}_duration_sec']
            n_seg = summary[f'{period.lower()}_n_segments']
            n_actual = len(period_segments[period])
            print(f"    {period:8s}: {dur:5.1f}s ({n_seg} events x ~{len(X_list)} worms = {n_actual} segments)")
        
        # Run 4-period analysis
        period_results = run_4period_analysis(
            period_segments,
            neuron_names=neuron_names,
            A_struct=A_struct,
            A_gap=A_gap,
            A_chem=A_chem,
            cook_neurons=cook_neurons,
            q_leifer=q_leifer,
            leifer_neurons=leifer_neurons,
            lags=args.lags,
            run_dir=run_dir,
            device=device,
            args=args,
        )
        
        # Generate comparison figures
        if not args.no_plots:
            print("\n  Generating 4-period comparison figures...")
            plot_4period_comparison(period_results, run_dir, X_list=X_list)
        
        # Add to config
        config_on_off = {
            'stimulus_periods': True,
            '4period_analysis': True,
            'period_lags': {p: period_results.get(p, {}).get('lags', []) 
                           for p in ['NOTHING', 'ON', 'SHOWING', 'OFF']},
        }
    else:
        config_on_off = {'stimulus_periods': False}
    
    # Step 5: Save config and finalize
    print("\n" + "-"*60)
    print("[5/5] SAVING RESULTS")
    print("-"*60)
    
    config = {
        'dataset': args.dataset,
        'p_max': args.p_max,
        'lags': args.lags,
        'approach': args.approach,
        'noise_std': args.noise_std,
        'hidden_dim': args.hidden_dim,
        'epochs': args.epochs,
        'n_folds': args.n_folds,
        'fdr_alpha': args.fdr_alpha,
        'tune_hp': args.tune_hp,
        'n_hp_trials': args.n_hp_trials,
        'device': device,
        'seed': args.seed,
        'timestamp': timestamp,
        'n_neurons': len(neuron_names),
        'approach_times': approach_times,
        **config_on_off,  # Merge ON/OFF config
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print(f"  ✓ config.json saved")
    for approach in results.keys():
        print(f"  ✓ result_{approach}.npz saved")
        print(f"  ✓ eval_cook_{approach}.csv saved")
        if eval_results[approach]['leifer'] is not None:
            print(f"  ✓ eval_leifer_{approach}.csv saved")
    if not args.no_plots:
        print(f"  ✓ figures/ directory with {len(list(fig_dir.glob('*.png')))} plots")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Results saved to: {run_dir}")
    print(f"Total runtime: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    
    # Summary
    print("\n=== Summary ===")
    for approach in ['A', 'B', 'C']:
        if approach in eval_results:
            df = eval_results[approach]['cook']
            best_row = df.loc[df['auroc_struct'].idxmax()]
            best_lag = int(best_row['lag'])
            # Also get null contrast for best lag
            nc = compute_null_contrast(results[approach].mu_hat[best_lag])
            
            # Edge count at best lag
            sig = results[approach].significant[best_lag]
            n_edges = int(sig.sum() - np.trace(sig))
            
            approach_names = {'A': 'Per-Lag 2-Block', 'B': 'Full Multi-Block', 'C': 'Minimal Multi-Block'}
            print(f"  {approach} ({approach_names[approach]}):")
            print(f"    Best AUROC = {best_row['auroc_struct']:.4f} at lag {best_lag} ({best_lag/SAMPLING_RATE:.2f}s)")
            print(f"    Null Contrast = {nc:.3f}, Edges = {n_edges}")


if __name__ == "__main__":
    main()
