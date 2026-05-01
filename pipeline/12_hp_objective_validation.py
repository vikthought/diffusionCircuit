#!/usr/bin/env python3
"""
SCRIPT 12: HP Objective Validation
===================================

Systematic comparison of different hyperparameter tuning objectives:
1. Edge Stability (bootstrap resampling) - theoretically grounded
2. Null Contrast (signal above shuffled null) - theoretically grounded  
3. Combined (stability × contrast) - both must be good
4. DSM Validation Loss (baseline - known to fail)

Goal: Identify which objective best correlates with biological evaluation
(Cook connectome, Leifer functional atlas).

Theoretical basis: See docs/THEORETICAL_FOUNDATIONS.md

Usage:
    # Local test (4 trials, quick)
    python pipeline/12_hp_objective_validation.py --trials 4 --quick
    
    # Full cluster run
    python pipeline/12_hp_objective_validation.py --trials 100 --bootstraps 5 --nulls 5
    
    # Test specific objectives
    python pipeline/12_hp_objective_validation.py --trials 10 --objectives stability,null_contrast

Outputs:
    - results/hp_objective_validation/hp_validation_results.csv
    - results/hp_objective_validation/correlation_summary.csv
    - results/hp_objective_validation/figures/objective_vs_evaluation.png
    - results/hp_objective_validation/figures/correlation_heatmap.png
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import warnings
import os
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import spearmanr


def setup_compute_environment(n_workers: int = 0) -> Dict:
    """
    Setup and report compute environment (GPU/CPU).
    
    Args:
        n_workers: Number of CPU workers for parallel operations (0 = auto)
    
    Returns:
        Dict with device info
    """
    info = {}
    
    # GPU detection
    if torch.cuda.is_available():
        info['device'] = 'cuda'
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['device'] = 'mps'  # Apple Silicon
        info['gpu_name'] = 'Apple Silicon (MPS)'
        info['gpu_memory'] = 'shared'
    else:
        info['device'] = 'cpu'
        info['gpu_name'] = None
        info['gpu_memory'] = None
    
    # CPU setup
    info['cpu_count'] = multiprocessing.cpu_count()
    if n_workers == 0:
        n_workers = min(4, info['cpu_count'])  # Default: 4 workers or available CPUs
    info['n_workers'] = n_workers
    
    # Set PyTorch thread count for CPU operations
    if info['device'] == 'cpu':
        torch.set_num_threads(info['cpu_count'])
        info['torch_threads'] = torch.get_num_threads()
    else:
        info['torch_threads'] = torch.get_num_threads()
    
    return info
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import SBTG
from pipeline.models.sbtg import SBTGStructuredVolatilityEstimator

# Import utilities
from pipeline.utils.io import load_structural_connectome as _load_structural_connectome
from pipeline.utils.align import normalize_neuron_name
from pipeline.utils.leifer import load_leifer_atlas_data

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS_DIR = PROJECT_ROOT / "results" / "intermediate" / "datasets"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"
OUTPUT_DIR = PROJECT_ROOT / "results" / "hp_objective_validation"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_imputed_data() -> Tuple[List[np.ndarray], List[str]]:
    """Load full traces imputed dataset."""
    data_dir = DATASETS_DIR / "full_traces_imputed"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Imputed data not found: {data_dir}")
    
    # Load segments
    X_arr = np.load(data_dir / "X_segments.npy", allow_pickle=True)
    X_list = [X_arr[i] for i in range(X_arr.shape[0])]
    
    # Load neuron names
    std_file = data_dir / "standardization.json"
    if std_file.exists():
        with open(std_file) as f:
            data = json.load(f)
            neurons = data.get('node_order', [])
    else:
        neurons = [f"N{i}" for i in range(X_list[0].shape[1])]
    
    return X_list, neurons


def load_cook_connectome() -> Tuple[np.ndarray, List[str]]:
    """Load Cook structural connectome."""
    A_struct, nodes, _ = _load_structural_connectome(CONNECTOME_DIR)
    return A_struct, nodes


def load_leifer_data() -> Tuple[np.ndarray, List[str]]:
    """Load Leifer functional atlas q-values."""
    q_matrix, q_eq, neurons = load_leifer_atlas_data()
    
    if q_matrix is None:
        raise FileNotFoundError("Leifer atlas data not found. Run ensure_aligned_atlas() first.")
    
    return q_matrix, neurons


# =============================================================================
# HP CONFIGURATION SAMPLING
# =============================================================================

def sample_hp_configurations(n_configs: int, seed: int = 42, device: Optional[str] = None) -> List[Dict]:
    """
    Sample random HP configurations from the search space.
    
    We sample randomly (not optimize) to see correlation across the HP space.
    """
    np.random.seed(seed)
    
    configs = []
    for _ in range(n_configs):
        # Sample from reasonable ranges (informed by prior experiments)
        config = {
            'dsm_lr': 10 ** np.random.uniform(-4, -2.5),  # 1e-4 to 3e-3
            'dsm_epochs': int(np.random.choice([80, 100, 120, 140, 160])),
            'dsm_noise_std': np.random.choice([0.08, 0.12, 0.16, 0.20, 0.25, 0.30]),
            'dsm_hidden_dim': int(np.random.choice([64, 128, 256])),
            'structured_hidden_dim': int(np.random.choice([32, 64])),
            'structured_l1_lambda': 10 ** np.random.uniform(-3.5, -1.5),  # ~3e-4 to 3e-2
            'fdr_alpha': np.random.choice([0.1, 0.15, 0.2]),
            'model_type': np.random.choice(['linear', 'feature_bilinear']),
            # Fixed params
            'dsm_num_layers': 3,
            'dsm_batch_size': 128,
            'structured_num_layers': 2,
            'structured_init_scale': 0.1,
            'train_split': 'odd_even',
            'train_frac': 0.7,
            'fdr_method': 'by',
            'hac_max_lag': 5,
            'verbose': False,
            'inference_mode': 'in_sample',
        }
        
        # Add device if specified
        if device:
            config['device'] = device
        
        # Add model-specific params
        if config['model_type'] == 'feature_bilinear':
            config['feature_dim'] = int(np.random.choice([8, 16, 32]))
        
        configs.append(config)
    
    return configs


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def compute_dsm_val_loss(
    estimator: SBTGStructuredVolatilityEstimator,
    X_val_list: List[np.ndarray],
    device: Optional[torch.device] = None
) -> float:
    """
    Compute DSM loss on validation set (baseline objective).
    """
    # Use estimator's device to ensure consistency
    if device is None:
        device = estimator.device
    
    model = estimator.model
    noise_std = estimator.dsm_noise_std
    
    if model is None:
        return float('inf')

    model.eval()
    
    try:
        Z_raw, _, _ = estimator._build_windows_raw(X_val_list)
        
        if len(Z_raw) == 0:
            return float('inf')
        
        Z = (Z_raw - Z_raw.mean(axis=0, keepdims=True)) / (Z_raw.std(axis=0, keepdims=True) + 1e-8)
        Z_tensor = torch.tensor(Z, dtype=torch.float32, device=device, requires_grad=True)
        
        # Need gradients for score computation
        eps = torch.randn_like(Z_tensor) * noise_std
        noisy_Z = Z_tensor + eps
        
        output = model(noisy_Z)
        if isinstance(output, tuple):
            _, score = output
        else:
            score = output
        
        with torch.no_grad():
            target = -eps / (noise_std ** 2)
            loss = ((score - target) ** 2).sum(dim=1).mean()
            return loss.item()
            
    except Exception as e:
        print(f"  DSM loss error: {e}")
        return float('inf')


def compute_edge_stability(
    X_list: List[np.ndarray],
    params: Dict,
    n_bootstraps: int = 5,
    quick: bool = False
) -> float:
    """
    Measure edge consistency across bootstrap samples.
    
    Returns: stability_score (higher = more consistent = better)
    """
    edge_matrices = []
    
    # Quick mode: fewer epochs
    if quick:
        params = params.copy()
        params['dsm_epochs'] = min(params.get('dsm_epochs', 100), 30)
    
    for b in range(n_bootstraps):
        # Bootstrap sample of worms (with replacement)
        indices = np.random.choice(len(X_list), size=len(X_list), replace=True)
        X_boot = [X_list[i] for i in indices]
        
        try:
            # Train model
            estimator = SBTGStructuredVolatilityEstimator(**params)
            result = estimator.fit(X_boot)
            
            # Binary edge matrix
            edges = (result.sign_adj != 0).astype(float)
            edge_matrices.append(edges)
        except Exception as e:
            print(f"  Bootstrap {b} failed: {e}")
            continue
    
    if len(edge_matrices) < 2:
        return 0.0
    
    # Compute pairwise Jaccard agreement
    jaccard_scores = []
    for i in range(len(edge_matrices)):
        for j in range(i+1, len(edge_matrices)):
            intersection = (edge_matrices[i] * edge_matrices[j]).sum()
            union = ((edge_matrices[i] + edge_matrices[j]) > 0).sum()
            if union > 0:
                jaccard_scores.append(intersection / union)
    
    return np.mean(jaccard_scores) if jaccard_scores else 0.0


def shuffle_temporal(X_list: List[np.ndarray], seed: int = None) -> List[np.ndarray]:
    """
    Shuffle time indices independently for each neuron.
    
    This preserves marginal distributions but breaks all cross-neuron dependencies.
    Under this null, p(x) = prod_i p(x_i), so the precision matrix should be diagonal.
    """
    if seed is not None:
        np.random.seed(seed)
    
    X_shuffled = []
    for X in X_list:
        # X shape: (T, N)
        X_new = np.zeros_like(X)
        for i in range(X.shape[1]):  # For each neuron
            perm = np.random.permutation(X.shape[0])
            X_new[:, i] = X[perm, i]
        X_shuffled.append(X_new)
    
    return X_shuffled


def compute_null_contrast(
    X_list: List[np.ndarray],
    params: Dict,
    n_nulls: int = 5,
    quick: bool = False
) -> Dict[str, float]:
    """
    Measure edge strength relative to a shuffled-data null.
    
    Theoretical basis: Under the null (shuffled data), there are no cross-neuron
    dependencies, so any edges learned are spurious. Real edges should be 
    stronger than null edges.
    
    Returns:
        Dict with:
        - 'null_contrast': mean z-score of real edges vs null distribution
        - 'null_significant_fraction': fraction of edges with z > 2
        - 'null_mean_diff': mean(real_edges) - mean(null_edges)
    """
    # Quick mode: fewer epochs
    if quick:
        params = params.copy()
        params['dsm_epochs'] = min(params.get('dsm_epochs', 100), 30)
    
    # 1. Train on real data
    try:
        estimator_real = SBTGStructuredVolatilityEstimator(**params)
        result_real = estimator_real.fit(X_list)
        edges_real = np.abs(result_real.sign_adj)  # Use absolute edge weights
    except Exception as e:
        print(f"  Real model failed: {e}")
        return {'null_contrast': 0.0, 'null_significant_fraction': 0.0, 'null_mean_diff': 0.0}
    
    # 2. Train on shuffled null data
    null_edges_list = []
    for n in range(n_nulls):
        try:
            X_null = shuffle_temporal(X_list, seed=42 + n)
            estimator_null = SBTGStructuredVolatilityEstimator(**params)
            result_null = estimator_null.fit(X_null)
            null_edges_list.append(np.abs(result_null.sign_adj))
        except Exception as e:
            print(f"  Null {n} failed: {e}")
            continue
    
    if len(null_edges_list) == 0:
        return {'null_contrast': 0.0, 'null_significant_fraction': 0.0, 'null_mean_diff': 0.0}
    
    # 3. Compute statistics
    null_edges_stack = np.stack(null_edges_list, axis=0)
    null_mean = np.mean(null_edges_stack, axis=0)
    null_std = np.std(null_edges_stack, axis=0)
    
    # For z-score calculation, use a minimum std to avoid division issues
    # Use the median non-zero std as a floor
    non_zero_std = null_std[null_std > 0]
    if len(non_zero_std) > 0:
        std_floor = np.median(non_zero_std) * 0.1
    else:
        std_floor = 1e-4
    null_std_safe = np.maximum(null_std, std_floor)
    
    # Z-score: how many std above null mean?
    z_scores = (edges_real - null_mean) / null_std_safe
    
    # Mask diagonal (self-connections not meaningful)
    mask = ~np.eye(z_scores.shape[0], dtype=bool)
    z_off_diag = z_scores[mask]
    real_off_diag = edges_real[mask]
    null_mean_off_diag = null_mean[mask]
    
    # Metrics - use clipped z-scores to avoid extreme values
    z_clipped = np.clip(z_off_diag, -10, 10)  # Cap at reasonable range
    positive_z = z_clipped[z_clipped > 0]
    null_contrast = float(np.mean(positive_z)) if len(positive_z) > 0 else 0.0
    if np.isnan(null_contrast):
        null_contrast = 0.0
    
    null_significant_fraction = float(np.mean(z_off_diag > 2))  # Fraction with z > 2
    null_mean_diff = float(np.mean(real_off_diag) - np.mean(null_mean_off_diag))
    
    return {
        'null_contrast': null_contrast,
        'null_significant_fraction': null_significant_fraction,
        'null_mean_diff': null_mean_diff
    }


def compute_cross_worm_score(
    X_list: List[np.ndarray],
    params: Dict,
    n_folds: int = 5,
    quick: bool = False
) -> float:
    """
    Leave-one-worm-out cross-validation.
    
    Returns: mean prediction error on held-out worms (lower = better)
    """
    if quick:
        params = params.copy()
        params['dsm_epochs'] = min(params.get('dsm_epochs', 100), 30)
    
    prediction_errors = []
    n_folds = min(n_folds, len(X_list))
    
    for fold in range(n_folds):
        # Split: train on all except one worm
        X_train = [X_list[i] for i in range(len(X_list)) if i != fold]
        X_test = [X_list[fold]]
        
        try:
            # Train model
            estimator = SBTGStructuredVolatilityEstimator(**params)
            estimator.fit(X_train)
            
            # Evaluate DSM loss on held-out worm
            test_loss = compute_dsm_val_loss(estimator, X_test)
            if not np.isinf(test_loss):
                prediction_errors.append(test_loss)
        except Exception as e:
            print(f"  Fold {fold} failed: {e}")
            continue
    
    return np.mean(prediction_errors) if prediction_errors else float('inf')


# =============================================================================
# BIOLOGICAL EVALUATION
# =============================================================================

def align_and_evaluate(
    pred_adj: np.ndarray,
    pred_neurons: List[str],
    true_adj: np.ndarray,
    true_neurons: List[str],
    is_qvalue: bool = False
) -> Dict[str, float]:
    """
    Align neurons between prediction and ground truth, compute evaluation metrics.
    
    Args:
        pred_adj: Predicted adjacency matrix (model output)
        pred_neurons: Neuron names for prediction
        true_adj: Ground truth adjacency (or q-values)
        true_neurons: Neuron names for ground truth
        is_qvalue: If True, treat true_adj as q-values (lower = stronger connection)
    
    Returns:
        Dict with auroc, auprc, spearman, n_neurons
    """
    # Normalize neuron names
    pred_norm = [normalize_neuron_name(n) for n in pred_neurons]
    true_norm = [normalize_neuron_name(n) for n in true_neurons]
    
    # Find common neurons
    common = sorted(set(pred_norm) & set(true_norm))
    
    if len(common) < 10:
        return {'auroc': np.nan, 'auprc': np.nan, 'spearman': np.nan, 'n_neurons': len(common)}
    
    # Build index maps
    pred_idx = {n: i for i, n in enumerate(pred_norm)}
    true_idx = {n: i for i, n in enumerate(true_norm)}
    
    # Extract aligned submatrices
    pred_aligned = np.zeros((len(common), len(common)))
    true_aligned = np.zeros((len(common), len(common)))
    
    for i, ni in enumerate(common):
        for j, nj in enumerate(common):
            if i != j:  # Skip diagonal
                pred_aligned[i, j] = pred_adj[pred_idx[ni], pred_idx[nj]]
                true_aligned[i, j] = true_adj[true_idx[ni], true_idx[nj]]
    
    # Flatten (excluding diagonal)
    mask = ~np.eye(len(common), dtype=bool)
    pred_flat = np.abs(pred_aligned[mask])  # Use absolute values for edge strength
    true_flat = true_aligned[mask]
    
    # For q-values, convert to binary labels (significant = q < 0.05)
    if is_qvalue:
        true_binary = (true_flat < 0.05).astype(int)
        # Negate so that lower q-values (more significant) map to higher scores
        true_continuous = -true_flat
    else:
        # For structural connectome, nonzero = edge
        true_binary = (true_flat != 0).astype(int)
        true_continuous = true_flat
    
    results = {'n_neurons': len(common)}
    
    # AUROC
    try:
        if len(np.unique(true_binary)) > 1:
            results['auroc'] = roc_auc_score(true_binary, pred_flat)
        else:
            results['auroc'] = np.nan
    except Exception:
        results['auroc'] = np.nan
    
    # AUPRC
    try:
        if len(np.unique(true_binary)) > 1:
            results['auprc'] = average_precision_score(true_binary, pred_flat)
        else:
            results['auprc'] = np.nan
    except Exception:
        results['auprc'] = np.nan
    
    # Spearman correlation
    try:
        r, _ = spearmanr(pred_flat, true_continuous)
        results['spearman'] = r if not np.isnan(r) else 0.0
    except Exception:
        results['spearman'] = np.nan
    
    return results


# =============================================================================
# MAIN TRIAL EXECUTION
# =============================================================================

def run_trial(
    trial_idx: int,
    params: Dict,
    X_list: List[np.ndarray],
    model_neurons: List[str],
    struct_adj: np.ndarray,
    struct_neurons: List[str],
    leifer_q: np.ndarray,
    leifer_neurons: List[str],
    n_bootstraps: int = 5,
    n_nulls: int = 5,
    quick: bool = False,
    objectives: List[str] = None
) -> Dict:
    """
    Run a single HP configuration and compute ALL metrics.
    
    Args:
        objectives: List of objectives to compute. Options:
                   'stability', 'null_contrast', 'combined', 'dsm_val_loss'
                   If None, computes stability and null_contrast.
    """
    if objectives is None:
        objectives = ['stability', 'null_contrast']
    
    results = {'trial': trial_idx}
    
    # Log HP values
    for key in ['dsm_lr', 'dsm_epochs', 'dsm_noise_std', 'dsm_hidden_dim',
                'structured_hidden_dim', 'structured_l1_lambda', 'fdr_alpha', 'model_type']:
        results[key] = params.get(key)
    
    # Quick mode: fewer epochs for speed
    train_params = params.copy()
    if quick:
        train_params['dsm_epochs'] = min(train_params.get('dsm_epochs', 100), 30)
    
    # Initialize with defaults
    results['edge_stability'] = 0.0
    results['null_contrast'] = 0.0
    results['null_significant_fraction'] = 0.0
    results['combined_objective'] = 0.0
    results['dsm_val_loss'] = np.nan
    
    # 1. Edge Stability (bootstrap)
    if 'stability' in objectives or 'combined' in objectives:
        try:
            stability = compute_edge_stability(
                X_list, train_params, n_bootstraps=n_bootstraps, quick=quick
            )
            results['edge_stability'] = stability if stability is not None else 0.0
        except Exception as e:
            print(f"  Trial {trial_idx} stability failed: {e}")
    
    # 2. Null Contrast (shuffled data comparison)
    if 'null_contrast' in objectives or 'combined' in objectives:
        try:
            null_results = compute_null_contrast(
                X_list, train_params, n_nulls=n_nulls, quick=quick
            )
            results['null_contrast'] = null_results['null_contrast']
            results['null_significant_fraction'] = null_results['null_significant_fraction']
            results['null_mean_diff'] = null_results['null_mean_diff']
        except Exception as e:
            print(f"  Trial {trial_idx} null contrast failed: {e}")
    
    # 3. Combined objective (stability × contrast)
    if 'combined' in objectives:
        # Product of stability and contrast - both must be good
        results['combined_objective'] = results['edge_stability'] * results['null_contrast']
    
    # 4. Final model for biological evaluation
    try:
        estimator_full = SBTGStructuredVolatilityEstimator(**train_params)
        result = estimator_full.fit(X_list)
        
        # Edge statistics
        n_edges = int((result.sign_adj != 0).sum())
        n_possible = result.sign_adj.shape[0] * (result.sign_adj.shape[0] - 1)
        results['n_edges'] = n_edges
        results['edge_density'] = n_edges / n_possible if n_possible > 0 else 0
        
        # 5. Cook evaluation
        cook_metrics = align_and_evaluate(
            -result.sign_adj,  # Negate for sign convention
            model_neurons,
            struct_adj,
            struct_neurons,
            is_qvalue=False
        )
        results['cook_auroc'] = cook_metrics['auroc']
        results['cook_auprc'] = cook_metrics['auprc']
        results['cook_spearman'] = cook_metrics['spearman']
        results['cook_n_neurons'] = cook_metrics['n_neurons']
        
        # 6. Leifer evaluation
        leifer_metrics = align_and_evaluate(
            -result.sign_adj,  # Negate for sign convention
            model_neurons,
            leifer_q,
            leifer_neurons,
            is_qvalue=True
        )
        results['leifer_auroc'] = leifer_metrics['auroc']
        results['leifer_auprc'] = leifer_metrics['auprc']
        results['leifer_spearman'] = leifer_metrics['spearman']
        results['leifer_n_neurons'] = leifer_metrics['n_neurons']
        
    except Exception as e:
        print(f"  Trial {trial_idx} evaluation failed: {e}")
        for metric in ['n_edges', 'edge_density', 
                       'cook_auroc', 'cook_auprc', 'cook_spearman', 'cook_n_neurons',
                       'leifer_auroc', 'leifer_auprc', 'leifer_spearman', 'leifer_n_neurons']:
            results[metric] = np.nan
    
    return results


# =============================================================================
# ANALYSIS & VISUALIZATION
# =============================================================================

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlations between objectives and evaluations."""
    objectives = ['edge_stability', 'null_contrast', 'combined_objective', 
                  'null_significant_fraction', 'dsm_val_loss']
    evaluations = ['cook_auroc', 'cook_auprc', 'cook_spearman',
                   'leifer_auroc', 'leifer_auprc', 'leifer_spearman']
    
    results = []
    for obj in objectives:
        if obj not in df.columns:
            continue
        row = {'objective': obj}
        for eval_metric in evaluations:
            # Filter valid values
            if obj not in df.columns or eval_metric not in df.columns:
                row[eval_metric] = np.nan
                row[f'{eval_metric}_p'] = np.nan
                continue
            valid = df[[obj, eval_metric]].dropna()
            valid = valid[~np.isinf(valid[obj])]
            
            if len(valid) >= 3:
                r, p = spearmanr(valid[obj], valid[eval_metric])
                row[eval_metric] = r
                row[f'{eval_metric}_p'] = p
            else:
                row[eval_metric] = np.nan
                row[f'{eval_metric}_p'] = np.nan
        results.append(row)
    
    return pd.DataFrame(results)


def plot_objective_vs_evaluation(df: pd.DataFrame, output_dir: Path):
    """Generate scatter plots of each objective vs evaluation metrics."""
    objectives = [
        ('edge_stability', 'Edge Stability', False),     # higher = better
        ('null_contrast', 'Null Contrast', False),       # higher = better
        ('combined_objective', 'Combined (Stab × Contrast)', False),  # higher = better
        ('null_significant_fraction', 'Null Sig. Fraction', False),  # higher = better
    ]
    evaluations = [
        ('cook_auroc', 'Cook AUROC'),
        ('leifer_auroc', 'Leifer AUROC'),
    ]
    
    # Filter to objectives that exist in data
    objectives = [(n, l, b) for n, l, b in objectives if n in df.columns]
    
    n_obj = len(objectives)
    if n_obj == 0:
        print("  No objectives to plot")
        return
    
    fig, axes = plt.subplots(n_obj, len(evaluations), figsize=(12, 4 * n_obj))
    if n_obj == 1:
        axes = axes.reshape(1, -1)
    
    for i, (obj_name, obj_label, lower_better) in enumerate(objectives):
        for j, (eval_name, eval_label) in enumerate(evaluations):
            ax = axes[i, j]
            
            # Filter valid data
            valid = df[[obj_name, eval_name, 'model_type']].dropna()
            valid = valid[~np.isinf(valid[obj_name])]
            
            if len(valid) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel(obj_label)
                ax.set_ylabel(eval_label)
                continue
            
            # Color by model type
            colors = {'linear': 'blue', 'feature_bilinear': 'green', 'regime_gated': 'orange'}
            for model_type in valid['model_type'].unique():
                mask = valid['model_type'] == model_type
                ax.scatter(valid.loc[mask, obj_name], valid.loc[mask, eval_name],
                          c=colors.get(model_type, 'gray'), alpha=0.6, s=50,
                          label=model_type, edgecolors='white', linewidth=0.5)
            
            # Compute correlation
            r, p = spearmanr(valid[obj_name], valid[eval_name])
            
            # Add trend line
            z = np.polyfit(valid[obj_name], valid[eval_name], 1)
            x_line = np.linspace(valid[obj_name].min(), valid[obj_name].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5, linewidth=2)
            
            # Formatting
            ax.set_xlabel(obj_label)
            ax.set_ylabel(eval_label)
            
            # Title with correlation
            sign = "✓" if (lower_better and r < 0) or (not lower_better and r > 0) else "✗"
            ax.set_title(f'r = {r:.3f} (p={p:.3f}) {sign}')
            
            # Add reference line at AUROC = 0.5
            if 'auroc' in eval_name.lower():
                ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
            
            ax.grid(True, alpha=0.3)
            
            # Legend only on first plot
            if i == 0 and j == 0:
                ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle('HP Objective vs Biological Evaluation\n(✓ = expected direction, ✗ = wrong direction)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "objective_vs_evaluation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: objective_vs_evaluation.png")


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_dir: Path):
    """Generate heatmap of correlations between objectives and evaluations."""
    # Prepare data for heatmap
    objectives = ['edge_stability', 'null_contrast', 'combined_objective', 
                  'null_significant_fraction', 'dsm_val_loss']
    evaluations = ['cook_auroc', 'cook_auprc', 'cook_spearman',
                   'leifer_auroc', 'leifer_auprc', 'leifer_spearman']
    
    # Filter to objectives present in data
    objectives = [o for o in objectives if o in corr_df['objective'].values]
    
    if len(objectives) == 0:
        print("  No objectives for heatmap")
        return
    
    # Build matrix
    matrix = np.zeros((len(objectives), len(evaluations)))
    for i, obj in enumerate(objectives):
        row = corr_df[corr_df['objective'] == obj]
        if len(row) > 0:
            for j, eval_metric in enumerate(evaluations):
                if eval_metric in row.columns:
                    val = row[eval_metric].values[0]
                    matrix[i, j] = val if not np.isnan(val) else 0
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    # Labels
    obj_labels = {
        'edge_stability': 'Edge Stability\n(↑ = better)',
        'null_contrast': 'Null Contrast\n(↑ = better)',
        'combined_objective': 'Combined\n(↑ = better)',
        'null_significant_fraction': 'Null Sig. Frac.\n(↑ = better)',
        'dsm_val_loss': 'DSM Val Loss\n(↓ = better?)'
    }
    y_labels = [obj_labels.get(o, o) for o in objectives]
    eval_labels = ['Cook\nAUROC', 'Cook\nAUPRC', 'Cook\nSpearman',
                   'Leifer\nAUROC', 'Leifer\nAUPRC', 'Leifer\nSpearman']
    
    ax.set_xticks(range(len(evaluations)))
    ax.set_xticklabels(eval_labels, fontsize=10)
    ax.set_yticks(range(len(objectives)))
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # Add correlation values as text
    for i in range(len(objectives)):
        for j in range(len(evaluations)):
            val = matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=11)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman Correlation', fontsize=11)
    
    ax.set_title('Which HP Objective Best Predicts Biological Quality?\n(Green = positive correlation = GOOD)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: correlation_heatmap.png")


def plot_objective_distributions(df: pd.DataFrame, output_dir: Path):
    """Plot distributions of each objective colored by evaluation quality."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Use Cook AUROC as the quality metric for coloring
    df_valid = df.dropna(subset=['cook_auroc'])
    
    # 1. Edge Stability distribution
    ax = axes[0, 0]
    valid = df_valid[df_valid['edge_stability'] > 0] if 'edge_stability' in df_valid.columns else pd.DataFrame()
    if len(valid) > 0:
        scatter = ax.scatter(valid['edge_stability'], valid['cook_auroc'], 
                            c=valid['cook_auroc'], cmap='viridis', s=60, alpha=0.7)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('Edge Stability (Jaccard)')
        ax.set_ylabel('Cook AUROC')
        ax.set_title('Edge Stability vs Quality')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Cook AUROC')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Edge Stability vs Quality')
    
    # 2. Null Contrast distribution
    ax = axes[0, 1]
    valid = df_valid[df_valid['null_contrast'] > 0] if 'null_contrast' in df_valid.columns else pd.DataFrame()
    if len(valid) > 0:
        scatter = ax.scatter(valid['null_contrast'], valid['cook_auroc'],
                            c=valid['cook_auroc'], cmap='viridis', s=60, alpha=0.7)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('Null Contrast (z-score)')
        ax.set_ylabel('Cook AUROC')
        ax.set_title('Null Contrast vs Quality')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Cook AUROC')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Null Contrast vs Quality')
    
    # 3. Combined objective
    ax = axes[1, 0]
    valid = df_valid[df_valid['combined_objective'] > 0] if 'combined_objective' in df_valid.columns else pd.DataFrame()
    if len(valid) > 0:
        scatter = ax.scatter(valid['combined_objective'], valid['cook_auroc'],
                            c=valid['cook_auroc'], cmap='viridis', s=60, alpha=0.7)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('Combined (Stability × Contrast)')
        ax.set_ylabel('Cook AUROC')
        ax.set_title('Combined Objective vs Quality')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Cook AUROC')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Combined Objective vs Quality')
    
    # 4. Null significant fraction
    ax = axes[1, 1]
    valid = df_valid[df_valid['null_significant_fraction'] > 0] if 'null_significant_fraction' in df_valid.columns else pd.DataFrame()
    if len(valid) > 0:
        scatter = ax.scatter(valid['null_significant_fraction'], valid['cook_auroc'],
                            c=valid['cook_auroc'], cmap='viridis', s=60, alpha=0.7)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('Null Significant Fraction (z>2)')
        ax.set_ylabel('Cook AUROC')
        ax.set_title('Null Sig. Fraction vs Quality')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Cook AUROC')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Null Sig. Fraction vs Quality')
    
    plt.suptitle('HP Objective Distributions (colored by Cook AUROC)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "objective_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: objective_distributions.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="HP Objective Validation")
    parser.add_argument('--trials', type=int, default=50, help='Number of HP configurations to test')
    parser.add_argument('--bootstraps', type=int, default=5, help='Number of bootstrap samples for stability')
    parser.add_argument('--nulls', type=int, default=5, help='Number of null samples for null contrast')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer epochs)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--workers', type=int, default=0, help='Number of CPU workers (0=auto)')
    parser.add_argument('--objectives', type=str, default='stability,null_contrast,combined',
                        help='Comma-separated list of objectives: stability,null_contrast,combined,dsm_val_loss')
    parser.add_argument('--device', type=str, default=None, 
                        choices=['cpu', 'cuda', 'mps'], help='Force specific device')
    args = parser.parse_args()
    
    # Parse objectives
    objectives = [o.strip() for o in args.objectives.split(',')]
    print(f"  Objectives: {objectives}")
    
    # Setup compute environment
    compute_info = setup_compute_environment(args.workers)
    
    # Override device if specified
    if args.device:
        compute_info['device'] = args.device
        compute_info['forced'] = True
    else:
        compute_info['forced'] = False
    
    print("=" * 70)
    print("HP OBJECTIVE VALIDATION")
    print("=" * 70)
    print(f"  Trials: {args.trials}")
    print(f"  Bootstraps: {args.bootstraps}")
    print(f"  Nulls: {args.nulls}")
    print(f"  Objectives: {objectives}")
    print(f"  Quick mode: {args.quick}")
    print("")
    print("  Compute Environment:")
    device_str = compute_info['device'].upper()
    if compute_info.get('forced'):
        device_str += " (forced)"
    print(f"    Device: {device_str}")
    if compute_info['gpu_name']:
        print(f"    GPU: {compute_info['gpu_name']} ({compute_info['gpu_memory']})")
    print(f"    CPU cores: {compute_info['cpu_count']}")
    print(f"    PyTorch threads: {compute_info['torch_threads']}")
    print("=" * 70)
    
    # Output directory
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("\n[1] Loading data...")
    X_list, model_neurons = load_imputed_data()
    print(f"  Loaded {len(X_list)} worms, {len(model_neurons)} neurons")
    print(f"  Total timepoints: {sum(x.shape[0] for x in X_list)}")
    
    struct_adj, struct_neurons = load_cook_connectome()
    print(f"  Cook connectome: {len(struct_neurons)} neurons, {(struct_adj != 0).sum()} edges")
    
    leifer_q, leifer_neurons = load_leifer_data()
    n_significant = (leifer_q < 0.05).sum()
    print(f"  Leifer atlas: {len(leifer_neurons)} neurons, {n_significant} significant pairs (q<0.05)")
    
    # 2. Sample HP configurations
    print(f"\n[2] Sampling {args.trials} HP configurations...")
    hp_configs = sample_hp_configurations(
        args.trials, 
        seed=args.seed, 
        device=compute_info['device'] if compute_info.get('forced') else None
    )
    
    # Show distribution of model types
    model_types = [c['model_type'] for c in hp_configs]
    print(f"  Model types: {dict(pd.Series(model_types).value_counts())}")
    
    # 3. Run all trials
    print(f"\n[3] Running trials...")
    all_results = []
    
    for i, params in enumerate(tqdm(hp_configs, desc="Trials")):
        result = run_trial(
            trial_idx=i,
            params=params,
            X_list=X_list,
            model_neurons=model_neurons,
            struct_adj=struct_adj,
            struct_neurons=struct_neurons,
            leifer_q=leifer_q,
            leifer_neurons=leifer_neurons,
            n_bootstraps=args.bootstraps,
            n_nulls=args.nulls,
            quick=args.quick,
            objectives=objectives
        )
        all_results.append(result)
        
        # Progress update every 10 trials
        if (i + 1) % 10 == 0:
            valid_aurocs = [r['cook_auroc'] for r in all_results if not np.isnan(r.get('cook_auroc', np.nan))]
            if valid_aurocs:
                print(f"  Trials {i+1}: mean Cook AUROC = {np.mean(valid_aurocs):.3f}")
    
    # 4. Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save raw results
    df.to_csv(output_dir / 'hp_validation_results.csv', index=False)
    print(f"\n[4] Saved results to {output_dir / 'hp_validation_results.csv'}")
    
    # 5. Compute correlations
    print("\n[5] Computing correlations...")
    corr_df = compute_correlations(df)
    corr_df.to_csv(output_dir / 'correlation_summary.csv', index=False)
    
    # Print correlation summary
    print("\n" + "=" * 70)
    print("CORRELATION SUMMARY (Spearman r)")
    print("=" * 70)
    print(f"{'Objective':<25} | {'Cook AUROC':>12} | {'Leifer AUROC':>12} | {'Status':<20}")
    print("-" * 75)
    
    for _, row in corr_df.iterrows():
        obj = row['objective']
        cook_r = row.get('cook_auroc', np.nan)
        leifer_r = row.get('leifer_auroc', np.nan)
        
        # Interpretation - for all our new objectives, higher = better
        if obj in ['edge_stability', 'null_contrast', 'combined_objective', 'null_significant_fraction']:
            good = cook_r > 0.1 if not np.isnan(cook_r) else False
        elif obj == 'dsm_val_loss':
            good = cook_r < -0.1 if not np.isnan(cook_r) else False
        else:
            good = False
        
        status = "✓ Works!" if good else "✗ Doesn't work"
        
        cook_str = f"{cook_r:>12.3f}" if not np.isnan(cook_r) else f"{'N/A':>12}"
        leifer_str = f"{leifer_r:>12.3f}" if not np.isnan(leifer_r) else f"{'N/A':>12}"
        
        print(f"{obj:<25} | {cook_str} | {leifer_str} | {status}")
    
    print("=" * 75)
    
    # 6. Generate figures
    print("\n[6] Generating figures...")
    plot_objective_vs_evaluation(df, figures_dir)
    plot_correlation_heatmap(corr_df, figures_dir)
    plot_objective_distributions(df, figures_dir)
    
    # 7. Summary statistics
    print("\n[7] Summary Statistics")
    print("-" * 50)
    
    valid_df = df.dropna(subset=['cook_auroc'])
    print(f"  Valid trials: {len(valid_df)}/{len(df)}")
    print(f"  Cook AUROC: mean={valid_df['cook_auroc'].mean():.3f}, "
          f"std={valid_df['cook_auroc'].std():.3f}, "
          f"range=[{valid_df['cook_auroc'].min():.3f}, {valid_df['cook_auroc'].max():.3f}]")
    
    if 'leifer_auroc' in valid_df.columns:
        leifer_valid = valid_df.dropna(subset=['leifer_auroc'])
        if len(leifer_valid) > 0:
            print(f"  Leifer AUROC: mean={leifer_valid['leifer_auroc'].mean():.3f}, "
                  f"std={leifer_valid['leifer_auroc'].std():.3f}")
    
    # Best trial by each objective
    print("\n  Best trials by each objective:")
    
    # Best by stability (highest)
    if 'edge_stability' in df.columns:
        valid_stab = df[df['edge_stability'].notna() & (df['edge_stability'] > 0)]
        if len(valid_stab) > 0:
            best_idx = valid_stab['edge_stability'].idxmax()
            if pd.notna(best_idx):
                best_stab = valid_stab.loc[best_idx]
                print(f"    Edge Stability (highest): trial {int(best_stab['trial'])}, "
                      f"stability={best_stab['edge_stability']:.3f}, "
                      f"Cook AUROC={best_stab['cook_auroc']:.3f}")
    
    # Best by null contrast (highest)
    if 'null_contrast' in df.columns:
        valid_nc = df[df['null_contrast'].notna() & (df['null_contrast'] > 0)]
        if len(valid_nc) > 0:
            best_idx = valid_nc['null_contrast'].idxmax()
            if pd.notna(best_idx):
                best_nc = valid_nc.loc[best_idx]
                print(f"    Null Contrast (highest): trial {int(best_nc['trial'])}, "
                      f"contrast={best_nc['null_contrast']:.3f}, "
                      f"Cook AUROC={best_nc['cook_auroc']:.3f}")
    
    # Best by combined (highest)
    if 'combined_objective' in df.columns:
        valid_comb = df[df['combined_objective'].notna() & (df['combined_objective'] > 0)]
        if len(valid_comb) > 0:
            best_idx = valid_comb['combined_objective'].idxmax()
            if pd.notna(best_idx):
                best_comb = valid_comb.loc[best_idx]
                print(f"    Combined (highest): trial {int(best_comb['trial'])}, "
                      f"combined={best_comb['combined_objective']:.3f}, "
                      f"Cook AUROC={best_comb['cook_auroc']:.3f}")
    
    # Best by actual Cook AUROC (ground truth)
    if len(valid_df) > 0:
        best_cook = valid_df.loc[valid_df['cook_auroc'].idxmax()]
        print(f"    Cook AUROC (highest, ground truth): trial {int(best_cook['trial'])}, "
              f"AUROC={best_cook['cook_auroc']:.3f}, "
              f"model={best_cook['model_type']}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
