#!/usr/bin/env python3
"""
SCRIPT 03: Train Baseline Methods
==================================

Compute baseline connectivity methods for comparison with SBTG.

Methods:
- Pearson Correlation
- Cross-Correlation (lag-1)
- Partial Correlation (via precision matrix)
- Granger Causality (optional, slow)
- Graphical LASSO

Usage:
    python pipeline/03_train_baselines.py
    python pipeline/03_train_baselines.py --skip_granger  # Faster
    python pipeline/03_train_baselines.py --stimulus pentanedione

Outputs:
    results/baselines/{method}_{stimulus}.npz
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASETS_DIR = PROJECT_ROOT / "results" / "intermediate" / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "results" / "baselines"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"

from pipeline.config import STIMULI


# =============================================================================
# DATA LOADING
# =============================================================================

def load_timeseries_per_worm(stimulus: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load timeseries data, keeping worms separate.
    
    This is the correct approach that avoids spurious edges at worm boundaries.
    """
    stimulus_dir = DATASETS_DIR / stimulus
    
    if not stimulus_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {stimulus_dir}")
    
    segments_file = stimulus_dir / "X_segments.npy"
    if segments_file.exists():
        X_segments = np.load(segments_file, allow_pickle=True)
        # Convert object array elements to proper float arrays
        X_list = [np.asarray(X_segments[i], dtype=np.float32) for i in range(X_segments.shape[0])]
    else:
        raise FileNotFoundError(f"X_segments.npy not found")
    
    # Load neuron names (try standardization.json first, then neuron_names.json)
    std_file = stimulus_dir / "standardization.json"
    names_file = stimulus_dir / "neuron_names.json"
    
    if std_file.exists():
        with open(std_file) as f:
            data = json.load(f)
            neuron_names = data.get('node_order', [])
    elif names_file.exists():
        with open(names_file) as f:
            neuron_names = json.load(f)
    else:
        # Fallback to generic names
        neuron_names = [f"N{i}" for i in range(X_list[0].shape[1])]
    
    return X_list, neuron_names


# =============================================================================
# BASELINE METHODS
# =============================================================================

def compute_pearson_correlation(X_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise Pearson correlation.
    
    Averages across worms for robustness.
    """
    n_neurons = X_list[0].shape[1]
    corr_matrices = []
    
    for X in X_list:
        if X.shape[0] < 10:
            continue
        corr = np.corrcoef(X.T)
        if not np.isnan(corr).any():
            corr_matrices.append(corr)
    
    if len(corr_matrices) == 0:
        return np.zeros((n_neurons, n_neurons))
    
    return np.mean(corr_matrices, axis=0)


def compute_crosscorr_lag1(X_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute cross-correlation at lag 1 (directed).
    
    xcorr[i,j] = correlation(x_i(t), x_j(t+1))
    
    Averages across worms, respecting boundaries.
    """
    n_neurons = X_list[0].shape[1]
    xcorr_matrices = []
    
    for X in X_list:
        if X.shape[0] < 10:
            continue
        
        n_t = X.shape[0]
        X_t = X[:-1]  # x(t)
        X_t1 = X[1:]  # x(t+1)
        
        # Standardize
        X_t = (X_t - X_t.mean(0)) / (X_t.std(0) + 1e-8)
        X_t1 = (X_t1 - X_t1.mean(0)) / (X_t1.std(0) + 1e-8)
        
        # Cross-correlation: C[i,j] = (1/T) * sum_t x_i(t) * x_j(t+1)
        xcorr = X_t.T @ X_t1 / (n_t - 1)
        xcorr_matrices.append(xcorr)
    
    if len(xcorr_matrices) == 0:
        return np.zeros((n_neurons, n_neurons))
    
    return np.mean(xcorr_matrices, axis=0)


def compute_partial_correlation(X_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute partial correlation via precision matrix.
    
    Concatenates worms (safe for this method IF standardized first).
    """
    # Standardize each worm to avoid shifting baselines
    X_std_list = []
    for X in X_list:
        if X.shape[0] < 5: continue
        X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        X_std_list.append(X_z)
        
    if not X_std_list:
        return np.zeros((X_list[0].shape[1], X_list[0].shape[1]))

    X_concat = np.vstack(X_std_list)
    
    try:
        # Regularized precision matrix
        cov = np.cov(X_concat.T)
        prec = np.linalg.inv(cov + 0.01 * np.eye(cov.shape[0]))
        
        # Convert to partial correlation
        d = np.sqrt(np.diag(prec))
        partial_corr = -prec / np.outer(d, d)
        np.fill_diagonal(partial_corr, 0)
        
        return partial_corr
    except Exception:
        return np.zeros((X_list[0].shape[1], X_list[0].shape[1]))


def compute_glasso(X_list: List[np.ndarray]) -> np.ndarray:
    """
    Graphical LASSO for sparse precision matrix estimation.
    """
    # Standardize each worm
    X_std_list = []
    for X in X_list:
        if X.shape[0] < 5: continue
        X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        X_std_list.append(X_z)
        
    if not X_std_list:
        return np.zeros((X_list[0].shape[1], X_list[0].shape[1]))

    X_concat = np.vstack(X_std_list)
    
    try:
        model = GraphicalLassoCV(cv=3, max_iter=200)
        model.fit(X_concat)
        
        prec = model.precision_
        d = np.sqrt(np.diag(prec))
        partial_corr = -prec / np.outer(d, d)
        np.fill_diagonal(partial_corr, 0)
        
        return partial_corr
    except Exception:
        return np.zeros((X_list[0].shape[1], X_list[0].shape[1]))


def compute_granger_causality(X_list: List[np.ndarray], max_lag: int = 2) -> np.ndarray:
    """
    Pairwise Granger causality F-statistics.
    
    WARNING: Slow for many neurons. Consider skipping with --skip_granger.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    n_neurons = X_list[0].shape[1]
    gc_matrix = np.zeros((n_neurons, n_neurons))
    
    # Process each pair
    print(f"    Computing Granger for {n_neurons} neurons...")
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i == j:
                continue
            
            f_stats_per_worm = []
            
            # Compute Granger F-test for EACH worm separately
            for X in X_list:
                # Need sufficient length for lags
                if X.shape[0] < (3 * max_lag + 5):
                     continue
                     
                try:
                    data = np.column_stack([X[:, j], X[:, i]])  # [effect, cause]
                    result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                    # Get F-statistic from best lag
                    f_stats = [result[lag][0]['ssr_ftest'][0] for lag in range(1, max_lag + 1)]
                    f_stats_per_worm.append(max(f_stats))
                except Exception:
                    continue
            
            if f_stats_per_worm:
                # Average F-stat across worms (Robust to boundaries)
                gc_matrix[i, j] = np.mean(f_stats_per_worm)
            else:
                gc_matrix[i, j] = 0.0
    
    return gc_matrix


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute Baseline Methods")
    parser.add_argument('--stimulus', default='nacl', help='Stimulus to process')
    parser.add_argument('--skip_granger', action='store_true', help='Skip slow Granger causality')
    parser.add_argument('--all_stimuli', action='store_true', help='Process all stimuli')
    args = parser.parse_args()
    
    print("="*60)
    print("Baseline Methods Computation")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    stimuli = STIMULI if args.all_stimuli else [args.stimulus]
    
    for stimulus in stimuli:
        print(f"\n[Processing: {stimulus}]")
        
        # Load data
        try:
            X_list, neuron_names = load_timeseries_per_worm(stimulus)
            print(f"  Loaded: {len(X_list)} worms, {len(neuron_names)} neurons")
        except FileNotFoundError as e:
            print(f"  Skipping - {e}")
            continue
        
        results = {}
        
        # Pearson
        print("  Computing Pearson correlation...")
        results['pearson'] = compute_pearson_correlation(X_list)
        
        # Cross-correlation
        print("  Computing cross-correlation (lag-1)...")
        results['crosscorr'] = compute_crosscorr_lag1(X_list)
        
        # Partial correlation
        print("  Computing partial correlation...")
        results['partial_corr'] = compute_partial_correlation(X_list)
        
        # GLASSO
        print("  Computing Graphical LASSO...")
        results['glasso'] = compute_glasso(X_list)
        
        # Granger (optional)
        if not args.skip_granger:
            print("  Computing Granger causality (slow)...")
            results['granger'] = compute_granger_causality(X_list)
        
        # Save
        for method, matrix in results.items():
            outfile = OUTPUT_DIR / f"{method}_{stimulus}.npz"
            np.savez(outfile, matrix=matrix, neurons=neuron_names)
            print(f"  Saved: {outfile.name}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
