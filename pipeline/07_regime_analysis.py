#!/usr/bin/env python3
"""
SCRIPT 07: Regime-Gated Analysis (Latent State Interpretability)
================================================================

This script trains a Regime-Gated SBTG model to discover latent functional states
("regimes") in the C. elegans neural data.

Goals:
1. Train a 3-regime model (e.g., aiming for Baseline, Stimulus, Recovery).
2. Visualize Gating Variables alpha(t): When is each regime active?
3. Visualize Regime Graphs W_k: How does connectivity change between regimes?

Usage:
    python pipeline/07_regime_analysis.py --num_regimes 3 --epochs 200
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.io import loadmat

# Setup Paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.models.sbtg import SBTGStructuredVolatilityEstimator

# =============================================================================
# DATA LOADING (Robust Implementation, reused from Script 16)
# =============================================================================

NEUROPAL_HEAD_FILE = "Head_Activity_OH16230.mat"
NEUROPAL_TAIL_FILE = "Tail_Activity_OH16230.mat"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "regime_gated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# D/V COLLAPSE PATTERNS
from pipeline.utils.align import DV_COLLAPSE_PATTERNS


def collapse_dv_subtype(name: str) -> str:
    name = name.strip().upper()
    return DV_COLLAPSE_PATTERNS.get(name, name)

def collect_worm_trace(neuron_traces, worm_idx, num_worms):
    segments = []
    for offset in [worm_idx, worm_idx + num_worms]:
        if offset >= len(neuron_traces): continue
        arr = np.asarray(neuron_traces[offset], dtype=float)
        if arr.size == 0 or arr.ndim == 0: continue
        segments.append(arr)
    if not segments: return None
    min_len = min(seg.shape[-1] for seg in segments)
    cleaned = [np.squeeze(s)[-min_len:] for s in segments if np.squeeze(s).ndim > 0]
    if not cleaned: return None
    return np.mean(np.stack(cleaned, axis=0), axis=0)

def load_data(min_worms=15):
    print(f"Loading data from {DATA_DIR}...")
    
    # Load Head
    head_path = DATA_DIR / NEUROPAL_HEAD_FILE
    mat = loadmat(str(head_path), simplify_cells=True)
    raw_neurons = [collapse_dv_subtype(str(n)) for n in mat['neurons']]
    raw_traces = mat['norm_traces']
    num_worms = len(mat['files'])
    
    X_raw = [{} for _ in range(num_worms)]
    
    for w in range(num_worms):
        for n_idx, n_name in enumerate(raw_neurons):
            trace = collect_worm_trace(raw_traces[n_idx], w, num_worms)
            if trace is not None and len(trace) > 10:
                if n_name not in X_raw[w]: X_raw[w][n_name] = trace
    
    # Load Tail
    tail_path = DATA_DIR / NEUROPAL_TAIL_FILE
    if tail_path.exists():
        tail_mat = loadmat(str(tail_path), simplify_cells=True)
        tail_neurons = [collapse_dv_subtype(str(n)) for n in tail_mat['neurons']]
        tail_traces = tail_mat['norm_traces']
        tail_num = len(tail_mat['files'])
        for w in range(min(num_worms, tail_num)):
            for n_idx, n_name in enumerate(tail_neurons):
                trace = collect_worm_trace(tail_traces[n_idx], w, tail_num)
                if trace is not None and len(trace) > 10:
                    if n_name not in X_raw[w]: X_raw[w][n_name] = trace

    # Filter
    all_names = set()
    for w in X_raw: all_names.update(w.keys())
    counts = {n: sum(1 for w in X_raw if n in w) for n in all_names}
    common_neurons = sorted([n for n in all_names if counts[n] >= min_worms])
    
    print(f"  Neurons: {len(common_neurons)} (appearing in >= {min_worms} worms)")
    
    X_list = []
    for w_data in X_raw:
        lengths = [len(w_data[n]) for n in common_neurons if n in w_data]
        if not lengths: continue
        min_len = min(lengths)
        if min_len < 100: continue
        
        mat_data = np.zeros((min_len, len(common_neurons)))
        for i, n in enumerate(common_neurons):
            if n in w_data: mat_data[:, i] = w_data[n][-min_len:]
        
        # Z-score
        means = np.nanmean(mat_data, axis=0)
        stds = np.nanstd(mat_data, axis=0)
        stds[stds==0] = 1.0
        mat_data = (mat_data - means) / stds
        X_list.append(np.nan_to_num(mat_data))
        
    print(f"  Worms: {len(X_list)}")
    return X_list, common_neurons

# =============================================================================
# ANALYSIS & VISUALIZATION
# =============================================================================

def plot_gating_variables(alpha_eval, num_worms, num_regimes):
    """
    Plot average gating variable alpha(t) across all worms to see temporal structure.
    alpha_eval: (N_total_windows, K)
    """
    print("Plotting Gating Variables...")
    
    # Plot gating variables for the first 3 worms individually
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Rough estimate of frames per worm
    frames_per_worm = alpha_eval.shape[0] // num_worms
    time_axis = np.arange(frames_per_worm) / 4.0  # seconds
    
    for w in range(min(3, num_worms)):
        start = w * frames_per_worm
        end = start + frames_per_worm
        if end > alpha_eval.shape[0]: break
        
        alpha_w = alpha_eval[start:end]
        ax = axes[w]
        
        # Plot stacked area or lines
        ax.plot(time_axis[:len(alpha_w)], alpha_w, label=[f"Regime {k}" for k in range(num_regimes)])
        ax.set_title(f"Worm {w} Regime Probabilities")
        ax.set_ylabel("Probability")
        if w == 0: ax.legend(loc="upper right")
        
        # Add stimulus shading (approximate)
        # Based on 240s protocol: 60-70s, 120-130s, 180-190s
        for (start_s, end_s, color) in [(60, 70, 'red'), (120, 130, 'green'), (180, 190, 'blue')]:
            ax.axvspan(start_s, end_s, color=color, alpha=0.1)
            
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "alpha_timecourse_samples.png")
    plt.close()
    
    # Global Average
    # Truncate to min length to average
    min_len = frames_per_worm # approx
    
    alphas_stacked = []
    for w in range(num_worms):
        start = w * frames_per_worm
        end = start + frames_per_worm
        if end <= alpha_eval.shape[0]:
            alphas_stacked.append(alpha_eval[start:end])
            
    if alphas_stacked:
        alpha_avg = np.mean(np.stack(alphas_stacked, axis=0), axis=0)
        
        plt.figure(figsize=(15, 6))
        for k in range(num_regimes):
            plt.plot(time_axis[:len(alpha_avg)], alpha_avg[:, k], label=f"Regime {k}", linewidth=2)
            
        # Stimuli
        for (start_s, end_s, label) in [(60, 70, 'Butanone'), (120, 130, 'Pentanedione'), (180, 190, 'NaCl')]:
            plt.axvspan(start_s, end_s, color='gray', alpha=0.2, label=label if k==0 else None)
            plt.text((start_s+end_s)/2, 1.02, label, ha='center', va='bottom', transform=plt.gca().get_xaxis_transform())

        plt.title(f"Average Regime Probabilities (N={len(alphas_stacked)} worms)")
        plt.xlabel("Time (s)")
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "alpha_timecourse_average.png")
        plt.close()

def plot_regime_graphs(W_param, neurons):
    """
    Visualize the learned connectivity matrices W_k for each regime.
    W_param: (K, n, n)
    """
    print("Plotting Regime Graphs...")
    num_regimes = W_param.shape[0]
    
    fig, axes = plt.subplots(1, num_regimes, figsize=(6*num_regimes, 6))
    if num_regimes == 1: axes = [axes]
    
    # Determine common scale
    vmax = np.max(np.abs(W_param))
    
    for k in range(num_regimes):
        ax = axes[k]
        sns.heatmap(W_param[k], ax=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax, square=True,
                   cbar_kws={"label": "Interaction Strength"})
        ax.set_title(f"Regime {k} Connectivity")
        ax.set_xlabel("Source Neuron")
        ax.set_ylabel("Target Neuron")
        
        # Calculate stats
        density = np.mean(np.abs(W_param[k]) > 0.1) # Arbitrary threshold for stat
        ax.text(0.5, -0.1, f"Density(|w|>0.1): {density:.3f}", transform=ax.transAxes, ha='center')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "regime_connectivity_matrices.png")
    plt.close()
    
    # Difference Maps (Regime k vs Regime 0)
    if num_regimes > 1:
        fig, axes = plt.subplots(1, num_regimes-1, figsize=(6*(num_regimes-1), 6))
        if num_regimes-1 == 1: axes = [axes]
        
        for k in range(1, num_regimes):
            diff = W_param[k] - W_param[0]
            ax = axes[k-1]
            vmax_diff = np.max(np.abs(diff))
            sns.heatmap(diff, ax=ax, cmap="PuOr", vmin=-vmax_diff, vmax=vmax_diff, square=True)
            ax.set_title(f"Regime {k} - Regime 0")
            
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "regime_differences.png")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_regimes", type=int, default=3, help="Number of latent regimes")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Run with fewer epochs")
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 10
    
    np.random.seed(args.seed)
    
    # 1. Load Data
    X_list, neurons = load_data()
    
    # 2. Train Regime-Gated Model
    print(f"\nTraining Regime-Gated SBTG (K={args.num_regimes})...")
    estimator = SBTGStructuredVolatilityEstimator(
        model_type="regime_gated",
        num_regimes=args.num_regimes,
        dsm_epochs=args.epochs,
        dsm_lr=0.001,
        structured_l1_lambda=0.001,
        gate_entropy_lambda=0.01, # Encourage distinct regimes
        verbose=True,
        random_state=args.seed
    )
    
    result = estimator.fit(X_list)
    
    # 3. Analyze Results
    if result.gate_alpha_eval is not None and result.W_param is not None:
        print("\nAnalysis extracted successfully.")
        
        # Viz 1: Temporal dynamics of regimes
        plot_gating_variables(result.gate_alpha_eval, len(X_list), args.num_regimes)
        
        # Viz 2: Connectivity patterns
        plot_regime_graphs(result.W_param, neurons)
        
        # Save Raw Data for Quantitative Analysis
        np.save(OUTPUT_DIR / "alpha_eval.npy", result.gate_alpha_eval)
        np.save(OUTPUT_DIR / "W_param.npy", result.W_param)
        
        print(f"\nResults saved to {FIGURES_DIR}")
        print(f"Raw data saved to {OUTPUT_DIR}")
        print("Done.")
    else:
        print("Error: Regime-gated details not found in result.")

if __name__ == "__main__":
    main()
