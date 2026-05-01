#!/usr/bin/env python3
"""
Script 19: State-Dependent Connectivity Analysis

Analyzes how functional connectivity depends on locomotor state (forward vs backward)
based on AVA and AVB command interneuron activity.

Theory:
    AVA → Reverse locomotion (high AVA = backward)
    AVB → Forward locomotion (high AVB = forward)
    
    We segment time series into states and compare SBTG coupling strengths
    across states to identify state-dependent vs state-invariant connections.

Usage:
    python pipeline/19_state_dependent_analysis.py --result-dir results/multilag_separation/TIMESTAMP/

"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.neuron_types import get_neuron_type

# Constants
SAMPLING_RATE = 4  # Hz


# =============================================================================
# STATE SEGMENTATION
# =============================================================================

def identify_state_neurons(neuron_names: List[str]) -> Dict[str, Optional[int]]:
    """
    Identify AVA and AVB indices in neuron list.
    
    Returns:
        Dict with 'AVA' and 'AVB' indices (or None if not found)
    """
    indices = {'AVA': None, 'AVB': None}
    
    for i, name in enumerate(neuron_names):
        name_upper = str(name).upper().strip()
        if name_upper == 'AVA':
            indices['AVA'] = i
            print(f"  Found AVA at index {i}: {name}")
        elif name_upper == 'AVB':
            indices['AVB'] = i
            print(f"  Found AVB at index {i}: {name}")
    
    return indices


def segment_by_state(
    X: np.ndarray,
    ava_idx: int,
    avb_idx: int,
    forward_threshold: float = 0.5,
    backward_threshold: float = 0.5,
    smoothing_sigma: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Segment time series into forward/backward/transition states.
    
    Args:
        X: (T, n) calcium traces (z-scored)
        ava_idx: Index of AVA neuron
        avb_idx: Index of AVB neuron
        forward_threshold: Threshold for AVB to be considered "high"
        backward_threshold: Threshold for AVA to be considered "high"
        smoothing_sigma: Gaussian smoothing for state detection
        
    Returns:
        state_mask: (T,) array with 0=forward, 1=backward, 2=transition
        state_counts: Dict with counts for each state
    """
    T = X.shape[0]
    
    # Extract and smooth AVA/AVB traces
    ava_trace = gaussian_filter1d(X[:, ava_idx], sigma=smoothing_sigma)
    avb_trace = gaussian_filter1d(X[:, avb_idx], sigma=smoothing_sigma)
    
    # Initialize state mask
    state_mask = np.full(T, 2, dtype=int)  # Default: transition
    
    # Forward state: AVB high, AVA low
    forward_mask = (avb_trace > forward_threshold) & (ava_trace < -forward_threshold/2)
    state_mask[forward_mask] = 0
    
    # Backward state: AVA high, AVB low
    backward_mask = (ava_trace > backward_threshold) & (avb_trace < -backward_threshold/2)
    state_mask[backward_mask] = 1
    
    # Count states
    state_counts = {
        'forward': int(np.sum(state_mask == 0)),
        'backward': int(np.sum(state_mask == 1)),
        'transition': int(np.sum(state_mask == 2)),
    }
    
    return state_mask, state_counts


def load_traces_from_prepared_data(dataset: str = "full_traces_imputed") -> Tuple[List[np.ndarray], List[str]]:
    """
    Load prepared calcium traces and neuron names.
    
    Returns:
        X_list: List of (T, n) traces per worm
        neuron_names: List of neuron names
    """
    data_dir = PROJECT_ROOT / "results" / "intermediate" / "datasets" / dataset
    
    # Load standardization info
    std_file = data_dir / "standardization.json"
    with open(std_file) as f:
        std_data = json.load(f)
        neuron_names = std_data['node_order']
    
    # Load traces - try multiple file formats
    X_list = []
    
    # Try X_segments.npy (segmented by worm)
    segments_file = data_dir / "X_segments.npy"
    if segments_file.exists():
        X_segments = np.load(segments_file, allow_pickle=True)
        # X_segments is a list or array of traces
        if isinstance(X_segments, np.ndarray):
            if X_segments.dtype == object:
                # List of arrays stored as object array
                X_list = [x for x in X_segments if x is not None and len(x) > 0]
            elif X_segments.ndim == 3:
                # (n_worms, T, n_neurons)
                for i in range(X_segments.shape[0]):
                    X_list.append(X_segments[i])
        else:
            X_list = list(X_segments)
        
        print(f"  Loaded from X_segments.npy: {len(X_list)} traces")
        return X_list, neuron_names
    
    # Try Z_std.npy (standardized full dataset)
    z_std_file = data_dir / "Z_std.npy"
    if z_std_file.exists():
        Z_std = np.load(z_std_file)  # Should be (n_worms, T, n_neurons) or (T, n_neurons)
        
        if Z_std.ndim == 3:
            for i in range(Z_std.shape[0]):
                X_list.append(Z_std[i])
        else:
            X_list.append(Z_std)
        
        print(f"  Loaded from Z_std.npy: {len(X_list)} traces, shape: {Z_std.shape}")
        return X_list, neuron_names
    
    # Try train.npz/test.npz (older format)
    for split in ['train', 'test']:
        split_file = data_dir / f"{split}.npz"
        if split_file.exists():
            data = np.load(split_file)
            X = data['X']  # (n_worms, T, n_neurons) or (T, n_neurons)
            
            if X.ndim == 3:
                for i in range(X.shape[0]):
                    X_list.append(X[i])
            else:
                X_list.append(X)
    
    if X_list:
        print(f"  Loaded from train/test splits: {len(X_list)} traces")
    
    return X_list, neuron_names


# =============================================================================
# STATE-CONDITIONAL COUPLING ANALYSIS
# =============================================================================

def compute_state_conditional_coupling(
    mu_hat: np.ndarray,
    X_list: List[np.ndarray],
    state_masks: List[np.ndarray],
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean coupling strength for each state.
    
    Simplified approach: Weight the SBTG coupling matrix by correlation between
    edge activity and state occupancy.
    
    Args:
        mu_hat: (n, n) coupling matrix from SBTG
        X_list: List of traces
        state_masks: List of state masks (same length as X_list)
        lag: Time lag
        
    Returns:
        mu_forward: (n, n) coupling strength weighted by forward state
        mu_backward: (n, n) coupling strength weighted by backward state
        mu_transition: (n, n) coupling strength weighted by transition state
    """
    n = mu_hat.shape[0]
    
    # Compute correlation between edge activity and state
    state_correlations = {
        'forward': np.zeros((n, n)),
        'backward': np.zeros((n, n)),
        'transition': np.zeros((n, n)),
    }
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Collect edge activity and state indicators across all worms
            all_activity = []
            all_forward = []
            all_backward = []
            all_transition = []
            
            for X, mask in zip(X_list, state_masks):
                T = len(mask)
                if T < lag + 1:
                    continue
                
                # Compute edge activity: x_j(t) * x_i(t+lag)
                for t in range(T - lag):
                    activity = X[t, j] * X[t + lag, i]  # Can be positive or negative
                    state = mask[t]
                    
                    all_activity.append(activity)
                    all_forward.append(1.0 if state == 0 else 0.0)
                    all_backward.append(1.0 if state == 1 else 0.0)
                    all_transition.append(1.0 if state == 2 else 0.0)
            
            if len(all_activity) == 0:
                continue
            
            all_activity = np.array(all_activity)
            
            # Compute mean activity in each state
            for state_name, state_indicator in [
                ('forward', np.array(all_forward)),
                ('backward', np.array(all_backward)),
                ('transition', np.array(all_transition)),
            ]:
                if state_indicator.sum() > 0:
                    # Mean activity during this state
                    mean_activity = np.mean(all_activity[state_indicator > 0])
                    state_correlations[state_name][i, j] = mean_activity
    
    # Weight the SBTG coupling matrix by state-specific activity
    # Normalize so that forward + backward + transition ≈ original mu_hat
    total_activity = (np.abs(state_correlations['forward']) + 
                     np.abs(state_correlations['backward']) + 
                     np.abs(state_correlations['transition']))
    
    # Avoid division by zero
    total_activity = np.where(total_activity > 0, total_activity, 1.0)
    
    mu_forward = np.abs(mu_hat) * np.abs(state_correlations['forward']) / total_activity
    mu_backward = np.abs(mu_hat) * np.abs(state_correlations['backward']) / total_activity
    mu_transition = np.abs(mu_hat) * np.abs(state_correlations['transition']) / total_activity
    
    return mu_forward, mu_backward, mu_transition


def compare_state_coupling(
    mu_forward: np.ndarray,
    mu_backward: np.ndarray,
    sig_mask: np.ndarray,
    neuron_names: List[str],
) -> pd.DataFrame:
    """
    Compare coupling strengths between states for significant edges.
    
    Returns:
        DataFrame with edge-level comparisons
    """
    rows = []
    
    n = mu_forward.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j or not sig_mask[i, j]:
                continue
            
            mu_f = mu_forward[i, j]
            mu_b = mu_backward[i, j]
            
            # State dependence metrics
            abs_diff = abs(mu_f - mu_b)
            rel_diff = abs_diff / (abs(mu_f) + abs(mu_b) + 1e-10)
            
            # Directionality
            forward_dominant = mu_f > mu_b
            
            rows.append({
                'source': neuron_names[j],
                'target': neuron_names[i],
                'source_type': get_neuron_type(neuron_names[j]),
                'target_type': get_neuron_type(neuron_names[i]),
                'mu_forward': mu_f,
                'mu_backward': mu_b,
                'abs_diff': abs_diff,
                'rel_diff': rel_diff,
                'forward_dominant': forward_dominant,
            })
    
    return pd.DataFrame(rows)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_state_traces(
    X: np.ndarray,
    state_mask: np.ndarray,
    ava_idx: int,
    avb_idx: int,
    output_path: Path,
):
    """Plot example traces with state annotations."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    T = X.shape[0]
    time_sec = np.arange(T) / SAMPLING_RATE
    
    # Plot AVA
    ax = axes[0]
    ax.plot(time_sec, X[:, ava_idx], 'k-', lw=1, label='AVA (backward)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('AVA Activity (z-scored)', fontsize=12)
    ax.set_title('Command Interneuron Activity and Locomotor State', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot AVB
    ax = axes[1]
    ax.plot(time_sec, X[:, avb_idx], 'k-', lw=1, label='AVB (forward)')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('AVB Activity (z-scored)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot state
    ax = axes[2]
    state_colors = {0: 'blue', 1: 'red', 2: 'gray'}
    state_labels = {0: 'Forward', 1: 'Backward', 2: 'Transition'}
    
    for state_id in [0, 1, 2]:
        mask = state_mask == state_id
        ax.fill_between(time_sec, 0, 1, where=mask, 
                        color=state_colors[state_id], alpha=0.5, 
                        label=state_labels[state_id])
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    ax.legend(loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_state_comparison_scatter(
    mu_forward: np.ndarray,
    mu_backward: np.ndarray,
    sig_mask: np.ndarray,
    output_path: Path,
):
    """Scatter plot: forward vs backward coupling."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract significant edges
    forward_vals = mu_forward[sig_mask]
    backward_vals = mu_backward[sig_mask]
    
    # Scatter
    ax.scatter(forward_vals, backward_vals, alpha=0.6, s=50, 
              c='steelblue', edgecolors='k', linewidths=0.5)
    
    # Diagonal (state-invariant)
    max_val = max(forward_vals.max(), backward_vals.max())
    ax.plot([0, max_val], [0, max_val], 'k--', lw=2, label='State-invariant', alpha=0.5)
    
    # Axes
    ax.set_xlabel('Coupling Strength (Forward State)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coupling Strength (Backward State)', fontsize=14, fontweight='bold')
    ax.set_title('State-Dependent Connectivity\n(On diagonal = invariant, Off diagonal = state-specific)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add quadrant labels
    mid_x = max_val / 2
    mid_y = max_val / 2
    ax.text(mid_x * 1.5, mid_y * 0.5, 'Forward\nDominant', 
           ha='center', va='center', fontsize=12, alpha=0.5, style='italic')
    ax.text(mid_x * 0.5, mid_y * 1.5, 'Backward\nDominant', 
           ha='center', va='center', fontsize=12, alpha=0.5, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_state_dependence_heatmap(
    df_comparison: pd.DataFrame,
    output_path: Path,
    top_n: int = 30,
):
    """Heatmap of top state-dependent edges."""
    # Sort by state dependence
    df_sorted = df_comparison.sort_values('abs_diff', ascending=False).head(top_n)
    
    # Create matrix for heatmap
    data = np.column_stack([df_sorted['mu_forward'].values, df_sorted['mu_backward'].values])
    labels = [f"{row['source']}→{row['target']}" for _, row in df_sorted.iterrows()]
    
    fig, ax = plt.subplots(figsize=(8, max(10, top_n * 0.3)))
    
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', 
                   vmin=-data.max(), vmax=data.max())
    
    # Axes
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Forward', 'Backward'], fontsize=12)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(f'Top {top_n} State-Dependent Edges\n(Ranked by |Forward - Backward|)', 
                fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coupling Strength', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_celltype_by_state(
    df_comparison: pd.DataFrame,
    output_path: Path,
):
    """Bar plot of cell-type interactions by state."""
    # Define cell type pairs of interest
    pairs = ['S→I', 'S→M', 'I→M', 'I→I', 'M→M']
    
    # Map neuron types to abbreviations
    type_map = {'sensory': 'S', 'interneuron': 'I', 'motor': 'M'}
    df_comparison['pair'] = df_comparison.apply(
        lambda row: f"{type_map.get(row['source_type'], '?')}→{type_map.get(row['target_type'], '?')}", 
        axis=1
    )
    
    # Filter to pairs of interest
    df_filtered = df_comparison[df_comparison['pair'].isin(pairs)]
    
    # Compute mean coupling by pair and state
    summary = df_filtered.groupby('pair').agg({
        'mu_forward': 'mean',
        'mu_backward': 'mean',
    }).reindex(pairs)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(pairs))
    width = 0.35
    
    ax.bar(x - width/2, summary['mu_forward'], width, label='Forward', color='blue', alpha=0.7)
    ax.bar(x + width/2, summary['mu_backward'], width, label='Backward', color='red', alpha=0.7)
    
    ax.set_xlabel('Cell Type Interaction', fontsize=12)
    ax.set_ylabel('Mean Coupling Strength', fontsize=12)
    ax.set_title('Cell-Type Interactions by Locomotor State', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="State-dependent connectivity analysis")
    parser.add_argument("--result-dir", type=str, required=True,
                       help="Multi-lag result directory (e.g., results/multilag_separation/TIMESTAMP/)")
    parser.add_argument("--dataset", default="full_traces_imputed",
                       help="Dataset name for loading traces")
    parser.add_argument("--lag", type=int, default=1,
                       help="Lag to analyze (default: 1)")
    parser.add_argument("--forward-threshold", type=float, default=0.5,
                       help="Threshold for forward state detection (default: 0.5)")
    parser.add_argument("--backward-threshold", type=float, default=0.5,
                       help="Threshold for backward state detection (default: 0.5)")
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"ERROR: Result directory not found: {result_dir}")
        return
    
    print("="*60)
    print("STATE-DEPENDENT CONNECTIVITY ANALYSIS")
    print("="*60)
    print(f"Result dir: {result_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Lag: {args.lag}")
    print()
    
    # Create output directory
    output_dir = result_dir / "state_dependent_analysis"
    output_dir.mkdir(exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Load SBTG results
    print("[1/6] Loading SBTG results...")
    result_file = result_dir / "result_C.npz"
    if not result_file.exists():
        print(f"ERROR: Result file not found: {result_file}")
        return
    
    data = np.load(result_file, allow_pickle=True)
    mu_hat = data[f'mu_hat_lag{args.lag}']
    sig_mask = data[f'sig_lag{args.lag}'].astype(bool)  # Ensure boolean type
    neuron_names = list(data['neuron_names'])
    
    print(f"  Loaded lag-{args.lag} results: {mu_hat.shape} coupling matrix")
    print(f"  Significant edges: {sig_mask.sum()}")
    print()
    
    # Load traces
    print("[2/6] Loading calcium traces...")
    X_list, trace_neuron_names = load_traces_from_prepared_data(args.dataset)
    print(f"  Loaded {len(X_list)} traces")
    
    # Check neuron name alignment
    if neuron_names != trace_neuron_names:
        print("  WARNING: Neuron names may not match between SBTG results and traces")
    print()
    
    # Identify AVA and AVB
    print("[3/6] Identifying state neurons (AVA, AVB)...")
    state_indices = identify_state_neurons(neuron_names)
    
    if state_indices['AVA'] is None or state_indices['AVB'] is None:
        print("ERROR: Could not find both AVA and AVB neurons")
        return
    
    ava_idx = state_indices['AVA']
    avb_idx = state_indices['AVB']
    print()
    
    # Segment by state
    print("[4/6] Segmenting time series by locomotor state...")
    state_masks = []
    total_counts = {'forward': 0, 'backward': 0, 'transition': 0}
    
    for i, X in enumerate(X_list):
        mask, counts = segment_by_state(
            X, ava_idx, avb_idx,
            forward_threshold=args.forward_threshold,
            backward_threshold=args.backward_threshold
        )
        state_masks.append(mask)
        for key in counts:
            total_counts[key] += counts[key]
        
        if i == 0:
            # Plot first trace as example
            plot_state_traces(X, mask, ava_idx, avb_idx, 
                            fig_dir / "fig_state_segmentation_example.png")
    
    total_time = sum(total_counts.values())
    print(f"  Total time points: {total_time}")
    print(f"    Forward: {total_counts['forward']} ({100*total_counts['forward']/total_time:.1f}%)")
    print(f"    Backward: {total_counts['backward']} ({100*total_counts['backward']/total_time:.1f}%)")
    print(f"    Transition: {total_counts['transition']} ({100*total_counts['transition']/total_time:.1f}%)")
    print()
    
    # Compute state-conditional coupling
    print("[5/6] Computing state-conditional coupling...")
    mu_forward, mu_backward, mu_transition = compute_state_conditional_coupling(
        mu_hat, X_list, state_masks, args.lag
    )
    print(f"  Forward coupling range: [{mu_forward.min():.4f}, {mu_forward.max():.4f}]")
    print(f"  Backward coupling range: [{mu_backward.min():.4f}, {mu_backward.max():.4f}]")
    print()
    
    # Compare states
    print("[6/6] Comparing states and generating reports...")
    df_comparison = compare_state_coupling(mu_forward, mu_backward, sig_mask, neuron_names)
    df_comparison = df_comparison.sort_values('abs_diff', ascending=False)
    
    # Save comparison
    df_comparison.to_csv(output_dir / "state_comparison.csv", index=False)
    print(f"  Saved: {output_dir / 'state_comparison.csv'}")
    
    # Summary statistics
    n_forward_dominant = (df_comparison['forward_dominant']).sum()
    n_backward_dominant = (~df_comparison['forward_dominant']).sum()
    
    print(f"\n  Edge State Preference:")
    print(f"    Forward-dominant: {n_forward_dominant} ({100*n_forward_dominant/len(df_comparison):.1f}%)")
    print(f"    Backward-dominant: {n_backward_dominant} ({100*n_backward_dominant/len(df_comparison):.1f}%)")
    
    # Top state-dependent edges
    print(f"\n  Top 10 State-Dependent Edges:")
    for i, row in df_comparison.head(10).iterrows():
        state = "FWD" if row['forward_dominant'] else "BWD"
        print(f"    {row['source']}→{row['target']}: Δ={row['abs_diff']:.4f} ({state} dominant)")
    
    # Generate figures
    print("\n[Figures] Generating visualizations...")
    plot_state_comparison_scatter(mu_forward, mu_backward, sig_mask, 
                                  fig_dir / "fig_state_scatter.png")
    plot_state_dependence_heatmap(df_comparison, fig_dir / "fig_state_heatmap.png", top_n=30)
    plot_celltype_by_state(df_comparison, fig_dir / "fig_state_celltype.png")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Figures saved to: {fig_dir}")
    print()


if __name__ == "__main__":
    main()
