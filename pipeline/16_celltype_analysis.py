#!/usr/bin/env python3
"""
Cell-Type Statistical Analysis for Multi-Lag SBTG Results

This script performs principled statistical analysis on cell-type interactions:
1. Computes mean |μ̂| ± SEM for each cell-type pair × lag
2. Runs Mann-Whitney U tests for within-lag comparisons
3. Runs permutation tests for cross-lag comparisons
4. Generates clear figures with significance markers

Usage:
    python pipeline/16_celltype_analysis.py results/multilag_separation/20260119_134330
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.utils.neuron_types import get_neuron_type
from pipeline.utils.metrics import compute_weight_correlation
from pipeline.utils.io import load_structural_connectome as _load_structural_connectome
from pipeline.utils.display_names import FUNCTIONAL_LABEL, STRUCTURAL_LABEL
from typing import List, Tuple

# Constants
SAMPLING_RATE = 4  # Hz - matches NeuroPAL data sampling rate

# Paths for connectome
CONNECTOME_DIR = Path(__file__).parent.parent / 'results' / 'intermediate' / 'connectome'


def align_matrices_for_correlation(
    pred: np.ndarray,
    pred_neurons: List[str],
    gt: np.ndarray,
    gt_neurons: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Align predicted and ground truth matrices to common neurons."""
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


def load_results(result_dir: str, approach: str = 'C'):
    """Load results from NPZ file."""
    npz_path = Path(result_dir) / f'result_{approach}.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f"Results file not found: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    return data


def load_baselines(result_dir: str):
    """Load baseline matrices from NPZ file if available."""
    npz_path = Path(result_dir) / 'baselines.npz'
    if not npz_path.exists():
        return None
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        return dict(data)
    except Exception as e:
        print(f"Warning: Failed to load baselines.npz: {e}")
        return None


def get_neuron_type_indices(neuron_names):
    """Get indices for each neuron type."""
    type_indices = {'sensory': [], 'interneuron': [], 'motor': [], 'unknown': []}
    
    for i, name in enumerate(neuron_names):
        ntype = get_neuron_type(str(name))
        type_indices[ntype].append(i)
    
    return type_indices


def compute_celltype_stats(mu_hat, sig_mask, type_indices):
    """
    Compute mean |μ̂| and SEM for each cell-type pair.
    
    Returns dict with keys like 'S→I' containing:
        - 'weights': array of significant edge weights
        - 'mean': mean absolute weight
        - 'sem': standard error of mean
        - 'n': number of significant edges
    """
    type_abbrev = {'sensory': 'S', 'interneuron': 'I', 'motor': 'M'}
    type_order = ['sensory', 'interneuron', 'motor']
    
    results = {}
    
    for src_type in type_order:
        for tgt_type in type_order:
            src_idx = type_indices[src_type]
            tgt_idx = type_indices[tgt_type]
            
            if len(src_idx) == 0 or len(tgt_idx) == 0:
                continue
            
            # Get submatrix for this cell-type pair
            # mu_hat[i, j] = coupling from j -> i (source j, target i)
            # So rows are targets, cols are sources
            sub_mu = mu_hat[np.ix_(tgt_idx, src_idx)]
            sub_sig = sig_mask[np.ix_(tgt_idx, src_idx)]
            
            # Get significant edge weights
            weights = np.abs(sub_mu[sub_sig > 0])
            
            pair_key = f"{type_abbrev[src_type]}→{type_abbrev[tgt_type]}"
            
            if len(weights) > 0:
                results[pair_key] = {
                    'weights': weights,
                    'mean': np.mean(weights),
                    'sem': np.std(weights) / np.sqrt(len(weights)),
                    'n': len(weights),
                    'n_possible': len(src_idx) * len(tgt_idx)
                }
            else:
                results[pair_key] = {
                    'weights': np.array([]),
                    'mean': 0.0,
                    'sem': 0.0,
                    'n': 0,
                    'n_possible': len(src_idx) * len(tgt_idx)
                }
    
    return results


def mann_whitney_test(stats1, stats2):
    """Run Mann-Whitney U test on two sets of weights."""
    w1 = stats1['weights']
    w2 = stats2['weights']
    
    if len(w1) < 3 or len(w2) < 3:
        return None, None
    
    stat, pval = stats.mannwhitneyu(w1, w2, alternative='two-sided')
    return stat, pval


def permutation_test(weights1, weights2, n_permutations=10000):
    """
    Permutation test for difference in means.
    
    Returns p-value for the null hypothesis that the two distributions
    have the same mean.
    """
    if len(weights1) < 3 or len(weights2) < 3:
        return None
    
    observed_diff = np.mean(weights1) - np.mean(weights2)
    pooled = np.concatenate([weights1, weights2])
    n1 = len(weights1)
    
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(pooled)
        perm_diff = np.mean(pooled[:n1]) - np.mean(pooled[n1:])
        if np.abs(perm_diff) >= np.abs(observed_diff):
            count += 1
    
    return count / n_permutations


def plot_within_lag_comparison(all_lag_stats, lags, output_dir):
    """Create bar plots comparing cell-type pairs within each lag."""
    fig, axes = plt.subplots(1, len(lags), figsize=(4*len(lags), 5), sharey=True)
    if len(lags) == 1:
        axes = [axes]
    
    pair_order = ['S→S', 'S→I', 'S→M', 'I→S', 'I→I', 'I→M', 'M→S', 'M→I', 'M→M']
    colors = plt.cm.Set3(np.linspace(0, 1, 9))
    
    for ax_idx, lag in enumerate(lags):
        ax = axes[ax_idx]
        stats_dict = all_lag_stats[lag]
        
        means = [stats_dict.get(p, {}).get('mean', 0) for p in pair_order]
        sems = [stats_dict.get(p, {}).get('sem', 0) for p in pair_order]
        
        x = np.arange(len(pair_order))
        bars = ax.bar(x, means, yerr=sems, capsize=3, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Cell-Type Pair', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_order, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'Lag {lag} ({lag * 0.25:.2f}s)', fontsize=11)
        
        if ax_idx == 0:
            ax.set_ylabel('Mean |μ̂| ± SEM', fontsize=10)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Cell-Type Interaction Strength by Lag\n(Approach C: Minimal Multi-Block)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    out_path = Path(output_dir) / 'fig_celltype_within_lag.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_cross_lag_comparison(all_lag_stats, lags, output_dir):
    """Create line plots comparing each cell-type pair across lags."""
    pair_order = ['S→S', 'S→I', 'S→M', 'I→S', 'I→I', 'I→M', 'M→S', 'M→I', 'M→M']
    
    # Group pairs by source type for cleaner visualization
    pair_groups = {
        'Sensory→': ['S→S', 'S→I', 'S→M'],
        'Inter→': ['I→S', 'I→I', 'I→M'],
        'Motor→': ['M→S', 'M→I', 'M→M']
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    
    colors_by_target = {'S': '#2ecc71', 'I': '#3498db', 'M': '#e74c3c'}
    
    for ax_idx, (group_name, pairs) in enumerate(pair_groups.items()):
        ax = axes[ax_idx]
        
        for pair in pairs:
            means = []
            sems = []
            for lag in lags:
                s = all_lag_stats[lag].get(pair, {'mean': 0, 'sem': 0})
                means.append(s['mean'])
                sems.append(s['sem'])
            
            target = pair[-1]  # Last character is target type
            color = colors_by_target[target]
            
            ax.errorbar(lags, means, yerr=sems, marker='o', capsize=3,
                       label=pair, color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=10)
        ax.set_xticks(lags)
        ax.set_xticklabels([f'{l}\n({l*0.25:.2f}s)' for l in lags])
        ax.set_title(f'{group_name}X', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Mean |μ̂| ± SEM', fontsize=10)
    
    plt.suptitle('Cell-Type Interaction Strength Across Lags\n(Approach C: Minimal Multi-Block)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    out_path = Path(output_dir) / 'fig_celltype_cross_lag.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def run_statistical_tests(all_lag_stats, lags, output_dir):
    """Run and report statistical tests."""
    pair_order = ['S→S', 'S→I', 'S→M', 'I→S', 'I→I', 'I→M', 'M→S', 'M→I', 'M→M']
    
    results_rows = []
    
    # --- Within-lag comparisons (Mann-Whitney U) ---
    print("\n" + "="*60)
    print("WITHIN-LAG COMPARISONS (Mann-Whitney U Test)")
    print("="*60)
    
    for lag in lags:
        print(f"\n--- Lag {lag} ---")
        stats_dict = all_lag_stats[lag]
        
        # Compare all pairs to the overall mean
        all_weights = []
        for pair in pair_order:
            if pair in stats_dict and len(stats_dict[pair]['weights']) > 0:
                all_weights.extend(stats_dict[pair]['weights'])
        
        if len(all_weights) == 0:
            continue
        
        overall_mean = np.mean(all_weights)
        
        for pair in pair_order:
            if pair not in stats_dict or len(stats_dict[pair]['weights']) < 3:
                continue
            
            s = stats_dict[pair]
            # Compare this pair to all other pairs combined
            other_weights = []
            for p2 in pair_order:
                if p2 != pair and p2 in stats_dict:
                    other_weights.extend(stats_dict[p2]['weights'])
            
            if len(other_weights) < 3:
                continue
            
            stat, pval = mann_whitney_test(s, {'weights': np.array(other_weights)})
            
            sig_marker = ""
            if pval is not None:
                if pval < 0.001:
                    sig_marker = "***"
                elif pval < 0.01:
                    sig_marker = "**"
                elif pval < 0.05:
                    sig_marker = "*"
            
            results_rows.append({
                'comparison_type': 'within_lag',
                'lag': lag,
                'pair1': pair,
                'pair2': 'others',
                'mean1': s['mean'],
                'mean2': np.mean(other_weights),
                'n1': s['n'],
                'n2': len(other_weights),
                'stat': stat,
                'pval': pval,
                'significant': pval < 0.05 if pval else False
            })
            
            if pval is not None:
                print(f"  {pair} vs others: mean={s['mean']:.4f}, p={pval:.4f} {sig_marker}")
    
    # --- Cross-lag comparisons (Permutation test) ---
    print("\n" + "="*60)
    print("CROSS-LAG COMPARISONS (Permutation Test)")
    print("="*60)
    
    for pair in pair_order:
        print(f"\n--- {pair} ---")
        
        for i, lag1 in enumerate(lags):
            for lag2 in lags[i+1:]:
                s1 = all_lag_stats[lag1].get(pair, {})
                s2 = all_lag_stats[lag2].get(pair, {})
                
                w1 = s1.get('weights', np.array([]))
                w2 = s2.get('weights', np.array([]))
                
                if len(w1) < 3 or len(w2) < 3:
                    continue
                
                pval = permutation_test(w1, w2, n_permutations=5000)
                
                sig_marker = ""
                if pval is not None:
                    if pval < 0.001:
                        sig_marker = "***"
                    elif pval < 0.01:
                        sig_marker = "**"
                    elif pval < 0.05:
                        sig_marker = "*"
                
                results_rows.append({
                    'comparison_type': 'cross_lag',
                    'lag': f'{lag1}_vs_{lag2}',
                    'pair1': pair,
                    'pair2': pair,
                    'mean1': s1['mean'],
                    'mean2': s2['mean'],
                    'n1': s1['n'],
                    'n2': s2['n'],
                    'stat': None,
                    'pval': pval,
                    'significant': pval < 0.05 if pval else False
                })
                
                if pval is not None and pval < 0.1:  # Show marginally significant too
                    print(f"  Lag {lag1} vs {lag2}: {s1['mean']:.4f} vs {s2['mean']:.4f}, p={pval:.4f} {sig_marker}")
    
    # Save results
    df = pd.DataFrame(results_rows)
    out_path = Path(output_dir) / 'celltype_statistical_tests.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved statistical test results to: {out_path}")
    
    return df


def create_summary_figure(all_lag_stats, lags, test_results, output_dir):
    """Create a comprehensive summary figure."""
    fig = plt.figure(figsize=(16, 10))
    
    # Define grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25)
    
    pair_order = ['S→S', 'S→I', 'S→M', 'I→S', 'I→I', 'I→M', 'M→S', 'M→I', 'M→M']
    colors = plt.cm.Set3(np.linspace(0, 1, 9))
    
    # Panel A: Heatmap of mean weights
    ax_heat = fig.add_subplot(gs[0, 0])
    
    type_order = ['S', 'I', 'M']
    heat_data = np.zeros((3, 3, len(lags)))
    
    for lag_idx, lag in enumerate(lags):
        for i, src in enumerate(type_order):
            for j, tgt in enumerate(type_order):
                pair = f'{src}→{tgt}'
                if pair in all_lag_stats[lag]:
                    heat_data[j, i, lag_idx] = all_lag_stats[lag][pair]['mean']
    
    # Average across lags
    mean_heat = np.mean(heat_data, axis=2)
    
    im = ax_heat.imshow(mean_heat, cmap='Reds', aspect='equal')
    ax_heat.set_xticks([0, 1, 2])
    ax_heat.set_yticks([0, 1, 2])
    ax_heat.set_xticklabels(['Sensory', 'Inter', 'Motor'])
    ax_heat.set_yticklabels(['Sensory', 'Inter', 'Motor'])
    ax_heat.set_xlabel('Source Type', fontsize=10)
    ax_heat.set_ylabel('Target Type', fontsize=10)
    ax_heat.set_title('A. Mean Coupling Strength\n(averaged over lags)', fontsize=11, fontweight='bold')
    
    # Add values
    for i in range(3):
        for j in range(3):
            ax_heat.text(j, i, f'{mean_heat[i, j]:.3f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im, ax=ax_heat, shrink=0.8, label='Mean |μ̂|')
    
    # Panel B: Edge counts
    ax_counts = fig.add_subplot(gs[0, 1])
    
    count_data = np.zeros((3, 3))
    for i, src in enumerate(type_order):
        for j, tgt in enumerate(type_order):
            pair = f'{src}→{tgt}'
            total_n = sum(all_lag_stats[lag].get(pair, {}).get('n', 0) for lag in lags)
            count_data[j, i] = total_n / len(lags)  # Average per lag
    
    im2 = ax_counts.imshow(count_data, cmap='Blues', aspect='equal')
    ax_counts.set_xticks([0, 1, 2])
    ax_counts.set_yticks([0, 1, 2])
    ax_counts.set_xticklabels(['Sensory', 'Inter', 'Motor'])
    ax_counts.set_yticklabels(['Sensory', 'Inter', 'Motor'])
    ax_counts.set_xlabel('Source Type', fontsize=10)
    ax_counts.set_ylabel('Target Type', fontsize=10)
    ax_counts.set_title('B. Significant Edge Count\n(average per lag)', fontsize=11, fontweight='bold')
    
    for i in range(3):
        for j in range(3):
            ax_counts.text(j, i, f'{count_data[i, j]:.0f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im2, ax=ax_counts, shrink=0.8, label='# Edges')
    
    # Panel C: Significant cross-lag differences
    ax_sig = fig.add_subplot(gs[0, 2])
    
    sig_tests = test_results[
        (test_results['comparison_type'] == 'cross_lag') & 
        (test_results['pval'] < 0.05)
    ]
    
    if len(sig_tests) > 0:
        sig_text = "Significant Cross-Lag Differences (p < 0.05):\n\n"
        for _, row in sig_tests.iterrows():
            sig_text += f"• {row['pair1']} @ Lag {row['lag']}: p={row['pval']:.3f}\n"
    else:
        sig_text = "No significant cross-lag\ndifferences found (p < 0.05)"
    
    ax_sig.text(0.1, 0.9, sig_text, transform=ax_sig.transAxes, fontsize=10,
                verticalalignment='top', family='monospace')
    ax_sig.set_title('C. Cross-Lag Significance', fontsize=11, fontweight='bold')
    ax_sig.axis('off')
    
    # Panel D: Cross-lag trend lines (bottom row, full width)
    ax_trend = fig.add_subplot(gs[1, :])
    
    colors_by_target = {'S': '#2ecc71', 'I': '#3498db', 'M': '#e74c3c'}
    markers = ['o', 's', '^']
    
    for pair_idx, pair in enumerate(pair_order):
        means = [all_lag_stats[lag].get(pair, {}).get('mean', 0) for lag in lags]
        sems = [all_lag_stats[lag].get(pair, {}).get('sem', 0) for lag in lags]
        
        target = pair[-1]
        color = colors_by_target[target]
        marker = markers[pair_idx % 3]
        
        # Slight x offset for visibility
        x_offset = (pair_idx - 4) * 0.05
        
        ax_trend.errorbar([l + x_offset for l in lags], means, yerr=sems, 
                         marker=marker, capsize=2, label=pair, color=color,
                         linewidth=1.5, markersize=5, alpha=0.8)
    
    ax_trend.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=11)
    ax_trend.set_ylabel('Mean |μ̂| ± SEM', fontsize=11)
    ax_trend.set_title('D. Cell-Type Coupling Strength Across Lags', fontsize=11, fontweight='bold')
    ax_trend.set_xticks(lags)
    ax_trend.set_xticklabels([f'{l} ({l*0.25:.2f}s)' for l in lags])
    ax_trend.legend(loc='upper right', ncol=3, fontsize=8)
    ax_trend.grid(alpha=0.3)
    
    plt.suptitle('Cell-Type Interaction Analysis: Approach C (Minimal Multi-Block SBTG)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    out_path = Path(output_dir) / 'fig_celltype_summary.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_baseline_celltype_comparison(all_lag_stats, baseline_stats, output_dir):
    """
    Compare SBTG vs Baselines on cell-type interaction strength.
    Focus on S->M, S->I, I->M (main feedforward/feedback loops).
    """
    focus_pairs = ['S→I', 'S→M', 'I→M'] # Key functional pairs
    lags = sorted(list(all_lag_stats.keys()))
    
    fig, axes = plt.subplots(1, len(focus_pairs), figsize=(5*len(focus_pairs), 5), sharey=True)
    if len(focus_pairs) == 1: axes = [axes]
    
    # Method colors
    colors = {
        'SBTG': 'steelblue',
        'Pearson': 'gray',
        'VAR': 'coral',
        'Granger': 'forestgreen'
    }
    
    for ax_idx, pair in enumerate(focus_pairs):
        ax = axes[ax_idx]
        
        # SBTG
        means = [all_lag_stats[l].get(pair, {}).get('mean', 0) for l in lags]
        sems = [all_lag_stats[l].get(pair, {}).get('sem', 0) for l in lags]
        ax.errorbar(lags, means, yerr=sems, marker='o', label='SBTG', 
                   color=colors['SBTG'], lw=2.5, capsize=3)
        
        # Baselines
        for method, method_stats in baseline_stats.items():
            # Check if this method has data for these lags
            b_lags = sorted([l for l in lags if l in method_stats])
            if not b_lags: continue
            
            b_means = [method_stats[l].get(pair, {}).get('mean', 0) for l in b_lags]
            b_sems = [method_stats[l].get(pair, {}).get('sem', 0) for l in b_lags]
            
            color = colors.get(method.split('_')[0], 'black')
            ax.errorbar(b_lags, b_means, yerr=b_sems, marker='s', label=method,
                       color=color, lw=1.5, ls='--', capsize=2, alpha=0.7)
            
        ax.set_title(f'{pair} Interaction', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Lag', fontsize=11)
        ax.set_xticks(lags)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Mean Connection Strength (|W|)', fontsize=11)
            ax.legend(fontsize=9)
            
    plt.suptitle('SBTG vs Baselines: Cell-Type Interaction Strength', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = Path(output_dir) / 'fig_celltype_baseline_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# =============================================================================
# CLEAN SINGLE-APPROACH FIGURES (for paper, no legends)
# =============================================================================


def load_eval_csvs(result_dir: str, approach: str = 'C'):
    """Load evaluation CSVs."""
    eval_cook = pd.read_csv(Path(result_dir) / f'eval_cook_{approach}.csv')
    eval_leifer_path = Path(result_dir) / f'eval_leifer_{approach}.csv'
    eval_leifer = pd.read_csv(eval_leifer_path) if eval_leifer_path.exists() else None
    return eval_cook, eval_leifer


def plot_clean_auroc(eval_cook, eval_leifer, output_dir):
    """Clean AUROC vs Lag plot with no legend clutter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    lags = eval_cook['lag'].values
    lag_seconds = lags / SAMPLING_RATE
    
    # Panel 1: AUROC
    ax = axes[0]
    ax.plot(lag_seconds, eval_cook['auroc_struct'], 'o-', 
            lw=2.5, markersize=10, color='steelblue')
    for x, y in zip(lag_seconds, eval_cook['auroc_struct']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold',
                   color='steelblue')
    
    if eval_leifer is not None and len(eval_leifer) > 0:
        leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
        ax.plot(leifer_lag_sec, eval_leifer['auroc'], 's--', 
                lw=2, markersize=8, color='forestgreen')
        for x, y in zip(leifer_lag_sec, eval_leifer['auroc']):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0, -15), ha='center', fontsize=10, 
                       color='forestgreen', fontweight='bold')
    
    ax.axhline(0.5, color='gray', linestyle=':', lw=1.5)
    ax.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(f'AUROC vs Time Lag\n(Blue = {STRUCTURAL_LABEL}, Green = {FUNCTIONAL_LABEL})', 
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0.45, 0.75)
    ax.grid(True, alpha=0.3)
    
    # Mark best lag
    best_idx = eval_cook['auroc_struct'].idxmax()
    best_lag = eval_cook.loc[best_idx, 'lag'] / SAMPLING_RATE
    best_auroc = eval_cook.loc[best_idx, 'auroc_struct']
    ax.scatter([best_lag], [best_auroc], s=200, c='steelblue', 
               marker='*', edgecolors='black', linewidths=1.5, zorder=10)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lag_seconds)
    ax2.set_xticklabels([str(int(l)) for l in lags])
    ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=10)
    
    # Panel 2: AUPRC
    ax = axes[1]
    ax.plot(lag_seconds, eval_cook['auprc_struct'], 'o-', 
            lw=2.5, markersize=10, color='coral')
    for x, y in zip(lag_seconds, eval_cook['auprc_struct']):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold',
                   color='coral')
    
    if eval_leifer is not None and 'auprc' in eval_leifer.columns:
        leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
        ax.plot(leifer_lag_sec, eval_leifer['auprc'], 's--', 
                lw=2, markersize=8, color='darkgreen')
        for x, y in zip(leifer_lag_sec, eval_leifer['auprc']):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0, -15), ha='center', fontsize=10, 
                       color='darkgreen', fontweight='bold')
    
    ax.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax.set_ylabel('AUPRC', fontsize=12)
    ax.set_title(f'AUPRC vs Time Lag\n(Coral = {STRUCTURAL_LABEL}, Green = {FUNCTIONAL_LABEL})', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lag_seconds)
    ax2.set_xticklabels([str(int(l)) for l in lags])
    ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=10)
    
    plt.suptitle('Multi-Lag SBTG Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = Path(output_dir) / 'fig_clean_auroc.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_clean_edge_density(data, output_dir):
    """Clean edge density by lag plot."""
    lags = list(data['lags'])
    
    # Compute edge counts
    edge_counts = []
    for lag in lags:
        sig = data[f'sig_lag{lag}']
        edge_counts.append(int(sig.sum() - np.trace(sig)))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(range(len(lags)), edge_counts, color='coral', 
                  edgecolor='black', alpha=0.85, width=0.7)
    
    # Value labels
    for bar, val in zip(bars, edge_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Time Lag', fontsize=12)
    ax.set_ylabel('Number of FDR-Significant Edges', fontsize=12)
    ax.set_title('Detected Edges by Lag\n(FDR α = 0.1)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([f'Lag {l}\n({l/SAMPLING_RATE:.2f}s)' for l in lags], fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add summary stats
    total = sum(edge_counts)
    avg = total // len(lags)
    ax.text(0.98, 0.95, f'Total: {total:,}\nAverage: {avg:,}/lag', 
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    out_path = Path(output_dir) / 'fig_clean_edge_density.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_clean_edge_analysis_3panel(data, output_dir):
    """
    Clean 4-panel edge analysis: density, count, E:I balance, and Spearman correlation by lag.
    Single approach, no legend needed.
    """
    lags = list(data['lags'])
    lag_seconds = [l / SAMPLING_RATE for l in lags]
    neuron_names = list(data['neuron_names'])
    n = len(neuron_names)
    n_possible = n * (n - 1)  # Exclude diagonal
    
    # Load structural benchmark for Spearman correlation diagnostics.
    try:
        A_struct, cook_neurons, _ = _load_structural_connectome(CONNECTOME_DIR)
        has_connectome = True
    except Exception as e:
        print(f"  Warning: Could not load {STRUCTURAL_LABEL} connectome: {e}")
        has_connectome = False
        A_struct = None
        cook_neurons = None
    
    # Compute metrics for each lag
    densities = []
    edge_counts = []
    ei_ratios = []
    spearman_rhos = []
    
    for lag in lags:
        mu_hat = data[f'mu_hat_lag{lag}']
        sig = data[f'sig_lag{lag}']
        
        # Edge count (exclude diagonal)
        n_edges = sig.sum() - np.trace(sig)
        edge_counts.append(int(n_edges))
        
        # Density
        densities.append(n_edges / n_possible)
        
        # E:I ratio (excitatory = positive μ̂, inhibitory = negative μ̂)
        excit = ((mu_hat > 0) & (sig > 0)).sum()
        inhib = ((mu_hat < 0) & (sig > 0)).sum()
        ei_ratios.append(excit / max(1, inhib))
        
        # Spearman correlation against the structural benchmark.
        if has_connectome:
            # Align matrices
            mu_aligned, struct_aligned, common = align_matrices_for_correlation(
                mu_hat, neuron_names, A_struct, cook_neurons
            )
            if len(common) > 10:
                corr = compute_weight_correlation(
                    np.abs(mu_aligned), struct_aligned,
                    exclude_diagonal=True
                )
                spearman_rhos.append(corr.get('spearman_rho', np.nan))
            else:
                spearman_rhos.append(np.nan)
        else:
            spearman_rhos.append(np.nan)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Edge Density (top left)
    ax = axes[0, 0]
    ax.plot(lag_seconds, densities, 'o-', color='steelblue', lw=2.5, markersize=10)
    for x, y in zip(lag_seconds, densities):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold',
                   color='steelblue')
    
    ax.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax.set_ylabel('Edge Density\n(fraction of possible edges)', fontsize=12)
    ax.set_title('A. Edge Density vs Time Lag', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.8)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lag_seconds)
    ax2.set_xticklabels([str(int(l)) for l in lags])
    ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=10)
    
    # Panel 2: Edge Count (top right)
    ax = axes[0, 1]
    ax.plot(lag_seconds, edge_counts, 's-', color='coral', lw=2.5, markersize=10)
    for x, y in zip(lag_seconds, edge_counts):
        ax.annotate(f'{y:,}', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold',
                   color='coral')
    
    ax.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax.set_ylabel('Number of FDR-Significant Edges', fontsize=12)
    ax.set_title('B. Edge Count vs Time Lag\n(FDR α = 0.1)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add total
    total = sum(edge_counts)
    ax.text(0.98, 0.95, f'Total: {total:,}', transform=ax.transAxes, 
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lag_seconds)
    ax2.set_xticklabels([str(int(l)) for l in lags])
    ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=10)
    
    # Panel 3: E:I Balance (bottom left)
    ax = axes[1, 0]
    ax.plot(lag_seconds, ei_ratios, '^-', color='forestgreen', lw=2.5, markersize=10)
    for x, y in zip(lag_seconds, ei_ratios):
        ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold',
                   color='forestgreen')
    
    ax.axhline(1.0, color='gray', linestyle='--', lw=1.5)
    ax.text(0.98, 0.95, 'Balanced (E = I)', transform=ax.transAxes, 
            ha='right', va='top', fontsize=9, color='gray')
    
    ax.set_xlabel('Time Lag (seconds)', fontsize=12)
    ax.set_ylabel('E:I Ratio\n(Excitatory / Inhibitory)', fontsize=12)
    ax.set_title('C. Excitatory:Inhibitory Balance\nby Time Lag', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(lag_seconds)
    ax2.set_xticklabels([str(int(l)) for l in lags])
    ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=10)
    
    # Panel 4: Spearman Correlation (bottom right)
    ax = axes[1, 1]
    valid_rhos = [r for r in spearman_rhos if not np.isnan(r)]
    
    if len(valid_rhos) > 0:
        ax.plot(lag_seconds, spearman_rhos, 'D-', color='purple', lw=2.5, markersize=10)
        for x, y in zip(lag_seconds, spearman_rhos):
            if not np.isnan(y):
                ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=10, fontweight='bold',
                           color='purple')
        
        ax.axhline(0.0, color='gray', linestyle='--', lw=1.5)
        ax.set_xlabel('Time Lag (seconds)', fontsize=12)
        ax.set_ylabel(f'Spearman ρ\n(vs {STRUCTURAL_LABEL})', fontsize=12)
        ax.set_title('D. Correlation with Structural Connectome', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Secondary x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(lag_seconds)
        ax2.set_xticklabels([str(int(l)) for l in lags])
        ax2.set_xlabel('Time Lag (frames @ 4Hz)', fontsize=10)
    else:
        ax.text(0.5, 0.5, f'Spearman correlation\nnot available\n({STRUCTURAL_LABEL} not loaded)',
               transform=ax.transAxes, ha='center', va='center', fontsize=11)
        ax.set_title('D. Correlation with Structural Connectome', fontsize=13, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Edge Analysis by Time Lag', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = Path(output_dir) / 'fig_clean_edge_analysis.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_clean_celltype_significance(all_lag_stats, lags, output_dir, log_scale=False, 
                                      test_results=None):
    """
    Create a clean grid showing cell-type coupling strength across lags.
    Highlights significant within-lag differences dynamically from test results.
    
    Args:
        all_lag_stats: Dict of lag -> pair -> stats
        lags: List of lag values
        output_dir: Output directory
        log_scale: If True, apply log10 scaling to the data matrix
        test_results: DataFrame from run_statistical_tests (optional)
    """
    pair_order = ['S→S', 'S→I', 'S→M', 'I→S', 'I→I', 'I→M', 'M→S', 'M→I', 'M→M']
    
    # Build data matrix (9 pairs x len(lags))
    n_pairs = len(pair_order)
    n_lags = len(lags)
    
    data_matrix = np.zeros((n_pairs, n_lags))
    
    for i, pair in enumerate(pair_order):
        for j, lag in enumerate(lags):
            if pair in all_lag_stats[lag]:
                data_matrix[i, j] = all_lag_stats[lag][pair]['mean']
    
    # Store raw values for annotations
    raw_matrix = data_matrix.copy()
    
    # Apply log scaling if requested
    if log_scale:
        data_matrix = np.log10(data_matrix + 1e-6)  # Add small offset to avoid log(0)
        vmin, vmax = data_matrix.min(), data_matrix.max()
    else:
        # Dynamic vmin/vmax based on data
        vmin = max(0, raw_matrix.min() - 0.01)
        vmax = raw_matrix.max() + 0.01
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, max(6, n_pairs * 0.8)))
    
    im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    
    # Add value annotations (always show raw values for interpretability)
    for i in range(n_pairs):
        for j in range(n_lags):
            val = raw_matrix[i, j]
            scaled_val = data_matrix[i, j]
            threshold = (vmax + vmin) / 2
            color = 'white' if scaled_val > threshold else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                   fontsize=10 if n_lags > 6 else 11, fontweight='bold', color=color)
    
    # Build significance markers dynamically from test results
    sig_markers = {}
    if test_results is not None:
        within_lag = test_results[test_results['comparison_type'] == 'within_lag']
        for _, row in within_lag.iterrows():
            if row['pval'] is not None and row['pval'] < 0.05:
                pair = row['pair1']
                lag = row['lag']
                if pair in pair_order and lag in lags:
                    pair_idx = pair_order.index(pair)
                    lag_idx = lags.index(lag)
                    if row['pval'] < 0.001:
                        sig_markers[(pair_idx, lag_idx)] = '***'
                    elif row['pval'] < 0.01:
                        sig_markers[(pair_idx, lag_idx)] = '**'
                    else:
                        sig_markers[(pair_idx, lag_idx)] = '*'
    
    for (row, col), marker in sig_markers.items():
        ax.text(col, row + 0.38, marker, ha='center', va='top', 
               fontsize=12 if n_lags > 6 else 14, fontweight='bold', color='gold')
    
    # Axis labels - use SAMPLING_RATE for time conversion
    ax.set_xticks(range(n_lags))
    ax.set_xticklabels([f'Lag {l}\n({l/SAMPLING_RATE:.2f}s)' for l in lags], 
                       fontsize=9 if n_lags > 6 else 11)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_order, fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Time Lag', fontsize=13)
    ax.set_ylabel('Cell-Type Pair (Source → Target)', fontsize=13)
    
    # Title varies by scale
    scale_label = " (Log Scale)" if log_scale else ""
    ax.set_title(f'Cell-Type Coupling Strength Across Lags{scale_label}\n(Mean |μ̂| for FDR-Significant Edges)', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    if log_scale:
        cbar.set_label('log₁₀(Mean |μ̂|)', fontsize=12)
    else:
        cbar.set_label('Mean |μ̂|', fontsize=12)
    
    # Add legend for significance markers
    ax.text(1.15, 0.95, '*** p < 0.001\n** p < 0.01\n* p < 0.05', transform=ax.transAxes,
           fontsize=10, va='top', ha='left', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Filename varies by scale
    suffix = "_log" if log_scale else ""
    out_path = Path(output_dir) / f'fig_clean_celltype_by_lag{suffix}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_clean_type_interactions(data, type_indices, output_dir):
    """Clean neuron type interaction heatmaps."""
    lags = list(data['lags'])
    
    type_order = ['sensory', 'interneuron', 'motor']
    type_abbrev = {'sensory': 'S', 'interneuron': 'I', 'motor': 'M'}
    
    fig, axes = plt.subplots(1, len(lags), figsize=(4*len(lags), 4))
    if len(lags) == 1:
        axes = [axes]
    
    for ax_idx, lag in enumerate(lags):
        ax = axes[ax_idx]
        
        mu_hat = data[f'mu_hat_lag{lag}']
        sig_mask = data[f'sig_lag{lag}']
        
        # Compute mean |μ̂| for each type pair
        heat_data = np.zeros((3, 3))
        
        for i, src_type in enumerate(type_order):
            for j, tgt_type in enumerate(type_order):
                src_idx = type_indices[src_type]
                tgt_idx = type_indices[tgt_type]
                
                if len(src_idx) == 0 or len(tgt_idx) == 0:
                    continue
                
                sub_mu = mu_hat[np.ix_(tgt_idx, src_idx)]
                sub_sig = sig_mask[np.ix_(tgt_idx, src_idx)]
                
                weights = np.abs(sub_mu[sub_sig > 0])
                heat_data[j, i] = np.mean(weights) if len(weights) > 0 else 0
        
        im = ax.imshow(heat_data, cmap='Reds', aspect='equal', vmin=0, vmax=0.1)
        
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['S', 'I', 'M'])
        ax.set_yticklabels(['S', 'I', 'M'])
        ax.set_xlabel('Source Type', fontsize=10)
        ax.set_ylabel('Target Type', fontsize=10)
        ax.set_title(f'Lag {lag} ({lag/SAMPLING_RATE:.2f}s)', fontsize=11, fontweight='bold')
        
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{heat_data[i, j]:.3f}', ha='center', va='center', 
                       fontsize=9, color='white' if heat_data[i, j] > 0.05 else 'black')
    
    plt.suptitle('Cell-Type Coupling Strength by Lag\n(Mean |μ̂| for Significant Edges)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.25, 0.02, 0.5])
    plt.colorbar(im, cax=cbar_ax, label='Mean |μ̂|')
    
    out_path = Path(output_dir) / 'fig_clean_type_interactions.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_clean_summary(data, eval_cook, eval_leifer, output_dir):
    """Clean 4-panel summary for single approach (no legends needed)."""
    fig = plt.figure(figsize=(14, 10))
    
    lags = list(data['lags'])
    lag_seconds = [l / SAMPLING_RATE for l in lags]
    
    # Panel 1: AUROC (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    
    ax1.plot(lag_seconds, eval_cook['auroc_struct'], 'o-',
            color='steelblue', lw=2.5, markersize=10)
    
    if eval_leifer is not None:
        leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
        ax1.plot(leifer_lag_sec, eval_leifer['auroc'], 's--',
                color='forestgreen', lw=2, markersize=8)
    
    # Mark best
    best_idx = eval_cook['auroc_struct'].idxmax()
    best_lag = eval_cook.loc[best_idx, 'lag'] / SAMPLING_RATE
    best_auroc = eval_cook.loc[best_idx, 'auroc_struct']
    ax1.scatter([best_lag], [best_auroc], s=200, c='gold', 
               marker='*', edgecolors='black', linewidths=1.5, zorder=10)
    
    ax1.axhline(0.5, color='gray', linestyle=':', lw=1.5)
    ax1.set_xlabel('Time Lag (seconds)', fontsize=11)
    ax1.set_ylabel('AUROC', fontsize=11)
    ax1.set_title('A. Connectome Prediction Accuracy\n(★ = best lag)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0.45, 0.70)
    ax1.grid(True, alpha=0.3)
    
    ax1.text(0.02, 0.98, f'Blue: {STRUCTURAL_LABEL}\nGreen: {FUNCTIONAL_LABEL}', 
             transform=ax1.transAxes, va='top', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 2: Edge counts (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    
    edge_counts = []
    for lag in lags:
        sig = data[f'sig_lag{lag}']
        edge_counts.append(int(sig.sum() - np.trace(sig)))
    
    bars = ax2.bar(range(len(lags)), edge_counts, color='coral', 
                   edgecolor='black', alpha=0.8)
    
    for bar, val in zip(bars, edge_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Time Lag', fontsize=11)
    ax2.set_ylabel('Number of FDR-Significant Edges', fontsize=11)
    ax2.set_title('B. Detected Edges by Lag\n(FDR α = 0.1)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(lags)))
    ax2.set_xticklabels([f'{l}\n({l/SAMPLING_RATE:.2f}s)' for l in lags])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: AUPRC (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    
    ax3.plot(lag_seconds, eval_cook['auprc_struct'], 'o-',
            color='coral', lw=2.5, markersize=10)
    
    if eval_leifer is not None and 'auprc' in eval_leifer.columns:
        leifer_lag_sec = eval_leifer['lag'].values / SAMPLING_RATE
        ax3.plot(leifer_lag_sec, eval_leifer['auprc'], 's--',
                color='darkgreen', lw=2, markersize=8)
    
    ax3.set_xlabel('Time Lag (seconds)', fontsize=11)
    ax3.set_ylabel('AUPRC', fontsize=11)
    ax3.set_title('C. Precision-Recall Performance', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax3.text(0.02, 0.98, f'Coral: {STRUCTURAL_LABEL}\nGreen: {FUNCTIONAL_LABEL}', 
             transform=ax3.transAxes, va='top', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 4: Summary stats (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    best_cook_auroc = eval_cook['auroc_struct'].max()
    best_cook_lag = eval_cook.loc[eval_cook['auroc_struct'].idxmax(), 'lag']
    best_leifer_auroc = eval_leifer['auroc'].max() if eval_leifer is not None else 0
    best_leifer_lag = eval_leifer.loc[eval_leifer['auroc'].idxmax(), 'lag'] if eval_leifer is not None else 0
    total_edges = sum(edge_counts)
    
    summary_text = f"""
Summary Statistics
{'='*40}

Best {STRUCTURAL_LABEL} AUROC:     {best_cook_auroc:.3f} @ Lag {int(best_cook_lag)} ({best_cook_lag/SAMPLING_RATE:.2f}s)
Best {FUNCTIONAL_LABEL} AUROC:   {best_leifer_auroc:.3f} @ Lag {int(best_leifer_lag)} ({best_leifer_lag/SAMPLING_RATE:.2f}s)

Total FDR Edges:     {total_edges:,} (across all lags)
Average per Lag:     {total_edges // len(lags):,}

Neurons:             {data['neuron_names'].shape[0]}
Lags Analyzed:       {lags}
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('Multi-Lag SBTG Analysis Summary', 
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    
    out_path = Path(output_dir) / 'fig_clean_summary.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Cell-Type Statistical Analysis')
    parser.add_argument('result_dir', type=str, help='Path to results directory')
    parser.add_argument('--approach', type=str, default='C', choices=['A', 'B', 'C'],
                        help='Which approach to analyze (default: C)')
    parser.add_argument('--on-off', action='store_true',
                        help='Also analyze ON/OFF results if available')
    args = parser.parse_args()
    
    print("="*60)
    print("CELL-TYPE INTERACTION STATISTICAL ANALYSIS")
    print("="*60)
    print(f"Results directory: {args.result_dir}")
    print(f"Approach: {args.approach}")
    
    # Load results
    print("\n[1/5] Loading results...")
    data = load_results(args.result_dir, args.approach)
    
    neuron_names = data['neuron_names']
    lags = list(data['lags'])
    
    print(f"  Neurons: {len(neuron_names)}")
    print(f"  Lags: {lags}")
    
    # Get neuron type indices
    print("\n[2/5] Classifying neurons by type...")
    type_indices = get_neuron_type_indices(neuron_names)
    
    for ntype, indices in type_indices.items():
        if len(indices) > 0:
            print(f"  {ntype}: {len(indices)} neurons")
    
    # Compute statistics for each lag
    print("\n[3/5] Computing cell-type statistics...")
    all_lag_stats = {}
    
    for lag in lags:
        mu_hat = data[f'mu_hat_lag{lag}']
        sig_mask = data[f'sig_lag{lag}']
        
        stats_dict = compute_celltype_stats(mu_hat, sig_mask, type_indices)
        all_lag_stats[lag] = stats_dict
        
        print(f"\n  Lag {lag}:")
        for pair, s in stats_dict.items():
            if s['n'] > 0:
                print(f"    {pair}: n={s['n']:4d}, mean|μ̂|={s['mean']:.4f} ± {s['sem']:.4f}")
    
    # Run statistical tests
    print("\n[4/6] Running statistical tests...")
    test_results = run_statistical_tests(all_lag_stats, lags, args.result_dir)
    
    # Generate cell-type analysis figures
    print("\n[5/6] Generating cell-type analysis figures...")
    output_dir = Path(args.result_dir) / 'figures'
    
    plot_within_lag_comparison(all_lag_stats, lags, output_dir)
    plot_cross_lag_comparison(all_lag_stats, lags, output_dir)
    create_summary_figure(all_lag_stats, lags, test_results, output_dir)
    
    # Generate clean single-approach figures
    print("\n[6/6] Generating clean single-approach figures...")
    eval_cook, eval_leifer = load_eval_csvs(args.result_dir, args.approach)
    
    plot_clean_auroc(eval_cook, eval_leifer, output_dir)
    plot_clean_edge_density(data, output_dir)
    plot_clean_edge_analysis_3panel(data, output_dir)
    plot_clean_type_interactions(data, type_indices, output_dir)
    plot_clean_celltype_significance(all_lag_stats, lags, output_dir, test_results=test_results)
    plot_clean_celltype_significance(all_lag_stats, lags, output_dir, log_scale=True, test_results=test_results)
    plot_clean_summary(data, eval_cook, eval_leifer, output_dir)
    
    # Baseline comparison (NEW)
    print("\n[6.5/6] Processing baselines (if available)...")
    baselines = load_baselines(args.result_dir)
    if baselines:
        baseline_stats = {} # method -> lag -> stats
        
        # Identify methods and lags from keys like "Pearson_lag1" or "Pearson_1"
        methods = set()
        for key in baselines.keys():
            # Handle various formats: "Pearson_lag1", "Pearson_1", "pearson_lag_1"
            if '_lag' in key.lower():
                parts = key.split('_lag')
                methods.add(parts[0])
            elif any(key.endswith(f'_{i}') or key.endswith(f'_lag{i}') for i in lags):
                # Extract method name before the lag suffix
                for lag in lags:
                    for suffix in [f'_lag{lag}', f'_{lag}']:
                        if key.endswith(suffix):
                            methods.add(key[:-len(suffix)])
                            break
        
        if len(methods) == 0:
            print("  Warning: No baseline methods detected in baselines.npz")
            print(f"  Available keys: {list(baselines.keys())[:10]}...")
        else:
            print(f"  Found baseline methods: {sorted(list(methods))}")
            
            for method in methods:
                method_stats = {}
                for lag in lags:
                    # Try multiple key formats
                    possible_keys = [f"{method}_lag{lag}", f"{method}_{lag}", f"{method}_lag_{lag}"]
                    mat = None
                    
                    for key in possible_keys:
                        if key in baselines:
                            mat = baselines[key]
                            break
                    
                    if mat is not None:
                        # Create proxy significance mask: Top 10% absolute weights
                        # This is needed because baselines don't have FDR-corrected p-values
                        abs_w = np.abs(mat)
                        threshold = np.percentile(abs_w, 90)
                        sig_proxy = (abs_w > threshold).astype(float)
                        
                        method_stats[lag] = compute_celltype_stats(mat, sig_proxy, type_indices)
                    else:
                        print(f"  Warning: Could not find {method} for lag {lag}")
                
                if len(method_stats) > 0:
                    baseline_stats[method] = method_stats
            
            if len(baseline_stats) > 0:
                try:
                    plot_baseline_celltype_comparison(all_lag_stats, baseline_stats, output_dir)
                except Exception as e:
                    print(f"  Warning: Failed to plot baseline comparison: {e}")
            else:
                print("  No valid baseline data found for plotting")
    else:
        print("  No baselines.npz found, skipping baseline comparison.")
    if args.on_off:
        # Check for new 4-period structure first
        fourperiod_dir = Path(args.result_dir) / '4period_analysis'
        on_off_dir = Path(args.result_dir) / 'on_off_analysis'
        
        if fourperiod_dir.exists():
            print("\n" + "="*60)
            print("4-PERIOD STIMULUS ANALYSIS")
            print("="*60)
            
            for period in ['NOTHING', 'ON', 'SHOWING', 'OFF']:
                period_dir = fourperiod_dir / period
                result_file = period_dir / 'result.npz'
                
                if result_file.exists():
                    print(f"\n  Processing {period} period results...")
                    period_data = np.load(result_file, allow_pickle=True)
                    period_lags = list(period_data.get('lags', []))
                    
                    if len(period_lags) > 0:
                        # Load mu_hat and sig for each lag
                        period_all_stats = {}
                        for lag in period_lags:
                            mu_key = f'mu_hat_lag{lag}'
                            sig_key = f'sig_lag{lag}'
                            if mu_key in period_data and sig_key in period_data:
                                mu_hat = period_data[mu_key]
                                sig_mask = period_data[sig_key]
                                period_stats = compute_celltype_stats(mu_hat, sig_mask, type_indices)
                                period_all_stats[lag] = period_stats
                        
                        # Generate figures for this period
                        period_output = period_dir / 'figures'
                        period_output.mkdir(exist_ok=True)
                        
                        if len(period_all_stats) > 0:
                            plot_clean_celltype_significance(
                                period_all_stats, period_lags, period_output
                            )
                            plot_clean_celltype_significance(
                                period_all_stats, period_lags, period_output, log_scale=True
                            )
                            print(f"    ✓ Generated standard figures for {period}")
                        
                        # Load and plot Baselines (NEW)
                        period_baselines = load_baselines(period_dir)
                        if period_baselines:
                            period_baseline_stats = {}
                            methods = set()
                            for key in period_baselines.keys():
                                if '_lag' in key.lower():
                                    parts = key.split('_lag')
                                    methods.add(parts[0])
                            
                            if len(methods) > 0:
                                print(f"    Found baselines for {period}: {list(methods)}")
                                for method in methods:
                                    method_stats = {}
                                    for lag in period_lags:
                                        # Try multiple key formats
                                        possible_keys = [f"{method}_lag{lag}", f"{method}_{lag}"]
                                        mat = None
                                        for key in possible_keys:
                                            if key in period_baselines:
                                                mat = period_baselines[key]
                                                break
                                        
                                        if mat is not None:
                                            abs_w = np.abs(mat)
                                            # Use top 10% threshold as proxy for "significant" edges for strength comparison
                                            threshold = np.percentile(abs_w, 90)
                                            sig_proxy = (abs_w > threshold).astype(float)
                                            method_stats[lag] = compute_celltype_stats(mat, sig_proxy, type_indices)
                                    
                                    if len(method_stats) > 0:
                                        period_baseline_stats[method] = method_stats
                                
                                if len(period_baseline_stats) > 0:
                                    try:
                                        plot_baseline_celltype_comparison(period_all_stats, period_baseline_stats, period_output)
                                        print(f"    ✓ Generated baseline comparisons for {period}")
                                    except Exception as e:
                                        print(f"    Warning: Failed to plot baseline comparison for {period}: {e}")

                else:
                    print(f"  No results found for {period}")
        
        elif on_off_dir.exists():
            # Legacy ON/OFF analysis support
            print("\n" + "="*60)
            print("ON/OFF STIMULUS PERIOD ANALYSIS (Legacy)")
            print("="*60)
            
            for condition in ['ON', 'OFF']:
                cond_dir = on_off_dir / condition
                result_file = cond_dir / 'result.npz'
                
                if result_file.exists():
                    print(f"\n  Processing {condition} period results...")
                    cond_data = np.load(result_file, allow_pickle=True)
                    cond_lags = list(cond_data.get('lags', []))
                    
                    if len(cond_lags) > 0:
                        cond_all_stats = {}
                        for lag in cond_lags:
                            mu_key = f'mu_hat_lag{lag}'
                            sig_key = f'sig_lag{lag}'
                            if mu_key in cond_data and sig_key in cond_data:
                                mu_hat = cond_data[mu_key]
                                sig_mask = cond_data[sig_key]
                                cond_stats = compute_celltype_stats(mu_hat, sig_mask, type_indices)
                                cond_all_stats[lag] = cond_stats
                        
                        cond_output = cond_dir / 'figures'
                        cond_output.mkdir(exist_ok=True)
                        
                        if len(cond_all_stats) > 0:
                            plot_clean_celltype_significance(
                                cond_all_stats, cond_lags, cond_output
                            )
                            plot_clean_celltype_significance(
                                cond_all_stats, cond_lags, cond_output, log_scale=True
                            )
                            print(f"    ✓ Generated figures for {condition}")
                else:
                    print(f"  No results found for {condition}")
        else:
            print("\n  No 4period_analysis/ or on_off_analysis/ directory found.")
            print("  Run script 15 with --stimulus-periods first.")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
