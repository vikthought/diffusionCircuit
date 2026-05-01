#!/usr/bin/env python3
"""
06_leifer_analysis.py
=====================

Leifer optogenetic atlas analysis.
Creates aligned atlases if missing, then runs extended evaluations.

Four main analyses:

1. WT-only Edges (Extrasynaptic-Dependent)
   - Edges in WT but absent in unc-31 mutant
   
2. Dynamic Graph Collapse
   - Collapse operators: Prevalence, Peak, Average
   - Compares to static Leifer atlas
   
3. Path-2 Analysis
   - Tests indirect multi-synaptic paths (A→B→C)
   
4. Stratified False Positive Analysis
   - FPR on confirmed negatives (q_eq < 0.05)

References:
    Randi et al. (2023). Nature 623, 406–414.

Usage:
    python pipeline/06_leifer_analysis.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score, roc_auc_score
)
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = PROJECT_ROOT / "results" / "leifer_extended"
LEIFER_DIR = PROJECT_ROOT / "results" / "leifer_evaluation"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"
DATASETS_DIR = PROJECT_ROOT / "results" / "intermediate" / "datasets"

ALPHA = 0.05  # FDR threshold

# =============================================================================
# DATA LOADING
# =============================================================================

from pipeline.utils.leifer import ensure_aligned_atlas

def load_aligned_atlases() -> Tuple[Dict, Dict, List[str]]:
    """
    Load pre-aligned Leifer atlases for WT and unc-31.
    
    If aligned atlases don't exist, creates them using shared utility.
    """
    ensure_aligned_atlas()
    
    wt_file = LEIFER_DIR / "aligned_atlas_wild-type.npz"
    unc31_file = LEIFER_DIR / "aligned_atlas_unc-31.npz"
    
    wt_data = dict(np.load(wt_file, allow_pickle=True))
    unc31_data = dict(np.load(unc31_file, allow_pickle=True))
    
    neurons = list(wt_data['neuron_order'])
    
    return wt_data, unc31_data, neurons




def load_cook_connectome(neurons: List[str]) -> np.ndarray:
    """Load the Cook anatomical connectome aligned to our neuron order."""
    chem_file = CONNECTOME_DIR / "A_chem.npy"
    gap_file = CONNECTOME_DIR / "A_gap.npy"
    struct_file = CONNECTOME_DIR / "A_struct.npy"
    nodes_file = CONNECTOME_DIR / "nodes.json"
    
    if not nodes_file.exists():
        raise FileNotFoundError("Run pipeline/01_prepare_data.py first")
    
    with open(nodes_file, 'r') as f:
        connectome_nodes = json.load(f)
    
    # Try combined structural first
    if struct_file.exists():
        connectome = np.load(struct_file)
    elif chem_file.exists():
        chem = np.load(chem_file)
        gap = np.load(gap_file) if gap_file.exists() else np.zeros_like(chem)
        connectome = chem + 0.5 * gap
    else:
        raise FileNotFoundError("Run pipeline/01_prepare_data.py first")
    
    # Check alignment
    if connectome_nodes != neurons:
        print("  Warning: Connectome node order differs from Leifer alignment")
        # Need to reorder
        n = len(neurons)
        connectome_aligned = np.zeros((n, n))
        node_to_idx = {name: i for i, name in enumerate(connectome_nodes)}
        
        for i, ni in enumerate(neurons):
            for j, nj in enumerate(neurons):
                if ni in node_to_idx and nj in node_to_idx:
                    ci, cj = node_to_idx[ni], node_to_idx[nj]
                    connectome_aligned[i, j] = connectome[ci, cj]
        
        connectome = connectome_aligned
    
    return connectome


def load_connectivity_matrices(neurons: List[str]) -> Dict[str, np.ndarray]:
    """Load connectivity matrices computed from different methods/stimuli.
    
    Automatically subsets larger datasets to match the requested 'neurons' list.
    """
    matrices = {}
    target_n = len(neurons)
    neuron_to_idx = {name: i for i, name in enumerate(neurons)}
    
    # Load from datasets
    for stim in ["butanone", "pentanedione", "nacl", "full_traces"]:
        stim_dir = DATASETS_DIR / stim
        x_file = stim_dir / "X_segments.npy"
        meta_file = stim_dir / "standardization.json"
        
        if not x_file.exists() or not meta_file.exists():
            continue
        
        # Check neuron order
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        data_neurons = meta.get("node_order", [])
        
        # Find indices of target neurons in the dataset
        subset_indices = []
        found_neurons = 0
        for target_neuron in neurons:
             if target_neuron in data_neurons:
                 subset_indices.append(data_neurons.index(target_neuron))
                 found_neurons += 1
             else:
                 subset_indices.append(-1) # Missing
        
        # Proceed with whatever neurons were found; missing ones stay as NaN.
        if found_neurons == 0:
             continue
             
        # Load and concatenate segments
        X_segments = np.load(x_file, allow_pickle=True)
        valid_segments = []
        for s in X_segments:
            if s is not None and hasattr(s, 'shape') and len(s.shape) == 2:
                valid_segments.append(np.asarray(s, dtype=np.float64))
        
        if not valid_segments:
            continue
        
        X_full = np.vstack(valid_segments)
        X_full = X_full[~np.isnan(X_full).any(axis=1)]
        
        if X_full.shape[0] < 50:
             continue

        # Construct X with correct columns (handling missing)
        n_time = X_full.shape[0]
        X = np.full((n_time, target_n), np.nan) # Init with NaNs
        
        for tgt_i, src_i in enumerate(subset_indices):
            if src_i != -1:
                X[:, tgt_i] = X_full[:, src_i]
        
        # Pearson correlation (NaN columns from missing neurons yield NaN → zeroed)
        pearson = np.corrcoef(X.T)
        pearson = np.nan_to_num(pearson, nan=0)
        np.fill_diagonal(pearson, 0)
        matrices[f"{stim}_pearson"] = pearson
        
        # Cross-correlation
        X_t = X[:-1, :]
        X_tp1 = X[1:, :]
        xcorr = np.zeros((target_n, target_n))
        for i in range(target_n):
            # If neuron i is missing (all NaNs), std is NaN -> skip
            std_i = np.nanstd(X_t[:, i])
            
            for j in range(target_n):
                std_j = np.nanstd(X_tp1[:, j])
                
                if i != j and std_i > 0 and std_j > 0:
                    # Only compute pearsonr for neurons present in the data
                    if subset_indices[i] != -1 and subset_indices[j] != -1:
                         r, _ = stats.pearsonr(X_t[:, i], X_tp1[:, j])
                         xcorr[i, j] = r if not np.isnan(r) else 0
        matrices[f"{stim}_crosscorr"] = xcorr
        
        # Partial correlation - sensitive to NaNs/zeros (singular matrix)
        # Skip if missing neurons
        if found_neurons == target_n:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                lw.fit(X)
                precision = lw.precision_
                d = np.sqrt(np.diag(precision))
                partial = -precision / np.outer(d, d)
                np.fill_diagonal(partial, 0)
                matrices[f"{stim}_partial"] = partial
            except Exception:
                pass
    
    return matrices


def load_time_varying_connectivity(neurons: List[str], window_size: int = 100) -> Dict[str, np.ndarray]:
    """
    Load/compute time-varying connectivity matrices for dynamic analysis.
    
    Returns dict mapping method name to (n_windows, n_neurons, n_neurons) array.
    """
    n = len(neurons)
    time_varying = {}
    
    # Use full traces for time-varying analysis
    stim_dir = DATASETS_DIR / "full_traces"
    x_file = stim_dir / "X_segments.npy"
    meta_file = stim_dir / "standardization.json"
    
    if not x_file.exists():
        print("  Full traces data not found, skipping time-varying analysis")
        return time_varying
    
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    data_neurons = meta.get("node_order", [])
    
    if data_neurons != neurons:
        print("  Neuron order mismatch, skipping time-varying analysis")
        return time_varying
    
    # Load segments
    X_segments = np.load(x_file, allow_pickle=True)
    valid_segments = []
    for s in X_segments:
        if s is not None and hasattr(s, 'shape') and len(s.shape) == 2:
            arr = np.asarray(s, dtype=np.float64)
            arr = arr[~np.isnan(arr).any(axis=1)]
            if len(arr) >= window_size:
                valid_segments.append(arr)
    
    if not valid_segments:
        return time_varying
    
    # Concatenate all segments
    X_all = np.vstack(valid_segments)
    
    # Compute sliding window connectivity
    n_windows = (len(X_all) - window_size) // (window_size // 2)  # 50% overlap
    
    if n_windows < 5:
        print(f"  Too few windows ({n_windows}), skipping time-varying")
        return time_varying
    
    print(f"  Computing time-varying connectivity: {n_windows} windows of {window_size} frames...")
    
    # Pearson time-varying
    pearson_tv = np.zeros((n_windows, n, n))
    xcorr_tv = np.zeros((n_windows, n, n))
    
    for w in tqdm(range(n_windows), desc="  Computing windows"):
        start = w * (window_size // 2)
        end = start + window_size
        X_win = X_all[start:end]
        
        # Pearson
        corr = np.corrcoef(X_win.T)
        corr = np.nan_to_num(corr, nan=0)
        np.fill_diagonal(corr, 0)
        pearson_tv[w] = corr
        
        # Cross-correlation
        X_t = X_win[:-1, :]
        X_tp1 = X_win[1:, :]
        for i in range(n):
            for j in range(n):
                if i != j and np.std(X_t[:, i]) > 0 and np.std(X_tp1[:, j]) > 0:
                    r, _ = stats.pearsonr(X_t[:, i], X_tp1[:, j])
                    xcorr_tv[w, i, j] = r if not np.isnan(r) else 0
    
    time_varying["pearson"] = pearson_tv
    time_varying["crosscorr"] = xcorr_tv
    
    return time_varying


# =============================================================================
# ANALYSIS 1: WT-ONLY EDGES (EXTRASYNAPTIC)
# =============================================================================

def analyze_wt_only_edges(
    wt_data: Dict,
    unc31_data: Dict,
    connectivity: Dict[str, np.ndarray],
    output_dir: Path
) -> Dict:
    """
    Identify edges present in WT but absent in unc-31 (extrasynaptic-dependent).
    
    The unc-31 mutant lacks dense-core vesicle release, impairing extrasynaptic
    signaling. Edges that appear in WT but not unc-31 likely depend on 
    neuropeptide/monoamine signaling rather than synaptic transmission.
    
    We test: Do our methods preferentially score these extrasynaptic edges?
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 1: WT-ONLY EDGES (EXTRASYNAPTIC-DEPENDENT)")
    print("=" * 60)
    
    q_wt = wt_data['q']
    q_unc31 = unc31_data['q']
    
    n = q_wt.shape[0]
    
    # Define edge sets
    wt_connected = q_wt < ALPHA
    unc31_connected = q_unc31 < ALPHA
    
    # WT-only edges: connected in WT, NOT connected in unc-31
    # These are "extrasynaptic-dependent"
    wt_only = wt_connected & ~unc31_connected
    
    # Shared edges: connected in both (likely synaptic)
    shared = wt_connected & unc31_connected
    
    # unc-31-only edges: connected in unc-31 but not WT (rare/noise)
    unc31_only = ~wt_connected & unc31_connected
    
    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    
    print(f"\n  Edge counts (excluding diagonal):")
    print(f"    WT connected: {np.sum(wt_connected & mask)}")
    print(f"    unc-31 connected: {np.sum(unc31_connected & mask)}")
    print(f"    WT-only (extrasynaptic): {np.sum(wt_only & mask)}")
    print(f"    Shared (synaptic): {np.sum(shared & mask)}")
    print(f"    unc-31-only: {np.sum(unc31_only & mask)}")
    
    results = {
        "n_wt_connected": int(np.sum(wt_connected & mask)),
        "n_unc31_connected": int(np.sum(unc31_connected & mask)),
        "n_wt_only": int(np.sum(wt_only & mask)),
        "n_shared": int(np.sum(shared & mask)),
        "n_unc31_only": int(np.sum(unc31_only & mask)),
        "methods": {}
    }
    
    # For each connectivity method, compare scores on WT-only vs shared vs non-edges
    print("\n  Method performance on edge categories:")
    print(f"  {'Method':<30} {'WT-only':<12} {'Shared':<12} {'Non-edge':<12} {'Diff (WT-only - Shared)'}")
    print("  " + "-" * 80)
    
    for method_name, scores in connectivity.items():
        scores_abs = np.abs(scores)
        
        # Mean scores for each category
        wt_only_scores = scores_abs[wt_only & mask]
        shared_scores = scores_abs[shared & mask]
        non_edge_scores = scores_abs[~wt_connected & ~unc31_connected & mask & ~np.isnan(q_wt)]
        
        if len(wt_only_scores) == 0 or len(shared_scores) == 0:
            continue
        
        mean_wt_only = np.mean(wt_only_scores)
        mean_shared = np.mean(shared_scores)
        mean_non = np.mean(non_edge_scores) if len(non_edge_scores) > 0 else 0
        
        # Mann-Whitney U: WT-only scores > shared scores (one-sided)
        if len(wt_only_scores) > 5 and len(shared_scores) > 5:
            stat, pval = stats.mannwhitneyu(wt_only_scores, shared_scores, alternative='greater')
            sig = "*" if pval < 0.05 else ""
        else:
            pval = np.nan
            sig = ""
        
        diff = mean_wt_only - mean_shared
        
        print(f"  {method_name:<30} {mean_wt_only:<12.4f} {mean_shared:<12.4f} {mean_non:<12.4f} {diff:+.4f} {sig}")
        
        results["methods"][method_name] = {
            "mean_wt_only": float(mean_wt_only),
            "mean_shared": float(mean_shared),
            "mean_non_edge": float(mean_non),
            "diff": float(diff),
            "mannwhitney_p": float(pval) if not np.isnan(pval) else None,
        }
    
    # Generate visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Score distributions
    ax = axes[0]
    categories = []
    scores_list = []
    labels = []
    
    best_method = max(connectivity.keys(), key=lambda m: np.mean(np.abs(connectivity[m][wt_only & mask])) if np.sum(wt_only & mask) > 0 else 0)
    scores = np.abs(connectivity[best_method])
    
    categories.extend(['WT-only\n(extrasynaptic)'] * np.sum(wt_only & mask))
    scores_list.extend(scores[wt_only & mask].tolist())
    
    categories.extend(['Shared\n(synaptic)'] * np.sum(shared & mask))
    scores_list.extend(scores[shared & mask].tolist())
    
    non_mask = ~wt_connected & ~unc31_connected & mask & ~np.isnan(q_wt)
    n_sample = min(np.sum(non_mask), 500)  # Subsample for visibility
    idx = np.random.choice(np.where(non_mask.flatten())[0], n_sample, replace=False)
    categories.extend(['Non-edge'] * n_sample)
    scores_list.extend(scores.flatten()[idx].tolist())
    
    df_plot = pd.DataFrame({'Category': categories, 'Score': scores_list})
    sns.boxplot(data=df_plot, x='Category', y='Score', ax=ax, palette=['red', 'blue', 'gray'])
    ax.set_ylabel(f"|{best_method}| Score")
    ax.set_title("Score Distribution by Edge Category")
    
    # Panel B: Venn diagram (simplified as stacked bar)
    ax = axes[1]
    n_wt_only = np.sum(wt_only & mask)
    n_shared = np.sum(shared & mask)
    n_unc31_only = np.sum(unc31_only & mask)
    
    bars = ax.bar(['WT-only\n(extrasynaptic)', 'Shared\n(both)', 'unc-31-only'], 
                   [n_wt_only, n_shared, n_unc31_only],
                   color=['red', 'purple', 'blue'])
    ax.set_ylabel("Number of Edges")
    ax.set_title("Edge Categories: WT vs unc-31")
    
    for bar, count in zip(bars, [n_wt_only, n_shared, n_unc31_only]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "wt_only_edges_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved: {output_dir / 'wt_only_edges_analysis.png'}")
    
    return results


# =============================================================================
# ANALYSIS 2: DYNAMIC GRAPH COLLAPSE
# =============================================================================

def analyze_dynamic_collapse(
    time_varying: Dict[str, np.ndarray],
    wt_data: Dict,
    output_dir: Path
) -> Dict:
    """
    Evaluate time-varying connectivity using collapse operators.
    
    Collapse operators:
    - Prevalence: fraction of windows with |score| > threshold
    - Peak: max(|score|) across windows
    - Average: mean(|score|) across windows
    
    This tests whether stable edges (high prevalence) or transient edges (high peak)
    better match the functional atlas.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2: DYNAMIC GRAPH COLLAPSE")
    print("=" * 60)
    
    if not time_varying:
        print("  No time-varying data available")
        return {}
    
    q = wt_data['q']
    q_eq = wt_data['q_eq']
    
    # Create labels
    n = q.shape[0]
    labels = np.full_like(q, np.nan)
    labels[q < ALPHA] = 1
    labels[(q_eq < ALPHA) & np.isnan(labels)] = 0
    
    mask = ~np.eye(n, dtype=bool) & ~np.isnan(labels)
    y_true = labels[mask].flatten()
    
    results = {"methods": {}}
    all_results_for_plot = []
    
    for method_name, tv_scores in time_varying.items():
        n_windows = tv_scores.shape[0]
        tv_abs = np.abs(tv_scores)
        
        print(f"\n  Method: {method_name} ({n_windows} windows)")
        
        # Collapse operators
        prevalence_threshold = np.nanpercentile(tv_abs, 75)  # Top 25%
        prevalence = np.mean(tv_abs > prevalence_threshold, axis=0)
        peak = np.nanmax(tv_abs, axis=0)
        average = np.nanmean(tv_abs, axis=0)
        
        collapses = {
            "Prevalence": prevalence,
            "Peak": peak,
            "Average": average,
        }
        
        method_results = {}
        
        for collapse_name, collapsed_scores in collapses.items():
            y_score = collapsed_scores[mask].flatten()
            
            # Remove NaN
            valid = ~np.isnan(y_score)
            y_t = y_true[valid]
            y_s = y_score[valid]
            
            if len(y_t) == 0 or y_t.sum() == 0:
                continue
            
            # Compute metrics
            precision, recall, _ = precision_recall_curve(y_t, y_s)
            auprc = auc(recall, precision)
            
            fpr, tpr, _ = roc_curve(y_t, y_s)
            auroc = auc(fpr, tpr)
            
            print(f"    {collapse_name}: AUROC={auroc:.3f}, AUPRC={auprc:.3f}")
            
            method_results[collapse_name] = {
                "AUROC": float(auroc),
                "AUPRC": float(auprc),
            }
            
            all_results_for_plot.append({
                "Method": method_name,
                "Collapse": collapse_name,
                "AUROC": auroc,
                "AUPRC": auprc,
            })
        
        results["methods"][method_name] = method_results
    
    # Visualization
    if all_results_for_plot:
        df_plot = pd.DataFrame(all_results_for_plot)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # AUROC
        ax = axes[0]
        sns.barplot(data=df_plot, x='Collapse', y='AUROC', hue='Method', ax=ax)
        ax.set_title("AUROC by Collapse Operator")
        ax.set_ylim([0.4, 0.8])
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.legend(title='Method')
        
        # AUPRC
        ax = axes[1]
        sns.barplot(data=df_plot, x='Collapse', y='AUPRC', hue='Method', ax=ax)
        ax.set_title("AUPRC by Collapse Operator")
        random_baseline = y_true.sum() / len(y_true) if len(y_true) > 0 else 0.16
        ax.axhline(random_baseline, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax.legend(title='Method')
        
        plt.tight_layout()
        plt.savefig(output_dir / "dynamic_collapse_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  ✓ Saved: {output_dir / 'dynamic_collapse_analysis.png'}")
    
    return results


# =============================================================================
# ANALYSIS 3: PATH-2 (INDIRECT CONNECTIONS)
# =============================================================================

def analyze_path2(
    connectivity: Dict[str, np.ndarray],
    wt_data: Dict,
    connectome: np.ndarray,
    output_dir: Path
) -> Dict:
    """
    Test if functional methods capture indirect multi-synaptic paths.
    
    The Leifer paper explicitly states that signals propagate along
    "indirect, multisynaptic, and recursive paths". This analysis tests
    whether pairs connected by path-2 (A→B→C) in the anatomical connectome
    show functional connectivity.
    
    We compare:
    - Direct anatomical: A→C exists
    - Path-2 anatomical: A→B and B→C exist (but not A→C)
    - Path-2 functional: A→C in Leifer atlas but not in Cook anatomy
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 3: PATH-2 (INDIRECT CONNECTIONS)")
    print("=" * 60)
    
    n = connectome.shape[0]
    
    # Binary anatomical connectivity
    anat_binary = (connectome > 0).astype(int)
    
    # Path-2 matrix: reachable in exactly 2 hops
    # path2[i,j] = sum over k of (anat[i,k] * anat[k,j])
    path2 = anat_binary @ anat_binary
    np.fill_diagonal(path2, 0)
    
    # Categories
    direct_anat = anat_binary > 0
    path2_only = (path2 > 0) & ~direct_anat  # Path-2 but no direct connection
    
    # Functional connectivity
    q = wt_data['q']
    func_connected = q < ALPHA
    
    # Path-2 functional: connected functionally but not anatomically
    path2_func = func_connected & ~direct_anat
    
    mask = ~np.eye(n, dtype=bool)
    
    print(f"\n  Connection counts (excluding diagonal):")
    print(f"    Direct anatomical: {np.sum(direct_anat & mask)}")
    print(f"    Path-2 anatomical (no direct): {np.sum(path2_only & mask)}")
    print(f"    Functional (Leifer q<0.05): {np.sum(func_connected & mask)}")
    print(f"    Functional but no direct anatomy: {np.sum(path2_func & mask)}")
    
    # Partition functional-only edges by presence of path-2 anatomical support
    path2_func_with_anat = path2_func & (path2 > 0)
    path2_func_no_anat = path2_func & ~(path2 > 0)
    
    print(f"\n  Of functional-only edges ({np.sum(path2_func & mask)}):")
    print(f"    With path-2 anatomical support: {np.sum(path2_func_with_anat & mask)}")
    print(f"    No anatomical support (extrasynaptic?): {np.sum(path2_func_no_anat & mask)}")
    
    results = {
        "n_direct_anat": int(np.sum(direct_anat & mask)),
        "n_path2_anat": int(np.sum(path2_only & mask)),
        "n_func_connected": int(np.sum(func_connected & mask)),
        "n_path2_func": int(np.sum(path2_func & mask)),
        "n_path2_func_with_anat": int(np.sum(path2_func_with_anat & mask)),
        "n_path2_func_no_anat": int(np.sum(path2_func_no_anat & mask)),
        "methods": {}
    }
    
    # Compare mean scores across direct, path-2, and unconnected edge groups
    print(f"\n  {'Method':<30} {'Direct':<10} {'Path-2':<10} {'Unconnected':<12}")
    print("  " + "-" * 65)
    
    for method_name, scores in connectivity.items():
        scores_abs = np.abs(scores)
        
        direct_scores = scores_abs[direct_anat & mask]
        path2_scores = scores_abs[path2_only & mask]
        unconnected_scores = scores_abs[~direct_anat & ~(path2 > 0) & mask]
        
        mean_direct = np.mean(direct_scores) if len(direct_scores) > 0 else 0
        mean_path2 = np.mean(path2_scores) if len(path2_scores) > 0 else 0
        mean_unconnected = np.mean(unconnected_scores) if len(unconnected_scores) > 0 else 0
        
        print(f"  {method_name:<30} {mean_direct:<10.4f} {mean_path2:<10.4f} {mean_unconnected:<12.4f}")
        
        results["methods"][method_name] = {
            "mean_direct": float(mean_direct),
            "mean_path2": float(mean_path2),
            "mean_unconnected": float(mean_unconnected),
        }
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Venn-like breakdown
    ax = axes[0]
    categories = ['Direct\nAnatomical', 'Path-2\nAnatomical', 'Functional\n(Leifer)', 
                  'Func & Path-2', 'Func no Anat']
    counts = [
        np.sum(direct_anat & mask),
        np.sum(path2_only & mask),
        np.sum(func_connected & mask),
        np.sum(path2_func_with_anat & mask),
        np.sum(path2_func_no_anat & mask),
    ]
    colors = ['blue', 'lightblue', 'green', 'purple', 'red']
    
    bars = ax.bar(categories, counts, color=colors)
    ax.set_ylabel("Number of Edges")
    ax.set_title("Connection Categories: Direct vs Path-2 vs Functional")
    ax.tick_params(axis='x', rotation=30)
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=9)
    
    # Panel B: Score comparison
    ax = axes[1]
    best_method = "full_traces_crosscorr" if "full_traces_crosscorr" in connectivity else list(connectivity.keys())[0]
    scores = np.abs(connectivity[best_method])
    
    data = []
    for cat, cat_mask in [('Direct', direct_anat & mask), 
                           ('Path-2', path2_only & mask),
                           ('None', ~direct_anat & ~(path2 > 0) & mask)]:
        vals = scores[cat_mask]
        if len(vals) > 500:
            vals = np.random.choice(vals, 500, replace=False)
        data.extend([{'Category': cat, 'Score': v} for v in vals])
    
    df_plot = pd.DataFrame(data)
    sns.boxplot(data=df_plot, x='Category', y='Score', ax=ax, palette=['blue', 'lightblue', 'gray'])
    ax.set_ylabel(f"|{best_method}| Score")
    ax.set_title("Connectivity Score by Anatomical Path Length")
    
    plt.tight_layout()
    plt.savefig(output_dir / "path2_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved: {output_dir / 'path2_analysis.png'}")
    
    return results


# =============================================================================
# ANALYSIS 4: STRATIFIED FALSE POSITIVE ANALYSIS
# =============================================================================

def analyze_stratified_fp(
    connectivity: Dict[str, np.ndarray],
    wt_data: Dict,
    output_dir: Path
) -> Dict:
    """
    Focused analysis on false positives against confirmed negatives (q_eq < 0.05).
    
    These are the highest-confidence "no connection" labels from the Leifer atlas.
    A good method should have very low scores on these pairs.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 4: STRATIFIED FALSE POSITIVE ANALYSIS")
    print("=" * 60)
    
    q = wt_data['q']
    q_eq = wt_data['q_eq']
    
    n = q.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    # Categories by confidence level
    confirmed_positive = (q < 0.01) & mask  # Very high confidence positive
    positive = (q < 0.05) & ~(q < 0.01) & mask  # Moderate confidence positive
    
    confirmed_negative = (q_eq < 0.01) & mask  # Very high confidence negative
    negative = (q_eq < 0.05) & ~(q_eq < 0.01) & mask  # Moderate confidence negative
    
    # Ambiguous: neither positive nor negative
    ambiguous = ~(q < 0.05) & ~(q_eq < 0.05) & mask & ~np.isnan(q)
    
    print(f"\n  Stratified label counts:")
    print(f"    Confirmed positive (q < 0.01): {np.sum(confirmed_positive)}")
    print(f"    Positive (0.01 ≤ q < 0.05): {np.sum(positive)}")
    print(f"    Confirmed negative (q_eq < 0.01): {np.sum(confirmed_negative)}")
    print(f"    Negative (0.01 ≤ q_eq < 0.05): {np.sum(negative)}")
    print(f"    Ambiguous: {np.sum(ambiguous)}")
    
    results = {"counts": {
        "confirmed_positive": int(np.sum(confirmed_positive)),
        "positive": int(np.sum(positive)),
        "confirmed_negative": int(np.sum(confirmed_negative)),
        "negative": int(np.sum(negative)),
        "ambiguous": int(np.sum(ambiguous)),
    }, "methods": {}}
    
    # For each method, compute FPR at different thresholds
    print(f"\n  False positive rates on confirmed negatives:")
    print(f"  {'Method':<30} {'FPR@10%':<10} {'FPR@15%':<10} {'FPR@20%':<10}")
    print("  " + "-" * 60)
    
    all_results_for_plot = []
    
    for method_name, scores in connectivity.items():
        scores_abs = np.abs(scores)
        
        # At different density thresholds
        fprs = {}
        for density in [0.10, 0.15, 0.20]:
            # Threshold to get this density
            n_edges = int(density * np.sum(mask))
            sorted_scores = np.sort(scores_abs[mask].flatten())[::-1]
            threshold = sorted_scores[n_edges - 1] if n_edges > 0 else 0
            
            predicted_edges = (scores_abs >= threshold) & mask
            
            # FPR on confirmed negatives
            fp = np.sum(predicted_edges & confirmed_negative)
            total_neg = np.sum(confirmed_negative)
            fpr = fp / total_neg if total_neg > 0 else 0
            
            fprs[f"FPR@{int(density*100)}%"] = fpr
            
            all_results_for_plot.append({
                "Method": method_name,
                "Density": f"{int(density*100)}%",
                "FPR": fpr,
            })
        
        print(f"  {method_name:<30} {fprs['FPR@10%']:<10.3f} {fprs['FPR@15%']:<10.3f} {fprs['FPR@20%']:<10.3f}")
        
        # Also compute score distributions
        method_results = {**fprs}
        
        # Mean scores on each category
        method_results["mean_confirmed_positive"] = float(np.mean(scores_abs[confirmed_positive]))
        method_results["mean_confirmed_negative"] = float(np.mean(scores_abs[confirmed_negative]))
        
        # Separation test: confirmed-positive vs confirmed-negative scores
        if np.sum(confirmed_positive) > 10 and np.sum(confirmed_negative) > 10:
            pos_scores = scores_abs[confirmed_positive]
            neg_scores = scores_abs[confirmed_negative]
            stat, pval = stats.mannwhitneyu(pos_scores, neg_scores, alternative='greater')
            method_results["separation_p"] = float(pval)
            method_results["separation_auc"] = float(roc_auc_score(
                np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))]),
                np.concatenate([pos_scores, neg_scores])
            ))
        
        results["methods"][method_name] = method_results
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: FPR by density
    ax = axes[0]
    df_plot = pd.DataFrame(all_results_for_plot)
    sns.barplot(data=df_plot, x='Density', y='FPR', hue='Method', ax=ax)
    ax.set_ylabel("False Positive Rate on Confirmed Negatives")
    ax.set_title("FPR on q_eq < 0.01 Pairs (Highest Confidence Negatives)")
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Panel B: Score distributions
    ax = axes[1]
    best_method = "full_traces_crosscorr" if "full_traces_crosscorr" in connectivity else list(connectivity.keys())[0]
    scores = np.abs(connectivity[best_method])
    
    data = []
    for cat, cat_mask in [('Confirmed\nPositive', confirmed_positive),
                           ('Confirmed\nNegative', confirmed_negative),
                           ('Ambiguous', ambiguous)]:
        vals = scores[cat_mask]
        if len(vals) > 300:
            vals = np.random.choice(vals, 300, replace=False)
        data.extend([{'Category': cat, 'Score': v} for v in vals])
    
    df_plot = pd.DataFrame(data)
    sns.boxplot(data=df_plot, x='Category', y='Score', ax=ax, 
                palette=['green', 'red', 'gray'])
    ax.set_ylabel(f"|{best_method}| Score")
    ax.set_title("Score Distribution by Confidence Level")
    
    plt.tight_layout()
    plt.savefig(output_dir / "stratified_fp_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  ✓ Saved: {output_dir / 'stratified_fp_analysis.png'}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("LEIFER EXTENDED ANALYSIS")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading aligned Leifer atlases...")
    wt_data, unc31_data, neurons = load_aligned_atlases()
    print(f"  Loaded WT and unc-31 atlases, {len(neurons)} neurons")
    
    print("\n[2/6] Loading Cook anatomical connectome...")
    connectome = load_cook_connectome(neurons)
    print(f"  Loaded connectome shape: {connectome.shape}")
    
    print("\n[3/6] Loading connectivity matrices...")
    connectivity = load_connectivity_matrices(neurons)
    print(f"  Loaded {len(connectivity)} connectivity methods")
    
    print("\n[4/6] Loading time-varying connectivity...")
    time_varying = load_time_varying_connectivity(neurons, window_size=100)
    print(f"  Loaded {len(time_varying)} time-varying methods")
    
    # Run analyses
    all_results = {}
    
    print("\n" + "=" * 80)
    all_results["wt_only_edges"] = analyze_wt_only_edges(
        wt_data, unc31_data, connectivity, OUTPUT_DIR
    )
    
    print("\n" + "=" * 80)
    all_results["dynamic_collapse"] = analyze_dynamic_collapse(
        time_varying, wt_data, OUTPUT_DIR
    )
    
    print("\n" + "=" * 80)
    all_results["path2"] = analyze_path2(
        connectivity, wt_data, connectome, OUTPUT_DIR
    )
    
    print("\n" + "=" * 80)
    all_results["stratified_fp"] = analyze_stratified_fp(
        connectivity, wt_data, OUTPUT_DIR
    )
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    with open(OUTPUT_DIR / "extended_analysis_results.json", 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)
    print(f"  ✓ Saved: {OUTPUT_DIR / 'extended_analysis_results.json'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n1. WT-only edges (extrasynaptic-dependent):")
    print(f"   Found {all_results['wt_only_edges']['n_wt_only']} edges in WT but not unc-31")
    
    if all_results['dynamic_collapse']:
        print("\n2. Dynamic collapse:")
        for method, collapses in all_results['dynamic_collapse'].get('methods', {}).items():
            best_collapse = max(collapses.items(), key=lambda x: x[1].get('AUROC', 0))
            print(f"   {method}: best collapse = {best_collapse[0]} (AUROC={best_collapse[1]['AUROC']:.3f})")
    
    print("\n3. Path-2 analysis:")
    print(f"   Functional edges with path-2 support: {all_results['path2']['n_path2_func_with_anat']}")
    print(f"   Functional edges with no anatomical support: {all_results['path2']['n_path2_func_no_anat']}")
    
    print("\n4. Stratified FP analysis:")
    print(f"   Confirmed negatives (q_eq < 0.01): {all_results['stratified_fp']['counts']['confirmed_negative']}")
    
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

