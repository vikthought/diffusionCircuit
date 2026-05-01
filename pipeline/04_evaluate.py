#!/usr/bin/env python3
"""
SCRIPT 04: Evaluate Connectivity
================================

Unified evaluation of SBTG and baseline methods against ground truth:
- Cook_Synapses_2019 structural benchmark (anatomy)
- Randi_Optogenetics_2023 functional benchmark (optogenetic stimulation)

Usage:
    python pipeline/04_evaluate.py --against cook
    python pipeline/04_evaluate.py --against leifer
    python pipeline/04_evaluate.py --against both --plot

Outputs:
    results/evaluation/report.csv
    results/evaluation/figures/*.png (with --plot)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.io import load_structural_connectome as _load_structural_connectome
from pipeline.utils.metrics import compute_weight_correlation

# =============================================================================
# CONFIGURATION
# =============================================================================

SBTG_DIR = PROJECT_ROOT / "results" / "sbtg_training" / "models"
BASELINES_DIR = PROJECT_ROOT / "results" / "baselines"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"
LEIFER_DIR = PROJECT_ROOT / "results" / "leifer_evaluation"
OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"

# Edge density for F1 comparison (match number of predicted edges)
EDGE_DENSITY = 0.15


# =============================================================================
# DATA LOADING
# =============================================================================

DATASETS_DIR = PROJECT_ROOT / "results" / "intermediate" / "datasets"


def load_cook_connectome() -> Tuple[np.ndarray, List[str]]:
    """Load Cook_Synapses_2019 structural benchmark."""
    A_struct, nodes, _ = _load_structural_connectome(CONNECTOME_DIR)
    return A_struct, nodes


from pipeline.utils.leifer import load_leifer_atlas_data

def load_leifer_atlas() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """Load aligned Randi_Optogenetics_2023 functional atlas using shared utility."""
    return load_leifer_atlas_data()


def load_leifer_dff() -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """Load functional-atlas dFF amplitudes for weight correlation analysis."""
    wt_file = LEIFER_DIR / "aligned_atlas_wild-type.npz"
    if not wt_file.exists():
        return None, None
    data = np.load(wt_file, allow_pickle=True)
    return data.get('dff', None), list(data.get('neuron_order', []))


def load_model_neuron_names(dataset: str = "nacl") -> List[str]:
    """Load neuron names from prepared dataset.
    
    Args:
        dataset: Dataset name (e.g., 'nacl', 'full_traces_imputed')
    """
    # Try standardization.json first
    std_file = DATASETS_DIR / dataset / "standardization.json"
    if std_file.exists():
        with open(std_file) as f:
            data = json.load(f)
            return data.get('node_order', [])
    
    # Fallback to neuron_names.json
    names_file = DATASETS_DIR / dataset / "neuron_names.json"
    if names_file.exists():
        with open(names_file) as f:
            return json.load(f)
    return []


def load_sbtg_results(dataset: str = "nacl") -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Load SBTG model results with neuron names.
    
    Args:
        dataset: Dataset name to filter models (e.g., 'nacl', 'full_traces_imputed')
    
    Returns:
        results: Dict[name -> weight matrix (abs(mu_hat))]
        significance: Dict[name -> significance mask]
        neuron_names: List of neuron names
        
    For models with volatility_adj, creates separate entries:
        - name: Uses mean test edges only (sign_adj != 0)
        - name_combined: Uses mean + volatility edges
        - name_vol_only: Uses volatility-only edges
    """
    results = {}
    significance = {}
    neuron_names = load_model_neuron_names(dataset)
    
    if not SBTG_DIR.exists():
        return results, significance, neuron_names
    
    # Filter by dataset name to avoid mixing disparate models
    # Supports both stimulus-specific (e.g., 'nacl') and full-trace datasets (e.g., 'full_traces_imputed')
    pattern = f"*{dataset}*.npz"
    model_files = list(SBTG_DIR.glob(pattern))
    
    if not model_files:
        print(f"  No SBTG models found for dataset '{dataset}' (matching {pattern})")
        print(f"  Available files: {[f.name for f in SBTG_DIR.glob('*.npz')]}")
    
    for f in model_files:
        data = np.load(f, allow_pickle=True)
        name = f.stem
        if 'sign_adj' not in data:
            continue
            
        sign_adj = data['sign_adj']
        mu_hat = data['mu_hat'] if 'mu_hat' in data else (sign_adj != 0).astype(float)
        
        # Mean test edges only (original behavior)
        mean_mask = (sign_adj != 0)
        results[f"sbtg_{name}"] = np.abs(mu_hat)
        significance[f"sbtg_{name}"] = mean_mask
        
        # If volatility_adj is available, create additional entries
        if 'volatility_adj' in data:
            vol_adj = data['volatility_adj'].astype(bool)
            vol_only_mask = vol_adj & ~mean_mask
            combined_mask = mean_mask | vol_adj
            
            # Combined: mean + volatility edges
            results[f"sbtg_{name}_combined"] = np.abs(mu_hat)  # Same weights, different mask
            significance[f"sbtg_{name}_combined"] = combined_mask
            
            # Volatility-only edges (for separate analysis)
            if vol_only_mask.sum() > 0:
                results[f"sbtg_{name}_vol_only"] = np.abs(mu_hat)
                significance[f"sbtg_{name}_vol_only"] = vol_only_mask
    
    return results, significance, neuron_names


def load_baseline_results(stimulus: str = "nacl") -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load baseline method results with neuron names.
    
    Note: Baselines are always stimulus-specific (e.g., 'nacl'), not full traces.
    """
    results = {}
    neuron_names = load_model_neuron_names(stimulus)
    
    for f in BASELINES_DIR.glob(f"*_{stimulus}.npz"):
        data = np.load(f)
        method = f.stem.replace(f"_{stimulus}", "")
        results[method] = np.abs(data['matrix'])
    
    return results, neuron_names


def align_matrices(
    scores: np.ndarray,
    score_neurons: List[str],
    ground_truth: np.ndarray,
    gt_neurons: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align score matrix to ground truth by finding common neurons.
    
    Returns aligned matrices on the intersection of neurons.
    """
    # Find common neurons
    common = [n for n in score_neurons if n in gt_neurons]
    
    if len(common) == 0:
        return np.array([]), np.array([])
    
    # Get indices
    score_idx = [score_neurons.index(n) for n in common]
    gt_idx = [gt_neurons.index(n) for n in common]
    
    # Extract submatrices
    scores_aligned = scores[np.ix_(score_idx, score_idx)]
    gt_aligned = ground_truth[np.ix_(gt_idx, gt_idx)]
    
    return scores_aligned, gt_aligned


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_vs_binary(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    name: str = "method"
) -> Dict:
    """
    Evaluate connectivity scores against binary ground truth.
    
    Args:
        scores: (n, n) continuous connectivity scores
        ground_truth: (n, n) binary adjacency matrix
        
    Returns:
        Dict with AUROC, AUPRC, F1, precision, recall
    """
    # Flatten, exclude diagonal
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    y_score = np.abs(scores[mask])
    y_true = (ground_truth[mask] > 0).astype(int)
    
    # Handle NaN
    valid = ~np.isnan(y_score) & ~np.isnan(y_true)
    y_score = y_score[valid]
    y_true = y_true[valid]
    
    if len(y_true) == 0 or y_true.sum() == 0:
        return {'name': name, 'auroc': 0.5, 'auprc': 0.0, 'f1': 0.0}
    
    # Compute metrics
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    
    # F1 at matched density
    n_positive = y_true.sum()
    
    # Check if scores are already binary/discrete (e.g. from FDR control)
    unique_scores = np.unique(y_score)
    is_discrete = len(unique_scores) <= 3 and np.all(np.isin(unique_scores, [0, 1]))
    
    if is_discrete:
        # For discrete/SBTG graphs, use the graph AS IS (don't re-threshold)
        y_pred = y_score.astype(int)
    else:
        # For continuous scores (baselines), threshold to match density
        threshold = np.percentile(y_score, 100 * (1 - n_positive / len(y_score)))
        y_pred = (y_score >= threshold).astype(int)
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'name': name,
        'auroc': auroc,
        'auprc': auprc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'n_true_edges': int(y_true.sum()),
        'n_predicted': int(y_pred.sum())
    }


def evaluate_vs_leifer(
    scores: np.ndarray,
    q: np.ndarray,
    name: str = "method",
    alpha: float = 0.05
) -> Dict:
    """
    Evaluate against the functional benchmark atlas.
    
    q-values are interpreted as: q < alpha indicates a functional connection.
    """
    n = scores.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    y_score = np.abs(scores[mask])
    y_true = (q[mask] < alpha).astype(int)
    
    # Handle NaN
    valid = ~np.isnan(y_score) & ~np.isnan(q[mask])
    y_score = y_score[valid]
    y_true = y_true[valid]
    
    if len(y_true) == 0 or y_true.sum() == 0:
        return {'name': name, 'auroc': 0.5, 'auprc': 0.0}
    
    return {
        'name': name,
        'auroc': roc_auc_score(y_true, y_score),
        'auprc': average_precision_score(y_true, y_score),
        'n_functional_edges': int(y_true.sum())
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_roc_curves(
    results: Dict[str, np.ndarray],
    ground_truth: np.ndarray,
    output_path: Path,
    title: str = "ROC Curves"
):
    """Plot ROC curves for all methods."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n = ground_truth.shape[0]
    mask = ~np.eye(n, dtype=bool)
    y_true = (ground_truth[mask] > 0).astype(int)
    
    for name, scores in results.items():
        y_score = np.abs(scores[mask])
        valid = ~np.isnan(y_score)
        
        if valid.sum() == 0:
            continue
        
        fpr, tpr, _ = roc_curve(y_true[valid], y_score[valid])
        auc = roc_auc_score(y_true[valid], y_score[valid])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_comparison_bar(
    metrics_df: pd.DataFrame,
    output_path: Path
):
    """Bar chart comparing methods."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    for i, metric in enumerate(['auroc', 'auprc', 'f1']):
        if metric not in metrics_df.columns:
            continue
        ax = axes[i]
        sns.barplot(data=metrics_df, x='name', y=metric, ax=ax)
        ax.set_title(metric.upper())
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Connectivity Methods")
    parser.add_argument('--against', choices=['cook', 'leifer', 'both'], 
                        default='both', help='Ground truth to evaluate against')
    parser.add_argument('--stimulus', default='nacl', help='Stimulus for baselines (e.g., nacl, butanone)')
    parser.add_argument('--dataset', default=None, 
                        help='Dataset name for SBTG models (e.g., full_traces_imputed). '
                             'If not specified, uses --stimulus value.')
    parser.add_argument('--plot', action='store_true', help='Generate visualizations')
    args = parser.parse_args()
    
    # Determine dataset for SBTG models
    sbtg_dataset = args.dataset if args.dataset else args.stimulus
    
    print("="*60)
    print("Connectivity Evaluation")
    print("="*60)
    print(f"  SBTG dataset: {sbtg_dataset}")
    print(f"  Baseline stimulus: {args.stimulus}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    if args.plot:
        figures_dir.mkdir(exist_ok=True)
    
    # Load methods with neuron names
    print("\n[1] Loading connectivity matrices...")
    all_methods = {}
    model_neurons = []
    
    sbtg_results, sbtg_significance, sbtg_neurons = load_sbtg_results(sbtg_dataset)
    if sbtg_results:
        print(f"  SBTG: {len(sbtg_results)} models ({len(sbtg_neurons)} neurons)")
        all_methods.update(sbtg_results)
        model_neurons = sbtg_neurons
    
    baseline_results, baseline_neurons = load_baseline_results(args.stimulus)
    if baseline_results:
        print(f"  Baselines: {len(baseline_results)} methods ({len(baseline_neurons)} neurons)")
        all_methods.update(baseline_results)
        if not model_neurons:
            model_neurons = baseline_neurons
    
    if not all_methods:
        print("  No methods found. Run training scripts first.")
        return
    
    all_metrics = []
    
    # Evaluate vs Cook
    if args.against in ['cook', 'both']:
        print("\n[2] Evaluating vs Cook Connectome...")
        A_cook, cook_neurons = load_cook_connectome()
        print(f"  Connectome: {len(cook_neurons)} neurons, {(A_cook > 0).sum()} edges")
        
        # Find intersection
        common = [n for n in model_neurons if n in cook_neurons]
        print(f"  Common neurons: {len(common)}")
        
        if len(common) > 0:
            for name, scores in all_methods.items():
                scores_aligned, cook_aligned = align_matrices(
                    scores, model_neurons, A_cook, cook_neurons
                )
                
                if scores_aligned.size == 0:
                    print(f"  Skipping {name}: no overlap")
                    continue
                
                metrics = evaluate_vs_binary(scores_aligned, cook_aligned, name)
                metrics['benchmark'] = 'cook'
                metrics['n_neurons'] = len(common)
                
                # Weight correlation (comparing predicted weights to synapse counts)
                # Use significance mask for SBTG models to only correlate on predicted edges
                sig_mask = sbtg_significance.get(name, None)
                if sig_mask is not None:
                    # Align significance mask to common neurons
                    sig_idx = [model_neurons.index(n) for n in common]
                    sig_mask_aligned = sig_mask[np.ix_(sig_idx, sig_idx)]
                else:
                    sig_mask_aligned = None
                weight_corr = compute_weight_correlation(
                    scores_aligned, cook_aligned, exclude_diagonal=True,
                    significance_mask=sig_mask_aligned
                )
                metrics['spearman_r'] = weight_corr.get('spearman_rho', np.nan)
                metrics['r_squared'] = weight_corr.get('r_squared', np.nan)
                metrics['pearson_r'] = weight_corr.get('pearson_r', np.nan)
                
                all_metrics.append(metrics)
                print(f"  {name}: AUROC={metrics['auroc']:.3f}, AUPRC={metrics['auprc']:.3f}, F1={metrics['f1']:.3f}, Spearman={metrics['spearman_r']:.3f}")
    
    # Evaluate vs Leifer
    if args.against in ['leifer', 'both']:
        print("\n[3] Evaluating vs Leifer Atlas...")
        q, q_eq, leifer_neurons = load_leifer_atlas()
        
        if q is not None:
            print(f"  Atlas: {len(leifer_neurons)} neurons, {(q < 0.05).sum()} functional edges")
            
            common = [n for n in model_neurons if n in leifer_neurons]
            print(f"  Common neurons: {len(common)}")
            
            if len(common) > 0:
                for name, scores in all_methods.items():
                    scores_aligned, q_aligned = align_matrices(
                        scores, model_neurons, q, leifer_neurons
                    )
                    
                    if scores_aligned.size == 0:
                        print(f"  Skipping {name}: no overlap")
                        continue
                    
                    metrics = evaluate_vs_leifer(scores_aligned, q_aligned, name)
                    metrics['benchmark'] = 'leifer'
                    metrics['n_neurons'] = len(common)
                    
                    # Weight correlation against dff amplitudes
                    dff, dff_neurons = load_leifer_dff()
                    if dff is not None:
                        scores_dff, dff_aligned = align_matrices(
                            scores, model_neurons, dff, dff_neurons
                        )
                        if scores_dff.size > 0:
                            weight_corr = compute_weight_correlation(
                                scores_dff, dff_aligned, 
                                exclude_diagonal=True, only_overlapping=True
                            )
                            metrics['spearman_r'] = weight_corr.get('spearman_rho', np.nan)
                            metrics['r_squared'] = weight_corr.get('r_squared', np.nan)
                            metrics['pearson_r'] = weight_corr.get('pearson_r', np.nan)
                    
                    all_metrics.append(metrics)
                    print(f"  {name}: AUROC={metrics['auroc']:.3f}, AUPRC={metrics['auprc']:.3f}, Spearman={metrics.get('spearman_r', float('nan')):.3f}")
        else:
            print("  Leifer atlas not available")
    
    # Save results
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR / 'evaluation_results.csv'}")
        
        if args.plot:
            plot_comparison_bar(df, figures_dir / "comparison.png")
            print(f"Saved: {figures_dir / 'comparison.png'}")
    else:
        print("\nNo evaluations completed (check neuron overlap)")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
