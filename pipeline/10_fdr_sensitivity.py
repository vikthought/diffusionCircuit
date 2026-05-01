#!/usr/bin/env python3
"""
10_fdr_sensitivity.py
=====================

Analyze sensitivity of SBTG results to FDR control parameters:
- BH (Benjamini-Hochberg) vs BY (Benjamini-Yekutieli)
- Different alpha values (0.05, 0.10, 0.15, 0.20, 0.25)

This reuses a single trained model's p-values and applies different FDR thresholds
to understand the trade-off between edge detection power and false discovery control.

Usage:
    python pipeline/10_fdr_sensitivity.py

Outputs:
    results/figures/summary/fig15_fdr_sensitivity.png
    results/tables/fdr_sensitivity.csv
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import false_discovery_control

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
SBTG_DIR = RESULTS_DIR / "sbtg_training"
DATASETS_DIR = RESULTS_DIR / "intermediate" / "datasets"
CONNECTOME_DIR = RESULTS_DIR / "intermediate" / "connectome"
OUTPUT_DIR = RESULTS_DIR / "figures" / "summary"
TABLES_DIR = RESULTS_DIR / "tables"


def fdr_control(pvalues: np.ndarray, alpha: float, method: str = "bh") -> np.ndarray:
    """
    Apply FDR control to p-values.
    
    Args:
        pvalues: 1D array of p-values
        alpha: FDR threshold
        method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)
    
    Returns:
        Boolean array indicating which hypotheses are rejected
    """
    if len(pvalues) == 0:
        return np.array([], dtype=bool)
    
    # Handle NaN
    valid = ~np.isnan(pvalues)
    if valid.sum() == 0:
        return np.zeros(len(pvalues), dtype=bool)
    
    pvals_valid = pvalues[valid]
    
    # Use scipy's implementation
    if method == "bh":
        adjusted = false_discovery_control(pvals_valid, method='bh')
    elif method == "by":
        adjusted = false_discovery_control(pvals_valid, method='by')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reject where adjusted p-value < alpha
    reject_valid = adjusted < alpha
    
    # Map back to full array
    reject = np.zeros(len(pvalues), dtype=bool)
    reject[valid] = reject_valid
    
    return reject


def load_model_with_pvalues():
    """Find and load a model that has p-values saved."""
    models_dir = SBTG_DIR / "models"
    
    # Look for models with p_mean saved
    for f in sorted(models_dir.glob("*.npz"), key=lambda x: -x.stat().st_mtime):
        data = np.load(f, allow_pickle=True)
        if 'p_mean' in data.files:
            print(f"Using model: {f.name}")
            return {
                'path': f,
                'p_mean': data['p_mean'],
                'p_volatility': data.get('p_volatility', None),
                'mu_hat': data['mu_hat'],
                'sign_adj_original': data['sign_adj'],
                'hyperparams': json.loads(str(data['hyperparams'])) if 'hyperparams' in data else {}
            }
    
    return None


def apply_fdr_and_evaluate(p_mean, p_vol, mu_hat, method, alpha):
    """
    Apply FDR control with given parameters and compute metrics.
    
    Returns dict with edge counts and evaluation metrics.
    """
    n = p_mean.shape[0]
    mask = ~np.eye(n, dtype=bool)
    
    # Apply mean test FDR
    p_flat = p_mean[mask]
    reject_mean = fdr_control(p_flat, alpha, method)
    
    # Reconstruct sign_adj
    sign_adj = np.zeros((n, n), dtype=int)
    sign_adj[mask] = reject_mean * np.sign(mu_hat[mask])
    
    mean_edges = int(np.abs(sign_adj).sum())
    exc_edges = int((sign_adj > 0).sum())
    inh_edges = int((sign_adj < 0).sum())
    
    # Volatility test (on edges not selected by mean)
    vol_edges = 0
    if p_vol is not None:
        eligible = mask & (sign_adj == 0)
        p_vol_eligible = p_vol[eligible]
        reject_vol = fdr_control(p_vol_eligible, alpha, method)
        vol_edges = int(reject_vol.sum())
    
    combined_edges = mean_edges + vol_edges
    
    # Evaluate against Cook connectome
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import spearmanr
    
    # Get common neurons
    A_struct = np.load(CONNECTOME_DIR / "A_struct.npy")
    with open(CONNECTOME_DIR / "nodes.json") as f:
        struct_nodes = json.load(f)
    with open(DATASETS_DIR / "full_traces_imputed" / "neuron_names.json") as f:
        model_nodes = json.load(f)
    
    common = [n for n in model_nodes if n in struct_nodes]
    if len(common) < 10:
        return {
            'method': method,
            'alpha': alpha,
            'mean_edges': mean_edges,
            'exc_edges': exc_edges,
            'inh_edges': inh_edges,
            'vol_edges': vol_edges,
            'combined_edges': combined_edges,
            'auroc': np.nan,
            'auprc': np.nan,
            'spearman': np.nan
        }
    
    model_idx = [model_nodes.index(n) for n in common]
    struct_idx = [struct_nodes.index(n) for n in common]
    
    mu_aligned = np.abs(mu_hat[np.ix_(model_idx, model_idx)])
    gt_aligned = A_struct[np.ix_(struct_idx, struct_idx)]
    sign_aligned = sign_adj[np.ix_(model_idx, model_idx)]
    
    # Flatten excluding diagonal
    mask_common = ~np.eye(len(common), dtype=bool)
    y_score = mu_aligned[mask_common]
    y_true = (gt_aligned[mask_common] > 0).astype(int)
    sig_mask = (sign_aligned[mask_common] != 0)
    
    # AUROC on all pairs (using continuous scores)
    auroc = roc_auc_score(y_true, y_score) if y_true.sum() > 0 else 0.5
    auprc = average_precision_score(y_true, y_score) if y_true.sum() > 0 else 0.0
    
    # Spearman on significant edges only
    if sig_mask.sum() > 10:
        rho, _ = spearmanr(y_score[sig_mask], gt_aligned[mask_common][sig_mask])
    else:
        rho = np.nan
    
    return {
        'method': method,
        'alpha': alpha,
        'mean_edges': mean_edges,
        'exc_edges': exc_edges,
        'inh_edges': inh_edges,
        'vol_edges': vol_edges,
        'combined_edges': combined_edges,
        'edge_density': combined_edges / (n * (n-1)),
        'auroc': auroc,
        'auprc': auprc,
        'spearman': rho
    }


def create_sensitivity_figure(results_df):
    """Create comprehensive FDR sensitivity figure."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    bh_df = results_df[results_df['method'] == 'bh']
    by_df = results_df[results_df['method'] == 'by']
    
    # =========================================================================
    # Panel A: Edge count vs alpha
    # =========================================================================
    ax1 = axes[0, 0]
    ax1.plot(bh_df['alpha'], bh_df['combined_edges'], 'o-', color='#3498db', 
             linewidth=2, markersize=8, label='BH')
    ax1.plot(by_df['alpha'], by_df['combined_edges'], 's-', color='#e74c3c', 
             linewidth=2, markersize=8, label='BY')
    ax1.set_xlabel('FDR α', fontsize=12)
    ax1.set_ylabel('Number of Edges', fontsize=12)
    ax1.set_title('A. Edge Count vs FDR Threshold', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    # =========================================================================
    # Panel B: AUROC vs alpha
    # =========================================================================
    ax2 = axes[0, 1]
    ax2.plot(bh_df['alpha'], bh_df['auroc'], 'o-', color='#3498db', 
             linewidth=2, markersize=8, label='BH')
    ax2.plot(by_df['alpha'], by_df['auroc'], 's-', color='#e74c3c', 
             linewidth=2, markersize=8, label='BY')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax2.set_xlabel('FDR α', fontsize=12)
    ax2.set_ylabel('AUROC', fontsize=12)
    ax2.set_title('B. AUROC vs FDR Threshold', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim(0.45, 0.65)
    ax2.grid(alpha=0.3)
    
    # =========================================================================
    # Panel C: Spearman vs alpha
    # =========================================================================
    ax3 = axes[0, 2]
    ax3.plot(bh_df['alpha'], bh_df['spearman'], 'o-', color='#3498db', 
             linewidth=2, markersize=8, label='BH')
    ax3.plot(by_df['alpha'], by_df['spearman'], 's-', color='#e74c3c', 
             linewidth=2, markersize=8, label='BY')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('FDR α', fontsize=12)
    ax3.set_ylabel('Spearman ρ (significant edges)', fontsize=12)
    ax3.set_title('C. Weight Correlation vs FDR Threshold', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(alpha=0.3)
    
    # =========================================================================
    # Panel D: E:I ratio vs alpha
    # =========================================================================
    ax4 = axes[1, 0]
    bh_ratio = bh_df['exc_edges'] / np.maximum(bh_df['inh_edges'], 1)
    by_ratio = by_df['exc_edges'] / np.maximum(by_df['inh_edges'], 1)
    ax4.plot(bh_df['alpha'], bh_ratio, 'o-', color='#3498db', 
             linewidth=2, markersize=8, label='BH')
    ax4.plot(by_df['alpha'], by_ratio, 's-', color='#e74c3c', 
             linewidth=2, markersize=8, label='BY')
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Balanced')
    ax4.set_xlabel('FDR α', fontsize=12)
    ax4.set_ylabel('E:I Ratio', fontsize=12)
    ax4.set_title('D. E:I Ratio vs FDR Threshold', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(alpha=0.3)
    
    # =========================================================================
    # Panel E: BH vs BY comparison at each alpha
    # =========================================================================
    ax5 = axes[1, 1]
    alphas = bh_df['alpha'].values
    width = 0.015
    x = np.array(alphas)
    
    ax5.bar(x - width/2, bh_df['combined_edges'], width, color='#3498db', 
            edgecolor='black', label='BH')
    ax5.bar(x + width/2, by_df['combined_edges'], width, color='#e74c3c', 
            edgecolor='black', label='BY')
    ax5.set_xlabel('FDR α', fontsize=12)
    ax5.set_ylabel('Number of Edges', fontsize=12)
    ax5.set_title('E. BH vs BY Edge Discovery', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.set_xticks(alphas)
    
    # =========================================================================
    # Panel F: Summary text
    # =========================================================================
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Find best configurations
    best_auroc = results_df.loc[results_df['auroc'].idxmax()]
    best_spearman = results_df.loc[results_df['spearman'].idxmax()]
    
    summary_text = f"""
    FDR Sensitivity Analysis Summary
    ════════════════════════════════════
    
    Methods Compared:
    • BH: Benjamini-Hochberg (assumes independence)
    • BY: Benjamini-Yekutieli (arbitrary dependence)
    
    Alpha Values Tested:
    • 0.05, 0.10, 0.15, 0.20, 0.25
    
    Key Findings:
    ─────────────────────────────────
    Best AUROC:
      {best_auroc['method'].upper()} @ α={best_auroc['alpha']:.2f}
      AUROC = {best_auroc['auroc']:.3f}, Edges = {int(best_auroc['combined_edges'])}
    
    Best Spearman:
      {best_spearman['method'].upper()} @ α={best_spearman['alpha']:.2f}
      ρ = {best_spearman['spearman']:.3f}, Edges = {int(best_spearman['combined_edges'])}
    
    Current Config: BY @ α=0.20
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
    
    plt.suptitle('Figure 15: FDR Control Sensitivity Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = OUTPUT_DIR / "fig15_fdr_sensitivity.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("FDR SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Load model with p-values
    print("\n[1] Loading model with p-values...")
    model_data = load_model_with_pvalues()
    
    if model_data is None:
        print("ERROR: No model with p-values found. Run 02_train_sbtg.py first.")
        return
    
    print(f"  Original config: {model_data['hyperparams'].get('fdr_method', 'unknown')} @ α={model_data['hyperparams'].get('fdr_alpha', 'unknown')}")
    
    # Run sensitivity analysis
    print("\n[2] Running FDR sensitivity sweep...")
    
    methods = ['bh', 'by']
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    results = []
    
    for method in methods:
        for alpha in alphas:
            result = apply_fdr_and_evaluate(
                model_data['p_mean'],
                model_data['p_volatility'],
                model_data['mu_hat'],
                method,
                alpha
            )
            results.append(result)
            print(f"  {method.upper()} α={alpha:.2f}: {result['combined_edges']} edges, AUROC={result['auroc']:.3f}, ρ={result['spearman']:.3f}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLES_DIR / "fdr_sensitivity.csv"
    results_df.to_csv(table_path, index=False)
    print(f"\nSaved: {table_path}")
    
    # Create figure
    print("\n[3] Creating sensitivity figure...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_sensitivity_figure(results_df)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nEdge counts by configuration:")
    pivot = results_df.pivot(index='alpha', columns='method', values='combined_edges')
    print(pivot.to_string())
    
    print("\nAUROC by configuration:")
    pivot_auroc = results_df.pivot(index='alpha', columns='method', values='auroc')
    print(pivot_auroc.to_string())
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
