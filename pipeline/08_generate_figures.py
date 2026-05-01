#!/usr/bin/env python3
"""
08_generate_figures.py
======================

Generate summary figures for the SBTG connectivity analysis.

Prerequisites:
- `results/intermediate/datasets/full_traces_imputed/` for imputation and phase figures
- `results/sbtg_training/` and `results/evaluation/` from scripts 02 and 04
- `results/sbtg_temporal/` or the ability to retrain phase models in this script

Figures:
- Fig 1: Data overview
- Fig 2: Imputation statistics
- Fig 3: Data expansion impact
- Fig 4: Score/Jacobian conceptual figure
- Fig 5: SBTG vs baselines comparison
- Fig 6: Predicted vs anatomical connectivity
- Fig 7: E:I ratio temporal dynamics
- Fig 8: Aggregate network graph (networkx)
- Fig 9: Temporal phase network graphs
- Fig 10-12: Weight/correlation diagnostics
- Fig 14-19: Mean/volatility and transfer/direct analyses

Usage:
    python pipeline/08_generate_figures.py
    python pipeline/08_generate_figures.py --quick  # Skip slow network graphs
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.display_names import FUNCTIONAL_LABEL, STRUCTURAL_LABEL

# Directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "figures" / "summary"
CONNECTOME_DIR = RESULTS_DIR / "intermediate" / "connectome"
DATASETS_DIR = RESULTS_DIR / "intermediate" / "datasets"
SBTG_DIR = RESULTS_DIR / "sbtg_training"


def ensure_output_dir():
    """Create output directory if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def generate_phase_sbtg_data(force_regenerate: bool = False):
    """
    Generate phase-specific SBTG data for Fig 7 and Fig 9.
    
    IMPORTANT: This function trains actual SBTG models on each temporal phase,
    rather than using simple Pearson correlation. This ensures that the E:I
    ratios and edge predictions are methodologically consistent with the main
    SBTG results.
    
    Args:
        force_regenerate: If True, regenerate even if results already exist
    """
    from pipeline.models.sbtg import SBTGStructuredVolatilityEstimator
    from pipeline.config import OPTIMIZED_HYPERPARAMS
    try:
        from pipeline.configs.phase_optimal_params import PHASE_OPTIMAL_PARAMS
    except ImportError:
        PHASE_OPTIMAL_PARAMS = {}
    
    output_dir = RESULTS_DIR / 'sbtg_temporal'
    phase_results_path = output_dir / 'phase_results.json'
    
    if phase_results_path.exists() and not force_regenerate:
        print("\n[Pre] Phase SBTG data already exists, skipping regeneration")
        print("      (use --regenerate-phases to force regeneration)")
        return
    
    print("\n[Pre] Training SBTG models for each temporal phase...")
    print("      This may take several minutes per phase.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load imputed data
    imputed_dir = DATASETS_DIR / 'full_traces_imputed'
    if not imputed_dir.exists():
        print("  ERROR: Imputed dataset not found, skipping phase SBTG generation")
        return
    
    # X_segments is (n_worms,) object array, each element is (n_frames, n_neurons)
    X_segments = np.load(imputed_dir / 'X_segments.npy', allow_pickle=True)
    with open(imputed_dir / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    n_neurons = len(neuron_names)
    n_worms = len(X_segments)
    fps = 4.0  # 4 Hz sampling
    
    # Phase definitions in seconds (from METHODS.md experiment protocol)
    # Recording structure: 240 seconds total
    # - Baseline: 0-60s
    # - Stimulus 1 (butanone): 60.5-70.5s (10 seconds)
    # - Inter-stimulus 1: 70.5-120.5s
    # - Stimulus 2 (pentanedione): 120.5-130.5s (10 seconds)
    # - Inter-stimulus 2: 130.5-180.5s
    # - Stimulus 3 (NaCl): 180.5-190.5s (10 seconds)
    phases = {
        'baseline': (0, 60),              # Baseline period (60s)
        'butanone': (60.5, 70.5),         # Stimulus 1 (10s window)
        'pentanedione': (120.5, 130.5),   # Stimulus 2 (10s window)
        'nacl': (180.5, 190.5),           # Stimulus 3 (10s window)
    }
    
    # Use hyperparameters from the 150-trial HP search (Jan 2026)
    # The evaluation shows: regime_gated performs best across all phases.
    
    phase_results = {}
    
    for phase_name, (start_s, end_s) in phases.items():
        print(f"\n  === Training SBTG for phase: {phase_name} ({start_s}-{end_s}s) ===")
        
        # Get optimal params for this phase (or fallback to generic optimized)
        phase_hyperparams = PHASE_OPTIMAL_PARAMS.get(phase_name, OPTIMIZED_HYPERPARAMS).copy()
        
        phase_hyperparams.update({
            'dsm_epochs': 200,       # Use more epochs to compensate for less data per phase
            'verbose': True,
            'inference_mode': 'in_sample',
        })
        
        start_frame = int(start_s * fps)
        end_frame = int(end_s * fps)
        
        # Collect data from all worms for this phase as separate segments
        phase_segments = []
        for worm_idx in range(n_worms):
            worm_data = X_segments[worm_idx]
            if worm_data is None:
                continue
            n_frames = worm_data.shape[0]
            # Ensure we don't go past the end of the data
            s = min(start_frame, n_frames - 1)
            e = min(end_frame, n_frames)
            if e - s >= 10:  # Need at least 10 frames for meaningful training
                segment = worm_data[s:e, :]
                # Remove NaN rows at start/end
                nan_rows = np.any(np.isnan(segment), axis=1)
                segment_clean = segment[~nan_rows]
                if segment_clean.shape[0] >= 5:
                    phase_segments.append(segment_clean)
        
        if len(phase_segments) == 0:
            print(f"    WARNING: No valid data for phase {phase_name}")
            phase_results[phase_name] = {
                'n_positive': 0, 'n_negative': 0, 'n_total': 0, 
                'ei_ratio': 0.0, 'method': 'sbtg'
            }
            np.save(output_dir / f'sign_adj_{phase_name}.npy', np.zeros((n_neurons, n_neurons)))
            continue
        
        total_frames = sum(seg.shape[0] for seg in phase_segments)
        print(f"    Data: {len(phase_segments)} worm segments, {total_frames} total frames")
        
        # Check if we have enough data for SBTG training
        min_windows = 50  # Minimum windows needed for reliable SBTG
        estimated_windows = total_frames - len(phase_segments)  # Approximate
        
        if estimated_windows < min_windows:
            print(f"    WARNING: Only ~{estimated_windows} windows (need {min_windows}+)")
            print(f"    Falling back to Pearson correlation for this phase")
            
            # Fallback to Pearson correlation
            X_phase = np.vstack(phase_segments)
            corr = np.corrcoef(X_phase.T)
            np.fill_diagonal(corr, 0)
            corr = np.nan_to_num(corr, nan=0.0)
            
            abs_corr = np.abs(corr)
            non_zero = abs_corr[abs_corr > 0]
            if len(non_zero) > 0:
                threshold = np.percentile(non_zero, 85)
                sign_adj = np.zeros_like(corr)
                sign_adj[abs_corr >= threshold] = np.sign(corr[abs_corr >= threshold])
            else:
                sign_adj = np.zeros_like(corr)
            
            method_used = 'pearson_fallback'
        else:
            # Train SBTG model on this phase
            try:
                estimator = SBTGStructuredVolatilityEstimator(**phase_hyperparams)
                result = estimator.fit(phase_segments)
                
                # Negate sign_adj to match the functional-atlas dFF polarity convention.
                # (same as in 02_train_sbtg.py)
                sign_adj = -result.sign_adj
                method_used = 'sbtg'
                
                print(f"    SBTG training complete!")
                
            except Exception as e:
                print(f"    ERROR: SBTG training failed: {e}")
                print(f"    Falling back to Pearson correlation")
                
                X_phase = np.vstack(phase_segments)
                corr = np.corrcoef(X_phase.T)
                np.fill_diagonal(corr, 0)
                corr = np.nan_to_num(corr, nan=0.0)
                
                abs_corr = np.abs(corr)
                non_zero = abs_corr[abs_corr > 0]
                if len(non_zero) > 0:
                    threshold = np.percentile(non_zero, 85)
                    sign_adj = np.zeros_like(corr)
                    sign_adj[abs_corr >= threshold] = np.sign(corr[abs_corr >= threshold])
                else:
                    sign_adj = np.zeros_like(corr)
                
                method_used = 'pearson_fallback'
        
        # Save phase adjacency
        np.save(output_dir / f'sign_adj_{phase_name}.npy', sign_adj)
        
        # Compute E:I stats from SBTG output
        n_positive = int((sign_adj > 0).sum())
        n_negative = int((sign_adj < 0).sum())
        
        # E:I ratio calculation
        if n_negative > 0:
            ei_ratio = n_positive / n_negative
        elif n_positive > 0:
            ei_ratio = float('inf')  # All excitatory
        else:
            ei_ratio = 0.0  # No edges
        
        phase_results[phase_name] = {
            'n_positive': n_positive,
            'n_negative': n_negative,
            'n_total': n_positive + n_negative,
            'ei_ratio': round(ei_ratio, 4) if ei_ratio != float('inf') else 'inf',
            'method': method_used,
        }
        
        ei_display = f"{ei_ratio:.3f}" if ei_ratio != float('inf') else "∞"
        print(f"    Result: {n_positive + n_negative} edges (E:{n_positive}, I:{n_negative}), E:I = {ei_display}")
        print(f"    Method: {method_used}")
    
    # Save phase results
    with open(phase_results_path, 'w') as f:
        json.dump(phase_results, f, indent=2)
    
    print(f"\n  ✓ Saved phase_results.json and sign_adj_*.npy to {output_dir}")


# =============================================================================
# FIGURE 2: IMPUTATION STATISTICS
# =============================================================================

def create_fig2_imputation_stats():
    """Create figure showing imputation statistics."""
    print("\n[Fig 2] Creating imputation statistics...")
    
    # Load imputation log
    imputation_log = pd.read_csv(DATASETS_DIR / 'full_traces_imputed' / 'imputation_log.csv')
    
    # Load neuron names
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Colors
    color_before = '#e74c3c'
    color_after = '#2ecc71'
    
    # Panel A: Worm coverage before/after
    ax1 = axes[0]
    complete_before = 6
    complete_after = 20
    total_worms = 21
    
    categories = ['Complete\nData', 'Partial\nData', 'Excluded']
    before_vals = [complete_before, total_worms - complete_before - 1, 1]
    after_vals = [complete_after, 0, 1]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, before_vals, width, label='Before Imputation', 
                    color=color_before, edgecolor='black')
    bars2 = ax1.bar(x + width/2, after_vals, width, label='After Imputation', 
                    color=color_after, edgecolor='black')
    
    ax1.set_ylabel('Number of Worms', fontsize=12)
    ax1.set_title('A. Worm Data Completeness', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 22)
    
    for bar in bars1:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                     int(bar.get_height()), ha='center', va='bottom', fontsize=11)
    for bar in bars2:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                     int(bar.get_height()), ha='center', va='bottom', fontsize=11)
    
    # Panel B: Imputation summary
    ax2 = axes[1]
    neurons_imputed = imputation_log['neuron'].nunique()
    worms_with_imputation = imputation_log['worm_idx'].nunique()
    total_imputations = len(imputation_log)
    
    summary_data = {
        'Total neurons': len(neuron_names),
        'Neurons needing imputation': neurons_imputed,
        'Worms receiving imputed data': worms_with_imputation,
        'Total worm×neuron imputations': total_imputations,
    }
    
    y_pos = np.arange(len(summary_data))
    values = list(summary_data.values())
    labels = list(summary_data.keys())
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    bars = ax2.barh(y_pos, values, color=colors, edgecolor='black', height=0.6)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel('Count', fontsize=12)
    ax2.set_title('B. Imputation Statistics', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 str(val), ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlim(0, max(values) * 1.2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_imputation_stats.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig2_imputation_stats.png")
    plt.close()


# =============================================================================
# FIGURE 3: DATA EXPANSION IMPACT
# =============================================================================

def create_fig3_data_expansion():
    """Create figure showing data expansion impact."""
    print("\n[Fig 3] Creating data expansion impact...")
    
    metrics_before = {'worms': 6, 'windows': 234, 'auroc_cook': 0.525, 'auroc_leifer': 0.529}
    metrics_after = {'worms': 20, 'windows': 18546, 'auroc_cook': 0.584, 'auroc_leifer': 0.643}
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    
    color_before = '#e74c3c'
    color_after = '#2ecc71'
    
    # Panel A: Worms
    ax1 = axes[0]
    categories = ['Before\nImputation', 'After\nImputation']
    values = [metrics_before['worms'], metrics_after['worms']]
    bars = ax1.bar(categories, values, color=[color_before, color_after], edgecolor='black', linewidth=1)
    ax1.set_ylabel('Number of Worms', fontsize=12)
    ax1.set_title('A. Training Dataset Size', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 25)
    
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(val), ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax1.annotate('', xy=(1, 20), xytext=(0, 6),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax1.text(0.5, 14, '3.3×', fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    
    # Panel B: Training Windows
    ax2 = axes[1]
    values = [metrics_before['windows'], metrics_after['windows']]
    bars = ax2.bar(categories, values, color=[color_before, color_after], edgecolor='black', linewidth=1)
    ax2.set_ylabel('Training Windows', fontsize=12)
    ax2.set_title('B. Training Data Volume', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 22000)
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                 f'{val:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.annotate('', xy=(1, 18546), xytext=(0, 234),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.text(0.5, 10000, '79×', fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    
    # Panel C: AUROC Improvement
    ax3 = axes[2]
    width = 0.35
    x = np.arange(2)
    benchmarks = [f'{STRUCTURAL_LABEL}\n(Anatomy)', f'{FUNCTIONAL_LABEL}\n(Function)']
    
    before_auroc = [metrics_before['auroc_cook'], metrics_before['auroc_leifer']]
    after_auroc = [metrics_after['auroc_cook'], metrics_after['auroc_leifer']]
    
    ax3.bar(x - width/2, before_auroc, width, label='Before', color=color_before, edgecolor='black', linewidth=0.5)
    ax3.bar(x + width/2, after_auroc, width, label='After', color=color_after, edgecolor='black', linewidth=0.5)
    
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    ax3.set_ylabel('AUROC', fontsize=12)
    ax3.set_title('C. Model Performance', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(benchmarks, fontsize=11)
    ax3.set_ylim(0.45, 0.7)
    ax3.legend(loc='upper left', fontsize=10)
    
    for i, (b, a) in enumerate(zip(before_auroc, after_auroc)):
        ax3.annotate(f'+{a-b:.2f}', xy=(i + width/2, a + 0.01), fontsize=10, 
                     ha='center', va='bottom', color='#27ae60', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_data_expansion_impact.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig3_data_expansion_impact.png")
    plt.close()


# =============================================================================
# FIGURE 5: SBTG VS BASELINES
# =============================================================================

def create_fig5_sbtg_vs_baselines():
    """Create figure comparing SBTG to baselines."""
    print("\n[Fig 5] Creating SBTG vs baselines comparison...")
    
    df = pd.read_csv(RESULTS_DIR / 'evaluation' / 'evaluation_results.csv')
    
    # Find SBTG imputed_best model dynamically
    sbtg_rows = df[df['name'].str.contains('imputed_best', na=False)]
    if len(sbtg_rows) == 0:
        # Fallback: find any SBTG model
        sbtg_rows = df[df['name'].str.startswith('sbtg_', na=False)]
    
    if len(sbtg_rows) == 0:
        print("  WARNING: No SBTG models found in evaluation results")
        return
    
    # Get the first SBTG model name
    sbtg_name = sbtg_rows['name'].iloc[0]
    print(f"  Using SBTG model: {sbtg_name}")
    
    # Methods to compare
    baseline_methods = ['pearson', 'crosscorr', 'granger', 'partial_corr', 'glasso']
    methods_cook = [sbtg_name] + baseline_methods
    methods_display = ['SBTG\n(imputed)', 'Pearson', 'Cross-corr', 'Granger', 'Partial\nCorr', 'GLASSO']
    
    cook = df[df['benchmark'] == 'cook']
    cook_auroc, cook_recall = [], []
    for m in methods_cook:
        row = cook[cook['name'] == m]
        if len(row) > 0:
            cook_auroc.append(row['auroc'].values[0])
            recall_val = row['recall'].values[0]
            cook_recall.append(recall_val * 100 if not pd.isna(recall_val) else 0)
        else:
            cook_auroc.append(0)
            cook_recall.append(0)
    
    leifer = df[df['benchmark'] == 'leifer']
    leifer_auroc = []
    for m in methods_cook:
        row = leifer[leifer['name'] == m]
        leifer_auroc.append(row['auroc'].values[0] if len(row) > 0 else 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(methods_display))]
    x = np.arange(len(methods_display))
    width = 0.7
    
    # Panel A: Cook AUROC
    ax1 = axes[0]
    bars1 = ax1.bar(x, cook_auroc, width, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random baseline')
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_title(f'A. {STRUCTURAL_LABEL}', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_display, fontsize=10)
    ax1.set_ylim(0.45, max(cook_auroc) * 1.1 if max(cook_auroc) > 0.5 else 0.65)
    ax1.legend(loc='upper right', fontsize=9)
    
    for bar, val in zip(bars1, cook_auroc):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Panel B: Functional-atlas AUROC
    ax2 = axes[1]
    bars2 = ax2.bar(x, leifer_auroc, width, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random baseline')
    ax2.set_ylabel('AUROC', fontsize=12)
    ax2.set_title(f'B. {FUNCTIONAL_LABEL}', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods_display, fontsize=10)
    ax2.set_ylim(0.45, max(leifer_auroc) * 1.1 if max(leifer_auroc) > 0.5 else 0.75)
    ax2.legend(loc='upper right', fontsize=9)
    
    for bar, val in zip(bars2, leifer_auroc):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Panel C: Recall
    ax3 = axes[2]
    bars3 = ax3.bar(x, cook_recall, width, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Recall (%)', fontsize=12)
    ax3.set_title(f'C. True Edges Recovered ({STRUCTURAL_LABEL})', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods_display, fontsize=10)
    ax3.set_ylim(0, max(cook_recall) * 1.2 if max(cook_recall) > 0 else 110)
    
    for bar, val in zip(bars3, cook_recall):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_sbtg_vs_baselines.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig5_sbtg_vs_baselines.png")
    plt.close()


# =============================================================================
# FIGURE 6: PREDICTED VS ANATOMICAL CONNECTIVITY
# =============================================================================

def create_fig6_predicted_vs_anatomical():
    """Create figure comparing predicted vs anatomical connectivity."""
    print("\n[Fig 6] Creating predicted vs anatomical connectivity...")
    
    # Load SBTG results
    models_dir = SBTG_DIR / 'models'
    imputed_models = list(models_dir.glob('*imputed_best*.npz'))
    
    if not imputed_models:
        print("  ERROR: No imputed model found")
        return
    
    model_path = imputed_models[0]
    print(f"  Loading: {model_path.name}")
    data = np.load(model_path, allow_pickle=True)
    sign_adj = data['sign_adj']
    
    # Load neuron names from the dataset (same that was used for training)
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    # Load structural connectome
    A_struct = np.load(CONNECTOME_DIR / 'A_struct.npy')
    with open(CONNECTOME_DIR / 'nodes.json') as f:
        struct_nodes = json.load(f)
    
    print(f"  SBTG shape: {sign_adj.shape}, neurons: {len(neuron_names)}")
    print(f"  Struct shape: {A_struct.shape}, neurons: {len(struct_nodes)}")
    
    # Find common neurons
    common_neurons = [n for n in neuron_names if n in struct_nodes]
    print(f"  Common neurons: {len(common_neurons)}")
    
    if len(common_neurons) == 0:
        # Try uppercase matching
        struct_nodes_upper = [n.upper() for n in struct_nodes]
        common_neurons = [n for n in neuron_names if n.upper() in struct_nodes_upper]
        print(f"  After uppercase matching: {len(common_neurons)}")
    
    if len(common_neurons) < 10:
        # Use all neurons from SBTG directly
        print("  Using direct SBTG matrix (no alignment)")
        sbtg_aligned = sign_adj
        struct_aligned = A_struct[:sign_adj.shape[0], :sign_adj.shape[0]]
        common_neurons = neuron_names[:min(len(neuron_names), len(struct_nodes))]
    else:
        sbtg_idx = [neuron_names.index(n) for n in common_neurons]
        struct_idx = [struct_nodes.index(n) if n in struct_nodes else 
                      struct_nodes.index(n.upper()) if n.upper() in [s.upper() for s in struct_nodes] else -1 
                      for n in common_neurons]
        struct_idx = [i for i in struct_idx if i >= 0]
        common_neurons = common_neurons[:len(struct_idx)]
        sbtg_idx = sbtg_idx[:len(struct_idx)]
        
        sbtg_aligned = sign_adj[np.ix_(sbtg_idx, sbtg_idx)]
        struct_aligned = A_struct[np.ix_(struct_idx, struct_idx)]
    
    n = sbtg_aligned.shape[0]
    print(f"  Aligned matrix size: {n}x{n}")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: SBTG Prediction
    ax1 = axes[0]
    im1 = ax1.imshow(sbtg_aligned, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title('A. SBTG Predicted Connectivity', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Target Neuron', fontsize=11)
    ax1.set_ylabel('Source Neuron', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='Sign (+1 exc, -1 inh)')
    
    # Panel B: Structural Ground Truth
    ax2 = axes[1]
    struct_binary = (struct_aligned > 0).astype(float)
    im2 = ax2.imshow(struct_binary, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    ax2.set_title('B. Anatomical Connectome', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Target Neuron', fontsize=11)
    ax2.set_ylabel('Source Neuron', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='Connection')
    
    # Panel C: Overlap (TP/FP/FN)
    ax3 = axes[2]
    sbtg_binary = (np.abs(sbtg_aligned) > 0).astype(int)
    struct_binary_int = (struct_aligned > 0).astype(int)
    
    overlap = np.zeros_like(sbtg_binary, dtype=float)
    overlap[(sbtg_binary == 1) & (struct_binary_int == 1)] = 1  # TP
    overlap[(sbtg_binary == 1) & (struct_binary_int == 0)] = 2  # FP
    overlap[(sbtg_binary == 0) & (struct_binary_int == 1)] = 3  # FN
    
    colors = ['white', '#27ae60', '#e74c3c', '#3498db']
    cmap = ListedColormap(colors)
    
    im3 = ax3.imshow(overlap, cmap=cmap, vmin=0, vmax=3, aspect='auto')
    ax3.set_title('C. Prediction Accuracy', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Target Neuron', fontsize=11)
    ax3.set_ylabel('Source Neuron', fontsize=11)
    
    legend_elements = [
        mpatches.Patch(facecolor='#27ae60', label='True Positive'),
        mpatches.Patch(facecolor='#e74c3c', label='False Positive'),
        mpatches.Patch(facecolor='#3498db', label='False Negative'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Compute stats
    tp = ((sbtg_binary == 1) & (struct_binary_int == 1)).sum()
    fp = ((sbtg_binary == 1) & (struct_binary_int == 0)).sum()
    fn = ((sbtg_binary == 0) & (struct_binary_int == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    stats_text = f'TP={tp}, FP={fp}, FN={fn}\nRecall={recall*100:.1f}%, Prec={precision*100:.1f}%'
    ax3.text(0.02, 0.02, stats_text, transform=ax3.transAxes, fontsize=9, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_predicted_vs_anatomical.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig6_predicted_vs_anatomical.png")
    plt.close()


# =============================================================================
# FIGURE 7: E:I RATIO DYNAMICS
# =============================================================================

def create_fig7_ei_ratio_dynamics():
    """Create figure showing E:I ratio across phases (using SBTG results)."""
    print("\n[Fig 7] Creating E:I ratio temporal dynamics...")
    
    with open(RESULTS_DIR / 'sbtg_temporal' / 'phase_results.json') as f:
        phase_results = json.load(f)
    
    phases = ['baseline', 'butanone', 'pentanedione', 'nacl']
    phase_labels = ['Baseline\n(0-60s)', 'Butanone\n(60-80s)', 'Pentanedione\n(120-140s)', 'NaCl\n(180-200s)']
    
    # Handle case where ei_ratio can be 'inf' string in JSON
    ei_ratios = []
    for p in phases:
        r = phase_results[p]['ei_ratio']
        if r == 'inf':
            ei_ratios.append(float('inf'))
        else:
            ei_ratios.append(float(r))
    
    n_pos = [phase_results[p]['n_positive'] for p in phases]
    n_neg = [phase_results[p]['n_negative'] for p in phases]
    methods = [phase_results[p].get('method', 'unknown') for p in phases]
    
    # Check if all phases used SBTG (no fallback)
    all_sbtg = all(m.startswith('sbtg') for m in methods)
    if not all_sbtg:
        fallback_phases = [p for p, m in zip(phases, methods) if not m.startswith('sbtg')]
        print(f"  NOTE: Some phases used Pearson fallback: {fallback_phases}")
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Panel A: E:I Ratio
    ax1 = axes[0]
    # Cap infinite values for display
    ei_display = []
    for r in ei_ratios:
        if r == float('inf') or r > 100:
            ei_display.append(None)  # Will handle separately
        else:
            ei_display.append(r)
    
    # Find max finite value for scaling
    finite_ratios = [r for r in ei_ratios if r != float('inf') and r < 100]
    max_finite = max(finite_ratios) if finite_ratios else 5.0
    
    # Color based on E:I ratio (gray if 0, blue if <1, red if >1)
    colors = []
    display_vals = []
    for r in ei_ratios:
        if r == 0:
            colors.append('#95a5a6')  # Gray for no data
            display_vals.append(0)
        elif r == float('inf') or r > 100:
            colors.append('#c0392b')  # Dark red for all excitatory
            display_vals.append(max_finite * 1.3)  # Show as tall bar
        elif r < 1.0:
            colors.append('#3498db')  # Blue - inhibition dominant
            display_vals.append(r)
        else:
            colors.append('#e74c3c')  # Red - excitation dominant
            display_vals.append(r)
    
    bars = ax1.bar(phase_labels, display_vals, color=colors, edgecolor='black', linewidth=1)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Balanced (E:I = 1.0)')
    ax1.set_ylabel('E:I Ratio (Excitatory / Inhibitory)', fontsize=11)
    ax1.set_title('A. Excitation-Inhibition Balance', fontsize=12, fontweight='bold')
    
    # Dynamic y-limits based on data
    max_display = max(display_vals) if max(display_vals) > 0 else 5.0
    ax1.set_ylim(0, max_display * 1.2)
    ax1.legend(loc='upper right', fontsize=9)
    
    for i, (bar, val) in enumerate(zip(bars, ei_ratios)):
        if val == float('inf') or val > 100:
            # Show infinity symbol for all-excitatory
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_display * 0.03, 
                     '∞', ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkred')
        elif val > 0:
            color = 'darkblue' if val < 1.0 else 'darkred'
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_display * 0.03, 
                     f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
        else:
            # Show "N/A" or "0" for phases with no edges
            ax1.text(bar.get_x() + bar.get_width()/2, 0.1, 
                     'N/A', ha='center', va='bottom', fontsize=9, color='gray', style='italic')
    
    # Panel B: Edge counts (stacked bar)
    ax2 = axes[1]
    x = np.arange(len(phases))
    width = 0.6
    
    ax2.bar(x, n_pos, width, label='Excitatory (+)', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax2.bar(x, n_neg, width, bottom=n_pos, label='Inhibitory (-)', color='#3498db', edgecolor='black', linewidth=0.5)
    
    ax2.set_ylabel('Number of Edges', fontsize=11)
    ax2.set_title('B. Edge Composition by Phase', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(phase_labels, fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    
    max_edges = max([p + n for p, n in zip(n_pos, n_neg)]) if n_pos else 100
    ax2.set_ylim(0, max_edges * 1.15)
    
    for i, (pos, neg) in enumerate(zip(n_pos, n_neg)):
        total = pos + neg
        if total > 0:
            ax2.text(i, total + max_edges * 0.02, f'{total}', ha='center', fontsize=10)
        else:
            ax2.text(i, max_edges * 0.05, '0', ha='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_ei_ratio_dynamics.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig7_ei_ratio_dynamics.png")
    plt.close()


# =============================================================================
# FIGURE 8: AGGREGATE NETWORK GRAPH
# =============================================================================

def create_fig8_aggregate_network():
    """Create networkx visualization of aggregate connectivity."""
    print("\n[Fig 8] Creating aggregate network graph...")
    
    # Load SBTG results
    models_dir = SBTG_DIR / 'models'
    imputed_models = list(models_dir.glob('*imputed_best*.npz'))
    
    if not imputed_models:
        print("  ERROR: No imputed model found")
        return
    
    data = np.load(imputed_models[0], allow_pickle=True)
    sign_adj = data['sign_adj']
    
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    for i, name in enumerate(neuron_names):
        G.add_node(name)
    
    # Add edges (only significant ones)
    n_edges = 0
    edge_colors = []
    edge_list = []
    
    for i in range(len(neuron_names)):
        for j in range(len(neuron_names)):
            if sign_adj[i, j] != 0:
                G.add_edge(neuron_names[i], neuron_names[j])
                edge_list.append((neuron_names[i], neuron_names[j]))
                edge_colors.append('#e74c3c' if sign_adj[i, j] > 0 else '#3498db')
                n_edges += 1
    
    print(f"  Network: {len(G.nodes())} nodes, {n_edges} edges")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightgray', 
                           edgecolors='black', linewidths=0.5, ax=ax)
    
    # Draw edges with colors
    nx.draw_networkx_edges(G, pos, edge_list, edge_color=edge_colors, 
                           arrows=True, arrowsize=10, alpha=0.5, 
                           width=0.5, connectionstyle="arc3,rad=0.1", ax=ax)
    
    # Draw labels for top-degree nodes only
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:20]
    labels = {n: n for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Legend
    exc_patch = mpatches.Patch(color='#e74c3c', label='Excitatory')
    inh_patch = mpatches.Patch(color='#3498db', label='Inhibitory')
    ax.legend(handles=[exc_patch, inh_patch], loc='upper left', fontsize=12)
    
    ax.set_title(f'SBTG Functional Connectivity Network\n({len(G.nodes())} neurons, {n_edges} edges)', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_aggregate_network.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig8_aggregate_network.png")
    plt.close()


# =============================================================================
# FIGURE 9: TEMPORAL PHASE NETWORKS
# =============================================================================

def create_fig9_phase_networks():
    """Create networkx visualizations for each temporal phase."""
    print("\n[Fig 9] Creating temporal phase network graphs...")
    
    phases = ['baseline', 'butanone', 'pentanedione', 'nacl']
    phase_titles = ['Baseline (0-60s)', 'Butanone (60-80s)', 'Pentanedione (120-140s)', 'NaCl (180-200s)']
    
    # Load neuron names
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, (phase, title) in enumerate(zip(phases, phase_titles)):
        ax = axes[idx]
        
        # Load phase-specific adjacency
        sign_adj_path = RESULTS_DIR / 'sbtg_temporal' / f'sign_adj_{phase}.npy'
        if not sign_adj_path.exists():
            print(f"  WARNING: {sign_adj_path} not found")
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.axis('off')
            continue
        
        sign_adj = np.load(sign_adj_path)
        
        # Create graph
        G = nx.DiGraph()
        for name in neuron_names:
            G.add_node(name)
        
        edge_colors = []
        edge_list = []
        n_exc, n_inh = 0, 0
        
        for i in range(len(neuron_names)):
            for j in range(len(neuron_names)):
                if sign_adj[i, j] != 0:
                    G.add_edge(neuron_names[i], neuron_names[j])
                    edge_list.append((neuron_names[i], neuron_names[j]))
                    if sign_adj[i, j] > 0:
                        edge_colors.append('#e74c3c')
                        n_exc += 1
                    else:
                        edge_colors.append('#3498db')
                        n_inh += 1
        
        # Fixed layout for consistency
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray', 
                               edgecolors='black', linewidths=0.3, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_list, edge_color=edge_colors, 
                               arrows=True, arrowsize=5, alpha=0.3, 
                               width=0.3, ax=ax)
        
        ei_ratio = n_exc / n_inh if n_inh > 0 else 0
        ax.set_title(f'{title}\nEdges: {n_exc + n_inh} (E:I = {ei_ratio:.2f})', 
                     fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Common legend
    exc_patch = mpatches.Patch(color='#e74c3c', label='Excitatory')
    inh_patch = mpatches.Patch(color='#3498db', label='Inhibitory')
    fig.legend(handles=[exc_patch, inh_patch], loc='upper center', ncol=2, fontsize=12, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Phase-Specific Functional Connectivity Networks', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'fig9_phase_networks.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig9_phase_networks.png")
    plt.close()


# =============================================================================
# FIGURE 1: DATA OVERVIEW
# =============================================================================

def create_fig1_data_overview():
    """
    Create Figure 1: Data Overview showing raw traces, preprocessing stages,
    and worm neuron coverage.
    """
    from scipy.io import loadmat
    
    print("\n[Fig 1] Creating data overview figure...")
    
    # Load raw data from MAT file
    mat_file = DATA_DIR / "Head_Activity_OH16230.mat"
    if not mat_file.exists():
        print(f"  ERROR: {mat_file.name} not found, skipping Fig 1")
        return
    
    mat = loadmat(mat_file, simplify_cells=True)
    neuron_names = [str(n) for n in mat["neurons"]]
    norm_traces = mat["norm_traces"]  # list of (n_worms,) arrays per neuron
    fps = float(mat["fps"])
    stim_times = mat["stim_times"]  # (n_stim, 2) start/end in seconds
    stim_names = [str(s) for s in mat["stim_names"]]
    
    n_neurons = len(neuron_names)
    n_worms = len(norm_traces[0]) if len(norm_traces) > 0 else 0
    
    # Create figure with modified layout
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.8, 0.6, 1, 1], hspace=0.4, wspace=0.25)
    
    # =========================================================================
    # PANEL A: Raw and Z-scored traces
    # =========================================================================
    ax_trace = fig.add_subplot(gs[0, :])
    
    # Pick an example neuron and worm
    example_neuron_idx = 10  # Arbitrary
    example_worm_idx = 0
    
    # Get raw trace for this neuron/worm
    raw_trace = norm_traces[example_neuron_idx][example_worm_idx]
    if hasattr(raw_trace, 'tolist'):
        raw_trace = np.array(raw_trace)
    
    n_frames = len(raw_trace)
    time_s = np.arange(n_frames) / fps
    
    # Preprocessing stages
    trace_raw = raw_trace.copy()
    mean_val = np.nanmean(trace_raw)
    std_val = np.nanstd(trace_raw)
    trace_zscore = (trace_raw - mean_val) / (std_val + 1e-8)
    
    # Plot only raw and z-scored (offset)
    offset = 5
    ax_trace.plot(time_s, trace_raw, color='#e74c3c', linewidth=0.8, label='Raw ΔF/F₀', alpha=0.9)
    ax_trace.plot(time_s, trace_zscore + offset, color='#3498db', linewidth=0.8, label='Z-scored', alpha=0.9)
    ax_trace.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax_trace.axhline(y=offset, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Add stimulus shading
    stim_colors = ['#f1c40f', '#9b59b6', '#1abc9c']
    for stim_idx, (start, end) in enumerate(stim_times[:3]):
        ax_trace.axvspan(start, end, alpha=0.15, color=stim_colors[stim_idx % len(stim_colors)])
    
    for stim_idx, ((start, end), name) in enumerate(zip(stim_times[:3], stim_names[:3])):
        mid = (start + end) / 2
        ax_trace.text(mid, offset + 4, name.capitalize(), ha='center', fontsize=10, fontweight='bold')
    
    ax_trace.set_xlabel('Time (s)', fontsize=12)
    ax_trace.set_ylabel('Signal', fontsize=12)
    ax_trace.set_title(f'A. Preprocessing: Neuron "{neuron_names[example_neuron_idx]}"', 
                       fontsize=14, fontweight='bold')
    ax_trace.legend(loc='upper right', fontsize=10)
    ax_trace.set_xlim(0, time_s[-1])
    ax_trace.set_yticks([])
    
    # =========================================================================
    # PANEL A2: Lag-1 Windowing Illustration (Zoomed)
    # =========================================================================
    ax_lag = fig.add_subplot(gs[1, :])
    
    # Show a zoomed section (frames 100-110) to illustrate windowing
    zoom_start, zoom_end = 100, 112
    zoom_time = time_s[zoom_start:zoom_end]
    zoom_trace = trace_zscore[zoom_start:zoom_end]
    
    # Plot the zoomed trace with markers
    ax_lag.plot(zoom_time, zoom_trace, 'o-', color='#3498db', linewidth=2, markersize=10, label='z-scored values')
    
    # Add frame labels
    for i, (t, v) in enumerate(zip(zoom_time, zoom_trace)):
        ax_lag.annotate(f't={zoom_start+i}', (t, v), textcoords='offset points', 
                       xytext=(0, 15), ha='center', fontsize=9, color='#2c3e50')
    
    # Draw brackets showing window pairs
    bracket_y = zoom_trace.min() - 0.8
    for i in range(len(zoom_time) - 1):
        t1, t2 = zoom_time[i], zoom_time[i+1]
        # Draw bracket
        ax_lag.annotate('', xy=(t1, bracket_y), xytext=(t2, bracket_y),
                       arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2))
        # Window label
        ax_lag.text((t1 + t2) / 2, bracket_y - 0.4, f'Window {i}', ha='center', 
                   fontsize=8, color='#e74c3c', fontweight='bold')
    
    # Add annotation explaining the concept
    ax_lag.text(0.02, 0.95, 
                'Each window = [x(t), x(t+1)] for all neurons\n'
                'Creates pairs: "current state → next state"',
                transform=ax_lag.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
    
    ax_lag.set_xlabel('Time (s)', fontsize=12)
    ax_lag.set_ylabel('Z-scored signal', fontsize=12)
    ax_lag.set_title('A2. Lag-1 Windowing: Creating Training Examples', fontsize=14, fontweight='bold')
    ax_lag.set_ylim(bracket_y - 1, zoom_trace.max() + 1.5)
    
    # =========================================================================
    # PANEL B: Worm neuron coverage heatmap (moved down)
    # =========================================================================
    ax_coverage = fig.add_subplot(gs[2, 0])
    
    # Determine max worms across all neurons (they may have different counts)
    worm_counts_per_neuron = [len(norm_traces[i]) for i in range(n_neurons)]
    n_worms = max(worm_counts_per_neuron)
    
    # Build coverage matrix (n_worms x n_neurons) showing which neurons are measured
    coverage_matrix = np.zeros((n_worms, n_neurons))
    
    for neuron_idx in range(n_neurons):
        num_worms_this_neuron = len(norm_traces[neuron_idx])
        for worm_idx in range(num_worms_this_neuron):
            trace = norm_traces[neuron_idx][worm_idx]
            if trace is not None and hasattr(trace, '__len__') and len(trace) > 0:
                # Check if not all NaN
                if not np.all(np.isnan(trace)):
                    coverage_matrix[worm_idx, neuron_idx] = 1
    
    # Sort worms by coverage (most coverage at top)
    worm_coverage = coverage_matrix.sum(axis=1)
    sort_idx = np.argsort(-worm_coverage)
    coverage_sorted = coverage_matrix[sort_idx, :]
    
    # Sort neurons by coverage too
    neuron_coverage = coverage_matrix.sum(axis=0)
    neuron_sort_idx = np.argsort(-neuron_coverage)
    coverage_sorted = coverage_sorted[:, neuron_sort_idx]
    
    # Plot heatmap
    cmap = ListedColormap(['#2c3e50', '#2ecc71'])
    im = ax_coverage.imshow(coverage_sorted, aspect='auto', cmap=cmap, interpolation='nearest')
    ax_coverage.set_xlabel('Neurons (sorted by coverage)', fontsize=11)
    ax_coverage.set_ylabel('Worms (sorted by coverage)', fontsize=11)
    ax_coverage.set_title('B. Neuron Coverage per Worm', fontsize=13, fontweight='bold')
    
    # Add colorbar legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2c3e50', label='Missing'),
                       Patch(facecolor='#2ecc71', label='Measured')]
    ax_coverage.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # =========================================================================
    # PANEL C: Worm coverage histogram
    # =========================================================================
    ax_hist = fig.add_subplot(gs[2, 1])
    
    bins = np.arange(0, n_neurons + 10, 10)
    ax_hist.hist(worm_coverage, bins=bins, color='#3498db', edgecolor='black', alpha=0.8)
    ax_hist.axvline(x=worm_coverage.mean(), color='#e74c3c', linestyle='--', linewidth=2, 
                    label=f'Mean: {worm_coverage.mean():.0f}')
    ax_hist.set_xlabel('Number of Neurons Measured', fontsize=11)
    ax_hist.set_ylabel('Number of Worms', fontsize=11)
    ax_hist.set_title('C. Distribution of Neuron Coverage Across Worms', fontsize=13, fontweight='bold')
    ax_hist.legend(loc='upper left', fontsize=10)
    
    # =========================================================================
    # PANEL D: Neuron coverage histogram
    # =========================================================================
    ax_neuron_hist = fig.add_subplot(gs[3, 0])
    
    bins_neuron = np.arange(0, n_worms + 2, 1)
    ax_neuron_hist.hist(neuron_coverage, bins=bins_neuron, color='#9b59b6', edgecolor='black', alpha=0.8)
    ax_neuron_hist.axvline(x=neuron_coverage.mean(), color='#e74c3c', linestyle='--', linewidth=2,
                           label=f'Mean: {neuron_coverage.mean():.1f}')
    ax_neuron_hist.set_xlabel('Number of Worms With This Neuron', fontsize=11)
    ax_neuron_hist.set_ylabel('Number of Neurons', fontsize=11)
    ax_neuron_hist.set_title('D. Distribution of Worm Coverage Across Neurons', fontsize=13, fontweight='bold')
    ax_neuron_hist.legend(loc='upper left', fontsize=10)
    
    # =========================================================================
    # PANEL E: Summary statistics
    # =========================================================================
    ax_summary = fig.add_subplot(gs[3, 1])
    ax_summary.axis('off')
    
    # Calculate stats
    complete_worms = (worm_coverage == n_neurons).sum()
    partial_worms = ((worm_coverage > 0) & (worm_coverage < n_neurons)).sum()
    neurons_in_all = (neuron_coverage == n_worms).sum()
    neurons_in_most = (neuron_coverage >= n_worms * 0.75).sum()
    
    summary_text = f"""
    Data Summary
    ────────────────────────
    Total Worms: {n_worms}
    Total Neurons: {n_neurons}
    Recording Length: {n_frames / fps:.0f}s @ {fps} Hz
    
    Worm Coverage:
      • Complete (all neurons): {complete_worms}
      • Partial: {partial_worms}
      • Mean neurons/worm: {worm_coverage.mean():.1f}
    
    Neuron Coverage:
      • In all worms: {neurons_in_all}
      • In ≥75% worms: {neurons_in_most}
      • Mean worms/neuron: {neuron_coverage.mean():.1f}
    """
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                    fontsize=12, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    ax_summary.set_title('E. Data Summary', fontsize=13, fontweight='bold')
    
    plt.suptitle('Figure 1: Raw Data Overview & Preprocessing', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'fig1_data_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig1_data_overview.png")
    plt.close()


# =============================================================================
# FIGURE 4: SBTG METHOD INTUITION
# =============================================================================

def create_fig4_sbtg_intuition():
    """
    Create Figure 4: SBTG Method Intuition
    
    Panels:
    A. Score function intuition (gradient of log-density)
    B. Jacobian extraction (W matrix from scores)
    C. Cross-validation strategy (5-fold held-out)
    D. Edge detection (volatility test + FDR)
    """
    print("\n[Fig 4] Creating SBTG method intuition figure...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # PANEL A: Score Function Intuition
    # =========================================================================
    ax_score = fig.add_subplot(gs[0, 0])
    
    # Create a simple 2D density visualization
    np.random.seed(42)
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    
    # 2D Gaussian density
    Z = np.exp(-(X**2 + Y**2) / 2)
    
    # Plot density contours
    ax_score.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.7)
    ax_score.contour(X, Y, Z, levels=5, colors='#2c3e50', linewidths=0.5)
    
    # Add gradient arrows (score vectors point toward high density)
    arrow_x = [-2, 0, 2, -1.5, 1.5, 0, 0]
    arrow_y = [0, 2, 0, 1.5, -1.5, -2, 0]
    for ax_pt, ay_pt in zip(arrow_x, arrow_y):
        # Score = -gradient of potential = points toward center
        dx = -ax_pt * 0.4
        dy = -ay_pt * 0.4
        ax_score.arrow(ax_pt, ay_pt, dx, dy, head_width=0.15, head_length=0.08, 
                      fc='#e74c3c', ec='#c0392b', linewidth=1.5)
    
    ax_score.set_xlabel('Neuron i activity', fontsize=11)
    ax_score.set_ylabel('Neuron j activity', fontsize=11)
    ax_score.set_title('A. Score Function: ∇log p(x)', fontsize=13, fontweight='bold')
    
    # Add text annotation
    ax_score.text(0.02, 0.98, 
                 'Score = gradient of log-density\n'
                 'Points toward high-probability regions\n'
                 'Learned via denoising score matching',
                 transform=ax_score.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # =========================================================================
    # PANEL B: Jacobian → Connectivity Matrix
    # =========================================================================
    ax_jac = fig.add_subplot(gs[0, 1])
    
    # Create example matrices
    n = 5  # 5 neurons for illustration
    
    # Fake Jacobian (structured)
    W = np.array([
        [0.0, 0.8, 0.0, 0.3, 0.0],
        [0.7, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.4, 0.0, 0.6, 0.2],
        [0.2, 0.0, 0.5, 0.0, 0.7],
        [0.0, 0.0, 0.1, 0.6, 0.0],
    ])
    
    im = ax_jac.imshow(W, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax_jac.set_xticks(range(n))
    ax_jac.set_yticks(range(n))
    ax_jac.set_xticklabels([f'N{i+1}' for i in range(n)])
    ax_jac.set_yticklabels([f'N{i+1}' for i in range(n)])
    ax_jac.set_xlabel('From neuron (t)', fontsize=11)
    ax_jac.set_ylabel('To neuron (t+1)', fontsize=11)
    ax_jac.set_title('B. Connectivity Matrix W', fontsize=13, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_jac, shrink=0.8)
    cbar.set_label('Connection strength', fontsize=10)
    
    # Add annotation
    ax_jac.text(0.02, 0.98, 
               'W[i,j] = ∂score_i / ∂x_j\n'
               'Extracted from Jacobian of\n'
               'the learned score network',
               transform=ax_jac.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # =========================================================================
    # PANEL C: Cross-Validation Strategy
    # =========================================================================
    ax_cv = fig.add_subplot(gs[1, 0])
    ax_cv.axis('off')
    
    # Draw 5-fold CV diagram
    n_folds = 5
    fold_height = 0.12
    fold_gap = 0.02
    start_y = 0.85
    
    colors_train = '#3498db'
    colors_test = '#e74c3c'
    
    ax_cv.text(0.5, 0.98, 'C. 5-Fold Cross-Validation', fontsize=13, fontweight='bold',
              ha='center', va='top', transform=ax_cv.transAxes)
    
    for fold in range(n_folds):
        y = start_y - fold * (fold_height + fold_gap)
        
        # Draw fold label
        ax_cv.text(0.02, y - fold_height/2, f'Fold {fold+1}:', fontsize=10, 
                  va='center', fontweight='bold', transform=ax_cv.transAxes)
        
        # Draw segments
        segment_width = 0.15
        for seg in range(n_folds):
            x = 0.15 + seg * (segment_width + 0.01)
            color = colors_test if seg == fold else colors_train
            rect = plt.Rectangle((x, y - fold_height), segment_width, fold_height,
                                 facecolor=color, edgecolor='black', linewidth=1,
                                 transform=ax_cv.transAxes)
            ax_cv.add_patch(rect)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_train, edgecolor='black', label='Train'),
                       Patch(facecolor=colors_test, edgecolor='black', label='Test (held-out)')]
    ax_cv.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
                bbox_to_anchor=(0.5, 0.05))
    
    # Explanation
    ax_cv.text(0.5, 0.25, 
              'Each worm\'s windows are held out once.\n'
              'Scores computed ONLY on held-out data\n'
              '→ Prevents overfitting, ensures generalization.',
              ha='center', va='top', fontsize=11, transform=ax_cv.transAxes,
              bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
    
    # =========================================================================
    # PANEL D: Edge Detection (Volatility Test)
    # =========================================================================
    ax_edge = fig.add_subplot(gs[1, 1])
    
    # Simulate W values for two edges: one significant, one not
    np.random.seed(123)
    n_samples = 100
    
    # Edge 1: Strong connection (significant)
    w_strong = np.random.normal(0.6, 0.15, n_samples)
    
    # Edge 2: Noise (not significant)
    w_noise = np.random.normal(0.0, 0.2, n_samples)
    
    # Plot histograms
    bins = np.linspace(-0.5, 1.0, 30)
    ax_edge.hist(w_strong, bins=bins, alpha=0.7, color='#27ae60', label='Significant edge', edgecolor='black')
    ax_edge.hist(w_noise, bins=bins, alpha=0.7, color='#95a5a6', label='Non-significant', edgecolor='black')
    
    # Add zero line
    ax_edge.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Zero')
    
    # Add mean markers
    ax_edge.axvline(x=w_strong.mean(), color='#27ae60', linestyle='-', linewidth=2)
    ax_edge.axvline(x=w_noise.mean(), color='#95a5a6', linestyle='-', linewidth=2)
    
    ax_edge.set_xlabel('Estimated W[i,j] value', fontsize=11)
    ax_edge.set_ylabel('Frequency (across CV folds)', fontsize=11)
    ax_edge.set_title('D. Edge Detection via Volatility Test', fontsize=13, fontweight='bold')
    ax_edge.legend(loc='upper right', fontsize=9)
    
    # Annotation
    ax_edge.text(0.02, 0.98, 
                'Test: Is mean(W) significantly ≠ 0?\n'
                't-test across CV folds\n'
                'FDR correction (Benjamini-Hochberg)',
                transform=ax_edge.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.suptitle('Figure 4: SBTG Method Overview', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'fig4_sbtg_intuition.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig4_sbtg_intuition.png")
    plt.close()



# =============================================================================
# FIGURE 10: WEIGHT CORRELATION BAR CHART
# =============================================================================

def create_fig10_weight_correlation_bars():
    """Create bar chart comparing weight correlation (Spearman) across methods."""
    print("\n[Fig 10] Creating weight correlation bar chart...")
    
    eval_file = RESULTS_DIR / 'evaluation' / 'evaluation_results.csv'
    if not eval_file.exists():
        print("  Evaluation results not found, run 04_evaluate.py first")
        return
    
    df = pd.read_csv(eval_file)
    
    # Check if spearman_r column exists
    if 'spearman_r' not in df.columns:
        print("  spearman_r not in results, run 04_evaluate.py to update")
        return
    
    # Find SBTG imputed_best model
    sbtg_rows = df[df['name'].str.contains('imputed_best', na=False)]
    if len(sbtg_rows) == 0:
        sbtg_rows = df[df['name'].str.startswith('sbtg_', na=False)]
    
    if len(sbtg_rows) == 0:
        print("  WARNING: No SBTG models found")
        return
    
    sbtg_name = sbtg_rows['name'].iloc[0]
    print(f"  Using SBTG model: {sbtg_name}")
    
    # Methods to compare
    baseline_methods = ['pearson', 'crosscorr', 'granger', 'partial_corr', 'glasso']
    methods = [sbtg_name] + baseline_methods
    methods_display = ['SBTG\n(imputed)', 'Pearson', 'Cross-corr', 'Granger', 'Partial\nCorr', 'GLASSO']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(methods_display))]
    x = np.arange(len(methods_display))
    width = 0.7
    
    # Panel A: Cook Spearman
    ax1 = axes[0]
    cook = df[df['benchmark'] == 'cook']
    cook_spearman = []
    for m in methods:
        row = cook[cook['name'] == m]
        if len(row) > 0 and pd.notna(row['spearman_r'].values[0]):
            cook_spearman.append(row['spearman_r'].values[0])
        else:
            cook_spearman.append(0)
    
    bars1 = ax1.bar(x, cook_spearman, width, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_ylabel("Spearman ρ", fontsize=12)
    ax1.set_title(f'A. {STRUCTURAL_LABEL}\n(Predicted vs Synapse Counts)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_display, fontsize=10)
    ax1.set_ylim(-0.1, max(cook_spearman) * 1.2 if max(cook_spearman) > 0 else 0.3)
    
    for bar, val in zip(bars1, cook_spearman):
        if val != 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Panel B: Functional-atlas Spearman
    ax2 = axes[1]
    leifer = df[df['benchmark'] == 'leifer']
    leifer_spearman = []
    for m in methods:
        row = leifer[leifer['name'] == m]
        if len(row) > 0 and pd.notna(row['spearman_r'].values[0]):
            leifer_spearman.append(row['spearman_r'].values[0])
        else:
            leifer_spearman.append(0)
    
    bars2 = ax2.bar(x, leifer_spearman, width, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_ylabel("Spearman ρ", fontsize=12)
    ax2.set_title(f'B. {FUNCTIONAL_LABEL}\n(Predicted vs dFF Amplitude)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods_display, fontsize=10)
    ax2.set_ylim(-0.1, max(leifer_spearman) * 1.2 if max(leifer_spearman) > 0 else 0.3)
    
    for bar, val in zip(bars2, leifer_spearman):
        if val != 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig10_weight_correlation_bars.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig10_weight_correlation_bars.png")
    plt.close()


# =============================================================================
# FIGURE 11: WEIGHT SCATTER PLOTS
# =============================================================================

def create_fig11_weight_scatter():
    """Create scatter plots showing weight correlations."""
    print("\n[Fig 11] Creating weight correlation scatter plots...")
    
    from scipy.stats import pearsonr, spearmanr
    
    # Load SBTG results
    models_dir = SBTG_DIR / 'models'
    imputed_models = list(models_dir.glob('*imputed_best*.npz'))
    
    if not imputed_models:
        print("  ERROR: No imputed model found")
        return
    
    model_path = imputed_models[0]
    data = np.load(model_path, allow_pickle=True)
    # NOTE: Using abs() for unsigned magnitude comparison (E/I polarity not considered here)
    sbtg_weights = np.abs(data['mu_hat']) if 'mu_hat' in data else (data['sign_adj'] != 0).astype(float)
    
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: SBTG vs Cook synapse counts
    ax1 = axes[0]
    
    A_struct = np.load(CONNECTOME_DIR / 'A_struct.npy')
    with open(CONNECTOME_DIR / 'nodes.json') as f:
        struct_nodes = json.load(f)
    
    common = [n for n in neuron_names if n in struct_nodes]
    if len(common) > 5:
        sbtg_idx = [neuron_names.index(n) for n in common]
        cook_idx = [struct_nodes.index(n) for n in common]
        
        sbtg_aligned = sbtg_weights[np.ix_(sbtg_idx, sbtg_idx)]
        cook_aligned = A_struct[np.ix_(cook_idx, cook_idx)]
        
        mask = ~np.eye(len(common), dtype=bool)
        x_vals = sbtg_aligned[mask].flatten()
        y_vals = cook_aligned[mask].flatten()
        
        valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        x_plot = x_vals[valid]
        y_plot = y_vals[valid]
        
        if len(x_plot) > 10:
            ax1.scatter(x_plot, y_plot, alpha=0.3, s=10, c='#3498db')
            
            rho, _ = spearmanr(x_plot, y_plot)
            r, _ = pearsonr(x_plot, y_plot)
            
            ax1.text(0.05, 0.95, f'Spearman ρ = {rho:.3f}\nR² = {r**2:.3f}', 
                     transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.set_xlabel('SBTG |μ̂| (predicted weights)', fontsize=11)
    ax1.set_ylabel(f'{STRUCTURAL_LABEL} Synapse Count', fontsize=11)
    ax1.set_title('A. Predicted vs Anatomical Weights', fontsize=13, fontweight='bold')
    
    # Panel B: SBTG vs functional-atlas dFF
    ax2 = axes[1]
    
    leifer_file = RESULTS_DIR / 'leifer_evaluation' / 'aligned_atlas_wild-type.npz'
    if leifer_file.exists():
        leifer_data = np.load(leifer_file, allow_pickle=True)
        dff = leifer_data.get('dff', None)
        leifer_neurons = list(leifer_data.get('neuron_order', []))
        
        if dff is not None:
            common_l = [n for n in neuron_names if n in leifer_neurons]
            
            if len(common_l) > 5:
                sbtg_idx_l = [neuron_names.index(n) for n in common_l]
                dff_idx = [leifer_neurons.index(n) for n in common_l]
                
                sbtg_aligned_l = sbtg_weights[np.ix_(sbtg_idx_l, sbtg_idx_l)]
                dff_aligned = dff[np.ix_(dff_idx, dff_idx)]
                
                mask_l = ~np.eye(len(common_l), dtype=bool) & (dff_aligned > 0)
                x_dff = sbtg_aligned_l[mask_l].flatten()
                y_dff = dff_aligned[mask_l].flatten()
                
                valid_l = ~np.isnan(x_dff) & ~np.isnan(y_dff)
                x_dff_plot = x_dff[valid_l]
                y_dff_plot = y_dff[valid_l]
                
                if len(x_dff_plot) > 10:
                    ax2.scatter(x_dff_plot, y_dff_plot, alpha=0.3, s=10, c='#27ae60')
                    
                    rho_l, _ = spearmanr(x_dff_plot, y_dff_plot)
                    r_l, _ = pearsonr(x_dff_plot, y_dff_plot)
                    
                    ax2.text(0.05, 0.95, f'Spearman ρ = {rho_l:.3f}\nR² = {r_l**2:.3f}', 
                             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax2.set_xlabel('SBTG |μ̂| (predicted weights)', fontsize=11)
    ax2.set_ylabel(f'{FUNCTIONAL_LABEL} dFF (functional amplitude)', fontsize=11)
    ax2.set_title('B. Predicted vs Functional Weights', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig11_weight_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig11_weight_scatter.png")
    plt.close()



# =============================================================================
# FIGURE 12: SIGNED E/I CORRELATION (mu_hat vs dFF)
# =============================================================================

def create_fig12_signed_correlation():
    """
    Create figure showing signed correlation between SBTG mu_hat and
    Randi_Optogenetics_2023 dFF.
    
    This tests whether SBTG correctly predicts connection POLARITY:
    - Positive mu_hat should predict positive dFF (excitation)
    - Negative mu_hat should predict negative dFF (inhibition)
    """
    print("\n[Fig 12] Creating signed E/I correlation analysis...")
    
    from scipy.stats import pearsonr, spearmanr
    
    # Load SBTG results
    models_dir = SBTG_DIR / 'models'
    imputed_models = list(models_dir.glob('*imputed_best*.npz'))
    
    if not imputed_models:
        print("  ERROR: No imputed model found")
        return
    
    model_path = imputed_models[0]
    print(f"  Using model: {model_path.name}")
    
    data = np.load(model_path, allow_pickle=True)
    mu_hat = data['mu_hat']  # SIGNED values
    sign_adj = data['sign_adj']  # -1, 0, +1
    
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    # Load functional dFF reference
    leifer_file = RESULTS_DIR / 'leifer_evaluation' / 'aligned_atlas_wild-type.npz'
    if not leifer_file.exists():
        print(f"  ERROR: {FUNCTIONAL_LABEL} atlas not found")
        return
    
    leifer_data = np.load(leifer_file, allow_pickle=True)
    dff = leifer_data['dff']  # SIGNED values
    leifer_neurons = list(leifer_data.get('neuron_order', []))
    
    # Align neurons
    common = [n for n in neuron_names if n in leifer_neurons]
    if len(common) < 10:
        print(f"  ERROR: Only {len(common)} common neurons")
        return
    
    print(f"  Common neurons: {len(common)}")
    
    sbtg_idx = [neuron_names.index(n) for n in common]
    dff_idx = [leifer_neurons.index(n) for n in common]
    
    mu_aligned = mu_hat[np.ix_(sbtg_idx, sbtg_idx)]
    dff_aligned = dff[np.ix_(dff_idx, dff_idx)]
    sign_aligned = sign_adj[np.ix_(sbtg_idx, sbtg_idx)]
    
    # Mask: exclude diagonal, only where dFF is not NaN
    mask = ~np.eye(len(common), dtype=bool) & ~np.isnan(dff_aligned)
    
    mu_flat = mu_aligned[mask]
    dff_flat = dff_aligned[mask]
    sign_flat = sign_aligned[mask]
    
    print(f"  Total pairs: {len(mu_flat)}")
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: All pairs scatter (signed)
    ax1 = axes[0]
    colors = np.where(mu_flat > 0, '#e74c3c', '#3498db')  # Red=exc, Blue=inh
    ax1.scatter(mu_flat, dff_flat, alpha=0.3, s=8, c=colors)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    
    rho_all, p_all = spearmanr(mu_flat, dff_flat)
    r_all, _ = pearsonr(mu_flat, dff_flat)
    
    ax1.text(0.05, 0.95, f'Spearman ρ = {rho_all:.3f}\nPearson r = {r_all:.3f}\np < {p_all:.2e}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.set_xlabel('SBTG μ̂ (signed prediction)', fontsize=11)
    ax1.set_ylabel(f'{FUNCTIONAL_LABEL} dFF (signed amplitude)', fontsize=11)
    ax1.set_title('A. All Pairs: Signed Correlation', fontsize=12, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Predicted Excitatory (μ̂ > 0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Predicted Inhibitory (μ̂ < 0)'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # Panel B: Only significant edges (sign_adj != 0)
    ax2 = axes[1]
    sig_mask = sign_flat != 0
    if sig_mask.sum() > 10:
        mu_sig = mu_flat[sig_mask]
        dff_sig = dff_flat[sig_mask]
        sign_sig = sign_flat[sig_mask]
        
        colors_sig = np.where(sign_sig > 0, '#e74c3c', '#3498db')
        ax2.scatter(mu_sig, dff_sig, alpha=0.5, s=15, c=colors_sig)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax2.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        
        rho_sig, p_sig = spearmanr(mu_sig, dff_sig)
        r_sig, _ = pearsonr(mu_sig, dff_sig)
        
        # Quadrant analysis
        q1 = np.sum((mu_sig > 0) & (dff_sig > 0))  # True excitatory
        q2 = np.sum((mu_sig < 0) & (dff_sig > 0))  # Predicted inh, actual exc
        q3 = np.sum((mu_sig < 0) & (dff_sig < 0))  # True inhibitory
        q4 = np.sum((mu_sig > 0) & (dff_sig < 0))  # Predicted exc, actual inh
        
        accuracy = (q1 + q3) / len(mu_sig) * 100 if len(mu_sig) > 0 else 0
        
        ax2.text(0.05, 0.95, f'n = {len(mu_sig)} edges\nρ = {rho_sig:.3f}\nSign accuracy: {accuracy:.1f}%', 
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add quadrant counts
        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()
        ax2.text(xlim[1]*0.7, ylim[1]*0.8, f'Q1: {q1}', fontsize=9, color='#e74c3c')
        ax2.text(xlim[0]*0.7, ylim[1]*0.8, f'Q2: {q2}', fontsize=9, color='gray')
        ax2.text(xlim[0]*0.7, ylim[0]*0.8, f'Q3: {q3}', fontsize=9, color='#3498db')
        ax2.text(xlim[1]*0.7, ylim[0]*0.8, f'Q4: {q4}', fontsize=9, color='gray')
    
    ax2.set_xlabel('SBTG μ̂ (signed prediction)', fontsize=11)
    ax2.set_ylabel(f'{FUNCTIONAL_LABEL} dFF (signed amplitude)', fontsize=11)
    ax2.set_title('B. Significant Edges Only', fontsize=12, fontweight='bold')
    
    # Panel C: Confusion matrix for sign prediction
    ax3 = axes[2]
    
    # For edges where both have clear sign
    clear_mask = (np.abs(mu_flat) > 0.01) & (np.abs(dff_flat) > 0.01)
    if clear_mask.sum() > 20:
        pred_sign = np.sign(mu_flat[clear_mask])
        true_sign = np.sign(dff_flat[clear_mask])
        
        # Confusion: rows = predicted, cols = actual
        confusion = np.zeros((2, 2))
        confusion[0, 0] = np.sum((pred_sign == 1) & (true_sign == 1))   # Pred+, True+
        confusion[0, 1] = np.sum((pred_sign == 1) & (true_sign == -1))  # Pred+, True-
        confusion[1, 0] = np.sum((pred_sign == -1) & (true_sign == 1))  # Pred-, True+
        confusion[1, 1] = np.sum((pred_sign == -1) & (true_sign == -1)) # Pred-, True-
        
        # Normalize by column (true label)
        col_sums = confusion.sum(axis=0, keepdims=True)
        confusion_norm = confusion / np.maximum(col_sums, 1)
        
        im = ax3.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = f'{confusion_norm[i,j]:.2f}\n({int(confusion[i,j])})'
                ax3.text(j, i, text, ha='center', va='center', fontsize=11,
                        color='white' if confusion_norm[i,j] > 0.5 else 'black')
        
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['Excitation\n(dFF > 0)', 'Inhibition\n(dFF < 0)'])
        ax3.set_yticklabels(['Pred. Exc\n(μ̂ > 0)', 'Pred. Inh\n(μ̂ < 0)'])
        ax3.set_xlabel(f'Actual ({FUNCTIONAL_LABEL} dFF)', fontsize=11)
        ax3.set_ylabel('Predicted (SBTG)', fontsize=11)
        
        total_acc = (confusion[0,0] + confusion[1,1]) / confusion.sum() * 100
        ax3.set_title(f'C. Sign Confusion Matrix\n(Accuracy: {total_acc:.1f}%)', fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax3, label='Rate')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig12_signed_correlation.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig12_signed_correlation.png")
    plt.close()


# =============================================================================
# FIGURE 15: DIRECT vs TRANSFER TRAINING COMPARISON
# =============================================================================

def create_fig15_direct_vs_transfer():
    """
    Create figure comparing Direct vs Transfer learning for phase-specific SBTG.
    
    Direct training: Train from scratch on each stimulus phase
    Transfer training: Pre-train on baseline (0-60s), fine-tune on each stimulus
    
    Panels:
    A. Edge counts comparison (grouped bar)
    B. E:I ratio comparison
    C. Transfer contribution breakdown (% edges from baseline vs phase-specific)
    D. Learning curve comparison (if available)
    """
    print("\n[Fig 15] Creating Direct vs Transfer training comparison...")
    
    # Load comparison data from 05_temporal_analysis.py output
    sbtg_dir = RESULTS_DIR / 'stimulus_specific' / 'sbtg'
    comparison_file = sbtg_dir / 'direct_vs_transfer_comparison.csv'
    phase_results_file = sbtg_dir / 'phase_results.json'
    
    if not comparison_file.exists():
        print("  No direct vs transfer comparison found.")
        print("  Run: python pipeline/05_temporal_analysis.py --sbtg --transfer")
        return
    
    # Load comparison data
    comparison_df = pd.read_csv(comparison_file)
    
    if phase_results_file.exists():
        with open(phase_results_file) as f:
            phase_results = json.load(f)
    else:
        print("  No phase_results.json found, skipping detailed breakdown")
        phase_results = {}
    
    phases = comparison_df['phase'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # =========================================================================
    # PANEL A: Edge counts (grouped bar)
    # =========================================================================
    ax1 = axes[0, 0]
    
    x = np.arange(len(phases))
    width = 0.35
    
    direct_edges = comparison_df['direct_edges'].values
    transfer_edges = comparison_df['transfer_edges'].values
    
    bars1 = ax1.bar(x - width/2, direct_edges, width, label='Direct Training', 
                    color='#3498db', edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, transfer_edges, width, label='Transfer Learning',
                    color='#e74c3c', edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Number of Edges', fontsize=12)
    ax1.set_title('A. Total Edges by Training Method', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.capitalize() for p in phases])
    ax1.legend(loc='upper right', fontsize=10)
    
    # Annotate bars
    for bar, val in zip(bars1, direct_edges):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 str(int(val)), ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, transfer_edges):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 str(int(val)), ha='center', va='bottom', fontsize=10)
    
    max_edges = max(max(direct_edges), max(transfer_edges))
    ax1.set_ylim(0, max_edges * 1.2)
    
    # =========================================================================
    # PANEL B: E:I Ratio comparison
    # =========================================================================
    ax2 = axes[0, 1]
    
    direct_ei = comparison_df['direct_ei_ratio'].values
    transfer_ei = comparison_df['transfer_ei_ratio'].values
    
    bars1 = ax2.bar(x - width/2, direct_ei, width, label='Direct Training',
                    color='#3498db', edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, transfer_ei, width, label='Transfer Learning',
                    color='#e74c3c', edgecolor='black', linewidth=1)
    
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Balanced (E:I = 1.0)')
    ax2.set_ylabel('E:I Ratio', fontsize=12)
    ax2.set_title('B. Excitation-Inhibition Balance', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.capitalize() for p in phases])
    ax2.legend(loc='upper right', fontsize=10)
    
    # Annotate
    for bar, val in zip(bars1, direct_ei):
        color = 'darkred' if val > 1 else 'darkblue'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
    for bar, val in zip(bars2, transfer_ei):
        color = 'darkred' if val > 1 else 'darkblue'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
    
    # Guard against NaN/Inf in E:I ratios
    valid_ei = [v for v in list(direct_ei) + list(transfer_ei) if np.isfinite(v)]
    max_ei = max(valid_ei) if valid_ei else 2.0
    ax2.set_ylim(0, max_ei * 1.3)
    
    # =========================================================================
    # PANEL C: Transfer contribution breakdown (stacked bar)
    # =========================================================================
    ax3 = axes[1, 0]
    
    edges_from_baseline = comparison_df['edges_from_baseline'].values
    edges_from_phase = transfer_edges - edges_from_baseline
    
    bars1 = ax3.bar(x, edges_from_baseline, width=0.6, label='From Baseline Prior',
                    color='#9b59b6', edgecolor='black', linewidth=1)
    bars2 = ax3.bar(x, edges_from_phase, width=0.6, bottom=edges_from_baseline,
                    label='Phase-Specific', color='#f39c12', edgecolor='black', linewidth=1)
    
    ax3.set_ylabel('Number of Edges', fontsize=12)
    ax3.set_title('C. Transfer Learning: Edge Sources', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([p.capitalize() for p in phases])
    ax3.legend(loc='upper right', fontsize=10)
    
    # Annotate percentages
    for i, (base, phase, total) in enumerate(zip(edges_from_baseline, edges_from_phase, transfer_edges)):
        if total > 0:
            base_pct = 100 * base / total
            ax3.text(i, total + 10, f'{base_pct:.0f}% baseline',
                     ha='center', va='bottom', fontsize=9, color='#9b59b6', fontweight='bold')
    
    ax3.set_ylim(0, max(transfer_edges) * 1.25)
    
    # =========================================================================
    # PANEL D: Summary statistics table
    # =========================================================================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Phase', 'Direct Edges', 'Transfer Edges', 'Δ Edges', 'Direct E:I', 'Transfer E:I'])
    
    for i, phase in enumerate(phases):
        delta = int(transfer_edges[i] - direct_edges[i])
        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        table_data.append([
            phase.capitalize(),
            str(int(direct_edges[i])),
            str(int(transfer_edges[i])),
            delta_str,
            f"{direct_ei[i]:.3f}",
            f"{transfer_ei[i]:.3f}"
        ])
    
    # Add summary row
    total_direct = int(sum(direct_edges))
    total_transfer = int(sum(transfer_edges))
    avg_direct_ei = np.mean(direct_ei)
    avg_transfer_ei = np.mean(transfer_ei)
    total_delta = total_transfer - total_direct
    delta_str = f"+{total_delta}" if total_delta >= 0 else str(total_delta)
    
    table_data.append(['TOTAL/AVG', str(total_direct), str(total_transfer), 
                       delta_str, f"{avg_direct_ei:.3f}", f"{avg_transfer_ei:.3f}"])
    
    # Create table
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='center', cellLoc='center',
                      colColours=['#ecf0f1'] * 6)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Highlight last row (summary)
    for j in range(6):
        table[(len(phases) + 1, j)].set_facecolor('#d4edda')
        table[(len(phases) + 1, j)].set_text_props(fontweight='bold')
    
    ax4.set_title('D. Summary Statistics', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('Figure 15: Direct vs Transfer Training Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'fig15_direct_vs_transfer.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig15_direct_vs_transfer.png")
    plt.close()


# =============================================================================
# FIGURE 16: DIRECT TRAINING PHASE NETWORKS
# =============================================================================

def create_fig16_direct_networks():
    """Create networkx visualizations for direct training per phase."""
    print("\n[Fig 16] Creating direct training phase networks...")
    
    sbtg_adj_dir = RESULTS_DIR / 'stimulus_specific' / 'sbtg' / 'adjacencies'
    if not sbtg_adj_dir.exists():
        print("  No SBTG adjacencies found. Run: python pipeline/05_temporal_analysis.py --sbtg")
        return
    
    phases = ['baseline', 'butanone', 'pentanedione', 'nacl']
    phase_titles = ['Baseline (0-60s)', 'Butanone (60.5-70.5s)', 
                    'Pentanedione (120.5-130.5s)', 'NaCl (180.5-190.5s)']
    
    # Load neuron names
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, (phase, title) in enumerate(zip(phases, phase_titles)):
        ax = axes[idx]
        
        # Load direct training adjacency
        sign_adj_path = sbtg_adj_dir / f'{phase}_direct_sign_adj.npy'
        if not sign_adj_path.exists():
            print(f"  WARNING: {sign_adj_path.name} not found")
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', fontsize=14)
            ax.set_title(f'{title}\n(DIRECT)', fontsize=13, fontweight='bold')
            ax.axis('off')
            continue
        
        sign_adj = np.load(sign_adj_path)
        
        # Create graph
        G = nx.DiGraph()
        for name in neuron_names:
            G.add_node(name)
        
        edge_colors = []
        edge_list = []
        n_exc, n_inh = 0, 0
        
        for i in range(len(neuron_names)):
            for j in range(len(neuron_names)):
                if sign_adj[i, j] != 0:
                    G.add_edge(neuron_names[i], neuron_names[j])
                    edge_list.append((neuron_names[i], neuron_names[j]))
                    if sign_adj[i, j] > 0:
                        edge_colors.append('#e74c3c')
                        n_exc += 1
                    else:
                        edge_colors.append('#3498db')
                        n_inh += 1
        
        # Fixed layout for consistency
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray', 
                               edgecolors='black', linewidths=0.3, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_list, edge_color=edge_colors, 
                               arrows=True, arrowsize=5, alpha=0.4, 
                               width=0.4, ax=ax)
        
        # Label top hub nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]
        labels = {n: n for n in top_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
        
        ei_ratio = n_exc / n_inh if n_inh > 0 else float('inf')
        ax.set_title(f'{title}\nDIRECT: {n_exc + n_inh} edges (E:{n_exc}, I:{n_inh}, E:I={ei_ratio:.2f})', 
                     fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Common legend
    exc_patch = mpatches.Patch(color='#e74c3c', label='Excitatory')
    inh_patch = mpatches.Patch(color='#3498db', label='Inhibitory')
    fig.legend(handles=[exc_patch, inh_patch], loc='upper center', ncol=2, fontsize=12, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Figure 16: Direct Training - Phase-Specific Networks', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'fig16_direct_networks.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig16_direct_networks.png")
    plt.close()


# =============================================================================
# FIGURE 17: TRANSFER TRAINING PHASE NETWORKS
# =============================================================================

def create_fig17_transfer_networks():
    """Create networkx visualizations for transfer training per phase."""
    print("\n[Fig 17] Creating transfer training phase networks...")
    
    sbtg_adj_dir = RESULTS_DIR / 'stimulus_specific' / 'sbtg' / 'adjacencies'
    if not sbtg_adj_dir.exists():
        print("  No SBTG adjacencies found. Run: python pipeline/05_temporal_analysis.py --sbtg --transfer")
        return
    
    # Transfer only for stimulus phases (not baseline)
    phases = ['butanone', 'pentanedione', 'nacl']
    phase_titles = ['Butanone (60.5-70.5s)', 'Pentanedione (120.5-130.5s)', 'NaCl (180.5-190.5s)']
    
    # Load neuron names
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    # Load baseline for reference
    baseline_adj_path = sbtg_adj_dir / 'baseline_direct_sign_adj.npy'
    if baseline_adj_path.exists():
        baseline_adj = np.load(baseline_adj_path)
        baseline_edges = set()
        for i in range(len(neuron_names)):
            for j in range(len(neuron_names)):
                if baseline_adj[i, j] != 0:
                    baseline_edges.add((i, j))
    else:
        baseline_edges = set()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (phase, title) in enumerate(zip(phases, phase_titles)):
        ax = axes[idx]
        
        # Load transfer training adjacency
        sign_adj_path = sbtg_adj_dir / f'{phase}_transfer_sign_adj.npy'
        if not sign_adj_path.exists():
            print(f"  WARNING: {sign_adj_path.name} not found")
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', fontsize=14)
            ax.set_title(f'{title}\n(TRANSFER)', fontsize=13, fontweight='bold')
            ax.axis('off')
            continue
        
        sign_adj = np.load(sign_adj_path)
        
        # Create graph
        G = nx.DiGraph()
        for name in neuron_names:
            G.add_node(name)
        
        edge_colors = []
        edge_list = []
        n_exc, n_inh = 0, 0
        n_from_baseline = 0
        
        for i in range(len(neuron_names)):
            for j in range(len(neuron_names)):
                if sign_adj[i, j] != 0:
                    G.add_edge(neuron_names[i], neuron_names[j])
                    edge_list.append((neuron_names[i], neuron_names[j]))
                    
                    # Check if edge is from baseline
                    is_baseline_edge = (i, j) in baseline_edges
                    if is_baseline_edge:
                        n_from_baseline += 1
                    
                    if sign_adj[i, j] > 0:
                        # Lighter color for baseline-derived edges
                        edge_colors.append('#f5b7b1' if is_baseline_edge else '#e74c3c')
                        n_exc += 1
                    else:
                        edge_colors.append('#aed6f1' if is_baseline_edge else '#3498db')
                        n_inh += 1
        
        # Fixed layout for consistency
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray', 
                               edgecolors='black', linewidths=0.3, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_list, edge_color=edge_colors, 
                               arrows=True, arrowsize=5, alpha=0.5, 
                               width=0.5, ax=ax)
        
        # Label top hub nodes
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]
        labels = {n: n for n in top_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
        
        ei_ratio = n_exc / n_inh if n_inh > 0 else float('inf')
        n_phase_specific = (n_exc + n_inh) - n_from_baseline
        ax.set_title(f'{title}\nTRANSFER: {n_exc + n_inh} edges (E:{n_exc}, I:{n_inh})\n'
                     f'{n_from_baseline} baseline, {n_phase_specific} phase-specific', 
                     fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Legend with baseline vs phase-specific distinction
    exc_patch = mpatches.Patch(color='#e74c3c', label='Excitatory (phase)')
    exc_base_patch = mpatches.Patch(color='#f5b7b1', label='Excitatory (baseline)')
    inh_patch = mpatches.Patch(color='#3498db', label='Inhibitory (phase)')
    inh_base_patch = mpatches.Patch(color='#aed6f1', label='Inhibitory (baseline)')
    fig.legend(handles=[exc_patch, exc_base_patch, inh_patch, inh_base_patch], 
               loc='upper center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Figure 17: Transfer Learning - Phase-Specific Networks\n'
                 '(Pre-trained on Baseline, Fine-tuned on Stimulus)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(OUTPUT_DIR / 'fig17_transfer_networks.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig17_transfer_networks.png")
    plt.close()


# =============================================================================
# FIGURE 18: DIRECT vs TRANSFER SIDE-BY-SIDE
# =============================================================================

def create_fig18_direct_vs_transfer_sidebyside():
    """Create side-by-side comparison of direct vs transfer networks per phase."""
    print("\n[Fig 18] Creating direct vs transfer side-by-side comparison...")
    
    sbtg_adj_dir = RESULTS_DIR / 'stimulus_specific' / 'sbtg' / 'adjacencies'
    if not sbtg_adj_dir.exists():
        print("  No SBTG adjacencies found.")
        return
    
    phases = ['butanone', 'pentanedione', 'nacl']
    phase_titles = ['Butanone', 'Pentanedione', 'NaCl']
    
    # Load neuron names
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    
    for row, (phase, title) in enumerate(zip(phases, phase_titles)):
        for col, method in enumerate(['direct', 'transfer']):
            ax = axes[row, col]
            
            sign_adj_path = sbtg_adj_dir / f'{phase}_{method}_sign_adj.npy'
            if not sign_adj_path.exists():
                ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', fontsize=14)
                ax.set_title(f'{title} - {method.upper()}', fontsize=12, fontweight='bold')
                ax.axis('off')
                continue
            
            sign_adj = np.load(sign_adj_path)
            
            # Create graph
            G = nx.DiGraph()
            for name in neuron_names:
                G.add_node(name)
            
            edge_colors = []
            edge_list = []
            n_exc, n_inh = 0, 0
            
            for i in range(len(neuron_names)):
                for j in range(len(neuron_names)):
                    if sign_adj[i, j] != 0:
                        G.add_edge(neuron_names[i], neuron_names[j])
                        edge_list.append((neuron_names[i], neuron_names[j]))
                        if sign_adj[i, j] > 0:
                            edge_colors.append('#e74c3c')
                            n_exc += 1
                        else:
                            edge_colors.append('#3498db')
                            n_inh += 1
            
            # Use consistent layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            
            nx.draw_networkx_nodes(G, pos, node_size=40, node_color='lightgray', 
                                   edgecolors='black', linewidths=0.2, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_list, edge_color=edge_colors, 
                                   arrows=True, arrowsize=4, alpha=0.4, 
                                   width=0.3, ax=ax)
            
            # Labels for top hubs
            degrees = dict(G.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:8]
            labels = {n: n for n in top_nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)
            
            ei_ratio = n_exc / n_inh if n_inh > 0 else float('inf')
            method_label = 'DIRECT' if method == 'direct' else 'TRANSFER'
            ax.set_title(f'{title} - {method_label}\n{n_exc+n_inh} edges (E:{n_exc}, I:{n_inh}, E:I={ei_ratio:.2f})', 
                         fontsize=10, fontweight='bold')
            ax.axis('off')
    
    # Legend
    exc_patch = mpatches.Patch(color='#e74c3c', label='Excitatory')
    inh_patch = mpatches.Patch(color='#3498db', label='Inhibitory')
    fig.legend(handles=[exc_patch, inh_patch], loc='upper center', ncol=2, fontsize=11, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Figure 18: Direct vs Transfer Training - Side-by-Side Comparison', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'fig18_direct_vs_transfer_sidebyside.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig18_direct_vs_transfer_sidebyside.png")
    plt.close()


# =============================================================================
# FIGURE 19: STIMULUS-SPECIFIC NOVEL EDGES (Transfer edges not in baseline)
# =============================================================================

def create_fig19_novel_stimulus_edges():
    """
    Create figure showing edges that appear in transfer-trained stimulus graphs
    but NOT in the baseline graph.
    
    These represent "novel" connectivity patterns induced by the stimulus.
    """
    print("\n[Fig 19] Creating stimulus-specific novel edges figure...")
    
    sbtg_adj_dir = RESULTS_DIR / 'stimulus_specific' / 'sbtg' / 'adjacencies'
    if not sbtg_adj_dir.exists():
        print("  No SBTG adjacencies found. Run: python pipeline/05_temporal_analysis.py --sbtg --transfer")
        return
    
    # Load baseline (direct) for comparison
    baseline_path = sbtg_adj_dir / 'baseline_direct_sign_adj.npy'
    if not baseline_path.exists():
        print(f"  WARNING: baseline_direct_sign_adj.npy not found")
        return
    
    baseline_adj = np.load(baseline_path)
    baseline_mask = (baseline_adj != 0)  # Any edge in baseline
    
    # Load neuron names
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    phases = ['butanone', 'pentanedione', 'nacl']
    phase_titles = ['Butanone', 'Pentanedione', 'NaCl']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    all_novel_stats = []
    
    for idx, (phase, title) in enumerate(zip(phases, phase_titles)):
        ax = axes[idx]
        
        # Load transfer adjacency for this stimulus
        transfer_path = sbtg_adj_dir / f'{phase}_transfer_sign_adj.npy'
        if not transfer_path.exists():
            print(f"  WARNING: {transfer_path.name} not found")
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', fontsize=14)
            ax.set_title(f'{title}\n(Novel Edges)', fontsize=13, fontweight='bold')
            ax.axis('off')
            continue
        
        transfer_adj = np.load(transfer_path)
        transfer_mask = (transfer_adj != 0)
        
        # Find NOVEL edges: in transfer but NOT in baseline
        novel_mask = transfer_mask & ~baseline_mask
        novel_adj = transfer_adj * novel_mask  # Keep sign for novel edges only
        
        # Count statistics
        n_novel_exc = int((novel_adj > 0).sum())
        n_novel_inh = int((novel_adj < 0).sum())
        n_novel_total = n_novel_exc + n_novel_inh
        n_transfer_total = int(transfer_mask.sum())
        n_baseline_total = int(baseline_mask.sum())
        
        all_novel_stats.append({
            'phase': phase,
            'n_novel_exc': n_novel_exc,
            'n_novel_inh': n_novel_inh,
            'n_novel_total': n_novel_total,
            'n_transfer_total': n_transfer_total,
            'n_baseline_total': n_baseline_total,
            'pct_novel': n_novel_total / n_transfer_total * 100 if n_transfer_total > 0 else 0,
        })
        
        # Create graph with only novel edges
        G = nx.DiGraph()
        for name in neuron_names:
            G.add_node(name)
        
        edge_colors = []
        edge_list = []
        
        for i in range(len(neuron_names)):
            for j in range(len(neuron_names)):
                if novel_adj[i, j] != 0:
                    G.add_edge(neuron_names[i], neuron_names[j])
                    edge_list.append((neuron_names[i], neuron_names[j]))
                    if novel_adj[i, j] > 0:
                        edge_colors.append('#e74c3c')  # Red for excitatory
                    else:
                        edge_colors.append('#3498db')  # Blue for inhibitory
        
        # Use spring layout
        if len(edge_list) > 0:
            pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
            
            # Draw only nodes with novel edges
            nodes_with_edges = set()
            for src, tgt in edge_list:
                nodes_with_edges.add(src)
                nodes_with_edges.add(tgt)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, nodelist=list(nodes_with_edges), 
                                   node_size=60, node_color='lightyellow', 
                                   edgecolors='orange', linewidths=1.0, ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_list, edge_color=edge_colors, 
                                   arrows=True, arrowsize=6, alpha=0.6, 
                                   width=0.5, ax=ax)
            
            # Labels for top nodes with novel edges
            novel_degree = {}
            for node in nodes_with_edges:
                novel_degree[node] = sum(1 for e in edge_list if node in e)
            top_nodes = sorted(novel_degree, key=novel_degree.get, reverse=True)[:10]
            labels = {n: n for n in top_nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
        
        ei_ratio = n_novel_exc / n_novel_inh if n_novel_inh > 0 else float('inf')
        pct_novel = n_novel_total / n_transfer_total * 100 if n_transfer_total > 0 else 0
        
        ax.set_title(f'{title} - Novel Stimulus Edges\n'
                     f'{n_novel_total} novel edges (E:{n_novel_exc}, I:{n_novel_inh})\n'
                     f'{pct_novel:.1f}% of transfer graph not in baseline', 
                     fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Legend
    exc_patch = mpatches.Patch(color='#e74c3c', label='Excitatory (novel)')
    inh_patch = mpatches.Patch(color='#3498db', label='Inhibitory (novel)')
    fig.legend(handles=[exc_patch, inh_patch], loc='lower center', ncol=2, fontsize=11, 
               bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Figure 19: Stimulus-Specific Novel Edges\n'
                 '(Edges in transfer-trained graph that do NOT appear in baseline)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig19_novel_stimulus_edges.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig19_novel_stimulus_edges.png")
    plt.close()
    
    # Print summary
    print("\n  Novel Edge Summary:")
    for stats in all_novel_stats:
        print(f"    {stats['phase']}: {stats['n_novel_total']} novel edges "
              f"({stats['pct_novel']:.1f}% of transfer graph)")
    
    # Save stats to CSV (use RESULTS_DIR / tables to match other tables)
    tables_dir = RESULTS_DIR / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_df = pd.DataFrame(all_novel_stats)
    stats_df.to_csv(tables_dir / 'novel_stimulus_edges.csv', index=False)
    print(f"  Saved: novel_stimulus_edges.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate summary figures')
    parser.add_argument('--quick', action='store_true', help='Skip slow network figures')
    parser.add_argument('--regenerate-phases', action='store_true', 
                        help='Force regeneration of phase-specific SBTG data (trains new models)')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='Use pre-trained phase data from 05_temporal_analysis.py instead of regenerating')
    args = parser.parse_args()
    
    print("=" * 70)
    print("GENERATING SUMMARY FIGURES")
    print("=" * 70)
    
    ensure_output_dir()
    
    # Generate phase-specific SBTG data if needed (for Fig 7 and 9)
    # If --use-pretrained is set, skip regeneration entirely (use 05_temporal_analysis.py output)
    if not args.use_pretrained:
        generate_phase_sbtg_data(force_regenerate=args.regenerate_phases)
    else:
        print("\n[Pre] Using pre-trained phase data from 05_temporal_analysis.py")
    
    # Figure 1: Data overview (NEW)
    create_fig1_data_overview()
    
    # Always generate these
    create_fig2_imputation_stats()
    create_fig3_data_expansion()
    create_fig4_sbtg_intuition()
    create_fig5_sbtg_vs_baselines()
    create_fig6_predicted_vs_anatomical()
    create_fig7_ei_ratio_dynamics()
    
    # Network figures (can be slow)
    if not args.quick:
        create_fig8_aggregate_network()
        create_fig9_phase_networks()
        # NEW: Direct and Transfer network figures
        create_fig16_direct_networks()
        create_fig17_transfer_networks()
        create_fig18_direct_vs_transfer_sidebyside()
        create_fig19_novel_stimulus_edges()
    else:
        print("\n[--quick mode] Skipping network graph figures (Fig 8, 9, 16, 17, 18, 19)")
    
    # Weight correlation figures
    create_fig10_weight_correlation_bars()
    create_fig11_weight_scatter()
    create_fig12_signed_correlation()
    
    # Mean vs Volatility Test comparison
    create_fig14_mean_vs_volatility()
    
    # Direct vs Transfer training comparison (summary stats)
    create_fig15_direct_vs_transfer()
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")


# =============================================================================
# FIGURE 14: MEAN TEST vs VOLATILITY TEST COMPARISON
# =============================================================================

def create_fig14_mean_vs_volatility():
    """
    Create figure comparing Mean Test vs Volatility Test edge discovery.
    
    Panels:
    A. Edge count breakdown (Venn-style bar)
    B. AUROC comparison (mean-only vs combined vs vol-only)
    C. Spearman correlation comparison
    D. Top neurons with volatility-only connections
    """
    print("\n[Fig 14] Creating Mean vs Volatility test comparison...")
    
    # Look for model with volatility data
    models_dir = SBTG_DIR / 'models'
    vol_models = [f for f in models_dir.glob('*.npz') 
                  if 'volatility_adj' in np.load(f, allow_pickle=True).files]
    
    if not vol_models:
        print("  No models with volatility_adj found, skipping")
        return
    
    # Use most recent model with volatility data
    model_path = sorted(vol_models, key=lambda x: x.stat().st_mtime)[-1]
    print(f"  Using model: {model_path.name}")
    
    data = np.load(model_path, allow_pickle=True)
    sign_adj = data['sign_adj']
    vol_adj = data['volatility_adj'].astype(bool)
    mu_hat = data['mu_hat']
    
    with open(DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json') as f:
        neuron_names = json.load(f)
    
    n = len(neuron_names)
    n_possible = n * (n - 1)
    
    # Calculate edge sets
    mean_mask = (sign_adj != 0)
    vol_only_mask = vol_adj & ~mean_mask
    combined_mask = mean_mask | vol_adj
    
    n_mean = int(mean_mask.sum())
    n_vol_only = int(vol_only_mask.sum())
    n_combined = int(combined_mask.sum())
    n_exc = int((sign_adj > 0).sum())
    n_inh = int((sign_adj < 0).sum())
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # =========================================================================
    # PANEL A: Edge count breakdown
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    categories = ['Mean Test\n(signed)', 'Volatility\nOnly', 'Combined']
    counts = [n_mean, n_vol_only, n_combined]
    colors = ['#3498db', '#e74c3c', '#9b59b6']
    
    bars = ax1.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Edges', fontsize=12)
    ax1.set_title('A. Edge Discovery by Test Type', fontsize=14, fontweight='bold')
    
    for bar, count in zip(bars, counts):
        ax1.annotate(f'{count}\n({100*count/n_possible:.1f}%)', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(counts) * 1.25)
    
    # Add E/I breakdown annotation
    ax1.text(0.02, 0.98, f'Mean test breakdown:\n  • Excitatory: {n_exc}\n  • Inhibitory: {n_inh}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
    
    # =========================================================================
    # PANEL B: Load evaluation metrics and compare AUROC
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    eval_file = RESULTS_DIR / 'evaluation' / 'evaluation_results.csv'
    if eval_file.exists():
        df = pd.read_csv(eval_file)
        
        # Find matching entries for this model
        model_stem = model_path.stem
        
        # Extract AUROC for mean-only, combined, vol-only
        def get_auroc(suffix=''):
            name = f'sbtg_{model_stem}{suffix}'
            rows = df[df['name'] == name]
            if len(rows) > 0:
                cook_row = rows[rows['benchmark'] == 'cook']
                leifer_row = rows[rows['benchmark'] == 'leifer']
                return (cook_row['auroc'].values[0] if len(cook_row) > 0 else np.nan,
                        leifer_row['auroc'].values[0] if len(leifer_row) > 0 else np.nan)
            return (np.nan, np.nan)
        
        auroc_mean = get_auroc('')
        auroc_combined = get_auroc('_combined')
        auroc_vol_only = get_auroc('_vol_only')
        
        x = np.arange(3)
        width = 0.35
        
        cook_vals = [auroc_mean[0], auroc_combined[0], auroc_vol_only[0]]
        leifer_vals = [auroc_mean[1], auroc_combined[1], auroc_vol_only[1]]
        
        bars1 = ax2.bar(x - width/2, cook_vals, width, label=f'vs {STRUCTURAL_LABEL}', color='#3498db', edgecolor='black')
        bars2 = ax2.bar(x + width/2, leifer_vals, width, label=f'vs {FUNCTIONAL_LABEL}', color='#27ae60', edgecolor='black')
        
        ax2.set_ylabel('AUROC', fontsize=12)
        ax2.set_title('B. AUROC by Edge Set', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Mean Only', 'Combined', 'Vol Only'])
        ax2.legend(loc='upper right')
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax2.set_ylim(0.4, 0.75)
        
        # Annotate bars
        for bars in [bars1, bars2]:
            for bar in bars:
                if not np.isnan(bar.get_height()):
                    ax2.annotate(f'{bar.get_height():.3f}', 
                                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                 ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Run 04_evaluate.py first', ha='center', va='center', 
                 transform=ax2.transAxes)
    
    # =========================================================================
    # PANEL C: Spearman correlation comparison
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    if eval_file.exists():
        def get_spearman(suffix=''):
            name = f'sbtg_{model_stem}{suffix}'
            rows = df[df['name'] == name]
            if len(rows) > 0:
                cook_row = rows[rows['benchmark'] == 'cook']
                return cook_row['spearman_r'].values[0] if len(cook_row) > 0 else np.nan
            return np.nan
        
        spearman_mean = get_spearman('')
        spearman_combined = get_spearman('_combined')
        spearman_vol_only = get_spearman('_vol_only')
        
        x = np.arange(3)
        spearman_vals = [spearman_mean, spearman_combined, spearman_vol_only]
        
        bars = ax3.bar(x, spearman_vals, color=['#3498db', '#9b59b6', '#e74c3c'], 
                       edgecolor='black', linewidth=1.5)
        ax3.set_ylabel(f'Spearman ρ (vs {STRUCTURAL_LABEL})', fontsize=12)
        ax3.set_title('C. Weight Correlation by Edge Set', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Mean Only', 'Combined', 'Vol Only'])
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars, spearman_vals):
            if not np.isnan(val):
                ax3.annotate(f'{val:.3f}', 
                             xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                             ha='center', va='bottom' if val > 0 else 'top', fontsize=11, fontweight='bold')
    
    # =========================================================================
    # PANEL D: Top neurons with volatility-only connections
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    vol_out = vol_only_mask.sum(axis=1)
    vol_in = vol_only_mask.sum(axis=0)
    vol_total = vol_out + vol_in
    
    top_idx = np.argsort(vol_total)[::-1][:10]
    top_names = [neuron_names[i] for i in top_idx]
    top_counts = [int(vol_total[i]) for i in top_idx]
    
    bars = ax4.barh(top_names[::-1], top_counts[::-1], color='#e74c3c', edgecolor='black', linewidth=1)
    ax4.set_xlabel('Volatility-Only Edges (in + out)', fontsize=12)
    ax4.set_title('D. Top Neurons: Volatility Test', fontsize=14, fontweight='bold')
    
    for bar, count in zip(bars, top_counts[::-1]):
        ax4.annotate(f'{count}', xy=(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2),
                     ha='left', va='center', fontsize=10)
    ax4.set_xlim(0, max(top_counts) * 1.2)
    
    # Add biological interpretation
    ax4.text(0.98, 0.02, 
             'Volatility edges = sign-switching connections\n'
             'RIM/AVA: bidirectional motor command neurons',
             transform=ax4.transAxes, fontsize=9, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.9))
    
    plt.suptitle('Figure 14: Mean Test vs Volatility Test Edge Discovery', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(OUTPUT_DIR / 'fig14_mean_vs_volatility.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: fig14_mean_vs_volatility.png")
    plt.close()


if __name__ == "__main__":
    main()

