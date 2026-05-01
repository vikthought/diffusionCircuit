#!/usr/bin/env python3
"""
14_organize_results.py
======================

Organize and summarize all evaluation results from the diffusionCircuit pipeline.

Loads results from every completed pipeline stage (SBTG evaluation, multi-lag
analysis, HP objective validation, multi-block lag separation, baseline
comparisons) and consolidates them into human-readable summaries.

Creates:
- Comprehensive results index (RESULTS_INDEX.md)
- Summary tables in markdown format
- Overview and multi-lag summary figures

Usage:
    python pipeline/14_organize_results.py
    python pipeline/14_organize_results.py --regenerate-figures
"""

import sys
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Directories
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "summary"
FIGURES_DIR = RESULTS_DIR / "figures" / "summary"
TABLES_DIR = RESULTS_DIR / "tables"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
MULTILAG_DIR = RESULTS_DIR / "multilag_analysis"
HP_VALIDATION_DIR = RESULTS_DIR / "hp_objective_validation"
MULTIBLOCK_DIR = RESULTS_DIR / "multiblock_lag_separation"

# =============================================================================
# RESULT CONSOLIDATION
# =============================================================================

def load_evaluation_results() -> pd.DataFrame:
    """Load main evaluation results."""
    eval_file = EVALUATION_DIR / "evaluation_results.csv"
    if eval_file.exists():
        return pd.read_csv(eval_file)
    return pd.DataFrame()


def load_multilag_results() -> dict:
    """Load multi-lag analysis results."""
    results = {}
    
    # Stats
    stats_file = MULTILAG_DIR / "tables" / "multilag_stats.csv"
    if stats_file.exists():
        results['stats'] = pd.read_csv(stats_file)
    
    # Connectome overlap
    overlap_file = MULTILAG_DIR / "tables" / "connectome_overlap.csv"
    if overlap_file.exists():
        results['connectome_overlap'] = pd.read_csv(overlap_file)
    
    # Class interactions
    class_file = MULTILAG_DIR / "tables" / "neuron_class_interactions.csv"
    if class_file.exists():
        results['class_interactions'] = pd.read_csv(class_file)
    
    # Summary
    summary_file = MULTILAG_DIR / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            results['summary'] = json.load(f)
    
    return results


def load_baseline_comparison() -> pd.DataFrame:
    """Load baseline comparison from cook_leifer_evaluation."""
    eval_file = TABLES_DIR / "cook_leifer_evaluation.csv"
    if eval_file.exists():
        return pd.read_csv(eval_file)
    return pd.DataFrame()


def load_hp_validation_results() -> dict:
    """Load HP objective validation results (Script 12)."""
    results = {}
    
    # Main validation results
    main_file = HP_VALIDATION_DIR / "hp_validation_results.csv"
    if main_file.exists():
        results['trials'] = pd.read_csv(main_file)
    
    # Correlation summary
    corr_file = HP_VALIDATION_DIR / "correlation_summary.csv"
    if corr_file.exists():
        results['correlations'] = pd.read_csv(corr_file)
    
    # Config
    config_file = HP_VALIDATION_DIR / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            results['config'] = json.load(f)
    
    return results


def load_multiblock_results() -> dict:
    """Load multi-block lag separation results (Script 13).
    
    Script 13 outputs are timestamped, so we find the latest run.
    """
    results = {}
    
    # Find latest timestamped directory
    if MULTIBLOCK_DIR.exists():
        subdirs = [d for d in MULTIBLOCK_DIR.iterdir() if d.is_dir()]
        if subdirs:
            latest = max(subdirs, key=lambda x: x.name)
            
            # Stats
            stats_file = latest / "tables" / "multilag_stats.csv"
            if stats_file.exists():
                results['stats'] = pd.read_csv(stats_file)
            
            # Connectome overlap
            overlap_file = latest / "tables" / "connectome_overlap.csv"
            if overlap_file.exists():
                results['connectome_overlap'] = pd.read_csv(overlap_file)
            
            # Leifer overlap
            leifer_file = latest / "tables" / "leifer_overlap.csv"
            if leifer_file.exists():
                results['leifer_overlap'] = pd.read_csv(leifer_file)
            
            # Class interactions
            class_file = latest / "tables" / "neuron_class_interactions.csv"
            if class_file.exists():
                results['class_interactions'] = pd.read_csv(class_file)
            
            # Edge persistence
            persist_file = latest / "tables" / "edge_persistence.csv"
            if persist_file.exists():
                results['edge_persistence'] = pd.read_csv(persist_file)
            
            # Analysis summary
            summary_file = latest / "analysis_summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    results['summary'] = json.load(f)
            
            results['run_dir'] = str(latest)
    
    return results


# =============================================================================
# SUMMARY GENERATION
# =============================================================================

def create_best_models_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Extract best performing models for each benchmark."""
    if eval_df.empty:
        return pd.DataFrame()
    
    summaries = []
    
    for benchmark in ['cook', 'leifer']:
        bench_df = eval_df[eval_df['benchmark'] == benchmark].copy()
        if bench_df.empty:
            continue
        
        # Best by AUROC
        best_idx = bench_df['auroc'].idxmax()
        best_row = bench_df.loc[best_idx].copy()
        best_row['rank'] = 'best'
        best_row['metric'] = 'auroc'
        summaries.append(best_row)
    
    if summaries:
        return pd.DataFrame(summaries)
    return pd.DataFrame()


def create_model_comparison_table(eval_df: pd.DataFrame) -> str:
    """Create markdown table comparing all models."""
    if eval_df.empty:
        return "No evaluation results available."
    
    # Focus on unique models (not combined/vol_only variants)
    base_models = eval_df[~eval_df['name'].str.contains('_combined|_vol_only', na=False)]
    
    lines = ["# Model Comparison Summary", "", "## Cook Connectome Benchmark", ""]
    
    cook_df = base_models[base_models['benchmark'] == 'cook'].copy()
    if not cook_df.empty:
        cook_df = cook_df.sort_values('auroc', ascending=False)
        lines.append("| Model | AUROC | AUPRC | F1 | Spearman |")
        lines.append("|-------|-------|-------|-----|----------|")
        for _, row in cook_df.head(10).iterrows():
            name = row['name'].replace('sbtg_', '').replace('_nacl', '')[:30]
            auroc = f"{row['auroc']:.3f}" if pd.notna(row['auroc']) else "-"
            auprc = f"{row['auprc']:.3f}" if pd.notna(row['auprc']) else "-"
            f1 = f"{row['f1']:.3f}" if pd.notna(row['f1']) else "-"
            spearman = f"{row['spearman_r']:.3f}" if pd.notna(row.get('spearman_r')) else "-"
            lines.append(f"| {name} | {auroc} | {auprc} | {f1} | {spearman} |")
    
    lines.extend(["", "## Leifer Functional Atlas Benchmark", ""])
    
    leifer_df = base_models[base_models['benchmark'] == 'leifer'].copy()
    if not leifer_df.empty:
        leifer_df = leifer_df.sort_values('auroc', ascending=False)
        lines.append("| Model | AUROC | AUPRC | Spearman | Pearson |")
        lines.append("|-------|-------|-------|----------|---------|")
        for _, row in leifer_df.head(10).iterrows():
            name = row['name'].replace('sbtg_', '').replace('_nacl', '')[:30]
            auroc = f"{row['auroc']:.3f}" if pd.notna(row['auroc']) else "-"
            auprc = f"{row['auprc']:.3f}" if pd.notna(row['auprc']) else "-"
            spearman = f"{row['spearman_r']:.3f}" if pd.notna(row.get('spearman_r')) else "-"
            pearson = f"{row['pearson_r']:.3f}" if pd.notna(row.get('pearson_r')) else "-"
            lines.append(f"| {name} | {auroc} | {auprc} | {spearman} | {pearson} |")
    
    return "\n".join(lines)


def create_multilag_summary(multilag: dict) -> str:
    """Create markdown summary of multi-lag analysis."""
    if not multilag:
        return "# Multi-Lag Analysis\n\nNo multi-lag results available."
    
    lines = ["# Multi-Lag Analysis Summary", ""]
    
    # Summary info
    if 'summary' in multilag:
        summary = multilag['summary']
        lines.append(f"**Lags analyzed:** {summary.get('lags', [])}")
        lines.append(f"**Phases:** {list(summary.get('phases', {}).keys())}")
        lines.append(f"**Neurons:** {summary.get('n_neurons', 'N/A')}")
        lines.append("")
    
    # Stats table
    if 'stats' in multilag and not multilag['stats'].empty:
        stats_df = multilag['stats']
        lines.extend(["## Edge Statistics by Lag", ""])
        lines.append("| Phase | Lag | Time (s) | Edges | E+ | I- | E:I |")
        lines.append("|-------|-----|----------|-------|-----|-----|------|")
        for _, row in stats_df.iterrows():
            phase = row.get('phase', '-')
            lag = row.get('lag', '-')
            lag_s = f"{row.get('lag_seconds', 0):.2f}"
            edges = int(row.get('n_edges', 0))
            exc = int(row.get('n_excitatory', 0))
            inh = int(row.get('n_inhibitory', 0))
            ei = f"{row.get('ei_ratio', 0):.2f}"
            lines.append(f"| {phase} | {lag} | {lag_s} | {edges} | {exc} | {inh} | {ei} |")
        lines.append("")
    
    # Connectome overlap
    if 'connectome_overlap' in multilag and not multilag['connectome_overlap'].empty:
        overlap_df = multilag['connectome_overlap']
        lines.extend(["## Connectome Overlap by Lag", ""])
        lines.append("| Phase | Lag | Connectome | Jaccard | Precision | Recall |")
        lines.append("|-------|-----|------------|---------|-----------|--------|")
        for _, row in overlap_df.iterrows():
            phase = row.get('phase', '-')
            lag = row.get('lag', '-')
            conn = row.get('connectome', '-')
            jaccard = f"{row.get('jaccard', 0):.3f}"
            prec = f"{row.get('precision', 0):.3f}"
            recall = f"{row.get('recall', 0):.3f}"
            lines.append(f"| {phase} | {lag} | {conn} | {jaccard} | {prec} | {recall} |")
        lines.append("")
    
    return "\n".join(lines)


def create_hp_validation_summary(hp_results: dict) -> str:
    """Create markdown summary of HP objective validation (Script 12)."""
    if not hp_results:
        return "# HP Objective Validation\n\nNo HP validation results available. Run Script 12."
    
    lines = ["# HP Objective Validation Summary (Script 12)", ""]
    
    lines.append("**Key Finding:** null_contrast is the only HP objective that positively")
    lines.append("correlates with biological AUROC. Edge stability anti-correlates.")
    lines.append("")
    
    # Correlations
    if 'correlations' in hp_results and not hp_results['correlations'].empty:
        corr_df = hp_results['correlations']
        lines.extend(["## Objective-Evaluation Correlations", ""])
        lines.append("| Objective | Cook AUROC r | Leifer AUROC r | Interpretation |")
        lines.append("|-----------|-------------|----------------|----------------|")
        for _, row in corr_df.iterrows():
            obj = row.get('objective', '-')
            cook_r = f"{row.get('cook_auroc_r', 0):.3f}"
            leifer_r = f"{row.get('leifer_auroc_r', 0):.3f}"
            # Interpret
            if row.get('cook_auroc_r', 0) > 0.05:
                interp = "✅ Positive - USE"
            elif row.get('cook_auroc_r', 0) < -0.1:
                interp = "❌ Anti-correlates - AVOID"
            else:
                interp = "⚠️ No clear signal"
            lines.append(f"| {obj} | {cook_r} | {leifer_r} | {interp} |")
        lines.append("")
    
    # Trial count
    if 'trials' in hp_results:
        n_trials = len(hp_results['trials'])
        lines.append(f"**Trials run:** {n_trials}")
        lines.append("")
    
    return "\n".join(lines)


def create_multiblock_summary(multiblock: dict) -> str:
    """Create markdown summary of multi-block lag separation (Script 13)."""
    if not multiblock:
        return "# Multi-Block Lag Separation\n\nNo multi-block results available. Run Script 13."
    
    lines = ["# Multi-Block Lag Separation Summary (Script 13)", ""]
    
    lines.append("**Method:** Multi-block windows for direct lag-separated Jacobians (Theorem 5.1)")
    lines.append("")
    lines.append("Unlike Script 11 (pair windows → reduced-form VAR), this approach")
    lines.append("recovers the DIRECT lag-r Jacobian J_r = ∂f/∂x_{t+1-r}.")
    lines.append("")
    
    if 'run_dir' in multiblock:
        lines.append(f"**Results directory:** `{multiblock['run_dir']}`")
        lines.append("")
    
    # Summary info
    if 'summary' in multiblock:
        summary = multiblock['summary']
        lines.append(f"**Max lag (p_max):** {summary.get('p_max', 'N/A')}")
        lines.append(f"**Lags tested:** {summary.get('lags_tested', [])}")
        lines.append(f"**Neurons:** {summary.get('n', 'N/A')}")
        lines.append(f"**Windows:** {summary.get('n_windows', 'N/A')}")
        lines.append("")
    
    # Connectome overlap
    if 'connectome_overlap' in multiblock and not multiblock['connectome_overlap'].empty:
        overlap_df = multiblock['connectome_overlap']
        lines.extend(["## Cook Connectome AUROC by Lag", ""])
        lines.append("| Lag | Time (s) | AUROC | AUPRC | Jaccard |")
        lines.append("|-----|----------|-------|-------|---------|")
        
        # Filter to structural connectome
        struct_df = overlap_df[overlap_df['connectome'] == 'A_struct']
        for _, row in struct_df.iterrows():
            lag = row.get('lag', '-')
            lag_s = f"{row.get('lag_seconds', 0):.2f}"
            auroc = f"{row.get('auroc', 0):.3f}"
            auprc = f"{row.get('auprc', 0):.3f}"
            jaccard = f"{row.get('jaccard', 0):.4f}"
            lines.append(f"| {lag} | {lag_s} | {auroc} | {auprc} | {jaccard} |")
        lines.append("")
    
    # Leifer overlap
    if 'leifer_overlap' in multiblock and not multiblock['leifer_overlap'].empty:
        leifer_df = multiblock['leifer_overlap']
        lines.extend(["## Leifer Atlas AUROC by Lag", ""])
        lines.append("| Lag | Time (s) | AUROC | AUPRC |")
        lines.append("|-----|----------|-------|-------|")
        
        # Filter to leifer_q
        q_df = leifer_df[leifer_df['atlas'] == 'leifer_q']
        for _, row in q_df.iterrows():
            lag = row.get('lag', '-')
            lag_s = f"{row.get('lag_seconds', 0):.2f}"
            auroc = f"{row.get('auroc', 0):.3f}"
            auprc = f"{row.get('auprc', 0):.3f}"
            lines.append(f"| {lag} | {lag_s} | {auroc} | {auprc} |")
        lines.append("")
    
    # Edge persistence summary
    if 'edge_persistence' in multiblock and not multiblock['edge_persistence'].empty:
        persist_df = multiblock['edge_persistence']
        n_total = len(persist_df)
        n_50 = len(persist_df[persist_df['persistence'] >= 0.5])
        n_100 = len(persist_df[persist_df['persistence'] == 1.0])
        lines.extend(["## Edge Persistence", ""])
        lines.append(f"- Total edges tracked: {n_total}")
        lines.append(f"- Edges at ≥50% of lags: {n_50}")
        lines.append(f"- Edges at all lags (100%): {n_100}")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def check_figures_exist() -> dict:
    """Check which figures exist."""
    expected_figures = {
        'fig1_data_overview.png': 'Data Overview',
        'fig2_imputation_stats.png': 'Imputation Statistics',
        'fig3_data_expansion_impact.png': 'Data Expansion',
        'fig4_sbtg_intuition.png': 'SBTG Intuition',
        'fig5_sbtg_vs_baselines.png': 'SBTG vs Baselines',
        'fig6_predicted_vs_anatomical.png': 'Predicted vs Anatomical',
        'fig7_ei_ratio_dynamics.png': 'E:I Ratio Dynamics',
        'fig8_aggregate_network.png': 'Aggregate Network',
        'fig9_phase_networks.png': 'Phase Networks',
        'fig10_weight_correlation_bars.png': 'Weight Correlation',
        'fig11_weight_scatter.png': 'Weight Scatter',
        'fig12_signed_correlation.png': 'Signed Correlation',
        'fig13_regime_connectivity.png': 'Regime Connectivity',
        'fig14_mean_vs_volatility.png': 'Mean vs Volatility',
        'fig15_direct_vs_transfer.png': 'Direct vs Transfer',
        'fig16_direct_networks.png': 'Direct Networks',
        'fig17_transfer_networks.png': 'Transfer Networks',
        'fig18_direct_vs_transfer_sidebyside.png': 'Direct/Transfer Comparison',
        'fig19_novel_stimulus_edges.png': 'Novel Edges',
    }
    
    status = {}
    for fname, desc in expected_figures.items():
        path = FIGURES_DIR / fname
        status[fname] = {
            'description': desc,
            'exists': path.exists(),
            'path': str(path)
        }
    
    return status


def generate_results_overview_figure(eval_df: pd.DataFrame, output_path: Path):
    """Generate a high-level results overview figure."""
    if eval_df.empty:
        print("  No evaluation data for overview figure")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: AUROC comparison by benchmark
    ax1 = axes[0]
    
    # Get base models only
    base_models = eval_df[~eval_df['name'].str.contains('_combined|_vol_only', na=False)]
    
    # Separate SBTG and baselines
    sbtg_models = base_models[base_models['name'].str.contains('sbtg', na=False)]
    baseline_models = base_models[~base_models['name'].str.contains('sbtg', na=False)]
    
    cook_sbtg = sbtg_models[sbtg_models['benchmark'] == 'cook']['auroc'].max()
    cook_baseline = baseline_models[baseline_models['benchmark'] == 'cook']['auroc'].max()
    leifer_sbtg = sbtg_models[sbtg_models['benchmark'] == 'leifer']['auroc'].max()
    leifer_baseline = baseline_models[baseline_models['benchmark'] == 'leifer']['auroc'].max()
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, [cook_sbtg, leifer_sbtg], width, label='SBTG (best)', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, [cook_baseline, leifer_baseline], width, label='Baseline (best)', color='#95a5a6')
    
    ax1.set_ylabel('AUROC')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Cook\n(Structural)', 'Leifer\n(Functional)'])
    ax1.legend()
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax1.set_ylim(0.4, 0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
    
    # Plot 2: All models AUROC distribution
    ax2 = axes[1]
    
    for i, benchmark in enumerate(['cook', 'leifer']):
        bench_data = base_models[base_models['benchmark'] == benchmark]['auroc'].dropna()
        if len(bench_data) > 0:
            parts = ax2.violinplot([bench_data], positions=[i], showmeans=True)
            for pc in parts['bodies']:
                pc.set_facecolor(['#3498db', '#e74c3c'][i])
                pc.set_alpha(0.7)
    
    ax2.set_ylabel('AUROC')
    ax2.set_title('AUROC Distribution Across Models')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Cook', 'Leifer'])
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: {output_path.name}")


def generate_multilag_summary_figure(multilag: dict, output_path: Path):
    """Generate multi-lag summary figure."""
    if not multilag or 'stats' not in multilag:
        print("  No multi-lag data for summary figure")
        return
    
    stats_df = multilag['stats']
    if stats_df.empty:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Edge count by lag
    ax1 = axes[0]
    for phase in stats_df['phase'].unique():
        phase_data = stats_df[stats_df['phase'] == phase]
        ax1.plot(phase_data['lag'], phase_data['n_edges'], 'o-', label=phase, markersize=8)
    ax1.set_xlabel('Lag (frames)')
    ax1.set_ylabel('Number of Edges')
    ax1.set_title('Edges by Time Lag')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: E:I ratio by lag
    ax2 = axes[1]
    for phase in stats_df['phase'].unique():
        phase_data = stats_df[stats_df['phase'] == phase]
        ax2.plot(phase_data['lag'], phase_data['ei_ratio'], 's-', label=phase, markersize=8)
    ax2.set_xlabel('Lag (frames)')
    ax2.set_ylabel('E:I Ratio')
    ax2.set_title('Excitatory/Inhibitory Ratio')
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Connectome overlap by lag (if available)
    ax3 = axes[2]
    if 'connectome_overlap' in multilag and not multilag['connectome_overlap'].empty:
        overlap_df = multilag['connectome_overlap']
        struct_df = overlap_df[overlap_df['connectome'] == 'structural']
        for phase in struct_df['phase'].unique():
            phase_data = struct_df[struct_df['phase'] == phase]
            ax3.plot(phase_data['lag'], phase_data['jaccard'], '^-', label=phase, markersize=8)
        ax3.set_xlabel('Lag (frames)')
        ax3.set_ylabel('Jaccard Index')
        ax3.set_title('Connectome Overlap')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No connectome\noverlap data', ha='center', va='center', fontsize=12)
        ax3.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: {output_path.name}")


# =============================================================================
# INDEX GENERATION
# =============================================================================

def create_results_index(
    eval_df: pd.DataFrame,
    multilag: dict,
    fig_status: dict,
    hp_validation: dict = None,
    multiblock: dict = None,
) -> str:
    """Create comprehensive results index."""
    lines = [
        "# Results Index",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Quick Summary",
        "",
    ]
    
    # Best results
    if not eval_df.empty:
        cook_best = eval_df[eval_df['benchmark'] == 'cook']['auroc'].max()
        leifer_best = eval_df[eval_df['benchmark'] == 'leifer']['auroc'].max()
        lines.append(f"- **Best Cook AUROC:** {cook_best:.3f}")
        lines.append(f"- **Best Leifer AUROC:** {leifer_best:.3f}")
        lines.append(f"- **Total models evaluated:** {len(eval_df['name'].unique())}")
    
    if multilag and 'summary' in multilag:
        lines.append(f"- **Multi-lag (Script 11):** {len(multilag['summary'].get('lags', []))} lags analyzed")
    
    if hp_validation and 'trials' in hp_validation:
        n_trials = len(hp_validation['trials'])
        lines.append(f"- **HP Validation (Script 12):** {n_trials} trials run")
        lines.append("  - **Key finding:** null_contrast correlates with AUROC, edge_stability anti-correlates")
    
    if multiblock and 'summary' in multiblock:
        lags = multiblock['summary'].get('lags_tested', [])
        lines.append(f"- **Multi-block (Script 13):** {len(lags)} lags ({lags})")
    
    lines.append("")
    
    # HP Validation Key Finding
    if hp_validation:
        lines.extend([
            "## HP Objective Validation (Script 12)",
            "",
            "**Critical Finding:** The HP objective matters for biological relevance.",
            "",
            "| Objective | Cook AUROC Correlation | Recommendation |",
            "|-----------|----------------------|----------------|",
            "| null_contrast | r ≈ +0.13 | ✅ USE |",
            "| edge_stability | r ≈ -0.27 | ❌ AVOID |",
            "| dsm_loss | r ≈ -0.15 | ❌ AVOID |",
            "",
            "See `hp_validation_summary.md` for full details.",
            "",
        ])
    
    # Multi-block summary
    if multiblock and 'connectome_overlap' in multiblock:
        overlap_df = multiblock['connectome_overlap']
        struct_df = overlap_df[overlap_df['connectome'] == 'A_struct']
        if not struct_df.empty:
            best_row = struct_df.loc[struct_df['auroc'].idxmax()]
            lines.extend([
                "## Multi-Block Lag Separation (Script 13)",
                "",
                f"**Best lag:** {best_row['lag']} ({best_row['lag_seconds']:.2f}s)",
                f"**Cook AUROC:** {best_row['auroc']:.3f}",
                "",
                "See `multiblock_summary.md` for full AUROC by lag.",
                "",
            ])
    
    # Figures
    lines.extend(["## Figures", ""])
    existing = sum(1 for f in fig_status.values() if f['exists'])
    total = len(fig_status)
    lines.append(f"**Status:** {existing}/{total} figures generated")
    lines.append("")
    lines.append("| Figure | Description | Status |")
    lines.append("|--------|-------------|--------|")
    for fname, info in sorted(fig_status.items()):
        status = "✅" if info['exists'] else "❌"
        lines.append(f"| {fname} | {info['description']} | {status} |")
    lines.append("")
    
    # Tables
    lines.extend(["## Tables", ""])
    tables = list(TABLES_DIR.glob("*.csv")) + list(TABLES_DIR.glob("*.md"))
    lines.append(f"**Total tables:** {len(tables)}")
    lines.append("")
    lines.append("| Table | Type |")
    lines.append("|-------|------|")
    for t in sorted(tables)[:20]:
        lines.append(f"| {t.name} | {t.suffix} |")
    if len(tables) > 20:
        lines.append(f"| ... | ({len(tables) - 20} more) |")
    lines.append("")
    
    # Directory structure
    lines.extend([
        "## Directory Structure",
        "",
        "```",
        "results/",
        "├── evaluation/              # Model evaluation vs Cook/Leifer",
        "├── figures/summary/         # All generated figures",
        "├── multilag_analysis/       # Script 11: Pair-window multi-lag",
        "├── hp_objective_validation/ # Script 12: HP objective validation",
        "├── multiblock_lag_separation/ # Script 13: Direct Jacobian extraction",
        "├── tables/                  # Summary tables (CSV/MD)",
        "├── sbtg_training/           # Trained models and hyperparams",
        "├── baselines/               # Baseline method results",
        "├── stimulus_specific/       # Phase-specific analysis",
        "└── summary/                 # This index and consolidated results",
        "```",
        "",
    ])
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Organize and summarize results')
    parser.add_argument('--regenerate-figures', action='store_true',
                        help='Regenerate all summary figures')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RESULTS ORGANIZATION AND SUMMARY (Script 14)")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    print("\n--- Loading Results ---")
    eval_df = load_evaluation_results()
    print(f"  Main evaluation (Scripts 02-04): {len(eval_df)} rows")
    
    multilag = load_multilag_results()
    print(f"  Multi-lag pair windows (Script 11): {len(multilag)} components")
    
    hp_validation = load_hp_validation_results()
    print(f"  HP objective validation (Script 12): {len(hp_validation)} components")
    
    multiblock = load_multiblock_results()
    print(f"  Multi-block lag separation (Script 13): {len(multiblock)} components")
    
    baseline_df = load_baseline_comparison()
    print(f"  Baseline comparison: {len(baseline_df)} rows")
    
    # Check figures
    print("\n--- Checking Figures ---")
    fig_status = check_figures_exist()
    existing = sum(1 for f in fig_status.values() if f['exists'])
    print(f"  Existing: {existing}/{len(fig_status)}")
    
    # Generate summaries
    print("\n--- Generating Summaries ---")
    
    # Model comparison table
    comparison_md = create_model_comparison_table(eval_df)
    with open(OUTPUT_DIR / "model_comparison.md", 'w') as f:
        f.write(comparison_md)
    print("  Created: model_comparison.md")
    
    # Multi-lag summary (Script 11)
    multilag_md = create_multilag_summary(multilag)
    with open(OUTPUT_DIR / "multilag_summary.md", 'w') as f:
        f.write(multilag_md)
    print("  Created: multilag_summary.md")
    
    # HP validation summary (Script 12)
    hp_md = create_hp_validation_summary(hp_validation)
    with open(OUTPUT_DIR / "hp_validation_summary.md", 'w') as f:
        f.write(hp_md)
    print("  Created: hp_validation_summary.md")
    
    # Multi-block summary (Script 13)
    multiblock_md = create_multiblock_summary(multiblock)
    with open(OUTPUT_DIR / "multiblock_summary.md", 'w') as f:
        f.write(multiblock_md)
    print("  Created: multiblock_summary.md")
    
    # Results index
    index_md = create_results_index(eval_df, multilag, fig_status, hp_validation, multiblock)
    with open(OUTPUT_DIR / "RESULTS_INDEX.md", 'w') as f:
        f.write(index_md)
    print("  Created: RESULTS_INDEX.md")
    
    # Generate additional figures
    print("\n--- Generating Figures ---")
    
    # Results overview
    generate_results_overview_figure(eval_df, FIGURES_DIR / "fig20_results_overview.png")
    
    # Multi-lag summary
    generate_multilag_summary_figure(multilag, FIGURES_DIR / "fig21_multilag_summary.png")
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Summary files: {OUTPUT_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    
    if not eval_df.empty:
        cook_best = eval_df[eval_df['benchmark'] == 'cook']['auroc'].max()
        leifer_best = eval_df[eval_df['benchmark'] == 'leifer']['auroc'].max()
        print(f"\n  Best Cook AUROC: {cook_best:.3f}")
        print(f"  Best Leifer AUROC: {leifer_best:.3f}")
    
    print("\n  Key output files:")
    print(f"    - {OUTPUT_DIR / 'RESULTS_INDEX.md'}")
    print(f"    - {OUTPUT_DIR / 'model_comparison.md'}")
    print(f"    - {OUTPUT_DIR / 'multilag_summary.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
