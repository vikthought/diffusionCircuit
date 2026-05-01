#!/usr/bin/env python3
"""
SCRIPT 09: Generate Neuron Significance Tables
===============================================

Creates tables identifying the most significant excitatory and inhibitory
neurons discovered by SBTG for cross-referencing with C. elegans literature.

Tables Generated:
1. Global top E/I neurons (from imputed_best model)
2. Temporal/phase-specific top E/I neurons
3. Hub neurons (highest connectivity)

Usage:
    python pipeline/09_neuron_tables.py
    
Outputs:
    results/tables/top_neurons_global.md (and .csv)
    results/tables/top_neurons_by_phase.md (and .csv)
    results/tables/hub_neurons.md (and .csv)
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
SBTG_DIR = RESULTS_DIR / "sbtg_training"
TEMPORAL_DIR = RESULTS_DIR / "sbtg_temporal"
TABLES_DIR = RESULTS_DIR / "tables"
DATASETS_DIR = RESULTS_DIR / "intermediate" / "datasets"

# Import configurations
try:
    from pipeline.configs.phase_optimal_params import PHASE_OPTIMAL_PARAMS
except ImportError:
    PHASE_OPTIMAL_PARAMS = {}

from pipeline.config import OPTIMIZED_HYPERPARAMS


# =============================================================================
# NEURON METRICS COMPUTATION
# =============================================================================

def compute_neuron_metrics(mu_hat: np.ndarray, sign_adj: np.ndarray, 
                          neuron_names: List[str]) -> pd.DataFrame:
    """Compute significance metrics for all neurons."""
    n = len(neuron_names)
    metrics = []
    for i, name in enumerate(neuron_names):
        # Outgoing
        out_exc_strength = np.sum(mu_hat[i, :] * (sign_adj[i, :] == 1))
        out_inh_strength = np.sum(np.abs(mu_hat[i, :]) * (sign_adj[i, :] == -1))
        out_exc_count = np.sum(sign_adj[i, :] == 1)
        out_inh_count = np.sum(sign_adj[i, :] == -1)
        # Incoming
        in_exc_strength = np.sum(mu_hat[:, i] * (sign_adj[:, i] == 1))
        in_inh_strength = np.sum(np.abs(mu_hat[:, i]) * (sign_adj[:, i] == -1))
        in_exc_count = np.sum(sign_adj[:, i] == 1)
        in_inh_count = np.sum(sign_adj[:, i] == -1)
        # Totals
        total_connections = out_exc_count + out_inh_count + in_exc_count + in_inh_count
        total_strength = out_exc_strength + out_inh_strength + in_exc_strength + in_inh_strength
        strength = total_strength  # Standard weighted degree (Opsahl et al. 2010)
        metrics.append({
            'neuron': name,
            'out_exc_strength': out_exc_strength,
            'out_inh_strength': out_inh_strength,
            'in_exc_strength': in_exc_strength,
            'in_inh_strength': in_inh_strength,
            'out_exc_count': int(out_exc_count),
            'out_inh_count': int(out_inh_count),
            'in_exc_count': int(in_exc_count),
            'in_inh_count': int(in_inh_count),
            'total_connections': int(total_connections),
            'total_strength': total_strength,
            'strength': strength,
        })
    return pd.DataFrame(metrics)


# =============================================================================
# TABLE 1: GLOBAL TOP NEURONS
# =============================================================================

def generate_global_table():
    """Generate table of top E/I neurons from imputed_best model."""
    print("\n[Table 1] Global Top E/I Neurons")
    print("=" * 60)
    
    # Find imputed_best model
    models_dir = SBTG_DIR / "models"
    imputed_models = list(models_dir.glob("*imputed_best*.npz"))
    
    if not imputed_models:
        print("  ERROR: No imputed_best model found")
        return
    
    model_file = imputed_models[0]
    print(f"  Model: {model_file.name}")
    
    # Load model
    data = np.load(model_file, allow_pickle=True)
    mu_hat = data['mu_hat']
    sign_adj = data['sign_adj']
    
    # Load neuron names
    neuron_file = DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json'
    with open(neuron_file) as f:
        neuron_names = json.load(f)
    
    # Compute metrics
    df = compute_neuron_metrics(mu_hat, sign_adj, neuron_names)
    
    # Top excitatory neurons (by outgoing excitatory strength)
    top_exc = df.nlargest(20, 'out_exc_strength')[
        ['neuron', 'out_exc_strength', 'out_exc_count', 'in_exc_count', 'total_connections']
    ].copy()
    top_exc.columns = ['Neuron', 'Out E Strength', 'Out E Edges', 'In E Edges', 'Total Edges']
    
    # Top inhibitory neurons
    top_inh = df.nlargest(20, 'out_inh_strength')[
        ['neuron', 'out_inh_strength', 'out_inh_count', 'in_inh_count', 'total_connections']
    ].copy()
    top_inh.columns = ['Neuron', 'Out I Strength', 'Out I Edges', 'In I Edges', 'Total Edges']
    
    # Save markdown
    md_file = TABLES_DIR / 'top_neurons_global.md'
    with open(md_file, 'w') as f:
        f.write("# Top Excitatory and Inhibitory Neurons (Global)\n\n")
        f.write(f"**Model:** {model_file.name}\n")
        f.write(f"**Total Neurons:** {len(neuron_names)}\n\n")
        
        f.write("## Top 20 Excitatory Neurons (by outgoing strength)\n\n")
        f.write(top_exc.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Top 20 Inhibitory Neurons (by outgoing strength)\n\n")
        f.write(top_inh.to_markdown(index=False))
        f.write("\n")
    
    # Save CSV
    csv_file = TABLES_DIR / 'top_neurons_global.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"  Saved: {md_file.name}")
    print(f"  Saved: {csv_file.name}")
    print(f"\n  Top excitatory: {top_exc['Neuron'].iloc[0]} (strength: {top_exc['Out E Strength'].iloc[0]:.3f})")
    print(f"  Top inhibitory: {top_inh['Neuron'].iloc[0]} (strength: {top_inh['Out I Strength'].iloc[0]:.3f})")


# =============================================================================
# TABLE 2: PHASE-SPECIFIC TOP NEURONS (Baseline + Stimuli)
# =============================================================================

def generate_phase_tables():
    """
    Generate tables of top E/I neurons for each temporal phase.
    
    Uses the phase-specific SBTG results from 08_generate_figures.py:
    - sign_adj_baseline.npy
    - sign_adj_butanone.npy
    - sign_adj_pentanedione.npy
    - sign_adj_nacl.npy
    """
    print("\n[Table 2] Phase-Specific Top Neurons (SBTG-trained)")
    print("=" * 60)
    
    # Check if phase results exist
    phase_results_file = TEMPORAL_DIR / "phase_results.json"
    if not phase_results_file.exists():
        print("  ERROR: Phase results not found. Run:")
        print("    python pipeline/08_generate_figures.py --regenerate-phases")
        return
    
    with open(phase_results_file) as f:
        phase_results = json.load(f)
    
    # Load neuron names
    neuron_file = DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json'
    if not neuron_file.exists():
        print(f"  ERROR: Neuron names not found at {neuron_file}")
        return
    
    with open(neuron_file) as f:
        neuron_names = json.load(f)
    
    n_neurons = len(neuron_names)
    
    # Phase definitions
    phases = {
        'baseline': 'Baseline (0-60s)',
        'butanone': 'Butanone Stimulus (60.5-70.5s)',
        'pentanedione': 'Pentanedione Stimulus (120.5-130.5s)',
        'nacl': 'NaCl Stimulus (180.5-190.5s)',
    }
    
    all_phases_data = []
    
    for phase_key, phase_label in phases.items():
        sign_adj_file = TEMPORAL_DIR / f"sign_adj_{phase_key}.npy"
        
        if not sign_adj_file.exists():
            print(f"  WARNING: {sign_adj_file.name} not found, skipping")
            continue
        
        sign_adj = np.load(sign_adj_file)
        
        # For phase-specific analysis, we don't have mu_hat (coupling strengths)
        # We'll use the sign_adj counts directly
        # Create a synthetic mu_hat from sign_adj for edge counting
        # Note: This gives edge counts, not weighted strengths
        mu_hat = sign_adj.astype(float)  # -1, 0, +1
        
        phase_info = phase_results.get(phase_key, {})
        method = phase_info.get('method', 'unknown')
        ei_ratio = phase_info.get('ei_ratio', 0)
        n_total = phase_info.get('n_total', 0)
        
        print(f"\n  {phase_label}")
        print(f"    Method: {method}, Edges: {n_total}, E:I = {ei_ratio}")
        
        # Compute metrics
        df = compute_neuron_metrics(mu_hat, sign_adj, neuron_names)
        
        # Top excitatory sources (neurons with most outgoing excitatory edges)
        top_exc_out = df.nlargest(10, 'out_exc_count')[
            ['neuron', 'out_exc_count', 'in_exc_count', 'total_connections']
        ].copy()
        top_exc_out.columns = ['Neuron', 'Out E Edges', 'In E Edges', 'Total Edges']
        
        # Top inhibitory sources (neurons with most outgoing inhibitory edges)
        top_inh_out = df.nlargest(10, 'out_inh_count')[
            ['neuron', 'out_inh_count', 'in_inh_count', 'total_connections']
        ].copy()
        top_inh_out.columns = ['Neuron', 'Out I Edges', 'In I Edges', 'Total Edges']
        
        # Top excitatory targets (neurons receiving most excitatory input)
        top_exc_in = df.nlargest(10, 'in_exc_count')[
            ['neuron', 'in_exc_count', 'out_exc_count', 'total_connections']
        ].copy()
        top_exc_in.columns = ['Neuron', 'In E Edges', 'Out E Edges', 'Total Edges']
        
        # Top inhibitory targets (neurons receiving most inhibitory input)
        top_inh_in = df.nlargest(10, 'in_inh_count')[
            ['neuron', 'in_inh_count', 'out_inh_count', 'total_connections']
        ].copy()
        top_inh_in.columns = ['Neuron', 'In I Edges', 'Out I Edges', 'Total Edges']
        
        # Hub neurons (highest total connectivity)
        hubs = df.nlargest(10, 'total_connections')[
            ['neuron', 'total_connections', 'out_exc_count', 'out_inh_count', 
             'in_exc_count', 'in_inh_count']
        ].copy()
        hubs.columns = ['Neuron', 'Total Edges', 'Out E', 'Out I', 'In E', 'In I']
        
        all_phases_data.append({
            'phase_key': phase_key,
            'phase_label': phase_label,
            'method': method,
            'ei_ratio': ei_ratio,
            'n_total': n_total,
            'n_exc': phase_info.get('n_positive', 0),
            'n_inh': phase_info.get('n_negative', 0),
            'top_exc_out': top_exc_out,
            'top_inh_out': top_inh_out,
            'top_exc_in': top_exc_in,
            'top_inh_in': top_inh_in,
            'hubs': hubs,
            'full_metrics': df,
        })
    
    if not all_phases_data:
        print("  No phase data found!")
        return
    
    # Save main markdown file
    md_file = TABLES_DIR / 'top_neurons_by_phase.md'
    with open(md_file, 'w') as f:
        f.write("# Top Neurons by Temporal Phase (SBTG-Trained)\n\n")
        f.write("**Method:** Each phase was trained with a separate SBTG model using the same\n")
        f.write("hyperparameters as the best-performing imputed model (linear, 200 epochs).\n\n")
        f.write("---\n\n")
        
        for phase_data in all_phases_data:
            phase_label = phase_data['phase_label']
            method = phase_data['method']
            ei_ratio = phase_data['ei_ratio']
            n_total = phase_data['n_total']
            n_exc = phase_data['n_exc']
            n_inh = phase_data['n_inh']
            
            f.write(f"## {phase_label}\n\n")
            f.write(f"**Statistics:** {n_total} edges (E: {n_exc}, I: {n_inh}), ")
            f.write(f"E:I ratio = {ei_ratio}, Method: {method}\n\n")
            
            f.write("### Top 10 Excitatory Sources (most outgoing E edges)\n\n")
            f.write(phase_data['top_exc_out'].to_markdown(index=False))
            f.write("\n\n")
            
            f.write("### Top 10 Inhibitory Sources (most outgoing I edges)\n\n")
            f.write(phase_data['top_inh_out'].to_markdown(index=False))
            f.write("\n\n")
            
            f.write("### Top 10 Hub Neurons (highest total connectivity)\n\n")
            f.write(phase_data['hubs'].to_markdown(index=False))
            f.write("\n\n---\n\n")
    
    # Save summary comparison table
    summary_md = TABLES_DIR / 'phase_comparison.md'
    with open(summary_md, 'w') as f:
        f.write("# Phase Comparison: E:I Dynamics\n\n")
        
        # Summary table
        f.write("## Summary Statistics\n\n")
        f.write("| Phase | Total Edges | Excitatory | Inhibitory | E:I Ratio | Method |\n")
        f.write("|-------|-------------|------------|------------|-----------|--------|\n")
        for pd_data in all_phases_data:
            f.write(f"| {pd_data['phase_label']} | {pd_data['n_total']} | ")
            f.write(f"{pd_data['n_exc']} | {pd_data['n_inh']} | ")
            f.write(f"{pd_data['ei_ratio']} | {pd_data['method']} |\n")
        f.write("\n")
        
        # Top hubs comparison across phases
        f.write("## Top 5 Hub Neurons by Phase\n\n")
        f.write("| Rank | Baseline | Butanone | Pentanedione | NaCl |\n")
        f.write("|------|----------|----------|--------------|------|\n")
        for rank in range(5):
            row = [f"| {rank+1} |"]
            for pd_data in all_phases_data:
                hubs = pd_data['hubs']
                if rank < len(hubs):
                    neuron = hubs.iloc[rank]['Neuron']
                    edges = hubs.iloc[rank]['Total Edges']
                    row.append(f" {neuron} ({edges}) |")
                else:
                    row.append(" - |")
            f.write("".join(row) + "\n")
        f.write("\n")
        
        # Top excitatory sources comparison
        f.write("## Top 5 Excitatory Sources by Phase\n\n")
        f.write("| Rank | Baseline | Butanone | Pentanedione | NaCl |\n")
        f.write("|------|----------|----------|--------------|------|\n")
        for rank in range(5):
            row = [f"| {rank+1} |"]
            for pd_data in all_phases_data:
                top_exc = pd_data['top_exc_out']
                if rank < len(top_exc):
                    neuron = top_exc.iloc[rank]['Neuron']
                    edges = top_exc.iloc[rank]['Out E Edges']
                    row.append(f" {neuron} ({edges}) |")
                else:
                    row.append(" - |")
            f.write("".join(row) + "\n")
        f.write("\n")
        
        # Top inhibitory sources comparison
        f.write("## Top 5 Inhibitory Sources by Phase\n\n")
        f.write("| Rank | Baseline | Butanone | Pentanedione | NaCl |\n")
        f.write("|------|----------|----------|--------------|------|\n")
        for rank in range(5):
            row = [f"| {rank+1} |"]
            for pd_data in all_phases_data:
                top_inh = pd_data['top_inh_out']
                if rank < len(top_inh):
                    neuron = top_inh.iloc[rank]['Neuron']
                    edges = top_inh.iloc[rank]['Out I Edges']
                    row.append(f" {neuron} ({edges}) |")
                else:
                    row.append(" - |")
            f.write("".join(row) + "\n")
    
    # Save CSV with all phase metrics
    for pd_data in all_phases_data:
        csv_file = TABLES_DIR / f"neurons_{pd_data['phase_key']}.csv"
        pd_data['full_metrics'].to_csv(csv_file, index=False)
    
    print(f"\n  Saved: {md_file.name}")
    print(f"  Saved: {summary_md.name}")
    print(f"  Saved: neurons_*.csv (one per phase)")


# =============================================================================
# TABLE 3: HUB NEURONS
# =============================================================================

def generate_hub_table():
    """Generate table of hub neurons (highest connectivity)."""
    print("\n[Table 3] Hub Neurons")
    print("=" * 60)
    
    # Use imputed_best model
    models_dir = SBTG_DIR / "models"
    imputed_models = list(models_dir.glob("*imputed_best*.npz"))
    
    if not imputed_models:
        print("  ERROR: No imputed_best model found")
        return
    
    model_file = imputed_models[0]
    data = np.load(model_file, allow_pickle=True)
    mu_hat = data['mu_hat']
    sign_adj = data['sign_adj']
    
    neuron_file = DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json'
    with open(neuron_file) as f:
        neuron_names = json.load(f)
    
    df = compute_neuron_metrics(mu_hat, sign_adj, neuron_names)
    
    # Top hubs by strength
    hubs = df.nlargest(30, 'strength')[[
        'neuron', 'strength', 'total_connections', 'total_strength',
        'out_exc_count', 'out_inh_count', 'in_exc_count', 'in_inh_count'
    ]].copy()
    
    hubs.columns = [
        'Neuron', 'Strength', 'Total Edges', 'Total Strength',
        'Out E', 'Out I', 'In E', 'In I'
    ]
    
    # Save
    md_file = TABLES_DIR / 'hub_neurons.md'
    with open(md_file, 'w') as f:
        f.write("# Hub Neurons (Highest Connectivity)\n\n")
        f.write(f"**Model:** {model_file.name}\n\n")
        f.write("Strength = total_strength × sqrt(total_connections)\n\n")
        f.write("## Top 30 Hub Neurons\n\n")
        f.write(hubs.to_markdown(index=False))
        f.write("\n")
    
    print(f"  Saved: {md_file.name}")
    print(f"  Top hub: {hubs['Neuron'].iloc[0]} (score: {hubs['Strength'].iloc[0]:.2f})")


# =============================================================================
# TABLE 4: DALE'S LAW CONSISTENCY ANALYSIS
# =============================================================================

def generate_dales_law_analysis():
    """
    Analyze whether neurons maintain consistent E/I classification across phases
    using a strict 'decidability' criterion to filter out robust functional types.
    
    A neuron is 'Decidably E/I' in a phase if:
    1. It has enough connections (min_edges >= 5) to be confident.
    2. It has a strong bias (|bias| > 0.5) towards one sign.
    
    Categories:
    - Stable Excitatory: Decidably E in >=1 phase, never Decidably I.
    - Stable Inhibitory: Decidably I in >=1 phase, never Decidably E.
    - Context Switcher: Decidably E in some phase AND Decidably I in another.
    - Ambiguous: Never meets decidability criteria.
    """
    print("\n[Table 4] Dale's Law Consistency Analysis (Strict Decidability)")
    print("=" * 60)
    
    # Check if phase results exist
    phase_results_file = TEMPORAL_DIR / "phase_results.json"
    if not phase_results_file.exists():
        print("  ERROR: Phase results not found")
        return
    
    # Load neuron names
    neuron_file = DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json'
    with open(neuron_file) as f:
        neuron_names = json.load(f)
    
    phases = ['baseline', 'butanone', 'pentanedione', 'nacl']
    
    # Collect E/I data for each neuron across phases
    neuron_ei_data = {name: {} for name in neuron_names}
    
    for phase in phases:
        sign_adj_file = TEMPORAL_DIR / f"sign_adj_{phase}.npy"
        if not sign_adj_file.exists():
            continue
        
        sign_adj = np.load(sign_adj_file)
        
        for i, name in enumerate(neuron_names):
            out_exc = int(np.sum(sign_adj[i, :] == 1))
            out_inh = int(np.sum(sign_adj[i, :] == -1))
            total_out = out_exc + out_inh
            
            if total_out > 0:
                ei_bias = (out_exc - out_inh) / total_out
            else:
                ei_bias = 0.0
            
            neuron_ei_data[name][phase] = {
                'out_exc': out_exc,
                'out_inh': out_inh,
                'total_out': total_out,
                'ei_bias': ei_bias
            }
    # Classification Params
    MIN_EDGES = 5
    BIAS_THRESHOLD = 0.5  # > 75% agreement
    
    classification_data = []
    
    for name in neuron_names:
        decidable_states = []  # List of 'E', 'I' for phases where it was decidable
        phase_descriptions = []
        
        has_decidable_E = False
        has_decidable_I = False
        
        for phase in phases:
            if phase not in neuron_ei_data[name]:
                continue
                
            data = neuron_ei_data[name][phase]
            total = data['total_out']
            bias = data['ei_bias']
            
            state = 'Ambiguous'
            if total >= MIN_EDGES:
                if bias > BIAS_THRESHOLD:
                    state = 'E'
                    has_decidable_E = True
                    decidable_states.append('E')
                elif bias < -BIAS_THRESHOLD:
                    state = 'I'
                    has_decidable_I = True
                    decidable_states.append('I')
            
            # Formatting for display
            if state == 'E':
                desc = f"**{phase}**: E ({data['out_exc']}E/{data['out_inh']}I)"
            elif state == 'I':
                desc = f"**{phase}**: I ({data['out_exc']}E/{data['out_inh']}I)"
            else:
                desc = f"{phase}: - ({data['out_exc']}E/{data['out_inh']}I)"
            phase_descriptions.append(desc)
            
        # Determine Category
        if has_decidable_E and has_decidable_I:
            category = 'Context Switcher'
        elif has_decidable_E:
            category = 'Stable Excitatory'
        elif has_decidable_I:
            category = 'Stable Inhibitory'
        else:
            category = 'Ambiguous'
            
        classification_data.append({
            'Neuron': name,
            'Category': category,
            'Decidable_Phases': len(decidable_states),
            'Details': '<br>'.join(phase_descriptions)
        })
    
    df = pd.DataFrame(classification_data)
    
    # Save CSV
    df.to_csv(TABLES_DIR / 'dales_law_consistency.csv', index=False)
    print(f"  Saved: dales_law_consistency.csv")
    
    # Generate Markdown Report
    md_file = TABLES_DIR / 'dales_law_consistency.md'
    with open(md_file, 'w') as f:
        f.write(f"# Dale's Law Consistency Analysis (Strict)\n\n")
        f.write(f"**Criteria for Decidability**:\n")
        f.write(f"- Minimum Edges: {MIN_EDGES}\n")
        f.write(f"- Bias Threshold: > {BIAS_THRESHOLD} (i.e. > 75% consistency)\n\n")
        
        # Summary
        counts = df['Category'].value_counts()
        f.write("## Summary of Categories\n\n")
        f.write("| Category | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        total = len(df)
        for cat, count in counts.items():
            f.write(f"| {cat} | {count} | {count/total*100:.1f}% |\n")
        f.write("\n")
        
        # 1. Stable Neurons
        f.write("## 1. Stable Neurons\n")
        f.write("Neurons that are decidably E or I in at least one phase and NEVER flip.\n\n")
        stable_df = df[df['Category'].str.startswith('Stable')].sort_values(['Category', 'Neuron'])
        if not stable_df.empty:
            f.write(stable_df[['Neuron', 'Category', 'Details']].to_markdown(index=False))
        else:
            f.write("*No stable neurons found.*")
        f.write("\n\n")
        
        # 2. Context Switchers
        f.write("## 2. Context Switchers (Functional Flip)\n")
        f.write("Neurons that are decidably Excitatory in some phases and Inhibitory in others.\n\n")
        switch_df = df[df['Category'] == 'Context Switcher']
        if not switch_df.empty:
            f.write(switch_df[['Neuron', 'Details']].to_markdown(index=False))
        else:
            f.write("*No robust context switchers found under strict criteria.*")
        f.write("\n\n")
        
        # 3. Ambiguous
        f.write("## 3. Ambiguous / Likely Silent\n")
        f.write("Neurons that never meet the strict decidability criteria in any phase (low activity or mixed signaling).\n\n")
        amb_df = df[df['Category'] == 'Ambiguous'].sort_values('Neuron')
        # Just list names to save space if many
        names = amb_df['Neuron'].tolist()
        if names:
            f.write(", ".join(names))
        else:
            f.write("*No ambiguous neurons found.*")
        f.write("\n")

    print(f"  Saved: {md_file.name}")


# =============================================================================
# TABLE 5: HYPERPARAMETER SUMMARY
# =============================================================================

def generate_hyperparameter_table():
    """
    Generate a table summarizing the hyperparameters used for the analysis.
    Compares the Global (Generic) settings with the Phase-Specific optimal parameters.
    """
    print("\n[Table 5] Hyperparameter Summary")
    print("=" * 60)
    
    # Define columns
    phases = ['Global (Generic)', 'Baseline', 'Butanone', 'Pentanedione', 'NaCl']
    
    # Collect params for each phase
    # Global uses OPTIMIZED_HYPERPARAMS
    # Phases use PHASE_OPTIMAL_PARAMS
    
    params_data = {}
    
    # 1. Global
    params_data['Global (Generic)'] = OPTIMIZED_HYPERPARAMS.copy()
    
    # 2. Phases
    for p in ['baseline', 'butanone', 'pentanedione', 'nacl']:
        label = p.capitalize()
        if p in PHASE_OPTIMAL_PARAMS:
            params_data[label] = PHASE_OPTIMAL_PARAMS[p].copy()
        else:
            params_data[label] = {"status": "Not Found"}
            
    # Define interesting keys to display
    keys_to_display = [
        'model_type',
        'num_regimes', 
        'dsm_lr',
        'dsm_epochs',
        'dsm_noise_std',
        'dsm_hidden_dim',
        'structured_hidden_dim',
        'structured_l1_lambda',
        'fdr_alpha',
        'feature_dim' # For feature_bilinear if present
    ]
    
    # Build Table Data
    table_rows = []
    
    for key in keys_to_display:
        row = {'Parameter': key}
        has_val = False
        
        for phase in phases:
            if phase in params_data:
                val = params_data[phase].get(key, '-')
                
                # Format numbers
                if isinstance(val, float):
                    if 'lambda' in key or 'lr' in key:
                        val = f"{val:.2e}"
                    elif 'noise' in key:
                         val = f"{val:.3f}"
                    elif int(val) == val:
                        val = int(val)
                    else:
                        val = f"{val:.3f}"
                
                row[phase] = val
                if val != '-':
                    has_val = True
        
        if has_val:
            table_rows.append(row)
            
    df = pd.DataFrame(table_rows)
    
    # Save CSV
    csv_file = TABLES_DIR / 'hyperparameters.csv'
    df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file.name}")
    
    # Save Markdown
    md_file = TABLES_DIR / 'hyperparameters.md'
    with open(md_file, 'w') as f:
        f.write("# Analysis Hyperparameters\n\n")
        f.write("**Source:** `pipeline/configs/phase_optimal_params.py` (150-trial HP search)\n\n")
        f.write("**Note:** 'Global' parameters are used for the main imputed model and general validation. ")
        f.write("Phase-specific parameters are used for temporal analysis (Figures 7 & 9).\n\n")
        
        f.write(df.to_markdown(index=False))
        f.write("\n")
        
    print(f"  Saved: {md_file.name}")


# =============================================================================
# TABLE 6: DIRECT vs TRANSFER TRAINING COMPARISON (Comprehensive)
# =============================================================================

def generate_direct_vs_transfer_table():
    """
    Generate comprehensive tables comparing Direct vs Transfer training approaches.
    
    Generates:
    1. Summary comparison table
    2. Per-phase E/I/hub analysis for DIRECT training
    3. Per-phase E/I/hub analysis for TRANSFER training
    4. Comparison markdown with all details
    
    Uses output from 05_temporal_analysis.py --sbtg --transfer
    """
    print("\n[Table 6] Direct vs Transfer Training Comparison (Comprehensive)")
    print("=" * 60)
    
    # Check for adjacency data
    sbtg_adj_dir = RESULTS_DIR / 'stimulus_specific' / 'sbtg' / 'adjacencies'
    comparison_file = RESULTS_DIR / 'stimulus_specific' / 'sbtg' / 'direct_vs_transfer_comparison.csv'
    
    if not sbtg_adj_dir.exists():
        print("  No SBTG adjacencies found. Run:")
        print("    python pipeline/05_temporal_analysis.py --sbtg --transfer")
        return
    
    # Load neuron names
    neuron_file = DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json'
    if not neuron_file.exists():
        print(f"  ERROR: Neuron names not found at {neuron_file}")
        return
    
    with open(neuron_file) as f:
        neuron_names = json.load(f)
    
    # Phase definitions
    phases = {
        'baseline': 'Baseline (0-60s)',
        'butanone': 'Butanone (60.5-70.5s)',
        'pentanedione': 'Pentanedione (120.5-130.5s)',
        'nacl': 'NaCl (180.5-190.5s)',
    }
    
    # Training methods to analyze
    methods = ['direct', 'transfer']
    
    # Collect all phase-method analysis results
    all_analysis = {}
    
    for phase_key, phase_label in phases.items():
        all_analysis[phase_key] = {}
        
        for method in methods:
            # Transfer not applicable for baseline
            if phase_key == 'baseline' and method == 'transfer':
                continue
            
            sign_adj_file = sbtg_adj_dir / f'{phase_key}_{method}_sign_adj.npy'
            mu_hat_file = sbtg_adj_dir / f'{phase_key}_{method}_mu_hat.npy'
            
            if not sign_adj_file.exists():
                continue
            
            sign_adj = np.load(sign_adj_file)
            
            # Use mu_hat if available, otherwise use sign_adj as proxy
            if mu_hat_file.exists():
                mu_hat = np.load(mu_hat_file)
            else:
                mu_hat = sign_adj.astype(float)
            
            # Compute neuron metrics
            df = compute_neuron_metrics(mu_hat, sign_adj, neuron_names)
            
            # Edge statistics
            n_exc = int((sign_adj > 0).sum())
            n_inh = int((sign_adj < 0).sum())
            n_total = n_exc + n_inh
            ei_ratio = n_exc / n_inh if n_inh > 0 else float('inf')
            
            # Top excitatory sources
            top_exc_out = df.nlargest(10, 'out_exc_count')[
                ['neuron', 'out_exc_count', 'in_exc_count', 'total_connections']
            ].copy()
            top_exc_out.columns = ['Neuron', 'Out E', 'In E', 'Total']
            
            # Top inhibitory sources
            top_inh_out = df.nlargest(10, 'out_inh_count')[
                ['neuron', 'out_inh_count', 'in_inh_count', 'total_connections']
            ].copy()
            top_inh_out.columns = ['Neuron', 'Out I', 'In I', 'Total']
            
            # Hub neurons (highest total connectivity)
            hubs = df.nlargest(10, 'total_connections')[
                ['neuron', 'total_connections', 'out_exc_count', 'out_inh_count', 
                 'in_exc_count', 'in_inh_count']
            ].copy()
            hubs.columns = ['Neuron', 'Total', 'Out E', 'Out I', 'In E', 'In I']
            
            all_analysis[phase_key][method] = {
                'n_exc': n_exc,
                'n_inh': n_inh,
                'n_total': n_total,
                'ei_ratio': ei_ratio,
                'top_exc_out': top_exc_out,
                'top_inh_out': top_inh_out,
                'hubs': hubs,
                'full_metrics': df,
            }
            
            print(f"  {phase_label} [{method.upper()}]: {n_total} edges (E:{n_exc}, I:{n_inh})")
    
    # =========================================================================
    # Save comprehensive markdown file
    # =========================================================================
    md_file = TABLES_DIR / 'direct_vs_transfer_comprehensive.md'
    with open(md_file, 'w') as f:
        f.write("# Direct vs Transfer Training: Comprehensive Analysis\n\n")
        f.write("**Direct Training:** Each phase trained from scratch (random initialization)\n")
        f.write("**Transfer Learning:** Pre-train on baseline (0-60s), then fine-tune on stimulus\n\n")
        f.write("---\n\n")
        
        # Summary table
        f.write("## Summary Statistics\n\n")
        f.write("| Phase | Method | Total Edges | Excitatory | Inhibitory | E:I Ratio |\n")
        f.write("|-------|--------|-------------|------------|------------|-----------|\n")
        
        for phase_key, phase_label in phases.items():
            for method in methods:
                if method in all_analysis.get(phase_key, {}):
                    data = all_analysis[phase_key][method]
                    ei_str = f"{data['ei_ratio']:.3f}" if data['ei_ratio'] != float('inf') else "∞"
                    f.write(f"| {phase_label} | {method.upper()} | {data['n_total']} | "
                           f"{data['n_exc']} | {data['n_inh']} | {ei_str} |\n")
        f.write("\n---\n\n")
        
        # Per-phase detailed analysis
        for phase_key, phase_label in phases.items():
            f.write(f"## {phase_label}\n\n")
            
            for method in methods:
                if method not in all_analysis.get(phase_key, {}):
                    continue
                
                data = all_analysis[phase_key][method]
                method_label = "DIRECT Training" if method == 'direct' else "TRANSFER Learning"
                
                f.write(f"### {method_label}\n\n")
                ei_str = f"{data['ei_ratio']:.3f}" if data['ei_ratio'] != float('inf') else "∞"
                f.write(f"**Statistics:** {data['n_total']} edges "
                       f"(E: {data['n_exc']}, I: {data['n_inh']}), E:I = {ei_str}\n\n")
                
                # Top excitatory sources
                f.write("#### Top 10 Excitatory Sources\n\n")
                f.write(data['top_exc_out'].to_markdown(index=False))
                f.write("\n\n")
                
                # Top inhibitory sources
                f.write("#### Top 10 Inhibitory Sources\n\n")
                f.write(data['top_inh_out'].to_markdown(index=False))
                f.write("\n\n")
                
                # Hub neurons
                f.write("#### Top 10 Hub Neurons\n\n")
                f.write(data['hubs'].to_markdown(index=False))
                f.write("\n\n")
            
            f.write("---\n\n")
        
        # Cross-method comparison tables
        f.write("## Cross-Method Comparison: Top Hubs\n\n")
        f.write("Comparing top 5 hub neurons across methods for each stimulus phase:\n\n")
        
        for phase_key in ['butanone', 'pentanedione', 'nacl']:
            phase_label = phases[phase_key]
            f.write(f"### {phase_label}\n\n")
            f.write("| Rank | DIRECT | TRANSFER |\n")
            f.write("|------|--------|----------|\n")
            
            for rank in range(5):
                row = [f"| {rank+1} |"]
                for method in ['direct', 'transfer']:
                    if method in all_analysis.get(phase_key, {}):
                        hubs = all_analysis[phase_key][method]['hubs']
                        if rank < len(hubs):
                            neuron = hubs.iloc[rank]['Neuron']
                            total = hubs.iloc[rank]['Total']
                            row.append(f" {neuron} ({total}) |")
                        else:
                            row.append(" - |")
                    else:
                        row.append(" N/A |")
                f.write("".join(row) + "\n")
            f.write("\n")
    
    print(f"  Saved: {md_file.name}")
    
    # =========================================================================
    # Save per-phase CSVs for both methods
    # =========================================================================
    for phase_key in phases.keys():
        for method in methods:
            if method in all_analysis.get(phase_key, {}):
                data = all_analysis[phase_key][method]
                
                # Save full metrics
                csv_file = TABLES_DIR / f'neurons_{phase_key}_{method}.csv'
                data['full_metrics'].to_csv(csv_file, index=False)
                
                # Save top neurons summary
                summary_file = TABLES_DIR / f'top_neurons_{phase_key}_{method}.csv'
                hubs = data['hubs'].copy()
                hubs['phase'] = phase_key
                hubs['method'] = method
                hubs.to_csv(summary_file, index=False)
    
    print(f"  Saved: neurons_*_direct.csv and neurons_*_transfer.csv")
    
    # =========================================================================
    # Also save original comparison summary if available
    # =========================================================================
    if comparison_file.exists():
        df = pd.read_csv(comparison_file)
        summary_rows = []
        for _, row in df.iterrows():
            phase = row['phase']
            direct_edges = int(row['direct_edges'])
            transfer_edges = int(row['transfer_edges'])
            baseline_edges = int(row.get('edges_from_baseline', 0))
            
            direct_ei = float(row['direct_ei_ratio'])
            transfer_ei = float(row['transfer_ei_ratio'])
            
            edge_diff = transfer_edges - direct_edges
            
            summary_rows.append({
                'Phase': phase.capitalize(),
                'Direct Edges': direct_edges,
                'Transfer Edges': transfer_edges,
                'Edge Δ': edge_diff,
                'Direct E:I': round(direct_ei, 3),
                'Transfer E:I': round(transfer_ei, 3),
                'From Baseline': baseline_edges,
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(TABLES_DIR / 'direct_vs_transfer_summary.csv', index=False)
        print(f"  Saved: direct_vs_transfer_summary.csv")


# =============================================================================
# TABLE 7: COOK AND LEIFER EVALUATION FOR EACH PHASE/METHOD
# =============================================================================

def generate_cook_leifer_evaluation_table():
    """
    Generate evaluation table comparing each phase/method graph against
    Cook et al. connectome and Leifer functional atlas.
    
    Evaluates:
    - Baseline (direct)
    - Butanone/Pentanedione/NaCl (direct and transfer)
    
    Metrics:
    - AUROC, AUPRC vs Cook anatomical connectome
    - AUROC, AUPRC vs Leifer functional atlas
    - F1, Precision, Recall at matched density
    """
    print("\n[Table 7] Cook and Leifer Evaluation per Phase/Method")
    print("=" * 60)
    
    # Import required utilities
    from sklearn.metrics import roc_auc_score, average_precision_score
    from pipeline.utils.io import load_structural_connectome as _load_structural_connectome
    from pipeline.utils.leifer import load_leifer_atlas_data
    
    CONNECTOME_DIR = RESULTS_DIR / "intermediate" / "connectome"
    sbtg_adj_dir = RESULTS_DIR / 'stimulus_specific' / 'sbtg' / 'adjacencies'
    
    if not sbtg_adj_dir.exists():
        print("  No SBTG adjacencies found. Run:")
        print("    python pipeline/05_temporal_analysis.py --sbtg --transfer")
        return
    
    # Load neuron names
    neuron_file = DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json'
    if not neuron_file.exists():
        print(f"  ERROR: Neuron names not found at {neuron_file}")
        return
    
    with open(neuron_file) as f:
        model_neurons = json.load(f)
    
    # Load Cook connectome
    try:
        A_struct, cook_neurons, _ = _load_structural_connectome(CONNECTOME_DIR)
        cook_available = True
        print(f"  Cook connectome: {len(cook_neurons)} neurons, {(A_struct > 0).sum()} edges")
    except Exception as e:
        print(f"  WARNING: Could not load Cook connectome: {e}")
        cook_available = False
        A_struct, cook_neurons = None, []
    
    # Load Leifer atlas
    try:
        q_leifer, q_eq, leifer_neurons = load_leifer_atlas_data()
        if q_leifer is not None:
            leifer_available = True
            n_functional = int((q_leifer < 0.05).sum())
            print(f"  Leifer atlas: {len(leifer_neurons)} neurons, {n_functional} functional edges (q<0.05)")
        else:
            leifer_available = False
            leifer_neurons = []
    except Exception as e:
        print(f"  WARNING: Could not load Leifer atlas: {e}")
        leifer_available = False
        q_leifer, leifer_neurons = None, []
    
    if not cook_available and not leifer_available:
        print("  ERROR: Neither Cook nor Leifer available for evaluation")
        return
    
    def align_and_evaluate(scores: np.ndarray, score_neurons: List[str], 
                           ground_truth: np.ndarray, gt_neurons: List[str],
                           is_qvalue: bool = False) -> Dict:
        """Align matrices and compute evaluation metrics."""
        # Find common neurons
        common = [n for n in score_neurons if n in gt_neurons]
        if len(common) < 10:
            return {'auroc': np.nan, 'auprc': np.nan, 'n_common': len(common)}
        
        # Get indices
        score_idx = [score_neurons.index(n) for n in common]
        gt_idx = [gt_neurons.index(n) for n in common]
        
        # Extract submatrices
        scores_aligned = scores[np.ix_(score_idx, score_idx)]
        gt_aligned = ground_truth[np.ix_(gt_idx, gt_idx)]
        
        n = len(common)
        mask = ~np.eye(n, dtype=bool)
        
        y_score = np.abs(scores_aligned[mask])
        
        if is_qvalue:
            # For Leifer, q < alpha means functional connection
            y_true = (gt_aligned[mask] < 0.05).astype(int)
        else:
            # For Cook, any edge > 0 is a connection
            y_true = (gt_aligned[mask] > 0).astype(int)
        
        # Handle NaN
        valid = ~np.isnan(y_score) & ~np.isnan(y_true.astype(float))
        y_score = y_score[valid]
        y_true = y_true[valid]
        
        if len(y_true) == 0 or y_true.sum() == 0:
            return {'auroc': 0.5, 'auprc': 0.0, 'n_common': len(common)}
        
        # Compute metrics
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        
        # F1 at matched density
        n_positive = y_true.sum()
        unique_scores = np.unique(y_score)
        is_discrete = len(unique_scores) <= 3 and np.all(np.isin(unique_scores, [0, 1]))
        
        if is_discrete:
            y_pred = y_score.astype(int)
        else:
            threshold = np.percentile(y_score, 100 * (1 - n_positive / len(y_score)))
            y_pred = (y_score >= threshold).astype(int)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'auroc': auroc,
            'auprc': auprc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'n_common': len(common),
            'n_true_edges': int(y_true.sum()),
            'n_predicted': int(y_pred.sum()),
        }
    
    # Phase/method combinations to evaluate
    phases = {
        'baseline': 'Baseline (0-60s)',
        'butanone': 'Butanone (60.5-70.5s)',
        'pentanedione': 'Pentanedione (120.5-130.5s)',
        'nacl': 'NaCl (180.5-190.5s)',
    }
    methods = ['direct', 'transfer']
    
    all_results = []
    
    for phase_key, phase_label in phases.items():
        for method in methods:
            # Transfer not applicable for baseline
            if phase_key == 'baseline' and method == 'transfer':
                continue
            
            sign_adj_path = sbtg_adj_dir / f'{phase_key}_{method}_sign_adj.npy'
            if not sign_adj_path.exists():
                continue
            
            sign_adj = np.load(sign_adj_path)
            
            # Convert to binary edge presence
            edge_matrix = (sign_adj != 0).astype(float)
            
            result = {
                'phase': phase_key,
                'phase_label': phase_label,
                'method': method.upper(),
                'n_edges': int((sign_adj != 0).sum()),
                'n_exc': int((sign_adj > 0).sum()),
                'n_inh': int((sign_adj < 0).sum()),
            }
            
            # Evaluate vs Cook
            if cook_available:
                cook_metrics = align_and_evaluate(edge_matrix, model_neurons, 
                                                   A_struct, cook_neurons, is_qvalue=False)
                result['cook_auroc'] = cook_metrics['auroc']
                result['cook_auprc'] = cook_metrics['auprc']
                result['cook_f1'] = cook_metrics.get('f1', np.nan)
                result['cook_precision'] = cook_metrics.get('precision', np.nan)
                result['cook_recall'] = cook_metrics.get('recall', np.nan)
                result['cook_n_common'] = cook_metrics['n_common']
            
            # Evaluate vs Leifer
            if leifer_available:
                leifer_metrics = align_and_evaluate(edge_matrix, model_neurons, 
                                                     q_leifer, leifer_neurons, is_qvalue=True)
                result['leifer_auroc'] = leifer_metrics['auroc']
                result['leifer_auprc'] = leifer_metrics['auprc']
                result['leifer_f1'] = leifer_metrics.get('f1', np.nan)
                result['leifer_precision'] = leifer_metrics.get('precision', np.nan)
                result['leifer_recall'] = leifer_metrics.get('recall', np.nan)
                result['leifer_n_common'] = leifer_metrics['n_common']
            
            all_results.append(result)
            
            print(f"  {phase_label} [{method.upper()}]: ", end='')
            if cook_available:
                print(f"Cook AUROC={result['cook_auroc']:.3f}", end=' ')
            if leifer_available:
                print(f"Leifer AUROC={result['leifer_auroc']:.3f}", end='')
            print()
    
    if not all_results:
        print("  No phase/method combinations found!")
        return
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    # =========================================================================
    # Save comprehensive markdown
    # =========================================================================
    md_file = TABLES_DIR / 'cook_leifer_evaluation.md'
    with open(md_file, 'w') as f:
        f.write("# Cook and Leifer Evaluation by Phase and Training Method\n\n")
        f.write("Evaluation of SBTG connectivity graphs against ground truth:\n\n")
        f.write("- **Cook et al. 2019**: Anatomical connectome from electron microscopy\n")
        f.write("- **Leifer et al.**: Functional atlas from optogenetic stimulation (q < 0.05)\n\n")
        f.write("---\n\n")
        
        # Summary table - Cook evaluation
        if cook_available:
            f.write("## Cook Connectome Evaluation\n\n")
            f.write("| Phase | Method | Edges | AUROC | AUPRC | F1 | Precision | Recall |\n")
            f.write("|-------|--------|-------|-------|-------|-----|-----------|--------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['phase_label']} | {row['method']} | {row['n_edges']} | "
                       f"{row['cook_auroc']:.3f} | {row['cook_auprc']:.3f} | "
                       f"{row['cook_f1']:.3f} | {row['cook_precision']:.3f} | "
                       f"{row['cook_recall']:.3f} |\n")
            f.write("\n")
        
        # Summary table - Leifer evaluation
        if leifer_available:
            f.write("## Leifer Functional Atlas Evaluation\n\n")
            f.write("| Phase | Method | Edges | AUROC | AUPRC | F1 | Precision | Recall |\n")
            f.write("|-------|--------|-------|-------|-------|-----|-----------|--------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['phase_label']} | {row['method']} | {row['n_edges']} | "
                       f"{row['leifer_auroc']:.3f} | {row['leifer_auprc']:.3f} | "
                       f"{row['leifer_f1']:.3f} | {row['leifer_precision']:.3f} | "
                       f"{row['leifer_recall']:.3f} |\n")
            f.write("\n")
        
        # Interpretation
        f.write("---\n\n")
        f.write("## Interpretation\n\n")
        f.write("- **AUROC**: Area under ROC curve (0.5 = random, 1.0 = perfect)\n")
        f.write("- **AUPRC**: Area under precision-recall curve (higher is better)\n")
        f.write("- **F1/Precision/Recall**: At matched edge density\n\n")
        
        # Compare direct vs transfer
        f.write("### Direct vs Transfer Comparison\n\n")
        for phase in ['butanone', 'pentanedione', 'nacl']:
            phase_df = df[df['phase'] == phase]
            if len(phase_df) == 2:
                direct_row = phase_df[phase_df['method'] == 'DIRECT'].iloc[0]
                transfer_row = phase_df[phase_df['method'] == 'TRANSFER'].iloc[0]
                
                f.write(f"**{phase.capitalize()}:**\n")
                if cook_available:
                    cook_diff = transfer_row['cook_auroc'] - direct_row['cook_auroc']
                    f.write(f"- Cook AUROC: DIRECT={direct_row['cook_auroc']:.3f}, "
                           f"TRANSFER={transfer_row['cook_auroc']:.3f} (Δ={cook_diff:+.3f})\n")
                if leifer_available:
                    leifer_diff = transfer_row['leifer_auroc'] - direct_row['leifer_auroc']
                    f.write(f"- Leifer AUROC: DIRECT={direct_row['leifer_auroc']:.3f}, "
                           f"TRANSFER={transfer_row['leifer_auroc']:.3f} (Δ={leifer_diff:+.3f})\n")
                f.write("\n")
    
    print(f"  Saved: {md_file.name}")
    
    # Save CSV
    csv_file = TABLES_DIR / 'cook_leifer_evaluation.csv'
    df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file.name}")


# =============================================================================
# TABLE 8: NOVEL STIMULUS EDGES - NEURON E/I CLASSIFICATION
# =============================================================================

def generate_novel_edges_classification_table():
    """
    Generate table classifying each neuron's E/I type based on NOVEL stimulus edges.
    
    Novel edges = edges in transfer-trained stimulus graph that are NOT in baseline.
    These represent stimulus-specific connectivity changes.
    
    For each neuron, computes:
    - Number of outgoing excitatory novel edges
    - Number of outgoing inhibitory novel edges
    - E/I ratio and classification (Excitatory/Inhibitory/Balanced/Silent)
    """
    print("\n[Table 8] Novel Stimulus Edges - Neuron E/I Classification")
    print("=" * 60)
    
    sbtg_adj_dir = RESULTS_DIR / 'stimulus_specific' / 'sbtg' / 'adjacencies'
    
    if not sbtg_adj_dir.exists():
        print("  No SBTG adjacencies found. Run:")
        print("    python pipeline/05_temporal_analysis.py --sbtg --transfer")
        return
    
    # Load baseline
    baseline_path = sbtg_adj_dir / 'baseline_direct_sign_adj.npy'
    if not baseline_path.exists():
        print("  ERROR: baseline_direct_sign_adj.npy not found")
        return
    
    baseline_adj = np.load(baseline_path)
    baseline_mask = (baseline_adj != 0)
    
    # Load neuron names
    neuron_file = DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json'
    if not neuron_file.exists():
        print(f"  ERROR: Neuron names not found")
        return
    
    with open(neuron_file) as f:
        neuron_names = json.load(f)
    
    stimuli = ['butanone', 'pentanedione', 'nacl']
    
    # Collect data for each neuron across stimuli
    neuron_data = []
    
    for i, neuron in enumerate(neuron_names):
        row = {'neuron': neuron}
        
        for stim in stimuli:
            transfer_path = sbtg_adj_dir / f'{stim}_transfer_sign_adj.npy'
            if not transfer_path.exists():
                row[f'{stim}_out_E'] = 0
                row[f'{stim}_out_I'] = 0
                row[f'{stim}_ratio'] = np.nan
                row[f'{stim}_class'] = 'N/A'
                continue
            
            transfer_adj = np.load(transfer_path)
            transfer_mask = (transfer_adj != 0)
            
            # Novel edges: in transfer but NOT in baseline
            novel_mask = transfer_mask & ~baseline_mask
            novel_adj = transfer_adj * novel_mask
            
            # Outgoing novel edges for this neuron
            out_exc = int((novel_adj[i, :] > 0).sum())
            out_inh = int((novel_adj[i, :] < 0).sum())
            total_out = out_exc + out_inh
            
            row[f'{stim}_out_E'] = out_exc
            row[f'{stim}_out_I'] = out_inh
            
            # Compute ratio and classification
            if total_out == 0:
                row[f'{stim}_ratio'] = np.nan
                row[f'{stim}_class'] = 'Silent'
            elif out_inh == 0:
                row[f'{stim}_ratio'] = float('inf')
                row[f'{stim}_class'] = 'Excitatory'
            else:
                ratio = out_exc / out_inh
                row[f'{stim}_ratio'] = ratio
                
                # Classification based on ratio
                if ratio > 1.5:
                    row[f'{stim}_class'] = 'Excitatory'
                elif ratio < 0.67:
                    row[f'{stim}_class'] = 'Inhibitory'
                else:
                    row[f'{stim}_class'] = 'Balanced'
        
        neuron_data.append(row)
    
    df = pd.DataFrame(neuron_data)
    
    # Filter to neurons with at least one novel edge in any stimulus
    df['total_novel'] = sum(df[f'{s}_out_E'] + df[f'{s}_out_I'] for s in stimuli)
    df_active = df[df['total_novel'] > 0].copy()
    
    print(f"  {len(df_active)} neurons with novel stimulus edges (of {len(df)} total)")
    
    # =========================================================================
    # Create summary classification table
    # =========================================================================
    summary_rows = []
    for _, row in df_active.iterrows():
        summary_rows.append({
            'Neuron': row['neuron'],
            'Butanone': row['butanone_class'],
            'Pentanedione': row['pentanedione_class'],
            'NaCl': row['nacl_class'],
            'But_E': row['butanone_out_E'],
            'But_I': row['butanone_out_I'],
            'Pent_E': row['pentanedione_out_E'],
            'Pent_I': row['pentanedione_out_I'],
            'NaCl_E': row['nacl_out_E'],
            'NaCl_I': row['nacl_out_I'],
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Sort by total novel edges
    summary_df['Total'] = (summary_df['But_E'] + summary_df['But_I'] + 
                           summary_df['Pent_E'] + summary_df['Pent_I'] +
                           summary_df['NaCl_E'] + summary_df['NaCl_I'])
    summary_df = summary_df.sort_values('Total', ascending=False)
    
    # =========================================================================
    # Save markdown
    # =========================================================================
    md_file = TABLES_DIR / 'novel_edges_neuron_classification.md'
    with open(md_file, 'w') as f:
        f.write("# Novel Stimulus Edges: Neuron E/I Classification\n\n")
        f.write("**Novel edges** = edges in transfer-trained stimulus graph that are NOT in baseline.\n")
        f.write("These represent stimulus-specific connectivity changes.\n\n")
        f.write("**Classification criteria** (based on outgoing E:I ratio):\n")
        f.write("- **Excitatory**: E:I > 1.5\n")
        f.write("- **Inhibitory**: E:I < 0.67\n")
        f.write("- **Balanced**: 0.67 ≤ E:I ≤ 1.5\n")
        f.write("- **Silent**: No novel edges for this stimulus\n\n")
        f.write("---\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        for stim in stimuli:
            col = f'{stim}_class'
            if col in df.columns:
                counts = df[col].value_counts()
                f.write(f"**{stim.capitalize()}:** ")
                parts = [f"{k}={v}" for k, v in counts.items() if k != 'Silent']
                f.write(", ".join(parts) + "\n\n")
        
        f.write("---\n\n")
        
        # Classification table (top 30 by total novel edges)
        f.write("## Neuron Classifications (Top 30 by Novel Edge Count)\n\n")
        
        display_cols = ['Neuron', 'Butanone', 'Pentanedione', 'NaCl', 
                        'But_E', 'But_I', 'Pent_E', 'Pent_I', 'NaCl_E', 'NaCl_I', 'Total']
        top30 = summary_df.head(30)[display_cols]
        f.write(top30.to_markdown(index=False))
        f.write("\n\n")
        
        # Cross-stimulus consistency analysis
        f.write("---\n\n")
        f.write("## Cross-Stimulus Consistency\n\n")
        f.write("Neurons that maintain consistent E/I classification across all stimuli:\n\n")
        
        # Find consistent neurons
        consistent_exc = []
        consistent_inh = []
        switchers = []
        
        for _, row in df_active.iterrows():
            classes = [row[f'{s}_class'] for s in stimuli]
            non_silent = [c for c in classes if c not in ['Silent', 'N/A']]
            
            if len(non_silent) >= 2:
                if all(c == 'Excitatory' for c in non_silent):
                    consistent_exc.append(row['neuron'])
                elif all(c == 'Inhibitory' for c in non_silent):
                    consistent_inh.append(row['neuron'])
                elif 'Excitatory' in non_silent and 'Inhibitory' in non_silent:
                    switchers.append(row['neuron'])
        
        f.write(f"**Consistently Excitatory ({len(consistent_exc)}):** {', '.join(consistent_exc[:15])}")
        if len(consistent_exc) > 15:
            f.write(f" ... (+{len(consistent_exc)-15} more)")
        f.write("\n\n")
        
        f.write(f"**Consistently Inhibitory ({len(consistent_inh)}):** {', '.join(consistent_inh[:15])}")
        if len(consistent_inh) > 15:
            f.write(f" ... (+{len(consistent_inh)-15} more)")
        f.write("\n\n")
        
        f.write(f"**Context Switchers ({len(switchers)}):** {', '.join(switchers[:15])}")
        if len(switchers) > 15:
            f.write(f" ... (+{len(switchers)-15} more)")
        f.write("\n\n")
        
        # Full table for all neurons with novel edges
        f.write("---\n\n")
        f.write("## Complete Classification Table\n\n")
        f.write(summary_df[display_cols].to_markdown(index=False))
        f.write("\n")
    
    print(f"  Saved: {md_file.name}")
    
    # Save CSV
    csv_file = TABLES_DIR / 'novel_edges_neuron_classification.csv'
    df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file.name}")
    
    # Print summary
    print(f"\n  Classification Summary:")
    for stim in stimuli:
        col = f'{stim}_class'
        if col in df.columns:
            exc_count = (df[col] == 'Excitatory').sum()
            inh_count = (df[col] == 'Inhibitory').sum()
            bal_count = (df[col] == 'Balanced').sum()
            print(f"    {stim}: E={exc_count}, I={inh_count}, Balanced={bal_count}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("NEURON SIGNIFICANCE TABLES")
    print("=" * 60)
    
    # Create output directory
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate tables
    generate_global_table()
    generate_phase_tables()  # Phase-specific tables (baseline, butanone, pentanedione, nacl)
    generate_hub_table()     # Global hub neurons from imputed_best model
    generate_dales_law_analysis()  # Dale's Law consistency check
    generate_hyperparameter_table() # Summary of hyperparameters
    generate_direct_vs_transfer_table()  # Direct vs Transfer training comparison
    generate_cook_leifer_evaluation_table()  # Cook and Leifer evaluation per phase/method
    generate_novel_edges_classification_table()  # Novel edges E/I classification per stimulus
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nTables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()

