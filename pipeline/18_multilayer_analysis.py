#!/usr/bin/env python3
"""
Script 18: Multilayer Connectome Analysis - Monoaminergic Evaluation

This script loads the Bentley et al. (2016) monoamine connectome and evaluates
whether our multi-lag SBTG functional connectivity predictions align better
with neuromodulatory networks at longer time lags.

Reference:
    Bentley B et al. (2016) "The Multilayer Connectome of Caenorhabditis elegans"
    PLoS Comput Biol 12(12): e1005283
    https://doi.org/10.1371/journal.pcbi.1005283

Theory:
    Neuromodulatory signaling (dopamine, serotonin, etc.) operates on slower
    timescales than synaptic transmission. If our multi-lag SBTG captures
    these dynamics, we expect:
    - Short lags (0.25-0.5s): Better prediction of synaptic connectivity
    - Long lags (2-5s): Better prediction of monoaminergic connectivity

Usage:
    # Phase 1: Load and validate monoamine data
    python pipeline/18_multilayer_analysis.py --validate
    
    # Phase 2: Evaluate against multi-lag results
    python pipeline/18_multilayer_analysis.py --result-dir results/multilag_separation/TIMESTAMP/

"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BENTLEY_DIR = DATA_DIR / "S1_Dataset"


# =============================================================================
# NEURON NAME NORMALIZATION
# =============================================================================

def normalize_bentley_name(name: str) -> str:
    """
    Normalize Bentley neuron name to match our naming convention.
    
    Bentley data uses:
    - Full L/R names: ADEL, ADER, RIML, RIMR
    - CEP subtypes: CEPDL, CEPDR, CEPVL, CEPVR
    
    Our data uses:
    - L/R collapsed: ADE, RIM
    - CEP collapsed: CEP
    
    Args:
        name: Raw neuron name from Bentley data
        
    Returns:
        Normalized name matching our convention
    """
    name = str(name).upper().strip()
    
    # Remove L/R suffix for bilateral neurons
    if len(name) > 1 and name[-1] in ['L', 'R']:
        base = name[:-1]
        # Don't collapse single-letter suffixes that are part of the name
        # e.g., PQR should stay PQR, not PQ
        # Check if this looks like a bilateral suffix
        if len(base) >= 2:
            name = base
    
    # Map CEP dorsal/ventral variants to single CEP class
    if name in ['CEPD', 'CEPV']:
        name = 'CEP'
    
    return name


def load_our_neuron_names(dataset: str = "full_traces_imputed") -> List[str]:
    """Load our 80 neuron names from the standardization file."""
    std_path = PROJECT_ROOT / "results" / "intermediate" / "datasets" / dataset / "standardization.json"
    
    if std_path.exists():
        with open(std_path) as f:
            data = json.load(f)
            return data['node_order']
    
    raise FileNotFoundError(f"Could not find standardization.json at {std_path}")


# =============================================================================
# MONOAMINE DATA LOADING
# =============================================================================

@dataclass
class MonoamineNetwork:
    """Container for monoamine network data."""
    edges_df: pd.DataFrame  # Raw edge dataframe
    adjacency: np.ndarray   # (n, n) binary adjacency aligned to our neurons
    neuron_names: List[str] # Our neuron names (80)
    n_raw_edges: int        # Total edges before filtering
    n_filtered_edges: int   # Edges surviving our neuron filter
    transmitter: str        # 'dopamine', 'serotonin', 'tyramine', 'octopamine', or 'all'
    
    @property
    def edge_density(self) -> float:
        """Fraction of possible edges that exist."""
        n = len(self.neuron_names)
        max_edges = n * (n - 1)  # Directed, no self-loops
        return self.n_filtered_edges / max_edges if max_edges > 0 else 0


def load_monoamine_edges(include_dop56: bool = False) -> pd.DataFrame:
    """
    Load monoamine edge list from Bentley CSV.
    
    Args:
        include_dop56: If True, use extended file including dop-5/dop-6 receptors
        
    Returns:
        DataFrame with columns: source, target, transmitter, receptor,
                                source_normalized, target_normalized
    """
    if include_dop56:
        csv_path = BENTLEY_DIR / "edge_lists" / "edgelist_MA_incl_dop-5_dop-6.csv"
    else:
        csv_path = BENTLEY_DIR / "edge_lists" / "edgelist_MA.csv"
    
    if not csv_path.exists():
        # Try to provide helpful error message
        print(f"\nERROR: Monoamine edge list not found at:")
        print(f"  {csv_path}")
        print(f"\nChecking if parent directories exist:")
        print(f"  DATA_DIR exists: {DATA_DIR.exists()}")
        print(f"  BENTLEY_DIR exists: {BENTLEY_DIR.exists()}")
        if BENTLEY_DIR.exists():
            print(f"  Contents of BENTLEY_DIR:")
            for item in BENTLEY_DIR.iterdir():
                print(f"    - {item.name}")
        raise FileNotFoundError(f"Monoamine edge list not found at {csv_path}")
    
    # Load CSV (no header)
    df = pd.read_csv(csv_path, header=None, 
                     names=['source', 'target', 'transmitter', 'receptor'])
    
    # Normalize neuron names
    df['source_normalized'] = df['source'].apply(normalize_bentley_name)
    df['target_normalized'] = df['target'].apply(normalize_bentley_name)
    
    return df


def build_monoamine_adjacency(
    edges_df: pd.DataFrame,
    neuron_names: List[str],
    transmitter: Optional[str] = None,
) -> Tuple[np.ndarray, int, int]:
    """
    Build adjacency matrix from monoamine edges.
    
    Args:
        edges_df: DataFrame with normalized source/target columns
        neuron_names: List of our neuron names (determines matrix order)
        transmitter: Filter to specific transmitter, or None for all
        
    Returns:
        adjacency: (n, n) binary matrix where A[i,j] = 1 means j → i
        n_raw_edges: Total edges for this transmitter
        n_filtered_edges: Edges where both source and target are in neuron_names
    """
    n = len(neuron_names)
    name_to_idx = {name: i for i, name in enumerate(neuron_names)}
    
    # Filter by transmitter if specified
    if transmitter is not None:
        df = edges_df[edges_df['transmitter'] == transmitter].copy()
    else:
        df = edges_df.copy()
    
    n_raw_edges = len(df)
    
    # Build adjacency
    adjacency = np.zeros((n, n), dtype=np.float32)
    n_filtered_edges = 0
    
    for _, row in df.iterrows():
        src = row['source_normalized']
        tgt = row['target_normalized']
        
        if src in name_to_idx and tgt in name_to_idx:
            i = name_to_idx[tgt]  # Target = row
            j = name_to_idx[src]  # Source = column
            if i != j:  # No self-loops
                adjacency[i, j] = 1
                n_filtered_edges += 1
    
    return adjacency, n_raw_edges, n_filtered_edges


def load_all_monoamine_networks(
    neuron_names: List[str],
    include_dop56: bool = False,
) -> Dict[str, MonoamineNetwork]:
    """
    Load all monoamine networks (per-transmitter and combined).
    
    Args:
        neuron_names: Our 80 neuron names
        include_dop56: Include extended dopamine network
        
    Returns:
        Dictionary with keys: 'dopamine', 'serotonin', 'tyramine', 'octopamine', 'all'
    """
    edges_df = load_monoamine_edges(include_dop56=include_dop56)
    
    networks = {}
    
    transmitters = ['dopamine', 'serotonin', 'tyramine', 'octopamine']
    
    for trans in transmitters:
        adj, n_raw, n_filt = build_monoamine_adjacency(edges_df, neuron_names, trans)
        networks[trans] = MonoamineNetwork(
            edges_df=edges_df[edges_df['transmitter'] == trans],
            adjacency=adj,
            neuron_names=neuron_names,
            n_raw_edges=n_raw,
            n_filtered_edges=n_filt,
            transmitter=trans,
        )
    
    # Combined network
    adj_all, n_raw_all, n_filt_all = build_monoamine_adjacency(edges_df, neuron_names, None)
    networks['all'] = MonoamineNetwork(
        edges_df=edges_df,
        adjacency=adj_all,
        neuron_names=neuron_names,
        n_raw_edges=n_raw_all,
        n_filtered_edges=n_filt_all,
        transmitter='all',
    )
    
    return networks


# =============================================================================
# VALIDATION
# =============================================================================

def validate_monoamine_data(networks: Dict[str, MonoamineNetwork], verbose: bool = True):
    """
    Validate loaded monoamine data against expected values from paper.
    
    Expected edge counts from Bentley paper (Table 2):
    - Dopamine: ~1752 edges (without dop-5/6)
    - Serotonin: ~540 edges
    - Tyramine: ~262 edges
    - Octopamine: ~72 edges
    """
    print("\n" + "="*60)
    print("MONOAMINE NETWORK VALIDATION")
    print("="*60)
    
    # Expected raw counts (from our exploration)
    expected = {
        'dopamine': 1752,
        'serotonin': 540,
        'tyramine': 262,
        'octopamine': 72,
    }
    
    print("\n[1] Edge Count Verification")
    print("-" * 40)
    
    total_raw = 0
    total_filtered = 0
    
    for trans in ['dopamine', 'serotonin', 'tyramine', 'octopamine']:
        net = networks[trans]
        exp = expected.get(trans, 'N/A')
        match = "✓" if net.n_raw_edges == exp else "✗"
        retention = 100 * net.n_filtered_edges / net.n_raw_edges if net.n_raw_edges > 0 else 0
        
        print(f"  {trans:12s}: {net.n_raw_edges:4d} raw {match} (expected {exp}), "
              f"{net.n_filtered_edges:3d} in our neurons ({retention:.1f}%)")
        
        total_raw += net.n_raw_edges
        total_filtered += net.n_filtered_edges
    
    print(f"  {'TOTAL':12s}: {total_raw:4d} raw, {total_filtered:3d} filtered")
    
    # Check hub neurons
    print("\n[2] Hub Neuron Verification")
    print("-" * 40)
    print("  Checking monoamine-producing neurons have high out-degree...")
    
    neuron_names = networks['all'].neuron_names
    adj_all = networks['all'].adjacency
    
    # Known monoamine sources in our data
    known_sources = {
        'ADE': 'dopamine',
        'CEP': 'dopamine', 
        'ADF': 'serotonin',
        'RIM': 'tyramine',
        'RIC': 'octopamine',
    }
    
    name_to_idx = {n: i for i, n in enumerate(neuron_names)}
    
    for neuron, trans in known_sources.items():
        if neuron in name_to_idx:
            idx = name_to_idx[neuron]
            out_degree = adj_all[:, idx].sum()  # Column = source
            in_degree = adj_all[idx, :].sum()   # Row = target
            print(f"  {neuron:4s} ({trans:10s}): out={int(out_degree):2d}, in={int(in_degree):2d}")
        else:
            print(f"  {neuron:4s} ({trans:10s}): NOT IN OUR NEURONS")
    
    # Sparsity check
    print("\n[3] Sparsity Check")
    print("-" * 40)
    
    for trans in ['dopamine', 'serotonin', 'tyramine', 'octopamine', 'all']:
        net = networks[trans]
        print(f"  {trans:12s}: density = {net.edge_density:.4f} "
              f"({net.n_filtered_edges} / {len(net.neuron_names)**2 - len(net.neuron_names)})")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


# =============================================================================
# PHASE 2: EVALUATION AGAINST MULTI-LAG RESULTS
# =============================================================================

def load_multilag_results(result_dir: Path) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """
    Load multi-lag SBTG results from result directory.
    
    Args:
        result_dir: Path to results (e.g., results/multilag_separation/TIMESTAMP/)
        
    Returns:
        mu_hat: Dict mapping lag -> (n, n) coupling matrix
        neuron_names: List of neuron names
    """
    from sklearn.metrics import roc_auc_score
    
    # Try approach C first, then others
    for approach in ['C', 'A', 'B']:
        result_file = result_dir / f"result_{approach}.npz"
        if result_file.exists():
            break
    else:
        raise FileNotFoundError(f"No result_*.npz found in {result_dir}")
    
    print(f"Loading: {result_file}")
    data = np.load(result_file, allow_pickle=True)
    
    # Extract mu_hat for each lag
    mu_hat = {}
    lags = []
    
    for key in data.keys():
        if key.startswith('mu_hat_lag'):
            lag = int(key.replace('mu_hat_lag', ''))
            mu_hat[lag] = data[key]
            lags.append(lag)
    
    # Get neuron names
    neuron_names = list(data['neuron_names']) if 'neuron_names' in data else None
    
    print(f"  Lags: {sorted(lags)}")
    print(f"  Matrix shape: {mu_hat[lags[0]].shape}")
    
    return mu_hat, neuron_names


def compute_auroc(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute AUROC between prediction scores and binary ground truth.
    
    Args:
        prediction: (n, n) continuous scores (higher = more confident edge)
        ground_truth: (n, n) binary adjacency (1 = edge exists)
        
    Returns:
        AUROC score (0.5 = random, 1.0 = perfect)
    """
    from sklearn.metrics import roc_auc_score
    
    n = prediction.shape[0]
    mask = ~np.eye(n, dtype=bool)  # Exclude diagonal
    
    y_true = ground_truth[mask].flatten()
    y_score = np.abs(prediction[mask]).flatten()  # Use absolute coupling strength
    
    # Need at least one positive and one negative
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    
    return roc_auc_score(y_true, y_score)


def evaluate_monoamine_prediction(
    mu_hat: Dict[int, np.ndarray],
    networks: Dict[str, MonoamineNetwork],
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Evaluate multi-lag predictions against monoamine networks.
    
    Args:
        mu_hat: Dict mapping lag -> (n, n) coupling matrix
        networks: Dict of MonoamineNetwork objects
        output_dir: Directory to save results and figures
        
    Returns:
        DataFrame with AUROC per lag per transmitter
    """
    import matplotlib.pyplot as plt
    
    lags = sorted(mu_hat.keys())
    transmitters = ['dopamine', 'serotonin', 'tyramine', 'octopamine', 'all']
    
    results = []
    
    print("\n" + "="*60)
    print("MONOAMINE PREDICTION EVALUATION")
    print("="*60)
    
    # Compute AUROC for each lag and transmitter
    for lag in lags:
        mu = mu_hat[lag]
        
        for trans in transmitters:
            net = networks[trans]
            auroc = compute_auroc(mu, net.adjacency)
            
            results.append({
                'lag': lag,
                'time_s': lag * 0.25,  # Assuming 4Hz
                'transmitter': trans,
                'auroc': auroc,
                'n_edges': net.n_filtered_edges,
            })
    
    df = pd.DataFrame(results)
    
    # Print results table
    print("\nAUROC by Lag and Transmitter:")
    print("-" * 60)
    
    pivot = df.pivot(index='lag', columns='transmitter', values='auroc')
    pivot = pivot[['dopamine', 'serotonin', 'tyramine', 'octopamine', 'all']]
    print(pivot.round(3).to_string())
    
    # Find best lag for each transmitter
    print("\nBest Lag per Transmitter:")
    print("-" * 40)
    for trans in transmitters:
        sub = df[df['transmitter'] == trans]
        best = sub.loc[sub['auroc'].idxmax()]
        print(f"  {trans:12s}: lag {int(best['lag'])} ({best['time_s']:.2f}s) AUROC={best['auroc']:.3f}")
    
    # Create figure
    if output_dir:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {
            'dopamine': '#e41a1c',
            'serotonin': '#377eb8', 
            'tyramine': '#4daf4a',
            'octopamine': '#984ea3',
            'all': '#333333',
        }
        
        for trans in transmitters:
            sub = df[df['transmitter'] == trans]
            linestyle = '-' if trans != 'all' else '--'
            linewidth = 2 if trans != 'all' else 3
            ax.plot(sub['time_s'], sub['auroc'], 
                    label=trans.capitalize(), 
                    color=colors[trans],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    marker='o')
        
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
        ax.set_xlabel('Lag (seconds)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title('Multi-Lag SBTG vs Monoaminergic Connectome', fontsize=14)
        ax.legend(loc='best')
        ax.set_ylim(0.4, 0.7)
        ax.grid(alpha=0.3)
        
        fig_path = output_dir / 'fig_auroc_monoamine_by_lag.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved figure: {fig_path}")
        
        # Save CSV
        csv_path = output_dir / 'eval_monoamine.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved results: {csv_path}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multilayer (monoamine) connectome analysis")
    parser.add_argument("--validate", action="store_true",
                        help="Validate monoamine data loading")
    parser.add_argument("--include-dop56", action="store_true",
                        help="Include extended dopamine network (dop-5/dop-6)")
    parser.add_argument("--dataset", default="full_traces_imputed",
                        help="Dataset to use for neuron names")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="Multi-lag result directory for evaluation")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MULTILAYER CONNECTOME ANALYSIS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Include dop-5/dop-6: {args.include_dop56}")
    
    # Load our neuron names
    try:
        neuron_names = load_our_neuron_names(args.dataset)
        print(f"Loaded {len(neuron_names)} neuron names")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Load monoamine networks
    print("\nLoading monoamine networks...")
    networks = load_all_monoamine_networks(neuron_names, args.include_dop56)
    
    if args.validate:
        validate_monoamine_data(networks)
    
    # Phase 2: Evaluation against multi-lag results
    if args.result_dir:
        result_dir = Path(args.result_dir)
        if not result_dir.exists():
            print(f"ERROR: Result directory not found: {result_dir}")
            return
        
        print(f"\n[Phase 2] Evaluating multi-lag results...")
        print(f"  Result dir: {result_dir}")
        
        try:
            mu_hat, _ = load_multilag_results(result_dir)
            evaluate_monoamine_prediction(mu_hat, networks, output_dir=result_dir)
        except Exception as e:
            print(f"ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone.")


if __name__ == "__main__":
    main()

