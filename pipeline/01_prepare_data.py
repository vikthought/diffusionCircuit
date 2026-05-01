#!/usr/bin/env python3
"""
SCRIPT 1: Prepare Data for SBTG Analysis

This script performs all data preparation steps needed for the SBTG analysis:
1. Aligns the Cook_Synapses_2019 connectome with NeuroPAL neuron names
2. Builds standardized lag-window datasets for each stimulus
3. Generates quality control visualizations

USAGE:
    python pipeline/01_prepare_data.py

OUTPUTS:
    - results/intermediate/connectome/     (adjacency matrices, node lists)
    - results/intermediate/datasets/       (lag-window datasets per stimulus)
    - results/figures/connectome/          (connectome heatmaps)
    - results/tables/connectome/           (alignment tables)

REQUIREMENTS:
    - data/Head_Activity_OH16230.mat       (NeuroPAL recordings)
    - data/SI 6 Cell class lists.xlsx      (Cook neuron list)
    - data/SI 7 Cell class connectome...xlsx (Cook adjacency matrices)

RUNTIME: ~2-3 minutes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat

# Add project root to path for pipeline.utils imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports from pipeline utils
from pipeline.utils.plotting import plot_connectome_heatmaps

# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum number of worms a neuron must appear in to be included.
# Lower values retain more neurons but reduce complete-coverage overlap.
MIN_WORMS = 15

# Which stimuli to process (set to None to process all)
STIMULI = None  # or ["nacl", "butanone", "pentanedione"]

# Use full traces (240s) instead of just stimulus windows (10s)
USE_FULL_TRACES = False  # Can be overridden via --full-traces flag

# Collapse D/V subtypes to parent cell classes (RMDD/RMDV -> RMD)
# This recovers ~20 neurons lost due to naming mismatch with Cook connectome
COLLAPSE_DV = True  # Can be disabled via --no-collapse-dv flag

# Include tail neurons from Tail_Activity_OH16230.mat
# This adds ~17 neurons to the analysis
INCLUDE_TAIL = True  # Can be disabled via --no-tail flag

# Impute missing neurons using donor worms
# When enabled, worms with missing neurons will borrow traces from other worms
# This allows using ALL worms for training instead of only "complete" ones
IMPUTE_MISSING = False  # Can be enabled via --impute-missing flag

# File names (should not need to change these)
NEUROPAL_HEAD_FILE = "Head_Activity_OH16230.mat"
NEUROPAL_TAIL_FILE = "Tail_Activity_OH16230.mat"
NEUROPAL_FILE = NEUROPAL_HEAD_FILE  # Backward compatibility alias
COOK_CONNECTOME_FILE = "SI 7 Cell class connectome adjacency matrices, corrected July 2020.xlsx"
COOK_CELL_LIST_FILE = "SI 6 Cell class lists.xlsx"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def locate_project_root(start: Path) -> Path:
    """
    Find the project root directory by looking for the 'data' folder.
    
    Args:
        start: Starting directory path
        
    Returns:
        Path to project root
        
    Raises:
        FileNotFoundError: If project root cannot be found
    """
    candidate = start.resolve()
    while True:
        if (candidate / "data").exists():
            return candidate
        if candidate.parent == candidate:
            raise FileNotFoundError(
                "Unable to locate project root. Please run from within the project directory."
            )
        candidate = candidate.parent


def normalize_name(name: str) -> str:
    """
    Normalize neuron class names to uppercase without whitespace.
    
    Args:
        name: Raw neuron name
        
    Returns:
        Normalized name (e.g., "  ase  " -> "ASE")
    """
    return name.strip().upper()


def ensure_directories(results_dir: Path) -> Dict[str, Path]:
    """
    Create all necessary output directories.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary mapping directory purposes to paths
    """
    dirs = {
        "connectome_intermediate": results_dir / "intermediate" / "connectome",
        "connectome_figures": results_dir / "figures" / "connectome",
        "connectome_tables": results_dir / "tables" / "connectome",
        "datasets": results_dir / "intermediate" / "datasets",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


# =============================================================================
# STEP 1: CONNECTOME ALIGNMENT
# =============================================================================

def load_cook_adjacency_with_labels(
    spreadsheet_path: Path,
    sheet_name: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load adjacency matrix from Cook et al. 2019 spreadsheet.
    
    The spreadsheets have a specific format:
    - Row 0: Excel metadata (skip)
    - Row 1: Column headers (neuron names)
    - Row 2+: Row label (neuron name) + adjacency values
    
    Args:
        spreadsheet_path: Path to Excel file
        sheet_name: Name of sheet to read
        
    Returns:
        Tuple of (adjacency_matrix, column_labels, row_labels)
    """
    print(f"  Loading sheet '{sheet_name}'...")
    
    # Read the raw spreadsheet
    df = pd.read_excel(spreadsheet_path, sheet_name=sheet_name, header=None)
    
    # Extract column labels from row 1
    col_labels = []
    for val in df.iloc[1, 2:]:
        if isinstance(val, str) and val.strip():
            col_labels.append(normalize_name(val))
        else:
            break  # Stop at first empty cell
    
    num_cols = len(col_labels)
    print(f"    Found {num_cols} column neurons")
    
    # Extract row labels and values
    row_labels = []
    matrix_rows = []
    
    for idx in range(2, len(df)):
        row = df.iloc[idx]
        
        # Get row label (check both column 0 and 1)
        label = None
        for col_idx in [1, 0]:
            val = row.iloc[col_idx]
            if isinstance(val, str) and val.strip():
                label = normalize_name(val)
                break
        
        if label is None:
            continue  # Skip rows without labels
        
        # Get adjacency values
        values = row.iloc[2:2 + num_cols].tolist()
        numeric_values = pd.to_numeric(
            pd.Series(values),
            errors="coerce"
        ).fillna(0.0).tolist()
        
        row_labels.append(label)
        matrix_rows.append(numeric_values)
    
    # Convert to numpy array
    adjacency = np.array(matrix_rows, dtype=float)
    print(f"    Matrix shape: {adjacency.shape} (rows={len(row_labels)}, cols={num_cols})")
    
    return adjacency, col_labels, row_labels


def align_matrices_to_common_neurons(
    A_chem: np.ndarray,
    col_labels_chem: List[str],
    row_labels_chem: List[str],
    A_gap: np.ndarray,
    col_labels_gap: List[str],
    row_labels_gap: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Align chemical and gap junction matrices to common neuron set.
    
    Args:
        A_chem: Chemical synapse matrix
        col_labels_chem: Column labels for chemical matrix
        row_labels_chem: Row labels for chemical matrix
        A_gap: Gap junction matrix
        col_labels_gap: Column labels for gap matrix
        row_labels_gap: Row labels for gap matrix
        
    Returns:
        Tuple of (aligned_A_chem, aligned_A_gap, common_neurons)
    """
    # Find neurons present in both row and column labels for each matrix
    chem_neurons = set(col_labels_chem) & set(row_labels_chem)
    gap_neurons = set(col_labels_gap) & set(row_labels_gap)
    
    # Find common neurons across both matrices
    common_neurons = sorted(chem_neurons & gap_neurons)
    print(f"  Chemical neurons (rows ∩ cols): {len(chem_neurons)}")
    print(f"  Gap junction neurons (rows ∩ cols): {len(gap_neurons)}")
    print(f"  Common neurons: {len(common_neurons)}")
    
    # Create aligned matrices
    n = len(common_neurons)
    aligned_chem = np.zeros((n, n), dtype=float)
    aligned_gap = np.zeros((n, n), dtype=float)
    
    # Build index maps
    chem_col_idx = {name: i for i, name in enumerate(col_labels_chem)}
    chem_row_idx = {name: i for i, name in enumerate(row_labels_chem)}
    gap_col_idx = {name: i for i, name in enumerate(col_labels_gap)}
    gap_row_idx = {name: i for i, name in enumerate(row_labels_gap)}
    
    # Fill aligned matrices
    for i, row_name in enumerate(common_neurons):
        for j, col_name in enumerate(common_neurons):
            # Chemical
            if row_name in chem_row_idx and col_name in chem_col_idx:
                aligned_chem[i, j] = A_chem[chem_row_idx[row_name], chem_col_idx[col_name]]
            # Gap junction
            if row_name in gap_row_idx and col_name in gap_col_idx:
                aligned_gap[i, j] = A_gap[gap_row_idx[row_name], gap_col_idx[col_name]]
    
    return aligned_chem, aligned_gap, common_neurons


def align_connectome(
    data_dir: Path,
    output_dirs: Dict[str, Path],
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Load and align Cook connectome data.
    
    Processes:
    - Chemical synapses (directed)
    - Gap junctions (symmetric)
    - Combined structural connectivity
    
    Args:
        data_dir: Path to data directory
        output_dirs: Dictionary of output directory paths
        
    Returns:
        Tuple of (node_list, adjacency_dict)
        where adjacency_dict has keys: 'chem', 'gap', 'struct'
    """
    print("\n" + "=" * 70)
    print("STEP 1: Aligning Cook Connectome with NeuroPAL")
    print("=" * 70)
    
    connectome_file = data_dir / COOK_CONNECTOME_FILE
    
    if not connectome_file.exists():
        raise FileNotFoundError(
            f"Cook connectome file not found: {connectome_file}\n"
            f"Please ensure it is in the data/ directory."
        )
    
    # Load chemical synapses
    print("\nLoading chemical synapses...")
    A_chem_raw, col_labels_chem, row_labels_chem = load_cook_adjacency_with_labels(
        connectome_file,
        "herm chem grouped"
    )
    
    # Load gap junctions
    print("\nLoading gap junctions...")
    A_gap_raw, col_labels_gap, row_labels_gap = load_cook_adjacency_with_labels(
        connectome_file,
        "herm gap jn grouped asymmetric"
    )
    
    # Align matrices to common neuron set
    print("\nAligning matrices to common neuron set...")
    A_chem, A_gap, nodes = align_matrices_to_common_neurons(
        A_chem_raw, col_labels_chem, row_labels_chem,
        A_gap_raw, col_labels_gap, row_labels_gap
    )
    
    print(f"\n✓ Loaded connectome with {len(nodes)} neuron classes")
    
    # Create combined structural matrix (chemical + gap)
    A_struct = A_chem + A_gap
    
    # Save adjacency matrices
    print("\nSaving adjacency matrices...")
    np.save(output_dirs["connectome_intermediate"] / "A_chem.npy", A_chem)
    np.save(output_dirs["connectome_intermediate"] / "A_gap.npy", A_gap)
    np.save(output_dirs["connectome_intermediate"] / "A_struct.npy", A_struct)
    
    # Save node list
    with open(output_dirs["connectome_intermediate"] / "nodes.json", "w") as f:
        json.dump(nodes, f, indent=2)
    
    print(f"  Saved to: {output_dirs['connectome_intermediate']}/")
    
    # Generate visualizations
    print("\nGenerating connectome visualizations...")
    plot_connectome_heatmaps(
        {"Chemical": A_chem, "Gap Junctions": A_gap, "Combined": A_struct},
        nodes,
        output_dirs["connectome_figures"]
    )
    
    # Create summary table
    summary_df = create_connectome_summary(A_chem, A_gap, A_struct, nodes)
    summary_df.to_csv(
        output_dirs["connectome_tables"] / "connectome_summary.csv",
        index=False
    )
    
    print(f"\n✓ Connectome alignment complete")
    print(f"  Total edges - Chemical: {int((A_chem > 0).sum())}, "
          f"Gap: {int((A_gap > 0).sum())}, "
          f"Combined: {int((A_struct > 0).sum())}")
    
    return nodes, {
        "chem": A_chem,
        "gap": A_gap,
        "struct": A_struct
    }





def create_connectome_summary(
    A_chem: np.ndarray,
    A_gap: np.ndarray,
    A_struct: np.ndarray,
    nodes: List[str],
) -> pd.DataFrame:
    """
    Create summary statistics table for connectome.
    
    Args:
        A_chem: Chemical synapse adjacency matrix
        A_gap: Gap junction adjacency matrix
        A_struct: Combined structural adjacency matrix
        nodes: List of neuron names
        
    Returns:
        DataFrame with per-neuron statistics
    """
    records = []
    
    # ADJACENCY CONVENTION: A[post, pre] = A[target, source]
    # This means:
    # - Row i contains all inputs TO neuron i (presynaptic partners)
    # - Column j contains all outputs FROM neuron j (postsynaptic partners)
    # Therefore:
    # - in_degree[i] = row sum = sum(A[i, :]) = # neurons projecting TO i
    # - out_degree[j] = col sum = sum(A[:, j]) = # neurons receiving FROM j
    
    for i, node in enumerate(nodes):
        # Chemical synapses
        # in_degree = row sum (inputs TO this neuron)
        chem_in = int((A_chem[i, :] > 0).sum())
        # out_degree = column sum (outputs FROM this neuron)  
        chem_out = int((A_chem[:, i] > 0).sum())
        
        # Gap junctions
        gap_in = int((A_gap[i, :] > 0).sum())
        gap_out = int((A_gap[:, i] > 0).sum())
        
        # Combined
        struct_in = int((A_struct[i, :] > 0).sum())
        struct_out = int((A_struct[:, i] > 0).sum())
        
        records.append({
            "neuron": node,
            "chem_out_degree": chem_out,
            "chem_in_degree": chem_in,
            "gap_out_degree": gap_out,
            "gap_in_degree": gap_in,
            "total_out_degree": struct_out,
            "total_in_degree": struct_in,
        })
    
    return pd.DataFrame(records)


# =============================================================================
# STEP 2: BUILD SBTG DATASETS
# =============================================================================

def load_tail_data(data_dir: Path) -> Dict:
    """
    Load NeuroPAL tail neuron recording from MATLAB file.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary with neuron names and traces, or None if file not found
    """
    tail_path = data_dir / NEUROPAL_TAIL_FILE
    
    if not tail_path.exists():
        print(f"  Warning: Tail file not found: {tail_path.name}")
        return None
    
    print(f"  Loading tail data from: {tail_path.name}")
    mat = loadmat(tail_path, simplify_cells=True)
    
    return {
        "neuron_names": [normalize_name(str(n)) for n in mat["neurons"]],
        "norm_traces": mat["norm_traces"],
        "fps": float(mat["fps"]),
        "stim_names": [str(s) for s in mat["stim_names"]],
        "stim_times": np.asarray(mat["stim_times"], dtype=float),
        "stims_per_worm": [np.asarray(row, dtype=int) for row in mat["stims"]],
        "worm_ids": [str(f) for f in mat["files"]],
    }


def merge_head_tail_data(head_data: Dict, tail_data: Dict) -> Dict:
    """
    Merge head and tail NeuroPAL recordings.
    
    Both recordings must have the same worms in the same order.
    
    Args:
        head_data: Head neuron recording data
        tail_data: Tail neuron recording data
        
    Returns:
        Merged data dictionary with all neurons
    """
    if tail_data is None:
        return head_data
    
    # Verify worm alignment
    if len(head_data["worm_ids"]) != len(tail_data["worm_ids"]):
        print(f"  Warning: Head has {len(head_data['worm_ids'])} worms, "
              f"tail has {len(tail_data['worm_ids'])} worms. Using head only.")
        return head_data
    
    # Merge neuron names and traces
    merged_names = list(head_data["neuron_names"]) + list(tail_data["neuron_names"])
    
    # Merge norm_traces (list of arrays per neuron)
    # Each entry is a list of traces for that neuron across worms
    merged_traces = list(head_data["norm_traces"]) + list(tail_data["norm_traces"])
    
    print(f"  Merged: {len(head_data['neuron_names'])} head + "
          f"{len(tail_data['neuron_names'])} tail = {len(merged_names)} neurons")
    
    return {
        "neuron_names": merged_names,
        "norm_traces": merged_traces,
        "fps": head_data["fps"],
        "stim_names": head_data["stim_names"],
        "stim_times": head_data["stim_times"],
        "stims_per_worm": head_data["stims_per_worm"],
        "worm_ids": head_data["worm_ids"],
        "source": "head+tail",
    }


def load_neuropal_data(
    data_dir: Path,
    include_tail: bool = False,
    collapse_dv: bool = False,
) -> Dict:
    """
    Load NeuroPAL recording from MATLAB file(s).
    
    Args:
        data_dir: Path to data directory
        include_tail: If True, also load and merge tail neurons
        collapse_dv: If True, collapse D/V subtypes (RMDD/RMDV -> RMD)
        
    Returns:
        Dictionary with neuron names, traces, stimulus info, etc.
    """
    # Import D/V collapsing utility
    from pipeline.utils.align import collapse_dv_subtypes
    
    neuropal_path = data_dir / NEUROPAL_HEAD_FILE
    
    if not neuropal_path.exists():
        raise FileNotFoundError(
            f"NeuroPAL file not found: {neuropal_path}\n"
            f"Please ensure it is in the data/ directory."
        )
    
    print(f"\nLoading NeuroPAL data from: {neuropal_path.name}")
    mat = loadmat(neuropal_path, simplify_cells=True)
    
    head_data = {
        "neuron_names": [normalize_name(str(n)) for n in mat["neurons"]],
        "norm_traces": mat["norm_traces"],
        "fps": float(mat["fps"]),
        "stim_names": [str(s) for s in mat["stim_names"]],
        "stim_times": np.asarray(mat["stim_times"], dtype=float),
        "stims_per_worm": [np.asarray(row, dtype=int) for row in mat["stims"]],
        "worm_ids": [str(f) for f in mat["files"]],
    }
    
    # Optionally load and merge tail data
    if include_tail:
        tail_data = load_tail_data(data_dir)
        data = merge_head_tail_data(head_data, tail_data)
    else:
        data = head_data
    
    # Optionally collapse D/V subtypes
    if collapse_dv:
        original_names = data["neuron_names"]
        collapsed_names = [collapse_dv_subtypes(n) for n in original_names]
        
        # Create mapping from collapsed name to list of original indices
        name_to_indices = {}
        for idx, (orig, collapsed) in enumerate(zip(original_names, collapsed_names)):
            if collapsed not in name_to_indices:
                name_to_indices[collapsed] = []
            name_to_indices[collapsed].append(idx)
            
        n_collapsed = sum(1 for indices in name_to_indices.values() if len(indices) > 1)
        if n_collapsed > 0:
            print(f"  D/V collapsing: Averaging variants for {n_collapsed} neuron classes")
            
        # Update data structure
        # We keep the original traces and indices, but expose the collapsed names
        # and a mapping mechanism. To fit the existing structure, we won't replace
        # "neuron_names" with a unique list yet, because other functions rely on
        # 1-to-1 mapping with traces.
        # Instead, we will handle the grouping in the consuming functions.
        # BUT validity requires us to present a unique list of neurons to the rest of the pipeline.
        
        unique_names = sorted(list(name_to_indices.keys()))
        data["neuron_names"] = unique_names
        data["name_to_indices"] = name_to_indices
        data["neuron_names_original"] = original_names
        data["dv_collapsed"] = True
    else:
        # Default 1-to-1 mapping
        data["name_to_indices"] = {n: [i] for i, n in enumerate(data["neuron_names"])}
        data["dv_collapsed"] = False
    
    return data


def compute_node_coverage(
    nodes: List[str],
    neuropal_data: Dict,
) -> pd.DataFrame:
    """
    Compute how many worms each neuron appears in.
    
    Args:
        nodes: List of neuron names (from connectome)
        neuropal_data: NeuroPAL recording data
        
    Returns:
        DataFrame with worm coverage counts per neuron
    """
    # Use the name_to_indices mapping created in load_neuropal_data
    name_to_indices = neuropal_data.get("name_to_indices")
    if name_to_indices is None:
         # Fallback if not collapsed (legacy support, though load_neuropal_data now always sets it)
         name_to_indices = {n: [i] for i, n in enumerate(neuropal_data["neuron_names"])}

    num_worms = len(neuropal_data["worm_ids"])
    
    records = []
    for node in nodes:
        if node not in name_to_indices:
            # Neuron not present in NeuroPAL recordings
            records.append({"node": node, "worm_count": 0})
            continue
        
        indices = name_to_indices[node]
        count = 0
        
        # Count how many worms have this neuron (averaging variants if needed)
        for worm_idx in range(num_worms):
            trace = collect_worm_trace(
                neuropal_data["norm_traces"], # Pass ALL traces
                indices,                      # Pass list of indices for this neuron
                worm_idx,
                num_worms
            )
            if trace is not None:
                count += 1
        
        records.append({"node": node, "worm_count": count})
    
    df = pd.DataFrame(records)
    df = df.sort_values("worm_count", ascending=False).reset_index(drop=True)
    
    return df


def collect_worm_trace(
    all_traces: List[np.ndarray],
    neuron_indices: List[int],
    worm_idx: int,
    num_worms: int,
) -> np.ndarray | None:
    """
    Collect and average traces for a neuron (and its variants) in a specific worm.
    
    Handles:
    1. Left/Right averaging (standard NeuroPAL structure)
    2. D/V subtype averaging (if multiple indices provided)
    
    Args:
        all_traces: List of ALL trace arrays (dataset["norm_traces"])
        neuron_indices: List of indices corresponding to this neuron class (e.g. [idx_RMDD, idx_RMDV])
        worm_idx: Which worm (0-indexed)
        num_worms: Total number of worms
        
    Returns:
        Averaged trace array, or None if no valid trace found for ANY variant
    """
    segments = []
    
    for idx in neuron_indices:
        neuron_traces = all_traces[idx]
        
        # Check left and right traces for this variant
        for offset in [worm_idx, worm_idx + num_worms]:
            if offset >= len(neuron_traces):
                continue
            
            arr = np.asarray(neuron_traces[offset], dtype=float)
            if arr.size == 0:
                continue
                
            segments.append(arr)
    
    if not segments:
        return None
    
    # Trim to shortest length and average
    min_len = min(seg.shape[-1] for seg in segments)
    stacked = np.stack([seg[-min_len:] for seg in segments], axis=0)
    
    return stacked.mean(axis=0)


def select_nodes_and_worms(
    coverage_df: pd.DataFrame,
    neuropal_data: Dict,
    min_worms: int,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Select neurons with sufficient coverage and worms with all selected neurons.
    
    Args:
        coverage_df: Node coverage dataframe
        neuropal_data: NeuroPAL recording data
        min_worms: Minimum worm count threshold
        
    Returns:
        Tuple of (selected_nodes, eligible_worms, name_to_idx)
    """
    # Select nodes with sufficient coverage
    selected_nodes = coverage_df[
        coverage_df["worm_count"] >= min_worms
    ]["node"].tolist()
    
    if not selected_nodes:
        raise ValueError(
            f"No neurons meet the min_worms={min_worms} threshold. "
            f"Try lowering the threshold."
        )
    
    print(f"  Selected {len(selected_nodes)} neurons "
          f"(coverage >= {min_worms} worms)")
    
    # Build name->indices mapping (already exists in data)
    name_to_indices = neuropal_data["name_to_indices"]
    
    # name_to_indices is Dict[str, List[int]] after D/V collapse
    
    # Find worms that have ALL selected neurons
    num_worms = len(neuropal_data["worm_ids"])
    eligible_worms = []
    
    for worm_idx in range(num_worms):
        has_all = True
        
        for node in selected_nodes:
            indices = name_to_indices[node]
            trace = collect_worm_trace(
                neuropal_data["norm_traces"],
                indices,
                worm_idx,
                num_worms
            )
            
            if trace is None:
                has_all = False
                break
        
        if has_all:
            eligible_worms.append(worm_idx)
    
    print(f"  Found {len(eligible_worms)}/{num_worms} worms with complete coverage")
    
    if not eligible_worms:
        raise ValueError(
            "No worms contain all selected neurons. "
            "Try lowering min_worms threshold."
        )
    
    return selected_nodes, eligible_worms, name_to_indices


def build_donor_map(
    selected_nodes: List[str],
    neuropal_data: Dict,
) -> Dict[str, Dict[int, int]]:
    """
    For each neuron, find which worms are missing it and identify donor worms.
    
    The donor selection strategy is:
    1. For each neuron N
    2. Find all worms that have N (donors) and all worms missing N
    3. For each missing worm, assign a donor (cycling through available donors)
    
    Args:
        selected_nodes: List of neurons we want to include
        neuropal_data: NeuroPAL recording data
        
    Returns:
        donor_map: {neuron_name: {missing_worm_idx: donor_worm_idx}}
    """
    name_to_indices = neuropal_data["name_to_indices"]
    num_worms = len(neuropal_data["worm_ids"])
    
    donor_map = {}
    
    for node in selected_nodes:
        if node not in name_to_indices:
            continue
            
        indices = name_to_indices[node]
        
        # Find worms that have this neuron
        worms_with_neuron = []
        worms_missing_neuron = []
        
        for worm_idx in range(num_worms):
            trace = collect_worm_trace(
                neuropal_data["norm_traces"],
                indices,
                worm_idx,
                num_worms
            )
            if trace is not None:
                worms_with_neuron.append(worm_idx)
            else:
                worms_missing_neuron.append(worm_idx)
        
        # Create donor assignments
        if worms_missing_neuron and worms_with_neuron:
            node_donor_map = {}
            for i, missing_worm in enumerate(worms_missing_neuron):
                # Cycle through donors
                donor = worms_with_neuron[i % len(worms_with_neuron)]
                node_donor_map[missing_worm] = donor
            donor_map[node] = node_donor_map
    
    return donor_map


def collect_worm_trace_with_imputation(
    all_traces: List[np.ndarray],
    neuron_indices: List[int],
    worm_idx: int,
    num_worms: int,
    donor_map: Dict[str, Dict[int, int]],
    node_name: str,
    neuropal_data: Dict,
) -> Tuple[np.ndarray | None, bool]:
    """
    Collect trace for a neuron, imputing from donor if missing.
    
    Args:
        all_traces: All trace data
        neuron_indices: Indices for this neuron class
        worm_idx: Target worm
        num_worms: Total worms
        donor_map: Donor mapping
        node_name: Name of this neuron
        neuropal_data: Full data dict (for donor lookups)
        
    Returns:
        (trace, was_imputed) tuple
    """
    # First try to get native trace
    trace = collect_worm_trace(all_traces, neuron_indices, worm_idx, num_worms)
    
    if trace is not None:
        return trace, False
    
    # Need to impute - check donor map
    if node_name in donor_map and worm_idx in donor_map[node_name]:
        donor_worm = donor_map[node_name][worm_idx]
        donor_trace = collect_worm_trace(
            all_traces, neuron_indices, donor_worm, num_worms
        )
        if donor_trace is not None:
            return donor_trace, True
    
    return None, False


def build_full_trace_dataset_with_imputation(
    selected_nodes: List[str],
    neuropal_data: Dict,
    name_to_indices: Dict[str, List[int]],
    output_dir: Path,
) -> None:
    """
    Build full-trace dataset using ALL worms via donor imputation.
    
    For worms missing certain neurons, we copy traces from donor worms that
    have those neurons. This allows us to use all 21 worms instead of just
    the ones with complete coverage.
    
    Args:
        selected_nodes: List of neuron names to include
        neuropal_data: NeuroPAL recording data
        name_to_indices: Mapping from neuron names to trace indices
        output_dir: Where to save dataset
    """
    print(f"\n  Building IMPUTED FULL TRACE dataset...")
    print(f"    Strategy: Donor-based imputation for missing neurons")
    
    # Create output directory
    stim_dir = output_dir / "full_traces_imputed"
    stim_dir.mkdir(parents=True, exist_ok=True)
    
    num_worms = len(neuropal_data["worm_ids"])
    fps = neuropal_data["fps"]
    
    # Build donor map
    print(f"    Building donor map...")
    donor_map = build_donor_map(selected_nodes, neuropal_data)
    
    n_neurons_need_imputation = len(donor_map)
    total_imputations = sum(len(v) for v in donor_map.values())
    print(f"    {n_neurons_need_imputation} neurons need imputation")
    print(f"    {total_imputations} total worm×neuron pairs will be imputed")
    
    # Collect data from ALL worms
    lag_windows = []
    segment_info = []
    time_series = []
    imputation_log = []
    
    node_indices_list = [name_to_indices[name] for name in selected_nodes]
    min_trace_len = 900  # Require at least 900 frames (~225s)
    
    for worm_idx in range(num_worms):
        # Build neuron × time matrix for this worm
        columns = []
        valid = True
        worm_imputations = 0
        
        for node_name, indices in zip(selected_nodes, node_indices_list):
            trace, was_imputed = collect_worm_trace_with_imputation(
                neuropal_data["norm_traces"],
                indices,
                worm_idx,
                num_worms,
                donor_map,
                node_name,
                neuropal_data,
            )
            
            if trace is None or len(trace) < min_trace_len:
                valid = False
                break
            
            columns.append(trace)
            
            if was_imputed:
                worm_imputations += 1
                donor = donor_map[node_name][worm_idx]
                imputation_log.append({
                    "worm_idx": worm_idx,
                    "worm_id": neuropal_data["worm_ids"][worm_idx],
                    "neuron": node_name,
                    "donor_worm_idx": donor,
                    "donor_worm_id": neuropal_data["worm_ids"][donor],
                })
        
        if not valid:
            print(f"    Warning: Worm {worm_idx} still missing data after imputation, skipping")
            continue
        
        # Truncate to common length
        min_len = min(col.shape[-1] for col in columns)
        matrix = np.stack([col[:min_len] for col in columns], axis=1)
        
        if matrix.shape[0] < min_trace_len:
            print(f"    Warning: Trace too short for worm {worm_idx} ({matrix.shape[0]} frames), skipping")
            continue
        
        # Create lag-1 windows from FULL trace
        x_t = matrix[:-1, :]
        x_tp1 = matrix[1:, :]
        windows = np.concatenate([x_t, x_tp1], axis=1)
        
        lag_windows.append(windows)
        time_series.append(matrix)
        
        segment_info.append({
            "worm_index": worm_idx,
            "worm_id": neuropal_data["worm_ids"][worm_idx],
            "start_frame": 0,
            "end_frame": matrix.shape[0],
            "frames_used": matrix.shape[0],
            "windows_created": windows.shape[0],
            "duration_seconds": matrix.shape[0] / fps,
            "imputed_neurons": worm_imputations,
        })
    
    worms_contributed = len(segment_info)
    
    if not lag_windows:
        print(f"    No data collected for imputed full traces. Skipping.")
        return
    
    # Stack all windows
    Z_raw = np.vstack(lag_windows)
    
    # Standardize (zero mean, unit variance)
    mean = np.nanmean(Z_raw, axis=0)
    std = np.nanstd(Z_raw, axis=0)
    std = np.maximum(std, 1e-6)
    Z_std = (Z_raw - mean) / std
    Z_std = np.nan_to_num(Z_std, nan=0.0)
    
    # Save outputs
    np.save(stim_dir / "Z_raw.npy", Z_raw)
    np.save(stim_dir / "Z_std.npy", Z_std)
    np.save(stim_dir / "X_segments.npy", np.array(time_series, dtype=object))
    
    # Save neuron names for downstream scripts
    with open(stim_dir / "neuron_names.json", "w") as f:
        json.dump(selected_nodes, f, indent=2)
    
    pd.DataFrame(segment_info).to_csv(stim_dir / "segments.csv", index=False)
    
    # Save imputation log
    if imputation_log:
        pd.DataFrame(imputation_log).to_csv(stim_dir / "imputation_log.csv", index=False)
    
    # Save metadata
    total_duration = sum(s["duration_seconds"] for s in segment_info)
    metadata = {
        "stimulus": "full_traces_imputed",
        "description": "Full 240s recordings with donor-based imputation",
        "imputation_enabled": True,
        "neurons_with_imputation": n_neurons_need_imputation,
        "total_imputations": total_imputations,
        "min_worms": MIN_WORMS,
        "node_order": selected_nodes,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window_count": int(Z_raw.shape[0]),
        "window_dim": int(Z_raw.shape[1]),
        "segment_count": len(segment_info),
        "worms_contributed": worms_contributed,
        "worms_total": num_worms,
        "total_duration_seconds": total_duration,
        "total_duration_minutes": total_duration / 60,
    }
    
    with open(stim_dir / "standardization.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    ✓ Created {Z_raw.shape[0]} windows from {worms_contributed}/{num_worms} worms")
    print(f"      Total duration: {total_duration/60:.1f} minutes")
    print(f"      Imputed {total_imputations} worm×neuron pairs")


def build_stimulus_dataset(
    stimulus_name: str,
    stim_idx: int,
    selected_nodes: List[str],
    eligible_worms: List[int],
    neuropal_data: Dict,
    name_to_idx: Dict[str, int],
    output_dir: Path,
) -> None:
    """
    Build lag-window dataset for a specific stimulus.
    
    Creates standardized lag-1 windows: z_t = [x_t, x_{t+1}]
    
    Args:
        stimulus_name: Name of stimulus (e.g., "NaCl")
        stim_idx: Index of stimulus in stim_times array
        selected_nodes: List of neuron names to include
        eligible_worms: List of worm indices to use
        neuropal_data: NeuroPAL recording data
        name_to_idx: Mapping from neuron names to trace indices
        output_dir: Where to save dataset
    """
    print(f"\n  Processing '{stimulus_name}'...")
    
    # Create output directory
    stim_dir = output_dir / stimulus_name.lower().replace(" ", "_")
    stim_dir.mkdir(parents=True, exist_ok=True)
    
    # name_to_idx contains the name -> indices mapping
    name_to_indices = name_to_idx
    node_indices_list = [name_to_indices[name] for name in selected_nodes]
    num_worms = len(neuropal_data["worm_ids"])
    fps = neuropal_data["fps"]
    
    # Collect data from each worm
    lag_windows = []
    segment_info = []
    time_series = []
    
    worms_attempted = 0
    worms_contributed = 0
    
    for worm_idx in eligible_worms:
        worms_attempted += 1
        
        # Get stimulus order for this worm
        order = neuropal_data["stims_per_worm"][worm_idx]
        
        # Find when this worm received this stimulus
        try:
            event_pos = int(np.where(order == (stim_idx + 1))[0][0])
        except IndexError:
            # Worm did not receive this stimulus
            continue
        
        # Get time window for this stimulus
        start_sec, end_sec = neuropal_data["stim_times"][event_pos]
        start_frame = int(round(start_sec * fps))
        end_frame = int(round(end_sec * fps))
        
        # Build neuron × time matrix for this worm
        columns = []
        for indices in node_indices_list:
            trace = collect_worm_trace(
                neuropal_data["norm_traces"],
                indices,
                worm_idx,
                num_worms
            )
            if trace is None:
                raise RuntimeError(
                    f"Missing trace for node in worm {worm_idx} despite eligibility check"
                )
            columns.append(trace)
        
        # Truncate to common length
        min_len = min(col.shape[-1] for col in columns)
        matrix = np.stack([col[-min_len:] for col in columns], axis=1)
        
        # Extract stimulus segment
        segment = matrix[max(0, start_frame):min(end_frame, matrix.shape[0]), :]
        
        if segment.shape[0] < 2:
            print(f"    Warning: Segment too short for worm {worm_idx}, skipping")
            continue
        
        # Create lag-1 windows
        x_t = segment[:-1, :]
        x_tp1 = segment[1:, :]
        windows = np.concatenate([x_t, x_tp1], axis=1)
        
        lag_windows.append(windows)
        time_series.append(segment)
        
        segment_info.append({
            "worm_index": worm_idx,
            "worm_id": neuropal_data["worm_ids"][worm_idx],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames_used": segment.shape[0],
            "windows_created": windows.shape[0],
        })
        
        worms_contributed += 1
    
    if not lag_windows:
        print(f"    No data collected for {stimulus_name}. Skipping.")
        print(f"    ({worms_contributed}/{worms_attempted} worms contributed)")
        return
    
    # Stack all windows
    Z_raw = np.vstack(lag_windows)
    
    # Standardize (zero mean, unit variance)
    mean = Z_raw.mean(axis=0)
    std = Z_raw.std(axis=0)
    std = np.maximum(std, 1e-6)  # Avoid division by zero
    Z_std = (Z_raw - mean) / std
    
    # Save outputs
    np.save(stim_dir / "Z_raw.npy", Z_raw)
    np.save(stim_dir / "Z_std.npy", Z_std)
    np.save(stim_dir / "X_segments.npy", np.array(time_series, dtype=object))
    
    pd.DataFrame(segment_info).to_csv(stim_dir / "segments.csv", index=False)
    
    # Save standardization metadata
    metadata = {
        "stimulus": stimulus_name,
        "stimulus_index": stim_idx + 1,
        "min_worms": MIN_WORMS,
        "node_order": selected_nodes,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window_count": int(Z_raw.shape[0]),
        "window_dim": int(Z_raw.shape[1]),
        "segment_count": len(segment_info),
        "worms_contributed": worms_contributed,
        "worms_attempted": worms_attempted,
    }
    
    with open(stim_dir / "standardization.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    ✓ Created {Z_raw.shape[0]} windows from {worms_contributed} worms")
    print(f"      ({worms_attempted - worms_contributed} worms skipped - "
          f"did not receive this stimulus)")


def build_full_trace_dataset(
    selected_nodes: List[str],
    eligible_worms: List[int],
    neuropal_data: Dict,
    name_to_idx: Dict[str, int],
    output_dir: Path,
) -> None:
    """
    Build lag-window dataset using FULL traces (240s) instead of stimulus windows.
    
    This creates a single unified dataset from all worms' complete recordings,
    providing ~38x more training data than stimulus-window datasets.
    
    Args:
        selected_nodes: List of neuron names to include
        eligible_worms: List of worm indices to use
        neuropal_data: NeuroPAL recording data
        name_to_idx: Mapping from neuron names to trace indices
        output_dir: Where to save dataset
    """
    print(f"\n  Building FULL TRACE dataset...")
    
    # Create output directory
    stim_dir = output_dir / "full_traces"
    stim_dir.mkdir(parents=True, exist_ok=True)
    
    # name_to_idx contains the name -> indices mapping
    name_to_indices = name_to_idx
    node_indices_list = [name_to_indices[name] for name in selected_nodes]
    num_worms = len(neuropal_data["worm_ids"])
    fps = neuropal_data["fps"]
    
    # Collect data from each worm
    lag_windows = []
    segment_info = []
    time_series = []
    
    worms_contributed = 0
    min_trace_len = 900  # Require at least 900 frames (~225s)
    
    for worm_idx in eligible_worms:
        # Build neuron × time matrix for this worm
        columns = []
        valid = True
        
        for indices in node_indices_list:
            trace = collect_worm_trace(
                neuropal_data["norm_traces"],
                indices,
                worm_idx,
                num_worms
            )
            if trace is None or len(trace) < min_trace_len:
                valid = False
                break
            columns.append(trace)
        
        if not valid:
            continue
        
        # Truncate to common length
        min_len = min(col.shape[-1] for col in columns)
        matrix = np.stack([col[:min_len] for col in columns], axis=1)
        
        if matrix.shape[0] < min_trace_len:
            print(f"    Warning: Trace too short for worm {worm_idx} ({matrix.shape[0]} frames), skipping")
            continue
        
        # Create lag-1 windows from FULL trace
        x_t = matrix[:-1, :]
        x_tp1 = matrix[1:, :]
        windows = np.concatenate([x_t, x_tp1], axis=1)
        
        lag_windows.append(windows)
        time_series.append(matrix)
        
        segment_info.append({
            "worm_index": worm_idx,
            "worm_id": neuropal_data["worm_ids"][worm_idx],
            "start_frame": 0,
            "end_frame": matrix.shape[0],
            "frames_used": matrix.shape[0],
            "windows_created": windows.shape[0],
            "duration_seconds": matrix.shape[0] / fps,
        })
        
        worms_contributed += 1
    
    if not lag_windows:
        print(f"    No data collected for full traces. Skipping.")
        return
    
    # Stack all windows
    Z_raw = np.vstack(lag_windows)
    
    # Standardize (zero mean, unit variance) handling NaNs
    mean = np.nanmean(Z_raw, axis=0)
    std = np.nanstd(Z_raw, axis=0)
    std = np.maximum(std, 1e-6)  # Avoid division by zero
    Z_std = (Z_raw - mean) / std
    Z_std = np.nan_to_num(Z_std, nan=0.0)
    
    # Save outputs
    np.save(stim_dir / "Z_raw.npy", Z_raw)
    np.save(stim_dir / "Z_std.npy", Z_std)
    np.save(stim_dir / "X_segments.npy", np.array(time_series, dtype=object))
    
    pd.DataFrame(segment_info).to_csv(stim_dir / "segments.csv", index=False)
    
    # Save standardization metadata
    total_duration = sum(s["duration_seconds"] for s in segment_info)
    metadata = {
        "stimulus": "full_traces",
        "description": "Full 240s recordings (not stimulus-specific)",
        "min_worms": MIN_WORMS,
        "node_order": selected_nodes,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window_count": int(Z_raw.shape[0]),
        "window_dim": int(Z_raw.shape[1]),
        "segment_count": len(segment_info),
        "worms_contributed": worms_contributed,
        "total_duration_seconds": total_duration,
        "total_duration_minutes": total_duration / 60,
    }
    
    with open(stim_dir / "standardization.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"    ✓ Created {Z_raw.shape[0]} windows from {worms_contributed} worms")
    print(f"      Total duration: {total_duration/60:.1f} minutes")
    print(f"      (Compare to ~5 min total with stimulus windows)")


def build_datasets(
    nodes: List[str],
    data_dir: Path,
    output_dir: Path,
    min_worms: int = MIN_WORMS,
    stimuli_filter: List[str] | None = None,
    use_full_traces: bool = False,
    include_tail: bool = INCLUDE_TAIL,
    collapse_dv: bool = COLLAPSE_DV,
    impute_missing: bool = False,
) -> None:
    """
    Build SBTG datasets for all stimuli.
    
    Args:
        nodes: List of neuron names from connectome
        data_dir: Path to data directory
        output_dir: Where to save datasets
        min_worms: Minimum worm coverage threshold
        stimuli_filter: Optional list of stimuli to process (None = all)
        use_full_traces: If True, also build dataset using full 240s traces
        include_tail: If True, include tail neurons from Tail_Activity file
        collapse_dv: If True, collapse D/V subtypes (RMDD/RMDV -> RMD)
        impute_missing: If True, impute missing neurons from donor worms
    """
    print("\n" + "=" * 70)
    print("STEP 2: Building SBTG Datasets")
    print("=" * 70)
    
    # Show configuration
    if include_tail:
        print("  Including tail neurons")
    if collapse_dv:
        print("  D/V subtype collapsing enabled")
    
    # Load NeuroPAL data with optional tail merge and D/V collapsing
    neuropal_data = load_neuropal_data(
        data_dir, 
        include_tail=include_tail, 
        collapse_dv=collapse_dv
    )
    
    print(f"\nNeuroPAL summary:")
    print(f"  Neurons: {len(neuropal_data['neuron_names'])}")
    print(f"  Worms: {len(neuropal_data['worm_ids'])}")
    print(f"  Stimuli: {', '.join(neuropal_data['stim_names'])}")
    print(f"  Sampling rate: {neuropal_data['fps']:.1f} Hz")
    
    # Compute node coverage
    print(f"\nComputing neuron coverage across worms...")
    coverage_df = compute_node_coverage(nodes, neuropal_data)
    
    # Select nodes and worms
    print(f"\nApplying coverage threshold (min_worms={min_worms})...")
    selected_nodes, eligible_worms, name_to_idx = select_nodes_and_worms(
        coverage_df,
        neuropal_data,
        min_worms
    )
    
    # Determine which stimuli to process
    if stimuli_filter is not None:
        stim_names = [
            s for s in neuropal_data["stim_names"]
            if s.lower() in [f.lower() for f in stimuli_filter]
        ]
    else:
        stim_names = neuropal_data["stim_names"]
    
    print(f"\nBuilding datasets for {len(stim_names)} stimuli:")
    
    # Build dataset for each stimulus
    for stim_idx, stim_name in enumerate(neuropal_data["stim_names"]):
        if stim_name not in stim_names:
            continue
        
        build_stimulus_dataset(
            stim_name,
            stim_idx,
            selected_nodes,
            eligible_worms,
            neuropal_data,
            name_to_idx,
            output_dir
        )
    
    # Optionally build full-trace dataset
    if use_full_traces:
        build_full_trace_dataset(
            selected_nodes,
            eligible_worms,
            neuropal_data,
            name_to_idx,
            output_dir
        )
    
    # Optionally build imputed full-trace dataset (uses ALL worms)
    if impute_missing:
        build_full_trace_dataset_with_imputation(
            selected_nodes,
            neuropal_data,
            name_to_idx,
            output_dir
        )
    
    print(f"\n✓ Dataset preparation complete")
    print(f"  Datasets saved to: {output_dir}/")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def reassemble_large_files(data_dir: Path):
    """
    Reassemble large files split for git storage.
    Checks for .part_* files and concatenates them into the original file if missing.
    """
    target_file = data_dir / NEUROPAL_HEAD_FILE
    
    # If file exists and is large enough (>80MB), assume valid
    if target_file.exists() and target_file.stat().st_size > 80_000_000:
        return

    # Look for parts
    parts = sorted(list(data_dir.glob(f"{NEUROPAL_HEAD_FILE}.part_*")))
    
    if not parts:
        return
        
    print(f"Reassembling {NEUROPAL_HEAD_FILE} from {len(parts)} parts...")
    
    with open(target_file, 'wb') as outfile:
        for part in parts:
            print(f"  Appending {part.name}...")
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
    
    print(f"✓ Reassembled {target_file.name} ({target_file.stat().st_size / 1e6:.1f} MB)")

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("PREPARE DATA FOR SBTG ANALYSIS")
    print("=" * 70)
    print("\nThis script prepares all data needed for SBTG analysis:")
    print("  1. Aligns Cook connectome with NeuroPAL neuron names")
    print("  2. Builds standardized lag-window datasets for each stimulus")
    print("  3. Generates quality control visualizations")
    print("\n" + "=" * 70)
    
    # Locate project root
    try:
        project_root = locate_project_root(Path.cwd())
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run this script from within the project directory:")
        print("  python pipeline/01_prepare_data.py")
        sys.exit(1)
    
    print(f"\nProject root: {project_root}")
    
    # Reassemble large data files if needed (for git checkouts)
    try:
        reassemble_large_files(project_root / "data")
    except Exception as e:
        print(f"Warning: Failed to reassemble large data: {e}")
    
    # Set up paths
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    
    # Create output directories
    output_dirs = ensure_directories(results_dir)
    
    try:
        # Step 1: Align connectome
        nodes, adjacencies = align_connectome(data_dir, output_dirs)
        
        # Step 2: Build datasets
        build_datasets(
            nodes,
            data_dir,
            output_dirs["datasets"],
            min_worms=MIN_WORMS,
            stimuli_filter=STIMULI,
            use_full_traces=USE_FULL_TRACES,
            include_tail=INCLUDE_TAIL,
            collapse_dv=COLLAPSE_DV,
            impute_missing=IMPUTE_MISSING,
        )
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print("\nData preparation complete. Next steps:")
        print("  1. Review figures in results/figures/connectome/")
        print("  2. Check dataset summaries in results/intermediate/datasets/")
        print("  3. Run: python pipeline/02_train_sbtg.py")
        print("\n" + "=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"\n{type(e).__name__}: {e}")
        print("\nPlease check:")
        print("  - Data files are in data/ directory")
        print("  - File names match expected names")
        print("  - MIN_WORMS threshold is not too high")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare data for SBTG analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Expansion Options:
  By default, this script now includes tail neurons and collapses D/V subtypes
  to maximize neuron coverage. Use --no-tail and --no-collapse-dv to disable.

  NOTE: OH15500 (7 additional worms) is NOT yet integrated. Future work.
"""
    )
    parser.add_argument(
        "--full-traces", 
        action="store_true",
        help="Also build dataset using full 240s traces (38x more data)"
    )
    parser.add_argument(
        "--no-tail",
        action="store_true",
        help="Disable tail neuron integration (reduces neurons by ~17)"
    )
    parser.add_argument(
        "--no-collapse-dv",
        action="store_true",
        help="Disable D/V subtype collapsing (reduces neurons by ~20)"
    )
    parser.add_argument(
        "--min-worms",
        type=int,
        default=15,
        help="Minimum worm coverage threshold (default: 15)"
    )
    parser.add_argument(
        "--impute-missing",
        action="store_true",
        help="Impute missing neurons using donor worms (enables ALL worms for training)"
    )
    args = parser.parse_args()
    
    # Update global config based on args
    if args.full_traces:
        USE_FULL_TRACES = True
        print("=" * 70)
        print("FULL TRACE MODE ENABLED")
        print("=" * 70)
        print("Will build additional dataset using complete 240s recordings")
        print("This provides ~38x more training data than stimulus windows")
        print("=" * 70)
    
    if args.no_tail:
        INCLUDE_TAIL = False
        print("Tail neuron integration DISABLED")
    
    if args.no_collapse_dv:
        COLLAPSE_DV = False
        print("D/V subtype collapsing DISABLED")
    
    MIN_WORMS = args.min_worms
    if args.min_worms != 15:
        print(f"MIN_WORMS threshold set to: {args.min_worms}")
    
    if args.impute_missing:
        IMPUTE_MISSING = True
        print("=" * 70)
        print("IMPUTATION MODE ENABLED")
        print("=" * 70)
        print("Missing neurons will be imputed from donor worms")
        print("This enables using ALL worms for training")
        print("=" * 70)
    
    # Set plotting style
    sns.set_context("talk")
    plt.style.use("seaborn-v0_8-colorblind")
    
    main()
