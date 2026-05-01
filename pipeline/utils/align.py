"""
Node alignment and direction convention enforcement.

DIRECTION CONVENTION:
====================
All adjacency matrices in this pipeline use A[post, pre] = A[target, source].

This means:
- A[i, j] > 0 implies j → i (neuron j is presynaptic to neuron i)
- Row i contains all inputs TO neuron i
- Column j contains all outputs FROM neuron j

For degree computations:
- in_degree[i] = sum(A[i, :])  = # of neurons projecting TO i (row sum)
- out_degree[j] = sum(A[:, j]) = # of neurons receiving FROM j (column sum)

NODE ORDER CONTRACT:
===================
Every saved matrix MUST include a `node_order` list.
Every load MUST verify node_order matches before comparing matrices.
"""

import re
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


# Direction convention: A[post, pre] = A[target, source]
# j → i is represented by A[i, j] > 0
DIRECTION_CONVENTION = "post_pre"


def normalize_neuron_name(name: str) -> str:
    """
    Normalize neuron class names to a canonical form.
    
    Rules:
    1. Strip whitespace
    2. Convert to uppercase
    3. Remove common suffixes (L/R handled separately by bilateral merge)
    
    Args:
        name: Raw neuron name
        
    Returns:
        Normalized name (e.g., "  ase  " -> "ASE", "AVAL" -> "AVAL")
    """
    return name.strip().upper()


def merge_bilateral_name(name: str) -> str:
    """
    Merge bilateral neuron names (L/R pairs) to a single class.
    
    Args:
        name: Normalized neuron name (e.g., "AVAL", "AVAR")
        
    Returns:
        Class name without L/R suffix (e.g., "AVA")
    """
    name = normalize_neuron_name(name)
    
    # Singletons that end in L or R but are NOT bilateral pairs
    # These should NOT have their suffix stripped.
    SINGLETONS_ENDING_IN_LR = {'PVR', 'AVL', 'PQR', 'DVA'}
    
    if name in SINGLETONS_ENDING_IN_LR:
        return name
        
    # Common bilateral patterns: AVAL/AVAR, RIML/RIMR, etc.
    # Remove trailing L or R if preceded by a letter/number
    if len(name) > 1 and name[-1] in ('L', 'R'):
        return name[:-1]
    
    return name


# D/V subtype patterns: suffix D or V that should be collapsed
# For example: RMDD/RMDV -> RMD, SAAD/SAAV -> SAA
# These are documented in WormAtlas and Cook et al. 2019
DV_COLLAPSE_PATTERNS = {
    # RMD subtypes
    'RMDD': 'RMD', 'RMDV': 'RMD',
    # SAA subtypes
    'SAAD': 'SAA', 'SAAV': 'SAA',
    # SAB subtypes
    'SABD': 'SAB', 'SABV': 'SAB',
    # SIA subtypes
    'SIAD': 'SIA', 'SIAV': 'SIA',
    # SIB subtypes
    'SIBD': 'SIB', 'SIBV': 'SIB',
    # SMB subtypes
    'SMBD': 'SMB', 'SMBV': 'SMB',
    # SMD subtypes
    'SMDD': 'SMD', 'SMDV': 'SMD',
    # URA subtypes
    'URAD': 'URA', 'URAV': 'URA',
    # URY subtypes
    'URYD': 'URY', 'URYV': 'URY',
    # CEP subtypes
    'CEPD': 'CEP', 'CEPV': 'CEP',
    # OLQ subtypes
    'OLQD': 'OLQ', 'OLQV': 'OLQ',
    # IL1 subtypes
    'IL1D': 'IL1', 'IL1V': 'IL1',
    # IL2 subtypes
    'IL2D': 'IL2', 'IL2V': 'IL2',
    # RME subtypes (RMED/RMEV -> RME) - note: RME already in Cook
    'RMED': 'RME', 'RMEV': 'RME',
}


def collapse_dv_subtypes(name: str) -> str:
    """
    Collapse D/V subtypes to their parent cell class for Cook alignment.
    
    NeuroPAL distinguishes dorsal (D) and ventral (V) variants of many neurons,
    but the Cook connectome uses pooled cell classes. This function maps
    D/V-specific names to their parent class.
    
    Examples:
        RMDD, RMDV -> RMD
        SAAD, SAAV -> SAA
        CEPD, CEPV -> CEP
    
    Args:
        name: Normalized neuron name (uppercase, stripped)
        
    Returns:
        Parent cell class name if a D/V variant, otherwise unchanged
    """
    name = normalize_neuron_name(name)
    return DV_COLLAPSE_PATTERNS.get(name, name)


def collapse_all_dv_subtypes(names: list) -> list:
    """
    Apply D/V collapsing to a list of neuron names.
    
    Args:
        names: List of neuron names
        
    Returns:
        List with D/V variants collapsed to parent classes
    """
    return [collapse_dv_subtypes(n) for n in names]


def find_common_neurons(
    list1: List[str],
    list2: List[str],
    normalize: bool = True,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Find neurons common to both lists and return aligned indices.
    
    Args:
        list1: First list of neuron names
        list2: Second list of neuron names
        normalize: If True, normalize names before comparison
        
    Returns:
        Tuple of:
        - common_neurons: Sorted list of common neuron names
        - indices1: Indices in list1 for each common neuron
        - indices2: Indices in list2 for each common neuron
    """
    if normalize:
        norm1 = [normalize_neuron_name(n) for n in list1]
        norm2 = [normalize_neuron_name(n) for n in list2]
    else:
        norm1 = list(list1)
        norm2 = list(list2)
    
    # Create mappings
    name_to_idx1 = {n: i for i, n in enumerate(norm1)}
    name_to_idx2 = {n: i for i, n in enumerate(norm2)}
    
    # Find common names (use normalized names for comparison)
    common_set = set(norm1) & set(norm2)
    common_neurons = sorted(list(common_set))
    
    # Get indices
    indices1 = [name_to_idx1[n] for n in common_neurons]
    indices2 = [name_to_idx2[n] for n in common_neurons]
    
    return common_neurons, indices1, indices2


def align_matrices(
    A: np.ndarray,
    source_neurons: List[str],
    target_neurons: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Align a matrix from source neuron order to target neuron order.
    
    Args:
        A: Matrix to align (shape: n_source × n_source)
        source_neurons: Neuron names in source order
        target_neurons: Neuron names in target order
        
    Returns:
        Tuple of:
        - A_aligned: Aligned matrix (shape: n_common × n_common)
        - common_neurons: Names of neurons in aligned matrix
    """
    common_neurons, src_indices, tgt_indices = find_common_neurons(
        source_neurons, target_neurons
    )
    
    if len(common_neurons) == 0:
        raise ValueError("No common neurons found between source and target")
    
    # Extract aligned submatrix
    A_aligned = A[np.ix_(src_indices, src_indices)]
    
    return A_aligned, common_neurons


def align_timeseries_to_connectome(
    X_list: List[np.ndarray],
    func_neurons: List[str],
    struct_neurons: List[str],
) -> Tuple[List[np.ndarray], np.ndarray, List[str], List[int], List[int]]:
    """
    Align functional timeseries data to structural connectome neurons.
    
    This is the canonical function for aligning calcium traces to the 
    Cook connectome. All scripts should use this instead of implementing
    their own alignment.
    
    Args:
        X_list: List of (T_w, n_neurons) timeseries arrays per worm
                Can also be a single (T, n_neurons) array
        func_neurons: Neuron names from functional recording
        struct_neurons: Neuron names from structural connectome
        
    Returns:
        Tuple of:
        - X_aligned: List of aligned (T_w, n_aligned) arrays
        - common_neurons: Sorted list of aligned neuron names
        - func_idx: Indices into func_neurons for aligned neurons
        - struct_idx: Indices into struct_neurons for aligned neurons
    """
    # Find common neurons
    common_neurons, func_idx, struct_idx = find_common_neurons(
        func_neurons, struct_neurons
    )
    
    if len(common_neurons) == 0:
        raise ValueError("No common neurons found between functional and structural data")
    
    # Handle both single array and list of arrays
    if isinstance(X_list, np.ndarray):
        if X_list.ndim == 2:
            # Single (T, n) array
            X_aligned = [X_list[:, func_idx]]
        elif X_list.ndim == 3:
            # (n_worms, T, n) array
            # NOTE: Using X[w][:, func_idx] instead of X[w, :, func_idx] to avoid transpose
            # due to numpy advanced indexing behavior with list indices
            X_aligned = [X_list[w][:, func_idx] for w in range(X_list.shape[0])]
        else:
            raise ValueError(f"Unexpected X_list shape: {X_list.shape}")
    else:
        # List of per-worm arrays
        X_aligned = [X[:, func_idx] for X in X_list]
    
    print(f"  Aligned {len(common_neurons)} neurons to structural connectome")
    
    return X_aligned, common_neurons, func_idx, struct_idx


def validate_node_order(
    node_order1: List[str],
    node_order2: List[str],
    context: str = "",
) -> bool:
    """
    Validate that two node orders match exactly.
    
    Args:
        node_order1: First node order
        node_order2: Second node order
        context: Description of comparison for error messages
        
    Returns:
        True if orders match
        
    Raises:
        ValueError: If orders don't match
    """
    if len(node_order1) != len(node_order2):
        raise ValueError(
            f"Node order length mismatch{' (' + context + ')' if context else ''}: "
            f"{len(node_order1)} vs {len(node_order2)}"
        )
    
    for i, (n1, n2) in enumerate(zip(node_order1, node_order2)):
        if normalize_neuron_name(n1) != normalize_neuron_name(n2):
            raise ValueError(
                f"Node order mismatch at index {i}{' (' + context + ')' if context else ''}: "
                f"'{n1}' vs '{n2}'"
            )
    
    return True


def assert_direction_convention(
    A_pred: np.ndarray,
    A_truth: np.ndarray,
    node_order: List[str],
    check_transpose: bool = True,
) -> Dict[str, Any]:
    """
    Assert that predicted and truth matrices follow the same direction convention.
    
    This function checks if A_pred and A_truth have the same orientation,
    or if one needs to be transposed. It does this by comparing overlap
    metrics in both orientations.
    
    Args:
        A_pred: Predicted adjacency matrix
        A_truth: Ground truth adjacency matrix
        node_order: Node order for both matrices
        check_transpose: If True, also check if transpose has better overlap
        
    Returns:
        Dict with:
        - convention: "consistent" or "needs_transpose"
        - overlap_original: Overlap count with original orientation
        - overlap_transposed: Overlap count with transposed orientation
        - recommendation: Human-readable recommendation
    """
    n = len(node_order)
    
    if A_pred.shape != (n, n) or A_truth.shape != (n, n):
        raise ValueError(f"Matrix shapes don't match node_order length {n}")
    
    # Binarize for comparison
    pred_binary = (A_pred != 0).astype(int)
    truth_binary = (A_truth != 0).astype(int)
    
    # Check overlap in original orientation
    overlap_original = np.sum(pred_binary & truth_binary)
    
    # Check overlap with transposed prediction
    overlap_transposed = np.sum(pred_binary.T & truth_binary)
    
    result = {
        'overlap_original': int(overlap_original),
        'overlap_transposed': int(overlap_transposed),
        'n_pred_edges': int(np.sum(pred_binary)),
        'n_truth_edges': int(np.sum(truth_binary)),
    }
    
    if overlap_transposed > overlap_original * 1.5:  # Significant improvement with transpose
        result['convention'] = 'needs_transpose'
        result['recommendation'] = (
            f"WARNING: Transposed orientation has {overlap_transposed} overlapping edges "
            f"vs {overlap_original} original. Consider transposing predicted matrix."
        )
    else:
        result['convention'] = 'consistent'
        result['recommendation'] = (
            f"Direction convention appears consistent: {overlap_original} overlapping edges."
        )
    
    return result


def save_matrix_with_node_order(
    A: np.ndarray,
    node_order: List[str],
    output_path: Path,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save a matrix with its node order (enforcing the node order contract).
    
    Args:
        A: Matrix to save
        node_order: Node order for the matrix
        output_path: Path to save (will create .npz and .json files)
        metadata: Optional additional metadata to save
    """
    output_path = Path(output_path)
    
    # Save matrix as .npz
    np.savez(
        output_path.with_suffix('.npz'),
        matrix=A,
        node_order=np.array(node_order, dtype=object),
    )
    
    # Save node order as JSON for easy inspection
    meta = {
        'node_order': node_order,
        'shape': list(A.shape),
        'n_edges': int(np.sum(A != 0)),
        'direction_convention': DIRECTION_CONVENTION,
    }
    if metadata:
        meta.update(metadata)
    
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def load_matrix_with_node_order(
    input_path: Path,
    expected_node_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load a matrix and verify its node order (enforcing the node order contract).
    
    Args:
        input_path: Path to load from (expects .npz file)
        expected_node_order: If provided, validate that loaded order matches
        
    Returns:
        Tuple of (matrix, node_order)
        
    Raises:
        ValueError: If expected_node_order provided and doesn't match
    """
    input_path = Path(input_path)
    
    data = np.load(input_path.with_suffix('.npz'), allow_pickle=True)
    
    A = data['matrix']
    node_order = list(data['node_order'])
    
    if expected_node_order is not None:
        validate_node_order(node_order, expected_node_order, context=str(input_path))
    
    return A, node_order


def compute_in_degree(A: np.ndarray) -> np.ndarray:
    """
    Compute in-degree for each neuron.
    
    In our convention A[post, pre]: in_degree = row sum.
    """
    return np.sum(A != 0, axis=1)


def compute_out_degree(A: np.ndarray) -> np.ndarray:
    """
    Compute out-degree for each neuron.
    
    In our convention A[post, pre]: out_degree = column sum.
    """
    return np.sum(A != 0, axis=0)

