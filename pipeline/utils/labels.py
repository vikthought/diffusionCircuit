"""
Leifer atlas labeling - single source of truth.

LABELING POLICY:
===============
When evaluating against the Leifer/Randi functional atlas:

1. POSITIVES: Edges with q < alpha (significant functional connection)
   - These are edges where optogenetic stimulation produced significant signal propagation
   
2. CONFIRMED NEGATIVES: Edges with q_eq < alpha (confirmed no connection)
   - These are edges where we have HIGH CONFIDENCE there is NO functional connection
   - NOT the same as q >= alpha (which could be underpowered)
   
3. AMBIGUOUS (ignore): Edges that are neither positive nor confirmed negative
   - These should be excluded from evaluation metrics

This is the ONLY correct way to use the Leifer atlas as ground truth.
Do NOT use all non-positives as negatives - this inflates specificity metrics.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np


@dataclass
class LeiferLabels:
    """
    Container for Leifer atlas labels.
    
    Attributes:
        q_matrix: q-values for functional connection tests
        q_eq_matrix: q-values for "no connection" tests
        amplitude_matrix: Signal amplitude estimates
        positives: Boolean mask of positive edges (q < alpha)
        confirmed_negatives: Boolean mask of confirmed negative edges (q_eq < alpha)
        ambiguous: Boolean mask of edges that are neither
        node_order: Neuron names in matrix order
        alpha: Significance threshold used
        genotype: "wild-type" or "unc-31"
    """
    q_matrix: np.ndarray
    q_eq_matrix: np.ndarray
    amplitude_matrix: Optional[np.ndarray]
    positives: np.ndarray
    confirmed_negatives: np.ndarray
    ambiguous: np.ndarray
    node_order: List[str]
    alpha: float
    genotype: str
    
    @property
    def n_positives(self) -> int:
        return int(np.sum(self.positives))
    
    @property
    def n_confirmed_negatives(self) -> int:
        return int(np.sum(self.confirmed_negatives))
    
    @property
    def n_ambiguous(self) -> int:
        return int(np.sum(self.ambiguous))
    
    @property
    def n_neurons(self) -> int:
        return len(self.node_order)
    
    @property
    def n_possible_edges(self) -> int:
        n = self.n_neurons
        return n * (n - 1)  # Excluding diagonal
    
    @property
    def prevalence(self) -> float:
        """Fraction of labeled edges that are positive."""
        n_labeled = self.n_positives + self.n_confirmed_negatives
        return self.n_positives / max(1, n_labeled)
    
    def get_evaluation_mask(self) -> np.ndarray:
        """
        Get mask of edges that should be used for evaluation.
        
        Returns:
            Boolean mask where True = include in evaluation (positive OR confirmed negative)
        """
        return self.positives | self.confirmed_negatives
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'node_order': self.node_order,
            'alpha': self.alpha,
            'genotype': self.genotype,
            'n_positives': self.n_positives,
            'n_confirmed_negatives': self.n_confirmed_negatives,
            'n_ambiguous': self.n_ambiguous,
            'prevalence': self.prevalence,
        }


def create_leifer_labels(
    q_matrix: np.ndarray,
    q_eq_matrix: np.ndarray,
    node_order: List[str],
    alpha: float = 0.05,
    genotype: str = "wild-type",
    amplitude_matrix: Optional[np.ndarray] = None,
) -> LeiferLabels:
    """
    Create Leifer labels from q-value matrices.
    
    Args:
        q_matrix: Q-values for "is there a connection?" test
                  Lower q = more significant = more likely connected
        q_eq_matrix: Q-values for "is there NO connection?" test
                     Lower q_eq = more confident there is NO connection
        node_order: Neuron names in matrix order
        alpha: Significance threshold (default 0.05)
        genotype: "wild-type" or "unc-31"
        amplitude_matrix: Optional signal amplitude estimates
        
    Returns:
        LeiferLabels object with properly labeled edges
    """
    n = len(node_order)
    
    if q_matrix.shape != (n, n) or q_eq_matrix.shape != (n, n):
        raise ValueError(f"Q-matrix shapes must match node_order length {n}")
    
    # Create diagonal mask (exclude self-connections)
    diag_mask = np.eye(n, dtype=bool)
    
    # POSITIVES: significant connection (q < alpha)
    # These are edges where optogenetic stimulation showed significant signal propagation
    positives = (q_matrix < alpha) & ~diag_mask
    
    # CONFIRMED NEGATIVES: confirmed no connection (q_eq < alpha)
    # These are edges where we have high confidence there is NO functional connection
    # This is NOT the same as "not positive" - that would be incorrect!
    confirmed_negatives = (q_eq_matrix < alpha) & ~diag_mask
    
    # Handle edge case: some edges might be both positive and confirmed negative
    # (shouldn't happen with proper testing, but be defensive)
    both = positives & confirmed_negatives
    if np.any(both):
        print(f"  WARNING: {np.sum(both)} edges are both positive and confirmed negative")
        # Resolve by giving priority to positive label
        confirmed_negatives = confirmed_negatives & ~positives
    
    # AMBIGUOUS: neither positive nor confirmed negative
    # These should be EXCLUDED from evaluation
    ambiguous = ~positives & ~confirmed_negatives & ~diag_mask
    
    return LeiferLabels(
        q_matrix=q_matrix,
        q_eq_matrix=q_eq_matrix,
        amplitude_matrix=amplitude_matrix,
        positives=positives,
        confirmed_negatives=confirmed_negatives,
        ambiguous=ambiguous,
        node_order=node_order,
        alpha=alpha,
        genotype=genotype,
    )


def load_leifer_labels_from_atlas(
    atlas_path: Path,
    alpha: float = 0.05,
) -> Tuple[LeiferLabels, LeiferLabels]:
    """
    Load Leifer labels from atlas files.
    
    Args:
        atlas_path: Path to directory containing aligned_atlas_*.npz files
        alpha: Significance threshold
        
    Returns:
        Tuple of (wild_type_labels, unc31_labels)
    """
    atlas_path = Path(atlas_path)
    
    labels = {}
    
    for genotype in ['wild-type', 'unc-31']:
        file_path = atlas_path / f'aligned_atlas_{genotype}.npz'
        
        if not file_path.exists():
            continue
        
        data = np.load(file_path, allow_pickle=True)
        
        q_matrix = data.get('q', data.get('q_matrix', None))
        q_eq_matrix = data.get('q_eq', data.get('q_eq_matrix', None))
        
        if q_matrix is None or q_eq_matrix is None:
            raise ValueError(f"Missing q or q_eq in {file_path}")
        
        # Get node order
        if 'neurons' in data:
            node_order = list(data['neurons'])
        elif 'node_order' in data:
            node_order = list(data['node_order'])
        else:
            raise ValueError(f"No node_order found in {file_path}")
        
        # Get amplitude if available
        amplitude = data.get('amplitude', data.get('IRF_median', None))
        
        labels[genotype] = create_leifer_labels(
            q_matrix=q_matrix,
            q_eq_matrix=q_eq_matrix,
            node_order=node_order,
            alpha=alpha,
            genotype=genotype,
            amplitude_matrix=amplitude,
        )
    
    wt = labels.get('wild-type')
    unc31 = labels.get('unc-31')
    
    return wt, unc31


def get_labeled_edges_for_evaluation(
    labels: LeiferLabels,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get arrays suitable for sklearn metrics.
    
    Only returns edges that are either positive OR confirmed negative.
    Ambiguous edges are excluded.
    
    Args:
        labels: LeiferLabels object
        
    Returns:
        Tuple of:
        - y_true: Binary array (1 = positive, 0 = confirmed negative)
        - mask: Boolean array indicating which edges are included
    """
    eval_mask = labels.get_evaluation_mask()
    
    # Flatten for sklearn
    y_true = labels.positives[eval_mask].astype(int)
    
    return y_true, eval_mask


def print_label_summary(labels: LeiferLabels, prefix: str = "") -> None:
    """Print a summary of Leifer labels."""
    print(f"{prefix}Leifer {labels.genotype} labels:")
    print(f"{prefix}  Neurons: {labels.n_neurons}")
    print(f"{prefix}  Positives (q < {labels.alpha}): {labels.n_positives}")
    print(f"{prefix}  Confirmed negatives (q_eq < {labels.alpha}): {labels.n_confirmed_negatives}")
    print(f"{prefix}  Ambiguous (excluded): {labels.n_ambiguous}")
    print(f"{prefix}  Prevalence (among labeled): {labels.prevalence:.3f}")

