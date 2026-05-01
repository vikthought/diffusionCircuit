"""
Evaluation metrics - single source of truth.

This module provides canonical implementations for all evaluation metrics used
in the pipeline. All scripts should import from here rather than implementing
their own metric calculations.

METRIC TYPES:
============
1. Binary metrics (after thresholding): Precision, Recall, F1
2. Ranking metrics (continuous scores): AUROC, AUPRC
3. Specificity metrics (on confirmed negatives): True Negative Rate
4. Density-controlled metrics: Metrics at matched edge density

RANDOM BASELINE:
===============
For a fair random baseline, we must match the predicted edge density.

Expected F1 at predicted density ρ and ground truth prevalence π:
    E[precision] = π  (by construction: random guess)
    E[recall] = ρ  (fraction of positives we'd hit by chance)
    E[F1] = 2 * π * ρ / (π + ρ)

Do NOT use ρ=0.5 as a default - this is almost never the actual predicted density!
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np

try:
    from sklearn.metrics import (
        precision_recall_curve,
        roc_curve,
        roc_auc_score,
        auc,
        average_precision_score,
        precision_score,
        recall_score,
        f1_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exclude_diagonal: bool = True,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    
    Args:
        y_true: Ground truth binary labels (n, n) or flattened
        y_pred: Predicted binary labels (n, n) or flattened
        exclude_diagonal: If True and inputs are 2D, exclude diagonal
        
    Returns:
        Dict with precision, recall, f1, and confusion matrix elements
    """
    # Handle 2D matrices
    if y_true.ndim == 2 and y_pred.ndim == 2:
        n = y_true.shape[0]
        if exclude_diagonal:
            mask = ~np.eye(n, dtype=bool)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        else:
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
    
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()
    
    # Confusion matrix elements
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    # Metrics
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-10, precision + recall)
    specificity = tn / max(1, tn + fp)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'n_pred_positive': tp + fp,
        'n_actual_positive': tp + fn,
        'n_total': tp + fp + tn + fn,
    }


def compute_auroc_auprc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    evaluation_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute AUROC and AUPRC from continuous scores.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Continuous prediction scores (higher = more likely positive)
        evaluation_mask: Optional mask of edges to include (for Leifer evaluation)
        
    Returns:
        Dict with AUROC, AUPRC, and related statistics
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for AUROC/AUPRC computation")
    
    # Flatten and apply mask
    if y_true.ndim == 2:
        y_true = y_true.flatten()
    if y_scores.ndim == 2:
        y_scores = y_scores.flatten()
    
    if evaluation_mask is not None:
        if evaluation_mask.ndim == 2:
            evaluation_mask = evaluation_mask.flatten()
        y_true = y_true[evaluation_mask]
        y_scores = y_scores[evaluation_mask]
    
    y_true = y_true.astype(int)
    
    # Handle edge cases
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return {
            'auroc': np.nan,
            'auprc': np.nan,
            'n_evaluated': len(y_true),
            'n_positives': int(n_pos),
            'n_negatives': int(n_neg),
            'warning': 'Cannot compute AUROC/AUPRC with only one class',
        }
    
    # Compute curves
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
    except Exception as e:
        return {
            'auroc': np.nan,
            'auprc': np.nan,
            'n_evaluated': len(y_true),
            'error': str(e),
        }
    
    # Prevalence (for context)
    prevalence = n_pos / len(y_true)
    
    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'prevalence': float(prevalence),
        'n_evaluated': len(y_true),
        'n_positives': int(n_pos),
        'n_negatives': int(n_neg),
        # Random baseline AUPRC = prevalence
        'random_auprc': float(prevalence),
    }


def compute_specificity_on_negatives(
    y_pred: np.ndarray,
    confirmed_negatives: np.ndarray,
) -> Dict[str, float]:
    """
    Compute specificity (true negative rate) on CONFIRMED negatives only.
    
    This is the correct way to evaluate against the Leifer atlas:
    - Use confirmed negatives (q_eq < alpha) as the negative set
    - NOT all non-positives (which includes ambiguous edges)
    
    Args:
        y_pred: Predicted binary labels (n, n)
        confirmed_negatives: Boolean mask of confirmed negative edges
        
    Returns:
        Dict with specificity and related statistics
    """
    # Flatten
    y_pred_flat = y_pred.flatten()
    neg_mask_flat = confirmed_negatives.flatten()
    
    # Get predictions on confirmed negatives
    pred_on_negatives = y_pred_flat[neg_mask_flat]
    
    n_confirmed_neg = len(pred_on_negatives)
    if n_confirmed_neg == 0:
        return {
            'specificity': np.nan,
            'n_confirmed_negatives': 0,
            'warning': 'No confirmed negatives to evaluate',
        }
    
    # True negatives = confirmed negatives that we correctly predict as negative
    true_negatives = int(np.sum(pred_on_negatives == 0))
    # False positives = confirmed negatives that we incorrectly predict as positive
    false_positives = int(np.sum(pred_on_negatives == 1))
    
    specificity = true_negatives / n_confirmed_neg
    
    return {
        'specificity': float(specificity),
        'false_positive_rate': float(1 - specificity),
        'true_negatives': true_negatives,
        'false_positives_on_confirmed_neg': false_positives,
        'n_confirmed_negatives': n_confirmed_neg,
    }


def compute_random_baseline_f1(
    predicted_density: float,
    ground_truth_prevalence: float,
) -> float:
    """
    Compute expected F1 for a random predictor at the given density.
    
    The correct formula for expected F1 at predicted density ρ and prevalence π:
        E[precision] = π (random guesses hit positives at rate π)
        E[recall] = ρ (we predict ρ fraction, so we hit ρ of positives on average)
        E[F1] = 2 * π * ρ / (π + ρ)
    
    Args:
        predicted_density: Fraction of edges predicted as positive (ρ)
        ground_truth_prevalence: Fraction of edges that are actually positive (π)
        
    Returns:
        Expected F1 score for a random predictor
        
    Note:
        DO NOT use a fixed density like 0.5 - use the ACTUAL predicted density!
    """
    rho = predicted_density
    pi = ground_truth_prevalence
    
    if rho <= 0 or pi <= 0:
        return 0.0
    
    return 2 * pi * rho / (pi + rho)


def compute_random_baseline_metrics(
    n_pred_edges: int,
    n_true_edges: int,
    n_possible_edges: int,
) -> Dict[str, float]:
    """
    Compute expected metrics for a random predictor.
    
    Args:
        n_pred_edges: Number of edges predicted as positive
        n_true_edges: Number of edges that are actually positive
        n_possible_edges: Total number of possible edges (excluding diagonal)
        
    Returns:
        Dict with expected precision, recall, F1 for random predictor
    """
    if n_possible_edges <= 0:
        return {'random_f1': 0.0, 'random_precision': 0.0, 'random_recall': 0.0}
    
    # Predicted density
    rho = n_pred_edges / n_possible_edges
    
    # Ground truth prevalence
    pi = n_true_edges / n_possible_edges
    
    # Expected metrics
    random_precision = pi  # Random guess hits positives at rate pi
    random_recall = rho  # We predict rho fraction, hitting rho of positives
    random_f1 = compute_random_baseline_f1(rho, pi)
    
    return {
        'random_precision': float(random_precision),
        'random_recall': float(random_recall),
        'random_f1': float(random_f1),
        'predicted_density': float(rho),
        'ground_truth_prevalence': float(pi),
    }


def compute_metrics_at_density(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_density: float,
    exclude_diagonal: bool = True,
) -> Dict[str, float]:
    """
    Compute metrics after thresholding scores to match a target density.
    
    This is useful for fair comparisons where different methods predict
    different numbers of edges. By matching density, we can compare
    precision/recall more fairly.
    
    Args:
        y_true: Ground truth binary labels (n, n)
        y_scores: Continuous prediction scores (n, n)
        target_density: Target fraction of edges to predict as positive
        exclude_diagonal: If True, exclude diagonal from calculations
        
    Returns:
        Dict with metrics at the target density
    """
    n = y_true.shape[0]
    
    if exclude_diagonal:
        mask = ~np.eye(n, dtype=bool)
        scores_flat = y_scores[mask]
        true_flat = y_true[mask]
    else:
        scores_flat = y_scores.flatten()
        true_flat = y_true.flatten()
    
    n_edges = len(scores_flat)
    n_to_select = int(target_density * n_edges)
    
    if n_to_select <= 0:
        return {'error': 'Target density too low'}
    
    # Find threshold that gives target density
    threshold = np.percentile(scores_flat, 100 * (1 - target_density))
    
    # Create predictions at this threshold
    y_pred = (scores_flat >= threshold).astype(int)
    
    # Compute metrics
    metrics = compute_binary_metrics(true_flat, y_pred, exclude_diagonal=False)
    metrics['threshold_used'] = float(threshold)
    metrics['target_density'] = target_density
    metrics['actual_density'] = float(np.mean(y_pred))
    
    return metrics


def compute_metrics_at_structural_density(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_structural_edges: int,
    exclude_diagonal: bool = True,
) -> Dict[str, float]:
    """
    Compute metrics after thresholding to match the structural connectome density.
    
    This is the recommended approach for Cook connectome comparison:
    threshold functional predictions to have the same number of edges as
    the structural connectome.
    
    Args:
        y_true: Ground truth binary labels (n, n)
        y_scores: Continuous prediction scores (n, n)
        n_structural_edges: Number of edges in structural connectome
        exclude_diagonal: If True, exclude diagonal
        
    Returns:
        Dict with metrics at structural density
    """
    n = y_true.shape[0]
    n_possible = n * (n - 1) if exclude_diagonal else n * n
    
    target_density = n_structural_edges / n_possible
    
    metrics = compute_metrics_at_density(
        y_true, y_scores, target_density, exclude_diagonal
    )
    metrics['n_structural_edges'] = n_structural_edges
    
    return metrics


def sweep_density_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    densities: Optional[np.ndarray] = None,
    n_points: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Compute metrics over a sweep of density thresholds.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Continuous prediction scores
        densities: Specific densities to evaluate (or None for automatic)
        n_points: Number of density points if automatic
        
    Returns:
        Dict with arrays of metrics at each density
    """
    if densities is None:
        densities = np.linspace(0.01, 0.5, n_points)
    
    precisions = []
    recalls = []
    f1s = []
    
    for density in densities:
        metrics = compute_metrics_at_density(y_true, y_scores, density)
        precisions.append(metrics.get('precision', np.nan))
        recalls.append(metrics.get('recall', np.nan))
        f1s.append(metrics.get('f1_score', np.nan))
    
    return {
        'densities': densities,
        'precisions': np.array(precisions),
        'recalls': np.array(recalls),
        'f1_scores': np.array(f1s),
    }


def compute_weight_correlation(
    predicted_weights: np.ndarray,
    ground_truth_weights: np.ndarray,
    exclude_diagonal: bool = True,
    only_overlapping: bool = False,
    significance_mask: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute weight-level correlation metrics between predicted and ground truth.
    
    Args:
        predicted_weights: (n, n) predicted connectivity weights
        ground_truth_weights: (n, n) ground truth weights (e.g., synapse counts)
        exclude_diagonal: If True, exclude self-connections (diagonal)
        only_overlapping: If True, compute only on edges where ground_truth > 0
        significance_mask: (n, n) boolean mask for significant edges (e.g., sign_adj != 0)
        
    Returns:
        Dict with pearson_r, r_squared, spearman_rho, n_pairs
    """
    from scipy.stats import pearsonr, spearmanr
    
    # Validate shapes
    if predicted_weights.shape != ground_truth_weights.shape:
        return {
            'pearson_r': np.nan,
            'r_squared': np.nan,
            'spearman_rho': np.nan,
            'n_pairs': 0,
            'error': 'Shape mismatch',
        }
    
    n = predicted_weights.shape[0]
    
    # Create mask
    if exclude_diagonal:
        mask = ~np.eye(n, dtype=bool)
    else:
        mask = np.ones((n, n), dtype=bool)
    
    if significance_mask is not None:
        mask = mask & significance_mask
    
    if only_overlapping:
        mask = mask & (ground_truth_weights > 0)
    
    # Flatten and extract values under mask
    pred = predicted_weights[mask].flatten()
    gt = ground_truth_weights[mask].flatten()
    
    # Remove NaN values
    valid = ~np.isnan(pred) & ~np.isnan(gt)
    pred = pred[valid]
    gt = gt[valid]
    
    n_pairs = len(pred)
    
    if n_pairs < 3:
        return {
            'pearson_r': np.nan,
            'r_squared': np.nan,
            'spearman_rho': np.nan,
            'n_pairs': n_pairs,
            'error': 'Too few valid pairs',
        }
    
    # Normalize to [0, 1] for interpretability
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-10)
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-10)
    
    # Compute correlations
    try:
        r, p_pearson = pearsonr(pred_norm, gt_norm)
        rho, p_spearman = spearmanr(pred_norm, gt_norm)
    except Exception as e:
        return {
            'pearson_r': np.nan,
            'r_squared': np.nan,
            'spearman_rho': np.nan,
            'n_pairs': n_pairs,
            'error': str(e),
        }
    
    return {
        'pearson_r': float(r),
        'r_squared': float(r ** 2),
        'spearman_rho': float(rho),
        'p_pearson': float(p_pearson),
        'p_spearman': float(p_spearman),
        'n_pairs': n_pairs,
    }
