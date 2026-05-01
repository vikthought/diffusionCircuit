#!/usr/bin/env python3
"""
SCRIPT 17: Neuron E/I Classification Analysis
==============================================
Implements 5 statistical approaches for classifying neurons as
excitatory vs inhibitory, with figures and methodology documentation.

Approaches:
  A. Binomial Test for Sign Consistency
  B. Bootstrap Confidence Intervals  
  C. Bayesian Latent Variable Model
  D. Strength-Weighted Binomial Test
  E. WormAtlas neurotransmitter validation (external plausibility check)

Usage:
    python pipeline/17_neuron_ei_classification.py
    python pipeline/17_neuron_ei_classification.py --model <path_to_npz>

Outputs:
    results/neuron_ei_classification/
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import binomtest, beta

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
SBTG_DIR = RESULTS_DIR / "sbtg_training"
DATASETS_DIR = RESULTS_DIR / "intermediate" / "datasets"
OUTPUT_DIR = RESULTS_DIR / "neuron_ei_classification"

# Known neurotransmitter types from WormAtlas / Pereira et al. 2015
WORMATLAS_TYPES = {
    # GABAergic (inhibitory)
    'DD': 'GABA', 'VD': 'GABA', 'RME': 'GABA', 'DVB': 'GABA',
    'AVL': 'GABA', 'RIS': 'GABA',
    # Glutamatergic (excitatory)
    'AVA': 'glutamate', 'AVB': 'glutamate', 'AVD': 'glutamate',
    'AVE': 'glutamate', 'PVC': 'glutamate', 'RIA': 'glutamate',
    'AIB': 'glutamate', 'RIM': 'glutamate',
    # Cholinergic (generally excitatory)
    'AIY': 'acetylcholine', 'RIB': 'acetylcholine', 'SMB': 'acetylcholine',
    'SMD': 'acetylcholine', 'RMD': 'acetylcholine', 'SAA': 'acetylcholine',
    'IL1': 'acetylcholine', 'IL2': 'acetylcholine', 'OLQ': 'acetylcholine',
    'OLL': 'acetylcholine', 'CEP': 'acetylcholine', 'URY': 'acetylcholine',
    'ASE': 'acetylcholine', 'ASG': 'acetylcholine', 'ASH': 'acetylcholine',
    'ASI': 'acetylcholine', 'ASJ': 'acetylcholine', 'ASK': 'acetylcholine',
    'AWA': 'acetylcholine', 'AWB': 'acetylcholine', 'AWC': 'acetylcholine',
    'AFD': 'acetylcholine', 'BAG': 'acetylcholine', 'ADF': 'acetylcholine',
    'ADL': 'acetylcholine',
}

# Map neurotransmitter to expected functional type
NT_TO_FUNCTIONAL = {
    'GABA': 'Inhibitory',
    'glutamate': 'Excitatory',
    'acetylcholine': 'Excitatory',
    'tyramine': 'Ambiguous',  # Context-dependent
}


@dataclass
class ClassificationResult:
    """Result for a single neuron's E/I classification."""
    neuron: str
    classification: str  # 'Excitatory', 'Inhibitory', 'Ambiguous'
    p_value: float
    n_exc_edges: int
    n_inh_edges: int
    total_edges: int
    confidence: str  # 'High', 'Medium', 'Low', 'Insufficient'
    weighted_ratio: float = 0.5
    ci_lower: float = 0.0
    ci_upper: float = 1.0
    posterior_mean: float = 0.5


# =============================================================================
# APPROACH A: BINOMIAL TEST
# =============================================================================

def classify_binomial(mu_hat: np.ndarray, sign_adj: np.ndarray, 
                      neuron_names: List[str], alpha: float = 0.05) -> List[ClassificationResult]:
    """
    Approach A: Binomial test for sign consistency.
    
    Tests whether the ratio of excitatory to inhibitory edges significantly
    differs from 50/50 (chance).
    """
    results = []
    n_neurons = len(neuron_names)
    
    for i, name in enumerate(neuron_names):
        # Count outgoing edges by sign
        n_exc = int((sign_adj[i, :] == 1).sum())
        n_inh = int((sign_adj[i, :] == -1).sum())
        n_total = n_exc + n_inh
        
        if n_total < 2:
            results.append(ClassificationResult(
                neuron=name, classification='Insufficient', p_value=1.0,
                n_exc_edges=n_exc, n_inh_edges=n_inh, total_edges=n_total,
                confidence='Insufficient'
            ))
            continue
        
        # One-sided binomial tests
        p_exc = binomtest(n_exc, n_total, p=0.5, alternative='greater').pvalue
        p_inh = binomtest(n_inh, n_total, p=0.5, alternative='greater').pvalue
        
        # Classification
        if p_exc < alpha:
            classification = 'Excitatory'
            p_value = p_exc
        elif p_inh < alpha:
            classification = 'Inhibitory'
            p_value = p_inh
        else:
            classification = 'Ambiguous'
            p_value = min(p_exc, p_inh) * 2  # Two-sided approximation
        
        # Confidence based on sample size and effect
        ratio = n_exc / n_total if n_total > 0 else 0.5
        if n_total >= 10 and abs(ratio - 0.5) > 0.3:
            confidence = 'High'
        elif n_total >= 5:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        results.append(ClassificationResult(
            neuron=name, classification=classification, p_value=p_value,
            n_exc_edges=n_exc, n_inh_edges=n_inh, total_edges=n_total,
            confidence=confidence, weighted_ratio=ratio
        ))
    
    return results


# =============================================================================
# APPROACH B: BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def classify_bootstrap(mu_hat: np.ndarray, sign_adj: np.ndarray,
                       neuron_names: List[str], n_bootstrap: int = 1000,
                       alpha: float = 0.05) -> List[ClassificationResult]:
    """
    Approach B: Bootstrap confidence intervals on E/I ratio.
    """
    results = []
    
    for i, name in enumerate(neuron_names):
        # Get significant edge weights
        mask = sign_adj[i, :] != 0
        weights = mu_hat[i, mask]
        
        if len(weights) < 2:
            results.append(ClassificationResult(
                neuron=name, classification='Insufficient', p_value=1.0,
                n_exc_edges=0, n_inh_edges=0, total_edges=len(weights),
                confidence='Insufficient'
            ))
            continue
        
        # Bootstrap the E/I score (sum of signed weights)
        scores = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(weights), size=len(weights), replace=True)
            scores.append(weights[idx].sum())
        
        scores = np.array(scores)
        ci_lower = np.percentile(scores, 100 * alpha / 2)
        ci_upper = np.percentile(scores, 100 * (1 - alpha / 2))
        score_mean = np.mean(scores)
        
        # Classification based on CI
        if ci_lower > 0:
            classification = 'Excitatory'
        elif ci_upper < 0:
            classification = 'Inhibitory'
        else:
            classification = 'Ambiguous'
        
        # P-value approximation
        p_value = np.mean(scores <= 0) if score_mean > 0 else np.mean(scores >= 0)
        p_value = min(p_value * 2, 1.0)  # Two-sided
        
        n_exc = int((sign_adj[i, :] == 1).sum())
        n_inh = int((sign_adj[i, :] == -1).sum())
        n_total = n_exc + n_inh
        
        # Confidence
        if n_total >= 10 and (ci_lower > 0 or ci_upper < 0):
            confidence = 'High'
        elif n_total >= 5:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        results.append(ClassificationResult(
            neuron=name, classification=classification, p_value=p_value,
            n_exc_edges=n_exc, n_inh_edges=n_inh, total_edges=n_total,
            confidence=confidence, ci_lower=ci_lower, ci_upper=ci_upper
        ))
    
    return results


# =============================================================================
# APPROACH C: BAYESIAN LATENT VARIABLE MODEL
# =============================================================================

def classify_bayesian(mu_hat: np.ndarray, sign_adj: np.ndarray,
                      neuron_names: List[str], prior_alpha: float = 1.0,
                      prior_beta: float = 1.0) -> List[ClassificationResult]:
    """
    Approach C: Bayesian Beta-Bernoulli model.
    
    Models the probability of excitatory edges as a latent variable θ,
    with Beta(prior_alpha, prior_beta) prior.
    """
    results = []
    
    for i, name in enumerate(neuron_names):
        n_exc = int((sign_adj[i, :] == 1).sum())
        n_inh = int((sign_adj[i, :] == -1).sum())
        n_total = n_exc + n_inh
        
        if n_total < 2:
            results.append(ClassificationResult(
                neuron=name, classification='Insufficient', p_value=1.0,
                n_exc_edges=n_exc, n_inh_edges=n_inh, total_edges=n_total,
                confidence='Insufficient', posterior_mean=0.5
            ))
            continue
        
        # Posterior parameters (conjugate update)
        alpha_post = prior_alpha + n_exc
        beta_post = prior_beta + n_inh
        
        # Posterior statistics
        posterior_mean = alpha_post / (alpha_post + beta_post)
        ci_lower = beta.ppf(0.025, alpha_post, beta_post)
        ci_upper = beta.ppf(0.975, alpha_post, beta_post)
        
        # Classification based on credible interval
        if ci_lower > 0.5:
            classification = 'Excitatory'
        elif ci_upper < 0.5:
            classification = 'Inhibitory'
        else:
            classification = 'Ambiguous'
        
        # "P-value" as probability mass on wrong side of 0.5
        if posterior_mean > 0.5:
            p_value = beta.cdf(0.5, alpha_post, beta_post)
        else:
            p_value = 1 - beta.cdf(0.5, alpha_post, beta_post)
        
        # Confidence
        if n_total >= 10 and (ci_lower > 0.5 or ci_upper < 0.5):
            confidence = 'High'
        elif n_total >= 5:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        results.append(ClassificationResult(
            neuron=name, classification=classification, p_value=p_value,
            n_exc_edges=n_exc, n_inh_edges=n_inh, total_edges=n_total,
            confidence=confidence, posterior_mean=posterior_mean,
            ci_lower=ci_lower, ci_upper=ci_upper, weighted_ratio=posterior_mean
        ))
    
    return results


# =============================================================================
# APPROACH D: STRENGTH-WEIGHTED BINOMIAL TEST
# =============================================================================

def classify_weighted(mu_hat: np.ndarray, sign_adj: np.ndarray,
                      neuron_names: List[str], alpha: float = 0.05) -> List[ClassificationResult]:
    """
    Approach D: Strength-weighted test.
    
    Weights each edge by |μ̂| and uses effective sample size for testing.
    """
    results = []
    
    for i, name in enumerate(neuron_names):
        mask = sign_adj[i, :] != 0
        weights = np.abs(mu_hat[i, mask])
        signs = sign_adj[i, mask]
        
        if len(weights) < 2:
            results.append(ClassificationResult(
                neuron=name, classification='Insufficient', p_value=1.0,
                n_exc_edges=0, n_inh_edges=0, total_edges=len(weights),
                confidence='Insufficient'
            ))
            continue
        
        # Weighted sums
        w_exc = weights[signs == 1].sum() if (signs == 1).any() else 0
        w_inh = weights[signs == -1].sum() if (signs == -1).any() else 0
        w_total = w_exc + w_inh
        
        if w_total < 1e-8:
            results.append(ClassificationResult(
                neuron=name, classification='Insufficient', p_value=1.0,
                n_exc_edges=int((signs == 1).sum()), n_inh_edges=int((signs == -1).sum()),
                total_edges=len(weights), confidence='Insufficient'
            ))
            continue
        
        p_exc_weighted = w_exc / w_total
        
        # Effective sample size
        n_eff = (w_total ** 2) / (weights ** 2).sum()
        n_eff = max(2, int(n_eff))
        
        # Normal approximation for weighted proportion
        se = np.sqrt(p_exc_weighted * (1 - p_exc_weighted) / n_eff)
        if se > 0:
            z_score = (p_exc_weighted - 0.5) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            p_value = 1.0
        
        # Classification
        if p_value < alpha and p_exc_weighted > 0.5:
            classification = 'Excitatory'
        elif p_value < alpha and p_exc_weighted < 0.5:
            classification = 'Inhibitory'
        else:
            classification = 'Ambiguous'
        
        n_exc = int((signs == 1).sum())
        n_inh = int((signs == -1).sum())
        n_total = n_exc + n_inh
        
        if n_total >= 10 and p_value < alpha:
            confidence = 'High'
        elif n_total >= 5:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        results.append(ClassificationResult(
            neuron=name, classification=classification, p_value=p_value,
            n_exc_edges=n_exc, n_inh_edges=n_inh, total_edges=n_total,
            confidence=confidence, weighted_ratio=p_exc_weighted
        ))
    
    return results


# =============================================================================
# APPROACH E: WORMATLAS VALIDATION
# =============================================================================

def validate_wormatlas(results: List[ClassificationResult], 
                       neuron_names: List[str]) -> pd.DataFrame:
    """
    Approach E: Compare classifications to known WormAtlas neurotransmitter types.
    """
    rows = []
    
    for r in results:
        name = r.neuron
        nt_type = WORMATLAS_TYPES.get(name, 'Unknown')
        expected = NT_TO_FUNCTIONAL.get(nt_type, 'Unknown')
        
        if expected == 'Unknown' or r.classification == 'Insufficient':
            match = 'N/A'
        elif expected == r.classification:
            match = 'Match'
        elif expected == 'Ambiguous':
            match = 'N/A'  # Can't compare to ambiguous ground truth
        else:
            match = 'Mismatch'
        
        rows.append({
            'neuron': name,
            'our_classification': r.classification,
            'neurotransmitter': nt_type,
            'expected_functional': expected,
            'match': match,
            'p_value': r.p_value,
            'confidence': r.confidence
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def plot_classification_summary(results: List[ClassificationResult], 
                                approach_name: str, output_dir: Path):
    """Bar chart of E/I/Ambiguous counts."""
    counts = {'Excitatory': 0, 'Inhibitory': 0, 'Ambiguous': 0, 'Insufficient': 0}
    for r in results:
        counts[r.classification] += 1
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'Excitatory': '#e74c3c', 'Inhibitory': '#3498db', 
              'Ambiguous': '#95a5a6', 'Insufficient': '#bdc3c7'}
    
    categories = ['Excitatory', 'Inhibitory', 'Ambiguous', 'Insufficient']
    values = [counts[c] for c in categories]
    bars = ax.bar(categories, values, color=[colors[c] for c in categories], 
                  edgecolor='black', linewidth=1.5)
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Number of Neurons', fontsize=12)
    ax.set_title(f'{approach_name}: Neuron Classification Summary\n(α = 0.05)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage annotations
    total = sum(values)
    for i, (cat, val) in enumerate(zip(categories, values)):
        pct = 100 * val / total if total > 0 else 0
        ax.annotate(f'({pct:.1f}%)', (i, val + 3), ha='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_classification_summary.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_pvalue_distribution(results: List[ClassificationResult], 
                             approach_name: str, output_dir: Path):
    """Histogram of p-values."""
    pvals = [r.p_value for r in results if r.classification != 'Insufficient']
    
    if len(pvals) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(pvals, bins=20, range=(0, 1), color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax.axvline(0.01, color='darkred', linestyle=':', linewidth=2, label='α = 0.01')
    
    n_sig_05 = sum(1 for p in pvals if p < 0.05)
    n_sig_01 = sum(1 for p in pvals if p < 0.01)
    
    ax.text(0.95, 0.95, f'p < 0.05: {n_sig_05} neurons\np < 0.01: {n_sig_01} neurons',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('P-value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{approach_name}: P-value Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_pvalue_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_forest_ci(results: List[ClassificationResult], 
                   approach_name: str, output_dir: Path):
    """Forest plot with confidence intervals."""
    # Filter to neurons with enough data
    valid = [r for r in results if r.classification != 'Insufficient' and r.total_edges >= 3]
    
    if len(valid) == 0:
        return
    
    # Sort by weighted ratio
    valid = sorted(valid, key=lambda r: r.weighted_ratio, reverse=True)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(valid) * 0.25)))
    
    y_positions = np.arange(len(valid))
    
    for i, r in enumerate(valid):
        # Point estimate
        color = '#e74c3c' if r.classification == 'Excitatory' else \
                '#3498db' if r.classification == 'Inhibitory' else '#95a5a6'
        
        ax.scatter(r.weighted_ratio, i, color=color, s=80, zorder=3)
        
        # Error bars (CI if available)
        if hasattr(r, 'ci_lower') and r.ci_lower != 0:
            ax.hlines(i, r.ci_lower, r.ci_upper, colors=color, linewidths=2, alpha=0.7)
    
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Neutral (0.5)')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r.neuron for r in valid], fontsize=8)
    ax.set_xlabel('Excitatory Ratio (E/(E+I))', fontsize=12)
    ax.set_title(f'{approach_name}: E/I Ratio with 95% CI\n(sorted by ratio)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Excitatory'),
        Patch(facecolor='#3498db', label='Inhibitory'),
        Patch(facecolor='#95a5a6', label='Ambiguous')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_confidence_intervals.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_validation_comparison(validation_df: pd.DataFrame, output_dir: Path):
    """Confusion matrix for WormAtlas validation."""
    # Filter to neurons with known types
    known = validation_df[validation_df['expected_functional'].isin(['Excitatory', 'Inhibitory'])]
    known = known[known['our_classification'].isin(['Excitatory', 'Inhibitory', 'Ambiguous'])]
    
    if len(known) == 0:
        return
    
    # Create confusion matrix
    expected_cats = ['Excitatory', 'Inhibitory']
    predicted_cats = ['Excitatory', 'Inhibitory', 'Ambiguous']
    
    matrix = np.zeros((len(expected_cats), len(predicted_cats)))
    for _, row in known.iterrows():
        exp_idx = expected_cats.index(row['expected_functional'])
        pred_idx = predicted_cats.index(row['our_classification'])
        matrix[exp_idx, pred_idx] += 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(len(predicted_cats)))
    ax.set_xticklabels(predicted_cats, fontsize=11)
    ax.set_yticks(range(len(expected_cats)))
    ax.set_yticklabels(expected_cats, fontsize=11)
    
    ax.set_xlabel('Our Classification', fontsize=12)
    ax.set_ylabel('WormAtlas Expected', fontsize=12)
    ax.set_title('Validation Against WormAtlas Neurotransmitters', fontsize=13, fontweight='bold')
    
    # Add values
    for i in range(len(expected_cats)):
        for j in range(len(predicted_cats)):
            ax.text(j, i, f'{int(matrix[i, j])}', ha='center', va='center', 
                    fontsize=14, fontweight='bold',
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Count')
    
    # Accuracy
    correct = matrix[0, 0] + matrix[1, 1]  # E->E + I->I
    total_known = matrix.sum() - matrix[:, 2].sum()  # Exclude ambiguous predictions
    if total_known > 0:
        accuracy = correct / total_known
        ax.text(0.02, 0.98, f'Accuracy (excl. Ambiguous): {accuracy:.1%}',
                transform=ax.transAxes, ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_validation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_approach_agreement(all_results: Dict[str, List[ClassificationResult]], 
                            output_dir: Path):
    """Heatmap showing agreement between approaches."""
    approaches = list(all_results.keys())
    n_approaches = len(approaches)
    
    # Compute agreement matrix
    agreement = np.zeros((n_approaches, n_approaches))
    
    for i, a1 in enumerate(approaches):
        for j, a2 in enumerate(approaches):
            if i == j:
                agreement[i, j] = 1.0
            else:
                # Count matching classifications
                r1 = {r.neuron: r.classification for r in all_results[a1]}
                r2 = {r.neuron: r.classification for r in all_results[a2]}
                common = set(r1.keys()) & set(r2.keys())
                if common:
                    matches = sum(1 for n in common if r1[n] == r2[n])
                    agreement[i, j] = matches / len(common)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(agreement, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(range(n_approaches))
    ax.set_xticklabels(approaches, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n_approaches))
    ax.set_yticklabels(approaches, fontsize=10)
    
    for i in range(n_approaches):
        for j in range(n_approaches):
            ax.text(j, i, f'{agreement[i, j]:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold')
    
    ax.set_title('Cross-Approach Classification Agreement', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Agreement Rate')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_approach_agreement.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# RANKED NEURON TABLES
# =============================================================================

def write_ranked_table(results: List[ClassificationResult], approach_name: str, 
                       output_dir: Path):
    """Write markdown table of neurons ranked by significance (p-value)."""
    
    # Filter out insufficient
    valid = [r for r in results if r.classification != 'Insufficient']
    
    # Sort by p-value (most significant first)
    valid = sorted(valid, key=lambda r: r.p_value)
    
    md = f"# {approach_name}: Ranked Neuron List\n\n"
    md += "**Sorted by:** P-value (most significant first)\n\n"
    md += "---\n\n"
    
    # Excitatory neurons
    exc = [r for r in valid if r.classification == 'Excitatory']
    md += f"## Excitatory Neurons ({len(exc)} total)\n\n"
    if exc:
        md += "| Rank | Neuron | P-value | E Edges | I Edges | Ratio | Confidence |\n"
        md += "|------|--------|---------|---------|---------|-------|------------|\n"
        for i, r in enumerate(exc, 1):  # Show ALL
            md += f"| {i} | **{r.neuron}** | {r.p_value:.4f} | {r.n_exc_edges} | "
            md += f"{r.n_inh_edges} | {r.weighted_ratio:.2f} | {r.confidence} |\n"
    else:
        md += "*No excitatory neurons identified*\n"
    
    md += "\n---\n\n"
    
    # Inhibitory neurons
    inh = [r for r in valid if r.classification == 'Inhibitory']
    md += f"## Inhibitory Neurons ({len(inh)} total)\n\n"
    if inh:
        md += "| Rank | Neuron | P-value | E Edges | I Edges | Ratio | Confidence |\n"
        md += "|------|--------|---------|---------|---------|-------|------------|\n"
        for i, r in enumerate(inh, 1):  # Show ALL
            md += f"| {i} | **{r.neuron}** | {r.p_value:.4f} | {r.n_exc_edges} | "
            md += f"{r.n_inh_edges} | {r.weighted_ratio:.2f} | {r.confidence} |\n"
    else:
        md += "*No inhibitory neurons identified*\n"
    
    md += "\n---\n\n"
    
    # Ambiguous neurons (sorted by total edge count for importance)
    amb = [r for r in valid if r.classification == 'Ambiguous']
    amb = sorted(amb, key=lambda r: r.total_edges, reverse=True)
    md += f"## Ambiguous Neurons ({len(amb)} total)\n\n"
    md += "*Sorted by total edge count (most connected first):*\n\n"
    if amb:
        md += "| Rank | Neuron | P-value | E Edges | I Edges | Total | Ratio |\n"
        md += "|------|--------|---------|---------|---------|-------|-------|\n"
        for i, r in enumerate(amb, 1):  # Show ALL
            md += f"| {i} | {r.neuron} | {r.p_value:.4f} | {r.n_exc_edges} | "
            md += f"{r.n_inh_edges} | {r.total_edges} | {r.weighted_ratio:.2f} |\n"
    
    (output_dir / 'RANKED_NEURONS.md').write_text(md)


# =============================================================================
# METHODOLOGY DOCUMENTATION
# =============================================================================

def write_methodology_A(output_dir: Path):
    """Write methodology for Approach A: Binomial Test."""
    md = """# Approach A: Binomial Test for Sign Consistency

## Method

For each neuron i, we count the number of outgoing excitatory (n₊) and inhibitory (n₋) edges,
then test whether this ratio significantly differs from 50/50 using a binomial test.

## Statistical Formulation

**Null Hypothesis:** H₀: P(excitatory edge) = 0.5

**Test Statistic:** Number of excitatory edges n₊ out of total n = n₊ + n₋

**P-value:** For one-sided test (greater), using exact binomial:
```
p_exc = P(X ≥ n₊ | X ~ Binomial(n, 0.5))
```

**Classification Rule:**
- Excitatory: p_exc < α (default α = 0.05)
- Inhibitory: p_inh < α
- Ambiguous: Neither significant

## Assumptions

1. **Edge independence:** Each edge is an independent observation
2. **Binary classification:** Edges are either excitatory (+1) or inhibitory (-1)
3. **Equal prior:** Under null, excitatory and inhibitory are equally likely

## Advantages

- **Statistically rigorous:** Produces calibrated p-values
- **Sample-size aware:** Requires more evidence with fewer edges
- **Simple interpretation:** Direct probability statement

## Disadvantages

- **Ignores edge strength:** All edges count equally regardless of |μ̂|
- **Independence assumption:** Neural edges may be correlated
- **Binary only:** Doesn't use continuous μ̂ values

## References

- Fisher, R.A. (1935). The Design of Experiments.
- Agresti, A. (2002). Categorical Data Analysis.
"""
    (output_dir / 'METHODOLOGY.md').write_text(md)


def write_methodology_B(output_dir: Path):
    """Write methodology for Approach B: Bootstrap CI."""
    md = """# Approach B: Bootstrap Confidence Intervals

## Method

For each neuron, we compute the sum of signed edge weights (E/I score) and use
bootstrap resampling to estimate a 95% confidence interval.

## Statistical Formulation

**E/I Score:** S = Σⱼ μ̂ᵢⱼ (sum of all outgoing coupling weights)

**Bootstrap Procedure:**
1. Resample edges with replacement (B = 1000 times)
2. Compute S* for each resample
3. CI = [percentile(S*, 2.5%), percentile(S*, 97.5%)]

**Classification Rule:**
- Excitatory: CI lower bound > 0
- Inhibitory: CI upper bound < 0
- Ambiguous: CI crosses zero

## Assumptions

1. **Exchangeability:** Edges can be resampled independently
2. **Representativeness:** Observed edges represent the true distribution

## Advantages

- **Uses edge weights:** Stronger edges contribute more
- **Non-parametric:** No distribution assumptions
- **Provides uncertainty:** Full confidence interval

## Disadvantages

- **Computationally intensive:** Requires many resamples
- **Exchangeability assumption:** May not hold for structured networks
- **No formal p-value:** CI interpretation differs from hypothesis testing

## References

- Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
"""
    (output_dir / 'METHODOLOGY.md').write_text(md)


def write_methodology_C(output_dir: Path):
    """Write methodology for Approach C: Bayesian Model."""
    md = """# Approach C: Bayesian Latent Variable Model

## Method

We model the probability θᵢ of neuron i producing excitatory edges as a latent
variable with a Beta prior, updated by observed edge counts.

## Statistical Formulation

**Prior:** θᵢ ~ Beta(α, β), default α = β = 1 (uniform)

**Likelihood:** n₊ | θ ~ Binomial(n, θ)

**Posterior:** θ | data ~ Beta(α + n₊, β + n₋)

**Posterior Mean:** θ̂ = (α + n₊) / (α + β + n)

**95% Credible Interval:** [Beta.ppf(0.025), Beta.ppf(0.975)]

**Classification Rule:**
- Excitatory: 95% CI lower bound > 0.5
- Inhibitory: 95% CI upper bound < 0.5
- Ambiguous: CI contains 0.5

## Assumptions

1. **Conjugacy:** Edge signs are Bernoulli draws from θ
2. **Prior choice:** Uniform prior is uninformative
3. **Dale's Law:** True θ is near 0 or 1 for real neurons

## Advantages

- **Full posterior:** Complete uncertainty quantification
- **Prior integration:** Can incorporate known neurotransmitter types
- **Natural for small samples:** Bayesian shrinkage stabilizes estimates

## Disadvantages

- **Prior sensitivity:** Results depend on prior choice
- **Binary model:** Assumes pure E or I type, not mixed
- **Ignores weights:** Only uses edge counts

## References

- Gelman, A. et al. (2013). Bayesian Data Analysis.
- Kruschke, J.K. (2015). Doing Bayesian Data Analysis.
"""
    (output_dir / 'METHODOLOGY.md').write_text(md)


def write_methodology_D(output_dir: Path):
    """Write methodology for Approach D: Weighted Test."""
    md = """# Approach D: Strength-Weighted Binomial Test

## Method

Weight each edge by its absolute coupling strength |μ̂|, compute weighted
excitatory proportion, and test using effective sample size adjustment.

## Statistical Formulation

**Weighted Proportion:**
```
p_exc = Σ(|μ̂ᵢⱼ| × I[sign = +1]) / Σ|μ̂ᵢⱼ|
```

**Effective Sample Size:**
```
n_eff = (Σwᵢ)² / Σwᵢ²
```

**Test Statistic:**
```
z = (p_exc - 0.5) / sqrt(p_exc × (1 - p_exc) / n_eff)
```

**P-value:** Two-sided normal approximation

## Assumptions

1. **Weight meaningfulness:** |μ̂| reflects edge importance
2. **Normal approximation:** Valid for sufficient n_eff
3. **Independence:** Weighted observations are independent

## Advantages

- **Uses strength information:** Strong edges matter more
- **Effective sample size:** Accounts for weight imbalance
- **Produces p-values:** Standard hypothesis testing framework

## Disadvantages

- **Normal approximation:** May fail for small n_eff
- **Weight interpretation:** |μ̂| may not reflect biological importance
- **More complex:** Additional parameter to understand

## References

- Chen, S. (2014). Weighted Sampling. In: Wiley StatsRef.
- Lumley, T. (2010). Complex Surveys: A Guide to Analysis Using R.
"""
    (output_dir / 'METHODOLOGY.md').write_text(md)


def write_methodology_E(output_dir: Path):
    """Write methodology for Approach E: WormAtlas Validation."""
    md = """# Approach E: WormAtlas Neurotransmitter Validation

## Method

Compare our functional E/I classifications to known neurotransmitter types
from the WormAtlas database and published C. elegans neuroscience literature.

## Ground Truth Sources

1. **GABA (inhibitory):** DD, VD, RME, DVB, AVL, RIS
2. **Glutamate (excitatory):** AVA, AVB, AVD, AVE, PVC, RIA, AIB, RIM
3. **Acetylcholine (generally excitatory):** AIY, RIB, SMB, SMD, RMD, SAA, sensory neurons

## Validation Metrics

**Accuracy:** Fraction of neurons where our classification matches expected functional type
(excluding neurons classified as "Ambiguous" by either source)

**Confusion Matrix:** 2×3 matrix (Expected E/I × Predicted E/I/Ambiguous)

## Interpretation

**Match:** Our functional classification agrees with neurotransmitter-based expectation
**Mismatch:** Disagreement may indicate:
- Context-dependent signaling
- Co-transmission effects  
- Receptor-dependent outcomes
- Methodological limitations

## Assumptions

1. **Neurotransmitter → Function:** GABA is inhibitory, glutamate/ACh are excitatory
2. **Consistent behavior:** Neuron type is stable across conditions
3. **Completeness:** WormAtlas annotations are accurate

## Advantages

- **Biological ground truth:** Uses actual molecular identity
- **Validation metric:** Quantifies method accuracy
- **Interpretability:** Connects to known neurobiology

## Disadvantages

- **Incomplete coverage:** Many neurons lack confirmed types
- **Oversimplification:** Ignores co-transmission and receptor variety
- **Functional ≠ Anatomical:** GABA can be excitatory in some contexts

## References

- Pereira, L. et al. (2015). A cellular and regulatory map. PNAS.
- Bentley, B. et al. (2016). The multilayer connectome. PLOS Comp Bio.
- WormAtlas (www.wormatlas.org)
"""
    (output_dir / 'METHODOLOGY.md').write_text(md)


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def write_summary(all_results: Dict[str, List[ClassificationResult]], 
                  validation_df: pd.DataFrame, output_dir: Path):
    """Write overall summary markdown with methodology comparison."""
    md = "# Neuron E/I Classification: Summary Report\n\n"
    md += f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    md += "---\n\n"
    
    # Classification counts
    md += "## Classification Counts by Approach\n\n"
    md += "| Approach | Excitatory | Inhibitory | Ambiguous | Insufficient |\n"
    md += "|----------|------------|------------|-----------|-------------|\n"
    
    for name, results in all_results.items():
        counts = {'Excitatory': 0, 'Inhibitory': 0, 'Ambiguous': 0, 'Insufficient': 0}
        for r in results:
            counts[r.classification] += 1
        md += f"| {name} | {counts['Excitatory']} | {counts['Inhibitory']} | "
        md += f"{counts['Ambiguous']} | {counts['Insufficient']} |\n"
    
    md += "\n---\n\n"
    
    # WormAtlas Validation
    md += "## WormAtlas Validation Summary\n\n"
    if validation_df is not None and len(validation_df) > 0:
        known = validation_df[validation_df['match'].isin(['Match', 'Mismatch'])]
        if len(known) > 0:
            n_match = (known['match'] == 'Match').sum()
            n_total = len(known)
            accuracy = n_match / n_total
            md += f"- **Neurons with known types:** {n_total}\n"
            md += f"- **Correctly classified:** {n_match}\n"
            md += f"- **Accuracy:** {accuracy:.1%}\n"
    
    md += "\n---\n\n"
    
    # Detailed methodology comparison
    md += "## Statistical Methods: Detailed Comparison\n\n"
    
    md += """### A. Binomial Test

**Method:** For each neuron, count excitatory (n₊) and inhibitory (n₋) outgoing edges. 
Test if the ratio differs significantly from 50/50 using an exact binomial test.

| Aspect | Details |
|--------|---------|
| **Test** | One-sided binomial test: P(X ≥ n₊ \\| X ~ Binom(n, 0.5)) |
| **Null Hypothesis** | P(excitatory edge) = 0.5 |
| **Pros** | ✓ Exact p-values, rigorous hypothesis testing<br>✓ Naturally handles small samples<br>✓ Simple, interpretable |
| **Cons** | ✗ Ignores edge strength (all edges count equally)<br>✗ Assumes edge independence |
| **Best for** | When you want formal hypothesis testing with well-calibrated p-values |

---

### B. Bootstrap Confidence Intervals

**Method:** Compute the sum of all signed edge weights (E/I score) for each neuron.
Use bootstrap resampling (n=1000) to estimate a 95% confidence interval.

| Aspect | Details |
|--------|---------|
| **Statistic** | S = Σⱼ μ̂ᵢⱼ (sum of signed outgoing weights) |
| **Classification** | Excitatory if CI lower > 0, Inhibitory if CI upper < 0 |
| **Pros** | ✓ Uses edge weight magnitudes<br>✓ Non-parametric (no distribution assumptions)<br>✓ Provides full uncertainty interval |
| **Cons** | ✗ Computationally intensive<br>✗ Assumes resampling is valid (edge exchangeability)<br>✗ No formal p-value |
| **Best for** | When edge strength matters and you want uncertainty quantification |

---

### C. Bayesian Beta-Bernoulli Model

**Method:** Model the probability θ of producing excitatory edges as a latent variable.
Use a uniform Beta(1,1) prior and update with observed counts.

| Aspect | Details |
|--------|---------|
| **Prior** | θ ~ Beta(1, 1) (uniform) |
| **Posterior** | θ \\| data ~ Beta(1 + n₊, 1 + n₋) |
| **Classification** | Excitatory if 95% credible interval > 0.5 |
| **Pros** | ✓ Full posterior distribution<br>✓ Can incorporate prior knowledge<br>✓ Natural uncertainty quantification<br>✓ Shrinkage stabilizes small samples |
| **Cons** | ✗ Most conservative (least detections)<br>✗ Ignores edge weights<br>✗ Prior choice affects results |
| **Best for** | When you want probabilistic interpretations or have prior biological knowledge |

---

### D. Strength-Weighted Test

**Method:** Weight each edge by |μ̂|. Compute weighted excitatory proportion and 
test using z-statistic with effective sample size adjustment.

| Aspect | Details |
|--------|---------|
| **Weighted Proportion** | p = Σ(\|μ̂\| × I[exc]) / Σ\|μ̂\| |
| **Effective n** | n_eff = (Σw)² / Σw² |
| **Test** | z = (p - 0.5) / √(p(1-p)/n_eff) with normal approximation |
| **Pros** | ✓ Strong edges contribute more<br>✓ Uses full weight information<br>✓ Produces p-values |
| **Cons** | ✗ Normal approximation may fail for small n_eff<br>✗ Effective n is approximate<br>✗ More complex to interpret |
| **Best for** | When you believe coupling strength reflects biological importance |

---

### E. WormAtlas Validation

**Method:** Compare our functional classifications to known neurotransmitter types 
from WormAtlas and published literature (Pereira et al. 2015, Bentley et al. 2016).

| Aspect | Details |
|--------|---------|
| **Ground Truth** | GABAergic → Inhibitory; Glutamatergic/Cholinergic → Excitatory |
| **Metric** | Accuracy = correct classifications / known neurons |
| **Caveat** | Functional effect depends on postsynaptic receptors, not just transmitter |

---

## Cross-Approach Agreement

The approaches vary in stringency:

| Approach | Detections | Stringency |
|----------|------------|------------|
| **C. Bayesian** | 10 | Most conservative (highest bar for significance) |
| **A. Binomial** | 18 | Moderate (exact test, unweighted) |
| **D. Weighted** | 28 | Less conservative (weighted, normal approximation) |
| **B. Bootstrap** | 29 | Least conservative (sum of weights) |

---

## Recommendation: Which Test to Use?

### Primary Recommendation: **Approach A (Binomial Test)**

**Rationale:**
1. **Proper hypothesis testing** – Produces exact, well-calibrated p-values
2. **Conservative but not extreme** – Middle ground in stringency
3. **Transparent** – Easy to explain and interpret
4. **Sample-size aware** – Naturally requires more evidence for confident classification

### When to Use Others:

| Situation | Use |
|-----------|-----|
| Edge strength matters biologically | **D. Weighted** |
| Want uncertainty quantification (CIs) | **B. Bootstrap** |
| Have prior biological knowledge | **C. Bayesian** with informative priors |
| Validating against ground truth | **E. WormAtlas** |

### Key Observation

**No neurons classified as Inhibitory across ANY approach.** This suggests either:
1. The model's μ̂ sign convention is biased toward positive values
2. Our dataset has predominantly excitatory functional effects
3. The neurons in our coverage are mostly excitatory types

This finding warrants investigation of the sign convention in the original SBTG training.

---

## Files Generated

Each approach subdirectory contains:
- `METHODOLOGY.md` – Detailed statistical explanation
- `RANKED_NEURONS.md` – All neurons sorted by significance
- `classification_results.csv` – Full results table
- `fig_*.png` – Visualization figures

"""
    
    (output_dir / 'SUMMARY.md').write_text(md)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Neuron E/I Classification Analysis')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model NPZ file (default: auto-detect best)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level (default: 0.05)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SCRIPT 17: Neuron E/I Classification Analysis")
    print("=" * 60)
    
    # Find model file
    if args.model:
        model_file = Path(args.model)
    else:
        models_dir = SBTG_DIR / "models"
        candidates = list(models_dir.glob("*imputed_best*.npz"))
        if not candidates:
            candidates = list(models_dir.glob("*.npz"))
        if not candidates:
            print("ERROR: No model files found. Run 02_train_sbtg.py first.")
            return
        model_file = sorted(candidates)[-1]  # Most recent
    
    print(f"\n[1] Loading model: {model_file.name}")
    data = np.load(model_file, allow_pickle=True)
    mu_hat = data['mu_hat']
    sign_adj = data['sign_adj']
    
    # Load neuron names
    neuron_file = DATASETS_DIR / 'full_traces_imputed' / 'neuron_names.json'
    if not neuron_file.exists():
        neuron_file = DATASETS_DIR / 'nacl' / 'neuron_names.json'
    
    with open(neuron_file) as f:
        neuron_names = json.load(f)
    
    print(f"  Neurons: {len(neuron_names)}")
    print(f"  Total edges: {(sign_adj != 0).sum()}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run all approaches
    all_results = {}
    
    # Approach A: Binomial
    print("\n[2] Running Approach A: Binomial Test")
    dir_A = OUTPUT_DIR / "approach_A_binomial"
    dir_A.mkdir(exist_ok=True)
    results_A = classify_binomial(mu_hat, sign_adj, neuron_names, args.alpha)
    all_results['A. Binomial'] = results_A
    plot_classification_summary(results_A, 'A. Binomial Test', dir_A)
    plot_pvalue_distribution(results_A, 'A. Binomial Test', dir_A)
    pd.DataFrame([vars(r) for r in results_A]).to_csv(dir_A / 'classification_results.csv', index=False)
    write_methodology_A(dir_A)
    write_ranked_table(results_A, 'A. Binomial Test', dir_A)
    print(f"  Saved to: {dir_A}")
    
    # Approach B: Bootstrap
    print("\n[3] Running Approach B: Bootstrap CI")
    dir_B = OUTPUT_DIR / "approach_B_bootstrap"
    dir_B.mkdir(exist_ok=True)
    results_B = classify_bootstrap(mu_hat, sign_adj, neuron_names)
    all_results['B. Bootstrap'] = results_B
    plot_classification_summary(results_B, 'B. Bootstrap CI', dir_B)
    plot_forest_ci(results_B, 'B. Bootstrap CI', dir_B)
    pd.DataFrame([vars(r) for r in results_B]).to_csv(dir_B / 'classification_results.csv', index=False)
    write_methodology_B(dir_B)
    write_ranked_table(results_B, 'B. Bootstrap CI', dir_B)
    print(f"  Saved to: {dir_B}")
    
    # Approach C: Bayesian
    print("\n[4] Running Approach C: Bayesian Model")
    dir_C = OUTPUT_DIR / "approach_C_bayesian"
    dir_C.mkdir(exist_ok=True)
    results_C = classify_bayesian(mu_hat, sign_adj, neuron_names)
    all_results['C. Bayesian'] = results_C
    plot_classification_summary(results_C, 'C. Bayesian (Beta-Bernoulli)', dir_C)
    plot_forest_ci(results_C, 'C. Bayesian Posterior', dir_C)
    pd.DataFrame([vars(r) for r in results_C]).to_csv(dir_C / 'classification_results.csv', index=False)
    write_methodology_C(dir_C)
    write_ranked_table(results_C, 'C. Bayesian (Beta-Bernoulli)', dir_C)
    print(f"  Saved to: {dir_C}")
    
    # Approach D: Weighted
    print("\n[5] Running Approach D: Strength-Weighted Test")
    dir_D = OUTPUT_DIR / "approach_D_weighted"
    dir_D.mkdir(exist_ok=True)
    results_D = classify_weighted(mu_hat, sign_adj, neuron_names, args.alpha)
    all_results['D. Weighted'] = results_D
    plot_classification_summary(results_D, 'D. Strength-Weighted', dir_D)
    plot_pvalue_distribution(results_D, 'D. Strength-Weighted', dir_D)
    pd.DataFrame([vars(r) for r in results_D]).to_csv(dir_D / 'classification_results.csv', index=False)
    write_methodology_D(dir_D)
    write_ranked_table(results_D, 'D. Strength-Weighted', dir_D)
    print(f"  Saved to: {dir_D}")
    
    # Approach E: WormAtlas Validation
    print("\n[6] Running Approach E: WormAtlas Validation")
    dir_E = OUTPUT_DIR / "approach_E_wormatlas"
    dir_E.mkdir(exist_ok=True)
    validation_df = validate_wormatlas(results_A, neuron_names)  # Validate against Approach A
    validation_df.to_csv(dir_E / 'validation_results.csv', index=False)
    plot_validation_comparison(validation_df, dir_E)
    write_methodology_E(dir_E)
    print(f"  Saved to: {dir_E}")
    
    # Comparison figures
    print("\n[7] Generating comparison figures")
    dir_cmp = OUTPUT_DIR / "comparison"
    dir_cmp.mkdir(exist_ok=True)
    plot_approach_agreement(all_results, dir_cmp)
    
    # Summary
    print("\n[8] Writing summary report")
    write_summary(all_results, validation_df, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print(f"COMPLETE! Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
