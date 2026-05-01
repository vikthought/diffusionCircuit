"""
Pipeline Configuration
======================

Centralizes shared constants and hyperparameters for the diffusionCircuit pipeline.
"""

from pathlib import Path

# =============================================================================
# DATASET CONSTANTS
# =============================================================================

STIMULI = ["nacl", "pentanedione", "butanone"]
MIN_WORMS = 15
INCLUDE_TAIL = True
COLLAPSE_DV = True
USE_FULL_TRACES = False

# =============================================================================
# MODEL HYPERPARAMETERS (DEFAULTS)
# =============================================================================

DEFAULT_HYPERPARAMS = {
    "window_length": 2,
    "dsm_hidden_dim": 128,
    "dsm_num_layers": 3,
    "dsm_noise_std": 0.1,
    "dsm_lr": 1e-3,
    "dsm_epochs": 100,
    "dsm_batch_size": 128,
    "structured_hidden_dim": 64,
    "structured_num_layers": 2,
    "structured_l1_lambda": 0.001,
    "structured_init_scale": 0.1,
    "train_frac": 0.8,
    "train_split": "prefix",
    "hac_max_lag": 5,
    "fdr_method": "by",
    "fdr_alpha": 0.2,
}

# =============================================================================
# OPTIMIZED HYPERPARAMS
# =============================================================================
# IMPORTANT: HP-tuning for DSM loss does NOT correlate with biological evaluation!
# Cluster results (50 trials) showed:
#   - feature_bilinear_default: AUROC 0.570 (BEST)
#   - regime_gated_hp_best: AUROC 0.484 (below random!)
#
# 
# Key insight: Very low L1 regularization (1e-5) causes overfitting.
# These params are empirically validated against Cook connectome.

# Try phase_optimal_params first (but constrained ranges)
try:
    from pipeline.configs.phase_optimal_params import PHASE_OPTIMAL_PARAMS
    if "baseline" in PHASE_OPTIMAL_PARAMS:
        OPTIMIZED_HYPERPARAMS = PHASE_OPTIMAL_PARAMS["baseline"].copy()
        # Ensure L1 regularization is not too low
        if OPTIMIZED_HYPERPARAMS.get("structured_l1_lambda", 0) < 5e-4:
            OPTIMIZED_HYPERPARAMS["structured_l1_lambda"] = 0.001  # Reset to safe value
    else:
        OPTIMIZED_HYPERPARAMS = None
except ImportError:
    OPTIMIZED_HYPERPARAMS = None

# Fallback: Use empirically-validated defaults (feature_bilinear performs best)
if OPTIMIZED_HYPERPARAMS is None:
    OPTIMIZED_HYPERPARAMS = {
        "dsm_lr": 1e-3,        # Standard LR
        "dsm_epochs": 100,     # Moderate epochs (prevent overfitting)
        "dsm_noise_std": 0.1,  # Conservative noise
        "dsm_hidden_dim": 128,
        "structured_hidden_dim": 64,
        "structured_l1_lambda": 0.001,  # CRITICAL: >= 5e-4 to prevent overfitting
        "model_type": "feature_bilinear",  # Best performer on Cook (AUROC 0.570)
        "feature_dim": 16,
    }

# Generic defaults for validation
OPTIMIZED_HYPERPARAMS.update({
    "dsm_batch_size": 128,
    "dsm_num_layers": 3,
    "structured_num_layers": 2,
    "structured_init_scale": 0.1,
    "train_split": "odd_even",
    "train_frac": 0.7,
    "hac_max_lag": 5,
    "fdr_method": "by",
})

# Model Configurations
MODEL_CONFIGS = {
    "linear": {
        "model_type": "linear",
        "description": "Linear structured score (baseline)"
    },
    "feature_bilinear": {
        "model_type": "feature_bilinear",
        "feature_dim": 16,
        "feature_hidden_dim": 64,
        "feature_num_layers": 2,
        "description": "Feature-bilinear coupling"
    },
    "regime_gated": {
        "model_type": "regime_gated",
        "num_regimes": 2,
        "gate_hidden_dim": 64,
        "gate_num_layers": 2,
        "description": "Regime-gated model (context switching)"
    }
}
