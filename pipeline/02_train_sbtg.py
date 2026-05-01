#!/usr/bin/env python3
"""
SCRIPT 02: Train SBTG Models
============================

Consolidated SBTG training script combining:
- Hyperparameter search (default objective: null_contrast)
- Model training with optimized hyperparameters
- Full traces training (240s recordings)

Usage:
    # Find optimal hyperparameters (default objective is null_contrast)
    python pipeline/02_train_sbtg.py --mode hp_search --n_trials 50
    
    # Train with best hyperparameters
    python pipeline/02_train_sbtg.py --mode train
    
    # Train on full 240s traces
    python pipeline/02_train_sbtg.py --mode full_traces
    
    # Quick test
    python pipeline/02_train_sbtg.py --mode hp_search --n_trials 3 --quick

Outputs:
    - results/sbtg_training/hyperparams/*.json (optimized hyperparameters)
    - results/sbtg_training/models/*.npz (trained models)
    - results/sbtg_training/figures/*.png (visualizations)
"""

import sys
import json
import argparse
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import SBTG
from pipeline.models.sbtg import SBTGStructuredVolatilityEstimator

# Import utilities
from pipeline.utils.align import DV_COLLAPSE_PATTERNS

# Check for Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Run: pip install optuna")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Import configuration
from pipeline.config import DEFAULT_HYPERPARAMS, MODEL_CONFIGS, OPTIMIZED_HYPERPARAMS

DATASETS_DIR = PROJECT_ROOT / "results" / "intermediate" / "datasets"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"
OUTPUT_DIR = PROJECT_ROOT / "results" / "sbtg_training"


# =============================================================================
# DATA LOADING
# =============================================================================

def collapse_dv_subtype(name: str) -> str:
    """Collapse D/V subtypes (e.g., RMDD -> RMD)."""
    return DV_COLLAPSE_PATTERNS.get(name, name)


def load_prepared_dataset(stimulus: str = "nacl") -> Tuple[List[np.ndarray], List[str]]:
    """
    Load pre-prepared dataset from 01_prepare_data.py.
    
    Returns:
        X_segments: List of (T, n_neurons) arrays, one per worm
        neuron_names: List of neuron names
    """
    stimulus_dir = DATASETS_DIR / stimulus
    
    if not stimulus_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {stimulus_dir}")
    
    # Load segments
    segments_file = stimulus_dir / "X_segments.npy"
    if segments_file.exists():
        X_segments_arr = np.load(segments_file, allow_pickle=True)
        # Convert to list of arrays
        X_segments = [X_segments_arr[i] for i in range(X_segments_arr.shape[0])]
    else:
        raise FileNotFoundError(f"X_segments.npy not found in {stimulus_dir}")
    
    # Load neuron names (try standardization.json first, then neuron_names.json)
    std_file = stimulus_dir / "standardization.json"
    names_file = stimulus_dir / "neuron_names.json"
    
    if std_file.exists():
        with open(std_file) as f:
            data = json.load(f)
            neuron_names = data.get('node_order', [])
    elif names_file.exists():
        with open(names_file) as f:
            neuron_names = json.load(f)
    else:
        # Fallback to generic names
        neuron_names = [f"N{i}" for i in range(X_segments[0].shape[1])]
    
    return X_segments, neuron_names


def load_structural_connectome() -> Tuple[np.ndarray, List[str]]:
    """Load the structural connectome."""
    A_struct, nodes, _ = _load_structural_connectome(CONNECTOME_DIR)
    return A_struct, nodes


# =============================================================================
# UNSUPERVISED VALIDATION LOSS
# =============================================================================

def compute_validation_dsm_loss(
    estimator: SBTGStructuredVolatilityEstimator,
    X_val_list: List[np.ndarray],
    device: Optional[torch.device] = None
) -> float:
    """
    Compute DSM loss on validation set.
    
    This is the UNSUPERVISED metric for hyperparameter selection.
    No ground truth (Cook/Leifer) is used.
    
    Uses the estimator's window building method to respect time_lag.
    
    Args:
        estimator: Trained SBTG estimator
        X_val_list: List of validation worm segments
        device: Torch device
        
    Returns:
        Average DSM loss on validation data
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = estimator.model
    noise_std = estimator.dsm_noise_std
    
    if model is None:
        raise ValueError("Estimator model is None. Make sure to call fit() with inference_mode='in_sample' or check internal state.")

    model.eval()
    
    try:
        # Use estimator's window building (respects time_lag)
        Z_raw, _, _ = estimator._build_windows_raw(X_val_list)
        
        if len(Z_raw) == 0:
            return float('inf')
        
        # Standardize
        Z = (Z_raw - Z_raw.mean(axis=0, keepdims=True)) / (Z_raw.std(axis=0, keepdims=True) + 1e-8)
        
        # Convert to tensor
        Z_tensor = torch.tensor(Z, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # Add noise
            eps = torch.randn_like(Z_tensor) * noise_std
            noisy_Z = Z_tensor + eps
            
            # Compute score
            output = model(noisy_Z)
            if isinstance(output, tuple):
                _, score = output
            else:
                score = output
            
            # DSM loss: || s(z + eps) + eps/sigma^2 ||^2
            target = -eps / (noise_std ** 2)
            loss = ((score - target) ** 2).sum(dim=1).mean()
            
            return loss.item()
            
    except Exception as e:
        return float('inf')


# =============================================================================
# HYPERPARAMETER SEARCH (OPTUNA)
# =============================================================================

def create_hp_objective(X_train: List[np.ndarray], X_val: List[np.ndarray], 
                         objective_type: str = "dsm_loss", extended: bool = False):
    """
    Create Optuna objective function.
    
    Args:
        X_train: Training data
        X_val: Validation data
        objective_type: "dsm_loss", "edge_density", or "null_contrast"
        extended: If True, use wider HP ranges and include regime_gated models
        
    Objective: 
    - dsm_loss: Minimize DSM validation loss (learning the noise).
    - edge_density: Maximize edge count while controlling density (heuristic).
    - null_contrast: Maximize signal above shuffled null (RECOMMENDED - correlates with biological AUROC per Script 12 validation).
    """
    def objective(trial):
        # Suggest hyperparameters (extended mode uses wider ranges)
        # NOTE: Objective-specific behavior is selected below.
        # DSM loss is useful for score quality but may not track biological metrics.
        # Constrained ranges prevent extreme deregularization that leads to overfitting.
        # See analysis: regime_gated_original_best (HP-tuned) got AUROC 0.484 (below random!)
        #              while feature_bilinear_default got AUROC 0.570 (best).
        if extended:
            # Extended mode: constrained ranges to prevent overfitting
            # Key insight: very low L1 lambda (1e-5) causes severe overfitting
            params = {
                'dsm_lr': trial.suggest_float('dsm_lr', 5e-5, 2e-3, log=True),  # Tightened: avoid very low LR
                'dsm_epochs': trial.suggest_int('dsm_epochs', 80, 200, step=20),  # Reduced max: prevent overfitting
                'dsm_noise_std': trial.suggest_float('dsm_noise_std', 0.08, 0.4, step=0.04),  # Tightened noise range
                'dsm_hidden_dim': trial.suggest_categorical('dsm_hidden_dim', [64, 128, 256]),
                'structured_hidden_dim': trial.suggest_categorical('structured_hidden_dim', [32, 64, 128]),
                # CRITICAL: L1 lambda >= 5e-4 to prevent overfitting (was 1e-5 which caused AUROC < 0.5)
                'structured_l1_lambda': trial.suggest_float('structured_l1_lambda', 5e-4, 0.05, log=True),
                'fdr_alpha': trial.suggest_categorical('fdr_alpha', [0.1, 0.15, 0.2]),  # Sensible range
                'model_type': trial.suggest_categorical('model_type', ['linear', 'feature_bilinear', 'regime_gated']),
                'train_split': trial.suggest_categorical('train_split', ['prefix', 'odd_even']),
            }
        else:
            # Standard mode (conservative ranges)
            params = {
                'dsm_lr': trial.suggest_float('dsm_lr', 1e-4, 2e-3, log=True),
                'dsm_epochs': trial.suggest_int('dsm_epochs', 80, 150, step=10),
                'dsm_noise_std': trial.suggest_float('dsm_noise_std', 0.08, 0.3, step=0.04),
                'dsm_hidden_dim': trial.suggest_categorical('dsm_hidden_dim', [64, 128, 256]),
                'structured_hidden_dim': trial.suggest_categorical('structured_hidden_dim', [32, 64, 128]),
                'structured_l1_lambda': trial.suggest_float('structured_l1_lambda', 5e-4, 0.02, log=True),
                'fdr_alpha': trial.suggest_categorical('fdr_alpha', [0.1, 0.15, 0.2]),
                'model_type': trial.suggest_categorical('model_type', ['linear', 'feature_bilinear']),
                'train_split': trial.suggest_categorical('train_split', ['prefix', 'odd_even']),
            }
        
        # Add model-specific params
        if params['model_type'] == 'feature_bilinear':
            params['feature_dim'] = trial.suggest_categorical('feature_dim', [8, 16, 32])
        elif params['model_type'] == 'regime_gated':
            params['num_regimes'] = trial.suggest_int('num_regimes', 2, 4)
            params['gate_hidden_dim'] = trial.suggest_categorical('gate_hidden_dim', [32, 64])
        
        # Fixed params
        params.update({
            'window_length': 2,
            'dsm_batch_size': 128,
            'dsm_num_layers': 3,
            'structured_num_layers': 2,
            'structured_init_scale': 0.1,
            'hac_max_lag': 5,
            'fdr_method': 'by',
            'train_frac': 0.8,
            'verbose': False,
            # CRITICAL: Use in-sample training for HP search
            # 1. We want a single model trained on all X_train for validation evaluation
            # 2. We don't need valid p-values here, just the loss on X_val
            # 3. Avoids training K models (cross-fitting) which is slow and sets self.model to the last fold only
            'inference_mode': 'in_sample',
        })
        
        try:
            # Train on train set
            estimator = SBTGStructuredVolatilityEstimator(**params)
            result = estimator.fit(X_train)
            
            # --- EVALUATION ---
            if objective_type == "dsm_loss":
                # Minimize Validation DSM Loss (score matching error on unseen data)
                # This measures how well the model learned the data distribution/gradients
                val_loss = compute_validation_dsm_loss(estimator, X_val, device=estimator.device)
                
                if np.isnan(val_loss) or np.isinf(val_loss):
                    print(f"Trial pruned due to NaN/Inf loss (LR={params['dsm_lr']:.2e}, Noise={params['dsm_noise_std']:.2f})")
                    return float('inf')

                trial.set_user_attr('val_loss', val_loss)
                trial.set_user_attr('model_type', params['model_type'])
                return val_loss
            
            elif objective_type == "null_contrast":
                # Null Contrast: signal strength above shuffled null distribution
                # RECOMMENDED per Script 12 validation: r=+0.13 correlation with Cook AUROC
                # Higher = stronger signal relative to noise
                sign_adj = result.sign_adj
                mu_hat = result.mu_hat
                
                # Get real signal strength (absolute mean coupling)
                n = sign_adj.shape[0]
                mask = ~np.eye(n, dtype=bool)  # Exclude diagonal
                real_signal = np.abs(mu_hat[mask]).mean()
                
                # Shuffle null: permute rows of X_val to destroy temporal structure
                null_signals = []
                for _ in range(5):  # 5 null shuffles
                    X_val_shuffled = []
                    for x in X_val:
                        perm = np.random.permutation(len(x))
                        X_val_shuffled.append(x[perm])
                    
                    # Refit on shuffled data (quick - reduced epochs)
                    params_null = params.copy()
                    params_null['dsm_epochs'] = min(params['dsm_epochs'] // 3, 30)
                    estimator_null = SBTGStructuredVolatilityEstimator(**params_null)
                    result_null = estimator_null.fit(X_val_shuffled)
                    
                    null_signal = np.abs(result_null.mu_hat[mask]).mean()
                    null_signals.append(null_signal)
                
                null_mean = np.mean(null_signals)
                null_std = np.std(null_signals) + 1e-8
                
                # Contrast: how many std above null is our signal?
                contrast = (real_signal - null_mean) / null_std
                
                trial.set_user_attr('null_contrast', contrast)
                trial.set_user_attr('real_signal', real_signal)
                trial.set_user_attr('null_mean', null_mean)
                trial.set_user_attr('model_type', params['model_type'])
                
                # Maximize contrast (return negative for minimization)
                return -contrast
                
            else: # edge_density
                # Compute edge statistics directly from sign_adj
                sign_adj = result.sign_adj
                n = sign_adj.shape[0]
                n_edges = int((sign_adj != 0).sum())
                n_possible = n * (n - 1)
                density = n_edges / n_possible if n_possible > 0 else 0
                
                # Objective: maximize meaningful edge discovery
                # Penalize trivial solutions (too sparse or too dense)
                target_density = 0.15
                density_penalty = abs(density - target_density) * 10
                
                # Score: edges discovered minus density penalty
                score = n_edges - density_penalty * 100
                
                trial.set_user_attr('n_edges', n_edges)
                trial.set_user_attr('density', density)
                trial.set_user_attr('model_type', params['model_type'])
                
                # Maximize edges (return negative for minimization)
                return -score
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    return objective


def run_hp_search(
    X_segments: List[np.ndarray],
    n_trials: int = 50,
    n_val_worms: int = 3,
    output_dir: Optional[Path] = None,
    objective_type: str = "dsm_loss",
    extended: bool = False
) -> Dict:
    """
    Run hyperparameter search using Optuna.
    
    Args:
        X_segments: List of worm data segments
        n_trials: Number of Optuna trials
        n_val_worms: Number of worms held out for validation
        output_dir: Where to save results
        objective_type: Optimization objective:
            - "dsm_loss": Minimize validation DSM loss (original)
            - "edge_density": Maximize edge count with density penalty
            - "null_contrast": Maximize signal above shuffled null (RECOMMENDED - per Script 12 validation)
        extended: If True, use extended HP search space with wider ranges
    
    Returns:
        Best hyperparameters dict
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required for HP search. Run: pip install optuna")
    
    # Split into train/val
    n_worms = len(X_segments)
    n_val = min(n_val_worms, max(1, n_worms // 5))  # At least 1, at most 20%
    train_idx = list(range(n_worms - n_val))
    val_idx = list(range(n_worms - n_val, n_worms))
    
    X_train = [X_segments[i] for i in train_idx]
    X_val = [X_segments[i] for i in val_idx]
    
    print(f"Train worms: {len(X_train)}, Val worms: {len(X_val)}")
    print(f"Objective: {objective_type}")
    if extended:
        print(f"Extended HP search: regime_gated models + wider ranges")
    
    # Create study
    # Configure TPESampler with more exploration (n_startup_trials)
    # n_startup_trials=20 forces 20 random samples before TPE kicks in
    # This prevents the model from converging too early to a local optimum
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
    
    study = optuna.create_study(direction="minimize", sampler=sampler)
    
    # Auto-detect number of CPUs for parallel optimization
    # n_jobs=-1 uses all available CPUs, n_jobs=1 disables parallelization
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 CPU free for system
    print(f"Using {n_jobs} parallel workers for hyperparameter optimization")
    
    study.optimize(
        create_hp_objective(X_train, X_val, objective_type=objective_type, extended=extended), 
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    # Get best params
    best_params = study.best_params.copy()
    best_params['best_val_loss'] = study.best_value
    
    print(f"\nBest hyperparameters (val={study.best_value:.4f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # Save
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "best_hyperparams.json", "w") as f:
            json.dump(best_params, f, indent=2)
        
        # Save study results
        df = study.trials_dataframe()
        df.to_csv(output_dir / "hp_search_results.csv", index=False)
    
    return best_params


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    X_segments: List[np.ndarray],
    hyperparams: Dict,
    model_config: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    stimulus: str = "nacl",
    tag: str = ""
):
    """
    Train SBTG model with given hyperparameters.

    Saved output convention:
    - `mu_hat` and `sign_adj` are negated before writing to disk so the stored
      polarity matches the functional-atlas sign convention used by downstream
      evaluation/figure scripts.
    
    Returns:
        Trained estimator and result object
    """
    # Merge configs
    params = DEFAULT_HYPERPARAMS.copy()
    params.update(hyperparams)
    if model_config:
        params.update(model_config)
    
    # Remove non-estimator params
    params.pop('best_val_loss', None)
    params.pop('description', None)
    params.pop('dataset_used', None)  # Metadata only, not an estimator param
    params['verbose'] = True
    
    print(f"Training {params.get('model_type', 'linear')} model...")
    print(f"  Epochs: {params.get('dsm_epochs')}")
    print(f"  LR: {params.get('dsm_lr')}")
    
    estimator = SBTGStructuredVolatilityEstimator(**params)
    result = estimator.fit(X_segments)
    
    # Save
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_type = params.get('model_type', 'linear')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save result
        tag_str = f"_{tag}" if tag else ""
        result_file = output_dir / f"{model_type}_{stimulus}{tag_str}_{timestamp}.npz"
        # Negate sign convention to match functional-atlas dFF polarity.
        # SBTG internally learns: positive mu_hat = source drives target (could be inhibitory if decreases activity)
        # Functional atlas dFF: positive = excitatory (increases activity), negative = inhibitory
        # We negate to align: positive mu_hat = excitatory, negative = inhibitory
        np.savez(
            result_file,
            sign_adj=-result.sign_adj,  # Negate sign convention
            mu_hat=-result.mu_hat,       # Negate coupling weights
            volatility_adj=result.volatility_adj,  # Edges from volatility test
            p_mean=result.p_mean,        # P-values from mean test
            p_volatility=result.p_volatility,  # P-values from volatility test
            W_param=result.W_param,
            hyperparams=json.dumps(params)
        )
        print(f"Saved: {result_file}")
    
    return estimator, result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SBTG Models")
    parser.add_argument('--mode', choices=['hp_search', 'train', 'full_traces'],
                        default='train', help='Training mode')
    parser.add_argument('--stimulus', default='nacl', help='Stimulus dataset')
    parser.add_argument('--model_type', choices=['linear', 'feature_bilinear', 'regime_gated'],
                        default='regime_gated', help='Model architecture')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--use_defaults', action='store_true', help='Ignore best_hyperparams.json and use defaults')
    parser.add_argument('--use_optimized', action='store_true', 
                        help='Use OPTIMIZED_HYPERPARAMS from config.py (150-trial HP search Jan 2026)')
    parser.add_argument('--tag', type=str, default='', help='Tag to append to output filename')
    parser.add_argument('--objective', choices=['dsm_loss', 'edge_density', 'null_contrast'], 
                        default='null_contrast', help='Optuna objective: dsm_loss, edge_density, or null_contrast (recommended per Script 12 validation)')
    parser.add_argument('--fdr_alpha', type=float, default=None, help='Override FDR alpha')
    parser.add_argument('--epochs', type=int, default=None, help='Override DSM epochs')
    parser.add_argument('--use_imputed', action='store_true', 
                        help='Use imputed full-trace dataset (all worms via donor imputation)')
    parser.add_argument('--hp_extended', action='store_true', default=True,
                        help='Run extended HP search (more epochs, wider ranges, discretized, n_startup_trials=20) [DEFAULT]')
    parser.add_argument('--hp_standard', dest='hp_extended', action='store_false',
                        help='Use standard HP search (narrower ranges, faster)')
    args = parser.parse_args()
    
    print("="*60)
    print(f"SBTG Training - Mode: {args.mode}")
    print("="*60)
    
    # Setup output directory
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    # Determine which dataset to use
    if args.use_imputed:
        dataset_name = "full_traces_imputed"
        print(f"\n[1] Loading IMPUTED data: {dataset_name}")
    else:
        dataset_name = args.stimulus
        print(f"\n[1] Loading data: {dataset_name}")
    
    X_segments, neuron_names = load_prepared_dataset(dataset_name)
    print(f"  Loaded: {len(X_segments)} worms, {len(neuron_names)} neurons")
    
    if args.mode == 'hp_search':
        # Hyperparameter search (unsupervised)
        print(f"\n[2] Running hyperparameter search ({args.n_trials} trials)")
        n_trials = 3 if args.quick else args.n_trials
        best_params = run_hp_search(
            X_segments,
            n_trials=n_trials,
            output_dir=output_dir / "hyperparams",
            objective_type=args.objective,
            extended=args.hp_extended
        )
        
    elif args.mode == 'train':
        # Load best hyperparams if available and not ignored
        hp_file = output_dir / "hyperparams" / "best_hyperparams.json"
        
        if args.use_defaults:
             print("\n[2] Using default hyperparameters (forced)")
             hyperparams = DEFAULT_HYPERPARAMS.copy()
        elif args.use_optimized:
             print("\n[2] Using OPTIMIZED_HYPERPARAMS from config.py (150-trial HP search Jan 2026)")
             hyperparams = OPTIMIZED_HYPERPARAMS.copy()
        elif hp_file.exists():
            print(f"\n[2] Loading optimized hyperparameters from {hp_file}")
            with open(hp_file) as f:
                hyperparams = json.load(f)
        else:
            # Fallback to OPTIMIZED_HYPERPARAMS if no hp_file found
            print("\n[2] No best_hyperparams.json found, using OPTIMIZED_HYPERPARAMS from config.py")
            hyperparams = OPTIMIZED_HYPERPARAMS.copy()
        
        # Override model type if specified
        if args.model_type:
            # If utilizing defaults, or if user explicitly overrides the optimized model type
            if args.use_defaults or args.model_type != hyperparams.get(('model_type'), 'regime_gated'):
                 hyperparams['model_type'] = args.model_type
        
        # Merge model-specific configuration (e.g. feature_dim, num_regimes)
        # This ensures that if we switch to 'regime_gated', we get the default 'num_regimes'
        model_type = hyperparams.get('model_type', 'regime_gated')
        if model_type in MODEL_CONFIGS:
            print(f"    Merging configuration for {model_type}")
            hyperparams.update(MODEL_CONFIGS[model_type])

        # Quick mode
        if args.quick:
            hyperparams['dsm_epochs'] = 10
        
        # Override epochs if specified
        if args.epochs is not None:
             hyperparams['dsm_epochs'] = args.epochs
             
        # Override alpha if specified (ensure it propagates to estimator init)
        if args.fdr_alpha is not None:
            hyperparams['fdr_alpha'] = args.fdr_alpha
            
        # Store actual dataset in hyperparams for documentation
        hyperparams['dataset_used'] = dataset_name
        
        print(f"\n[3] Training model...")
        estimator, result = train_model(
            X_segments,
            hyperparams,
            output_dir=output_dir / "models",
            stimulus=dataset_name,  # Use actual dataset name for accurate filenames
            tag=args.tag
        )

        # Print summary
        print(f"\n[4] Results:")
        print(f"  Sign edges: {(result.sign_adj != 0).sum()}")
        edge_density = (result.sign_adj != 0).sum() / (result.sign_adj.size - len(neuron_names))
        print(f"  Edge density: {edge_density:.3f}")
        
    elif args.mode == 'full_traces':
        # Full 240s traces mode
        print(f"\n[2] Training on full traces (240s recordings)")
        
        # Use specific hyperparams for full traces
        hyperparams = DEFAULT_HYPERPARAMS.copy()
        hyperparams['dsm_epochs'] = 150 if not args.quick else 10
        hyperparams['model_type'] = args.model_type
        
        # Store actual dataset in hyperparams for documentation
        hyperparams['dataset_used'] = dataset_name
        
        estimator, result = train_model(
            X_segments,
            hyperparams,
            output_dir=output_dir / "full_traces",
            stimulus=dataset_name  # Use actual dataset name for accurate filenames
        )
        
        print(f"\n[3] Results:")
        print(f"  Sign edges: {(result.sign_adj != 0).sum()}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
