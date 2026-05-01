#!/bin/bash
# =============================================================================
# SBTG Pipeline - Complete Execution Script
# =============================================================================
# 
# This script runs the complete SBTG functional connectome inference pipeline.
# 
# Pipeline Structure (12 Core Scripts):
#   01. Prepare Data        (Load, align, normalize)
#   02. Train SBTG          (Hyperparameter search + Training)
#   03. Train Baselines     (Pearson, GLASSO, etc.)
#   04. Evaluate            (Compare vs Cook & Leifer)
#   05. Temporal Analysis   (Phase-specific connectivity)
#   06. Leifer Analysis     (Extrasynaptic & Path-2)
#   07. Regime Analysis     (Latent state discovery)
#   08. Generate Figures    (Summary visualizations)
#   09. Neuron Tables       (Significance tables & Hyperparameters)
#   10. FDR Sensitivity     (Sensitivity analysis - Fig 15)
#   12. HP Objective Valid  (Validates null_contrast objective - SKIPPED, already validated)
#   14. Organize Results    (Consolidate all results + index)
#   15. Multi-Lag Analysis  (Unified: 3 approaches - pair windows, full multi-block, minimal multi-block)
#
# Usage:
#   ./run_pipeline.sh           # Run full pipeline with imputed data
#   ./run_pipeline.sh quick     # Run with --quick flag (fewer epochs/trials)
#   ./run_pipeline.sh clean     # Remove all results and start fresh
#   ./run_pipeline.sh impute    # Run with imputation enabled
#   ./run_pipeline.sh full      # Run BOTH original and imputed data with all analyses
#   ./run_pipeline.sh all       # Same as 'full' - generates all data for figures (5 HP trials, null_contrast objective)
#   ./run_pipeline.sh all --trials 20  # Same as 'all' but with 20 HP trials
#   # null_contrast objective is now DEFAULT (validated in Script 12 to correlate with AUROC)
#   ./run_pipeline.sh figures   # Just generate figures
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}SBTG Pipeline - Functional Connectome Inference${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Handle arguments
QUICK_FLAG=""
IMPUTE_FLAG=""
USE_IMPUTED_FLAG=""
RUN_FULL_MODE=false
FIGURES_ONLY=false
N_TRIALS=50  # Default number of HP search trials
NULL_CONTRAST_FLAG="--objective null_contrast"  # Use null_contrast objective for HP tuning (recommended per Script 12)

# Note: Extended HP search is now the default behavior
# To use standard (faster) HP search, use --hp_standard flag manually

if [ "$1" == "quick" ]; then
    echo -e "${YELLOW}[MODE] Quick execution selected (reduced epochs/trials)${NC}"
    QUICK_FLAG="--quick"
fi

if [ "$1" == "clean" ]; then
    echo -e "${RED}[warn] Cleaning results directory...${NC}"
    rm -rf results/intermediate results/sbtg_models results/baselines results/evaluation results/figures results/sbtg_training results/sbtg_temporal results/regime_gated results/stimulus_specific results/leifer_extended
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    exit 0
fi

if [ "$1" == "impute" ]; then
    echo -e "${YELLOW}[MODE] Imputation mode enabled${NC}"
    echo -e "${YELLOW}       Missing neurons will be imputed from donor worms${NC}"
    IMPUTE_FLAG="--impute-missing"
    USE_IMPUTED_FLAG="--use_imputed"
fi

# Note: Extended HP search is now the default behavior
# To use standard (faster) HP search, use --hp_standard flag manually

if [ "$1" == "full" ] || [ "$1" == "all" ]; then
    echo -e "${YELLOW}[MODE] FULL/ALL MODE - Running BOTH original and imputed data${NC}"
    echo -e "${YELLOW}       This generates all data needed for figures in script 08${NC}"
    RUN_FULL_MODE=true
    N_TRIALS=5  # Default to 5 trials for all mode (we have optimized hyperparams)
fi

if [ "$1" == "figures" ]; then
    echo -e "${YELLOW}[MODE] Generating figures only${NC}"
    FIGURES_ONLY=true
fi

# Check for --trials flag (second argument)
if [ "$2" == "--trials" ] && [ -n "$3" ]; then
    N_TRIALS=$3
    echo -e "${YELLOW}[OPTION] Using $N_TRIALS HP search trials${NC}"
fi

# Check for --null-contrast flag in any argument position
for arg in "$@"; do
    if [ "$arg" == "--null-contrast" ]; then
        NULL_CONTRAST_FLAG="--objective null_contrast"
        echo -e "${YELLOW}[OPTION] Using null_contrast objective for HP tuning (recommended per Script 12 validation)${NC}"
    fi
done

# =============================================================================
# SETUP: Activate virtual environment
# =============================================================================

echo -e "${YELLOW}[SETUP] Checking environment...${NC}"
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo -e "${YELLOW}No .venv found. Assuming global python or user handles env.${NC}"
fi

# Dependency check
if ! python -c "import numpy, optuna, scipy, pandas" &> /dev/null; then
    echo -e "${RED}ERROR: Missing required Python dependencies.${NC}"
    echo -e "Please install them by running: ${YELLOW}pip install -r requirements.txt${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Environment ready${NC}"
echo ""

# =============================================================================
# FIGURES ONLY MODE
# =============================================================================
if [ "$FIGURES_ONLY" = true ]; then
    echo -e "${BLUE}[08/14] GENERATE FIGURES${NC}"
    echo "---------------------------------------------------"
    # Use pre-trained phase data from 05_temporal_analysis.py if available
    python pipeline/08_generate_figures.py --use-pretrained
    echo -e "${GREEN}✓ Figure generation complete${NC}\n"
    exit 0
fi

# =============================================================================
# 1. OPTIONAL: Sanity Check
# =============================================================================
echo -e "${YELLOW}[PREline] Running pre-flight sanity check...${NC}"
python pipeline/sanity_check.py --pre-flight
echo -e "${GREEN}✓ Sanity check passed${NC}"
echo ""

# =============================================================================
# FULL MODE: Run both original and imputed data
# =============================================================================
if [ "$RUN_FULL_MODE" = true ]; then
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}PHASE 1: ORIGINAL DATA (6 worms)${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
    
    # 01. Prepare data (standard)
    echo -e "${BLUE}[01/14] PREPARE DATA (Original)${NC}"
    python pipeline/01_prepare_data.py
    echo -e "${GREEN}✓ Original data preparation complete${NC}\n"
    
    # 02. HP Search + Train (original)
    echo -e "${BLUE}[02/14] TRAIN SBTG (Original)${NC}"
    python pipeline/02_train_sbtg.py --mode hp_search --n_trials $N_TRIALS $QUICK_FLAG $NULL_CONTRAST_FLAG
    python pipeline/02_train_sbtg.py --mode train --tag original_best $QUICK_FLAG
    echo -e "${GREEN}✓ Original SBTG training complete${NC}\n"
    
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}PHASE 2: IMPUTED DATA (20 worms)${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
    
    # 01. Prepare data with imputation
    echo -e "${BLUE}[01/14] PREPARE DATA (Imputed)${NC}"
    python pipeline/01_prepare_data.py --impute-missing
    echo -e "${GREEN}✓ Imputed data preparation complete${NC}\n"
    
    # 02. HP Search + Train (imputed)
    echo -e "${BLUE}[02/14] TRAIN SBTG (Imputed)${NC}"
    python pipeline/02_train_sbtg.py --mode hp_search --use_imputed --n_trials $N_TRIALS $QUICK_FLAG $NULL_CONTRAST_FLAG
    python pipeline/02_train_sbtg.py --mode train --use_imputed --tag imputed_best $QUICK_FLAG
    python pipeline/02_train_sbtg.py --mode train --use_imputed --model_type linear --use_defaults --tag default $QUICK_FLAG
    python pipeline/02_train_sbtg.py --mode train --use_imputed --model_type feature_bilinear --use_defaults --tag default $QUICK_FLAG
    python pipeline/02_train_sbtg.py --mode train --use_imputed --model_type regime_gated --use_defaults --tag default $QUICK_FLAG
    echo -e "${GREEN}✓ Imputed SBTG training complete${NC}\n"
    
    # Continue with shared steps
    IMPUTE_FLAG=""
    USE_IMPUTED_FLAG=""
else
    # =============================================================================
    # SCRIPT 01: PREPARE DATA (standard mode)
    # =============================================================================
    echo -e "${BLUE}[01/14] PREPARE DATA${NC}"
    echo "---------------------------------------------------"
    python pipeline/01_prepare_data.py $IMPUTE_FLAG
    echo -e "${GREEN}✓ Data preparation complete${NC}\n"
    
    # =============================================================================
    # SCRIPT 02: TRAIN SBTG (standard mode)
    # =============================================================================
    echo -e "${BLUE}[02/14] TRAIN SBTG MODELS${NC}"
    echo "---------------------------------------------------"
    python pipeline/02_train_sbtg.py --mode hp_search --n_trials $N_TRIALS $QUICK_FLAG $USE_IMPUTED_FLAG $NULL_CONTRAST_FLAG
    
    echo "Training Best Model..."
    python pipeline/02_train_sbtg.py --mode train --tag best $QUICK_FLAG $USE_IMPUTED_FLAG
    
    echo "Training Linear (Baseline)..."
    python pipeline/02_train_sbtg.py --mode train --model_type linear --use_defaults --tag default $QUICK_FLAG $USE_IMPUTED_FLAG
    
    echo "Training Feature Bilinear (Variant)..."
    python pipeline/02_train_sbtg.py --mode train --model_type feature_bilinear --use_defaults --tag default $QUICK_FLAG $USE_IMPUTED_FLAG
    
    echo "Training Regime Gated (New Standard)..."
    python pipeline/02_train_sbtg.py --mode train --model_type regime_gated --use_defaults --tag default $QUICK_FLAG $USE_IMPUTED_FLAG
    echo -e "${GREEN}✓ SBTG training complete${NC}\n"
fi

# =============================================================================
# SCRIPT 03: TRAIN BASELINES
# =============================================================================
echo -e "${BLUE}[03/14] TRAIN BASELINE METHODS${NC}"
echo "---------------------------------------------------"
python pipeline/03_train_baselines.py
echo -e "${GREEN}✓ Baseline training complete${NC}\n"

# =============================================================================
# SCRIPT 04: EVALUATE
# =============================================================================
echo -e "${BLUE}[04/14] EVALUATION${NC}"
echo "---------------------------------------------------"
# Evaluate models from both original (nacl) and imputed (full_traces_imputed) datasets
if [ "$RUN_FULL_MODE" = true ]; then
    echo "Evaluating original (nacl) models..."
    python pipeline/04_evaluate.py --against both --dataset nacl --stimulus nacl --plot
    echo "Evaluating imputed (full_traces_imputed) models..."
    python pipeline/04_evaluate.py --against both --dataset full_traces_imputed --stimulus nacl --plot
else
    python pipeline/04_evaluate.py --against both --plot
fi
echo -e "${GREEN}✓ Evaluation complete${NC}\n"

# =============================================================================
# SCRIPT 05: TEMPORAL ANALYSIS
# =============================================================================
echo -e "${BLUE}[05/14] TEMPORAL ANALYSIS${NC}"
echo "---------------------------------------------------"
# Run correlation-based analysis first (fast)
python pipeline/05_temporal_analysis.py $QUICK_FLAG

# Run SBTG-based analysis with direct training and HP search
echo "Running SBTG direct training with HP search for each phase..."
python pipeline/05_temporal_analysis.py --sbtg --hp-search --n-trials $N_TRIALS $QUICK_FLAG

# Run SBTG transfer learning with HP search (pre-train on baseline, fine-tune on stimulus)
echo "Running SBTG transfer learning with HP search..."
python pipeline/05_temporal_analysis.py --sbtg --transfer --hp-search --n-trials $N_TRIALS $QUICK_FLAG

echo -e "${GREEN}✓ Temporal analysis complete (correlation + SBTG direct + SBTG transfer)${NC}\n"

# =============================================================================
# SCRIPT 06: LEIFER ANALYSIS
# =============================================================================
echo -e "${BLUE}[06/14] LEIFER ATLAS ANALYSIS${NC}"
echo "---------------------------------------------------"
python pipeline/06_leifer_analysis.py
echo -e "${GREEN}✓ Leifer analysis complete${NC}\n"

# =============================================================================
# SCRIPT 07: REGIME ANALYSIS
# =============================================================================
echo -e "${BLUE}[07/14] REGIME-GATED ANALYSIS${NC}"
echo "---------------------------------------------------"
python pipeline/07_regime_analysis.py --num_regimes 3 $QUICK_FLAG
echo -e "${GREEN}✓ Regime analysis complete${NC}\n"

# =============================================================================
# SCRIPT 08: GENERATE FIGURES
# =============================================================================
echo -e "${BLUE}[08/14] GENERATE FIGURES${NC}"
echo "---------------------------------------------------"
# Use pre-trained phase data from 05_temporal_analysis.py (avoids re-training)
python pipeline/08_generate_figures.py --use-pretrained
echo -e "${GREEN}✓ Figure generation complete${NC}\n"

# =============================================================================
# SCRIPT 09: NEURON TABLES
# =============================================================================
echo -e "${BLUE}[09/14] NEURON SIGNIFICANCE TABLES${NC}"
echo "---------------------------------------------------"
python pipeline/09_neuron_tables.py
echo -e "${GREEN}✓ Neuron tables complete${NC}\n"

# =============================================================================
# SCRIPT 10: FDR SENSITIVITY (Figure 15)
# =============================================================================
echo -e "${BLUE}[10/14] FDR SENSITIVITY ANALYSIS${NC}"
echo "---------------------------------------------------"
python pipeline/10_fdr_sensitivity.py
echo -e "${GREEN}✓ FDR analysis complete${NC}\n"

# =============================================================================
# SCRIPT 12: HP OBJECTIVE VALIDATION (OPTIONAL - Already Validated)
# =============================================================================
echo -e "${BLUE}[11/12] HP OBJECTIVE VALIDATION${NC}"
echo "---------------------------------------------------"
# SKIP by default - this was already run to validate that null_contrast > edge_stability > dsm_loss
# Key finding: null_contrast correlates with Cook AUROC (r=+0.13)
# Edge stability anti-correlates (r=-0.27), DSM loss anti-correlates
# Only run manually if you want to re-validate: python pipeline/12_hp_objective_validation.py --trials 100
echo "Skipping HP validation (already validated - null_contrast is the best objective)"
echo "To re-run: python pipeline/12_hp_objective_validation.py --trials 100"
echo -e "${GREEN}✓ HP objective validation skipped (use null_contrast)${NC}\n"

# =============================================================================
# SCRIPT 15: MULTI-LAG ANALYSIS (Unified - 3 Approaches)
# =============================================================================
echo -e "${BLUE}[12/13] MULTI-LAG ANALYSIS (Approach C)${NC}"
echo "---------------------------------------------------"
# Script 15 unifies the functionality of archived scripts 11 and 13:
#   Approach A: Per-Lag 2-Block (like old Script 11 - reduced-form pair windows)
#   Approach B: Full Multi-Block (like old Script 13 - Theorem 5.1 direct Jacobians)
#   Approach C: Minimal Multi-Block (per-lag conditioning on intermediate lags)
# Uses null_contrast objective for HP tuning (validated in Script 12)
# Auto-detects device (CUDA > MPS > CPU)
python pipeline/15_multilag_analysis.py \
    --dataset full_traces_imputed \
    --approach C \
    --p-max 5 \
    --lags 1 2 3 5 \
    --n-folds 5 \
    --fdr-alpha 0.1 \
    --tune-hp \
    --n-hp-trials $N_TRIALS \
    --seed 42
echo -e "${GREEN}✓ Multi-lag analysis complete${NC}\n"

# =============================================================================
# SCRIPT 16: CELL-TYPE INTERACTION ANALYSIS
# =============================================================================
echo -e "${BLUE}[13/13] CELL-TYPE STATISTICAL ANALYSIS${NC}"
echo "---------------------------------------------------"
# Analyzes interactions between sensory, interneuron, and motor neurons:
#   - Computes mean |μ̂| ± SEM for each cell-type pair × lag
#   - Mann-Whitney U tests for within-lag comparisons
#   - Permutation tests for cross-lag comparisons
#   - Generates comprehensive figures
# Find latest multilag results
MULTILAG_DIR=$(ls -td results/multilag_separation/*/ 2>/dev/null | head -1)
if [ -n "$MULTILAG_DIR" ]; then
    python pipeline/16_celltype_analysis.py "$MULTILAG_DIR" --approach C
    echo -e "${GREEN}✓ Cell-type analysis complete${NC}\n"
else
    echo -e "${YELLOW}⚠ No multilag results found, skipping cell-type analysis${NC}\n"
fi

# =============================================================================
# SCRIPT 14: ORGANIZE AND SUMMARIZE RESULTS
# =============================================================================
echo -e "${BLUE}[FINAL] ORGANIZE AND SUMMARIZE RESULTS${NC}"
echo "---------------------------------------------------"
# Consolidates results from all analyses including:
# - Main SBTG evaluation (Scripts 02-04)
# - Temporal dynamics (Scripts 05-07)
# - HP objective validation (Script 12)
# - Multi-lag analysis (Script 15 - all 3 approaches)
python pipeline/14_organize_results.py
echo -e "${GREEN}✓ Results organization complete${NC}\n"

# =============================================================================
# FINALIZE: Organize Figures
# =============================================================================
echo -e "${BLUE}Finalizing Figures...${NC}"
mkdir -p results/figures/summary
cp results/regime_gated/figures/regime_connectivity_matrices.png results/figures/summary/fig13_regime_connectivity.png 2>/dev/null || echo "Figure 13 not found to copy"
echo -e "${GREEN}✓ Figures organized${NC}\n"

# =============================================================================
# CONCLUSION
# =============================================================================
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${GREEN}PIPELINE SUCCESS! 🚀${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo "All results saved in results/"
echo "Summary figures in results/figures/summary/"
echo "Results index: results/summary/RESULTS_INDEX.md"
echo "Notebooks are available in pipeline/notebooks/ for detailed walkthroughs."

