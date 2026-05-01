#!/bin/bash
#SBATCH --job-name=sbtg_baseline_test
#SBATCH --output=logs/sbtg_baseline_test_%j.log
#SBATCH --time=12:00:00
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=ALL

# ==============================================================================
# SBTG vs Baselines Test Script (Post Baseline-Fix)
# ==============================================================================
#
# Purpose: Quick test to verify baseline fixes and compare SBTG vs baselines
#   on real C. elegans data with corrected VAR and Granger implementations.
#
# What's tested:
#   - SBTG Approach C with 50 HP trials (robust tuning)
#   - Pearson correlation (marginal lag-r effects)
#   - VAR-Ridge (NEW: marginal lag-r effects, not conditional)
#   - Granger causality (lag-1 only, warns for lag > 1)
#
# Additional Analysis:
#   - Cell-type interaction statistics (Script 16)
#   - Monoamine connectome evaluation (Script 18)
#     * Tests if slow monoamines (dopamine) align with long lags
#     * Tests if fast monoamines (tyramine) align with short lags
#
# Lags: 1, 2, 3, 5 (covering 0.25s to 1.25s at 4Hz)
#
# Usage:
#   sbatch synthetic_baseline_slurm.sh              # Default: 50 HP trials
#   sbatch synthetic_baseline_slurm.sh 30           # 30 HP trials
#
# Expected runtime: ~2-4 hours on GPU
#
# ==============================================================================

# Parse command line args
N_HP_TRIALS=${1:-50}     # Number of HP tuning trials for SBTG

# Fixed configuration
LAGS="1 2 3 5 8 12 16 20 "
P_MAX=5
DATASET="full_traces_imputed"
EPOCHS=100

# 1. Setup Project Environment
PROJECT_DIR="/path/to/your/project"
cd "$PROJECT_DIR" || exit 1

# 1.5. Load modules (adjust for your cluster)
# module load Python/3.x.x  # Adjust to your cluster

echo "============================================================"
echo "SBTG vs Baselines Test (Post Baseline-Fix)"
echo "============================================================"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "Project: $PROJECT_DIR"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Lags: $LAGS (0.25s, 0.5s, 0.75s, 1.25s)"
echo "  P_MAX: $P_MAX"
echo "  N_HP_TRIALS: $N_HP_TRIALS"
echo "  EPOCHS: $EPOCHS"
echo "  Approach: C (Minimal Multi-Block)"
echo ""
echo "Analysis Pipeline:"
echo "  1. Multi-lag SBTG + Baselines (Script 15)"
echo "  2. Cell-type statistics (Script 16)"
echo "  3. Monoamine evaluation (Script 18)"
echo ""
echo "Baseline Fixes Applied:"
echo "  ✓ VAR: Now uses Ridge regression (marginal lag-specific)"
echo "  ✓ Granger: Only computed for lag-1"
echo "  ✓ Pearson: Unchanged (already correct)"
echo "============================================================"

# 2. Virtual Environment
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
fi

source env/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# 3. Install Dependencies (if needed)
echo ""
echo "Checking dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Verify sklearn is available (needed for Ridge regression)
python -c "from sklearn.linear_model import Ridge; print('✓ sklearn available')" || {
    echo "ERROR: sklearn not available. Installing..."
    pip install scikit-learn
}

# 4. Environment Variables for Performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4

# PyTorch settings for better GPU utilization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# 5. Create logs directory
mkdir -p logs

# 6. Report GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "nvidia-smi not available"
echo ""

# ==============================================================================
# Run Multi-Lag Analysis with Baselines
# ==============================================================================

echo ""
echo "============================================================"
echo "Running Multi-Lag SBTG Analysis (Approach C) + Baselines"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Train SBTG Approach C with $N_HP_TRIALS HP trials per lag"
echo "  2. Compute Pearson baseline (marginal lag-r correlation)"
echo "  3. Compute VAR-Ridge baseline (NEW: marginal lag-r effects)"
echo "  4. Compute Granger baseline (lag-1 only, warns for lag > 1)"
echo "  5. Evaluate all methods against Cook and Leifer ground truth"
echo ""
echo "Expected behavior after fix:"
echo "  → VAR-Ridge AUROC should DECREASE at higher lags (fewer samples)"
echo "  → Pearson AUROC should DECREASE at higher lags"
echo "  → SBTG should MAINTAIN performance better (conditions on intermediates)"
echo "  → Gap between SBTG and baselines should WIDEN at higher lags"
echo ""

python -u pipeline/15_multilag_analysis.py \
    --dataset "$DATASET" \
    --approach C \
    --p-max "$P_MAX" \
    --lags $LAGS \
    --epochs "$EPOCHS" \
    --n-folds 5 \
    --fdr-alpha 0.1 \
    --tune-hp \
    --n-hp-trials "$N_HP_TRIALS" \
    --device auto \
    --seed 42

# Capture the output directory
RESULT_DIR=$(ls -td results/multilag_separation/*/ 2>/dev/null | head -1)

if [ -z "$RESULT_DIR" ]; then
    echo "ERROR: Could not find results directory"
    exit 1
fi

echo ""
echo "Results directory: $RESULT_DIR"

# ==============================================================================
# Generate Baseline Comparison Figures
# ==============================================================================

echo ""
echo "============================================================"
echo "Generating Comparison Figures"
echo "============================================================"
echo ""

# Check if baseline_metrics.csv exists
if [ -f "${RESULT_DIR}baseline_metrics.csv" ]; then
    echo "Baseline metrics found. Displaying summary:"
    echo ""
    python -c "
import sys, os, pandas as pd
result_dir = sys.argv[1]
df = pd.read_csv(os.path.join(result_dir, 'baseline_metrics.csv'))
print(df.to_string(index=False))
cook_path = os.path.join(result_dir, 'baseline_metrics_cook.csv')
if os.path.exists(cook_path):
    print('\nCook-specific metrics:')
    df_cook = pd.read_csv(cook_path)
    print(df_cook[['method', 'lag', 'auroc_struct', 'auprc_struct']].to_string(index=False))
leifer_path = os.path.join(result_dir, 'baseline_metrics_leifer.csv')
if os.path.exists(leifer_path):
    print('\nLeifer-specific metrics:')
    df_leifer = pd.read_csv(leifer_path)
    print(df_leifer[['method', 'lag', 'auroc', 'auprc']].to_string(index=False))
" "$RESULT_DIR"
else
    echo "Warning: baseline_metrics.csv not found"
fi

# Run cell-type analysis (includes baseline comparison plots if available)
echo ""
echo "Running cell-type analysis..."
python -u pipeline/16_celltype_analysis.py "$RESULT_DIR"

# ==============================================================================
# Monoamine Connectome Evaluation (Script 18)
# ==============================================================================

echo ""
echo "============================================================"
echo "Monoamine Connectome Evaluation (Script 18)"
echo "============================================================"
echo ""
echo "Evaluating multi-lag predictions against monoamine networks:"
echo "  - Dopamine (slow, metabotropic)"
echo "  - Serotonin (intermediate)"
echo "  - Tyramine (fast, ionotropic)"
echo "  - Octopamine"
echo ""
echo "Hypothesis: Slow monoamines should align better with long lags"
echo ""

python -u pipeline/18_multilayer_analysis.py \
    --result-dir "$RESULT_DIR" \
    --dataset full_traces_imputed

if [ $? -eq 0 ]; then
    echo "  ✓ Monoamine evaluation complete"
    
    # Check if monoamine results were created
    if [ -f "${RESULT_DIR}monoamine_auroc_by_lag.csv" ]; then
        echo ""
        echo "  Monoamine AUROC summary:"
        python -c "
import pandas as pd
df = pd.read_csv('${RESULT_DIR}monoamine_auroc_by_lag.csv')
pivot = df.pivot(index='lag', columns='transmitter', values='auroc')
print(pivot.to_string())
print()
print('Look for: Higher AUROC at longer lags for dopamine/serotonin')
"
    fi
else
    echo "  ⚠ Monoamine evaluation failed (non-critical, continuing...)"
fi

# ==============================================================================
# Create Summary Report
# ==============================================================================

echo ""
echo "============================================================"
echo "SUMMARY REPORT"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: $RESULT_DIR"
echo ""
echo "Key Files:"
echo "  ${RESULT_DIR}result_C.npz                        # SBTG results"
echo "  ${RESULT_DIR}eval_cook_C.csv                     # SBTG vs Cook AUROC"
echo "  ${RESULT_DIR}eval_leifer_C.csv                   # SBTG vs Leifer AUROC"
echo "  ${RESULT_DIR}baseline_metrics.csv                # All baselines AUROC (combined)"
echo "  ${RESULT_DIR}baseline_metrics_cook.csv           # Baselines vs Cook"
echo "  ${RESULT_DIR}baseline_metrics_leifer.csv         # Baselines vs Leifer"
echo "  ${RESULT_DIR}baselines.npz                       # Baseline matrices"
echo "  ${RESULT_DIR}monoamine_auroc_by_lag.csv          # Monoamine evaluation"
echo "  ${RESULT_DIR}figures/                            # All figures"
echo ""
echo "Check These Figures:"
echo "  ├── fig_auroc_C.png                               # SBTG AUROC (Cook & Leifer)"
echo "  ├── fig_auprc_C.png                               # SBTG AUPRC (Cook & Leifer)"
echo "  ├── fig_correlation_C.png                         # SBTG Spearman (Cook & Leifer)"
echo "  ├── fig_auroc_vs_lag_C.png                        # Combined AUROC plot"
echo "  ├── fig_baseline_comparison_cook.png              # SBTG vs all baselines"
echo "  ├── fig_clean_auroc.png                           # Clean AUROC by lag"
echo "  ├── fig_celltype_baseline_comparison.png          # Cell-type comparison"
echo "  └── fig_monoamine_auroc_by_lag.png                # Monoamine evaluation"
echo ""
echo "Expected Results (Post-Fix):"
echo "  ✓ VAR-Ridge AUROC decreases with lag (marginal effects)"
echo "  ✓ Pearson AUROC decreases with lag"
echo "  ✓ Granger only reported for lag-1"
echo "  ✓ SBTG maintains highest AUROC at all lags"
echo "  ✓ SBTG advantage INCREASES at longer lags"
echo "  ✓ Monoamines: Slower transmitters align better with longer lags"
echo ""
echo "If VAR-Ridge AUROC is still flat/increasing, check:"
echo "  - Ridge regression implementation in compute_var_baseline()"
echo "  - Sample size calculation (should decrease with lag)"
echo "  - sklearn import (required for Ridge)"
echo ""

exit 0
