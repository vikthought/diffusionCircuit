#!/bin/bash
#SBATCH --job-name=multilag_4period
#SBATCH --output=logs/multilag_4period_%j.log
#SBATCH --time=36:00:00
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

# ==============================================================================
# Multi-Lag SBTG Analysis with 4-Period Stimulus Analysis
# ==============================================================================
#
# Purpose: Comprehensive multi-lag connectivity analysis including:
#   1. Full trace analysis (Approach C with HP tuning)
#   2. 4-Period stimulus analysis (NOTHING, ON, SHOWING, OFF)
#   3. Cell-type statistical analysis for each period
#
# Extended lags: 1,2,3,5,8,10,15,20 (covering 0.25s to 5s at 4Hz)
#
# Usage:
#   sbatch multilag_slurm.sh              # Default: 20 HP trials
#   sbatch multilag_slurm.sh 30           # 30 HP trials
#   sbatch multilag_slurm.sh 20 50        # 20 HP trials, 50 epochs
#
# ==============================================================================

# Parse command line args
N_HP_TRIALS=${1:-20}     # Reduced default from 30 to 20 for faster runs
EPOCHS=${2:-100}         # Training epochs

# Fixed configuration
LAGS="1 2 3 5 8 10 15 20"
P_MAX=20
DATASET="full_traces_imputed"

# 1. Setup Project Environment
PROJECT_DIR="/path/to/your/project"
cd "$PROJECT_DIR" || exit 1

# 1.5. Load modules (adjust for your cluster)
# module load Python/3.x.x  # Adjust to your cluster

echo "============================================================"
echo "Multi-Lag SBTG Analysis with 4-Period Stimulus Analysis"
echo "============================================================"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo "Project: $PROJECT_DIR"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Lags: $LAGS"
echo "  P_MAX: $P_MAX"
echo "  N_HP_TRIALS: $N_HP_TRIALS"
echo "  EPOCHS: $EPOCHS"
echo "  Approach: C (Minimal Multi-Block)"
echo "  4-Period Analysis: NOTHING, ON, SHOWING, OFF"
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

# 4. Environment Variables for Performance
# Allow more threads for data loading but limit for BLAS to avoid over-subscription
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
# STEP 1: Multi-Lag Analysis with 4-Period Stimulus Analysis
# ==============================================================================

echo ""
echo "============================================================"
echo "STEP 1: Multi-Lag SBTG Analysis (Approach C)"
echo "============================================================"
echo ""
echo "This will run:"
echo "  1. Full trace analysis with HP tuning"
echo "  2. 4-Period stimulus analysis:"
echo "     - NOTHING: Baseline (no stimulus)"
echo "     - ON: Stimulus onset transition"
echo "     - SHOWING: Sustained stimulus"
echo "     - OFF: Stimulus offset transition"
echo "  3. Comparison figures across periods"
echo ""
echo "Lags: $LAGS"
echo "  = 0.25s, 0.50s, 0.75s, 1.25s, 2.0s, 2.5s, 3.75s, 5.0s"
echo ""

# Run Script 15 with stimulus-periods flag
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
    --stimulus-periods \
    --device auto \
    --seed 42

# Capture the output directory from the most recent run
RESULT_DIR=$(ls -td results/multilag_separation/*/ 2>/dev/null | head -1)

if [ -z "$RESULT_DIR" ]; then
    echo "ERROR: Could not find results directory"
    exit 1
fi

echo ""
echo "Results directory: $RESULT_DIR"

# ==============================================================================
# STEP 2: Cell-Type Analysis with 4-Period Support
# ==============================================================================

echo ""
echo "============================================================"
echo "STEP 2: Cell-Type Statistical Analysis"
echo "============================================================"
echo ""
echo "Generating cell-type coupling figures for:"
echo "  - Full traces"
echo "  - NOTHING periods (baseline)"
echo "  - ON periods (stimulus onset)"
echo "  - SHOWING periods (sustained stimulus)"
echo "  - OFF periods (stimulus offset)"
echo ""

python -u pipeline/16_celltype_analysis.py "$RESULT_DIR" --on-off

# ==============================================================================
# SUMMARY
# ==============================================================================

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: $RESULT_DIR"
echo ""
echo "Output Structure:"
echo "  $RESULT_DIR"
echo "  ├── result_C.npz                    # Full trace results"
echo "  ├── eval_cook_C.csv                 # AUROC vs Cook"
echo "  ├── config.json                     # Run configuration"
echo "  ├── figures/"
echo "  │   ├── fig_clean_auroc.png"
echo "  │   ├── fig_clean_celltype_by_lag.png"
echo "  │   └── ..."
echo "  └── 4period_analysis/"
echo "      ├── NOTHING/"
echo "      │   ├── result.npz"
echo "      │   ├── eval_cook.csv"
echo "      │   └── figures/"
echo "      ├── ON/"
echo "      │   ├── result.npz"
echo "      │   ├── eval_cook.csv"
echo "      │   └── figures/"
echo "      ├── SHOWING/"
echo "      │   ├── result.npz"
echo "      │   ├── eval_cook.csv"
echo "      │   └── figures/"
echo "      ├── OFF/"
echo "      │   ├── result.npz"
echo "      │   ├── eval_cook.csv"
echo "      │   └── figures/"
echo "      └── comparison/"
echo "          ├── fig_trace_with_periods.png"
echo "          ├── fig_auroc_4periods.png"
echo "          ├── fig_celltype_by_lag_combined.png"
echo "          └── eval_all_periods.csv"
echo ""

exit 0
