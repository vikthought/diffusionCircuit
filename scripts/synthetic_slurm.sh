#!/bin/bash
#SBATCH --job-name=synthetic_bench
#SBATCH --output=logs/synthetic_bench_%j.log
#SBATCH --time=12:00:00
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

# ==============================================================================
# Synthetic Benchmark for SBTG and Baselines
# ==============================================================================
#
# Purpose: Run comprehensive synthetic benchmarks including:
#   - SBTG variants (Linear, FeatureBilinear, Minimal)
#   - Classical baselines (VAR-LASSO, VAR-Ridge, VAR-LiNGAM, Poisson-GLM, 
#     PCMCI+, NOTEARS, DYNOTEARS)
#   - All 4 DGP families (var, poisson, hawkes, tanh) with multi-lag ground truth
#   - Low/high noise, short/long sequences
#
# Lean Configuration (default):
#   - 4 families × 2 noise × 2 lengths × 2 seeds = 32 dataset evaluations
#   - 3 SBTG methods × 2 stat configs = 6 SBTG fits per evaluation
#   - Total: ~192 SBTG fits + baselines
#   - Estimated runtime: 2-3 hours on GPU
#
# Usage:
#   sbatch synthetic_slurm.sh                     # Full lean benchmark (SBTG + baselines)
#   sbatch synthetic_slurm.sh --skip-baselines    # SBTG only (faster, ~1-2h)
#   sbatch synthetic_slurm.sh --mini              # Quick smoke test (~15min)
#   sbatch synthetic_slurm.sh --test-baselines    # Test baselines with detailed logging
#   sbatch synthetic_slurm.sh --check-deps        # Check package availability
#
# All arguments after the script name are passed directly to SyntheticTesting.py
#
# ==============================================================================

# Parse command line args (pass through to Python)
EXTRA_ARGS="${@}"

# Project setup
PROJECT_DIR="/path/to/your/project"
cd "$PROJECT_DIR" || exit 1

# Create logs directory
mkdir -p logs

# Load modules
# module load Python/3.x.x  # Adjust to your cluster

# ==============================================================================
# Resource Detection and Configuration
# ==============================================================================

# Detect available CPUs (use SLURM allocation or fallback to nproc)
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    TOTAL_CPUS=$SLURM_CPUS_PER_TASK
elif [ -n "$SLURM_CPUS_ON_NODE" ]; then
    TOTAL_CPUS=$SLURM_CPUS_ON_NODE
else
    TOTAL_CPUS=$(nproc)
fi

# Use ~half of CPUs for parallel workers (leave room for data loading threads)
MAX_WORKERS=$((TOTAL_CPUS / 2))
if [ $MAX_WORKERS -lt 1 ]; then
    MAX_WORKERS=1
fi

# Detect GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "None")
else
    GPU_COUNT=0
    GPU_INFO="None"
fi

# Set CUDA device if available
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
elif [ "$GPU_COUNT" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# Threading configuration for optimal performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# PyTorch specific settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "============================================================"
echo "Synthetic Benchmark for SBTG"
echo "============================================================"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start time: $(date)"
echo "============================================================"
echo ""
echo "Benchmark Configuration (Lean):"
echo "  - 4 DGP families: var, poisson, hawkes, tanh"
echo "  - 2 noise levels: low, high"
echo "  - 2 sequence lengths: short (300), long (800)"
echo "  - 2 seeds: 0, 1"
echo "  - Total evaluations: 32"
echo ""
echo "SBTG Methods (all using Optuna + null contrast, 20 HP trials):"
echo "  - SBTG-Linear (20 Optuna trials, null contrast objective)"
echo "  - SBTG-FeatureBilinear (20 Optuna trials, null contrast objective)"
echo "  - SBTG-Minimal (Approach C, 20 Optuna trials/lag, lags=[1,2])"
echo ""
echo "Statistical Config: BY FDR (alpha=0.10), HAC lags {5,7}"
echo ""
echo "Resource Configuration:"
echo "  Total CPUs: $TOTAL_CPUS"
echo "  Max Workers: $MAX_WORKERS"
echo "  GPU Count: $GPU_COUNT"
echo "  GPU Info: $GPU_INFO"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "============================================================"
echo "Extra args: $EXTRA_ARGS"
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


# Verify GPU is accessible from Python
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || echo "Could not verify PyTorch GPU"

# Run synthetic testing with detected resources
python -u pipeline/SyntheticTesting.py \
    --max-workers $MAX_WORKERS \
    $EXTRA_ARGS

EXIT_CODE=$?

echo "============================================================"
echo "Benchmark completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

# Results summary
RESULT_DIR="sbtg_results"
if [ -d "$RESULT_DIR" ]; then
    echo ""
    echo "============================================================"
    echo "RESULTS SUMMARY"
    echo "============================================================"
    echo "Output directory: $RESULT_DIR"
    echo ""
    echo "Generated files:"
    ls -la "$RESULT_DIR/" 2>/dev/null
    
    # Show metrics summary
    if [ -f "$RESULT_DIR/metrics.csv" ]; then
        echo ""
        echo "--- Metrics Overview ---"
        echo "Total rows: $(wc -l < "$RESULT_DIR/metrics.csv")"
        echo "Methods evaluated:"
        tail -n +2 "$RESULT_DIR/metrics.csv" | cut -d',' -f5 | sort | uniq -c | sort -rn
    fi
    
    # Show report summary
    if [ -f "$RESULT_DIR/report.md" ]; then
        echo ""
        echo "--- Report Summary ---"
        head -80 "$RESULT_DIR/report.md"
    fi
    
    # List figures
    echo ""
    echo "--- Generated Figures ---"
    ls -la "$RESULT_DIR"/*.png 2>/dev/null || echo "No figures generated"
fi

echo ""
echo "============================================================"
echo "Benchmark Complete"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

exit $EXIT_CODE
