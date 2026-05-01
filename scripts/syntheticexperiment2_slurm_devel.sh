#!/bin/bash
#SBATCH --job-name=synthetic_exp2_devel
#SBATCH --output=logs/synthetic_exp2_devel_%j.log
#SBATCH -p gpu_devel
#SBATCH -t 6:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=1           
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-cpu=4G
# ==============================================================================
# SyntheticExperiment2: Large-Scale Synthetic Benchmark (Reviewer Response)
# ==============================================================================
#
# Purpose: Run synthetic scaling experiments with n=80 neurons using SBTG and
# full classical baselines. This script targets reviewer requests on scalability.
#
# Default benchmark grid:
#   - 4 families × 2 noise × 2 lengths × 2 seeds = 32 dataset evaluations
#   - n_neurons = 80
#   - Full baseline suite enabled (unless --skip-baselines is passed)
#
# Usage:
#   sbatch scripts/syntheticexperiment2_slurm.sh
#   sbatch scripts/syntheticexperiment2_slurm.sh --skip-baselines
#   sbatch scripts/syntheticexperiment2_slurm.sh --mini --n-neurons 80
#   sbatch scripts/syntheticexperiment2_slurm.sh --medium --n-neurons 80
#   sbatch scripts/syntheticexperiment2_slurm.sh --n-seeds 4 --hp-trials 30
#   sbatch scripts/syntheticexperiment2_slurm.sh --check-deps
#   sbatch scripts/syntheticexperiment2_slurm.sh --medium --run-external-baselines
#
# All arguments after script name are passed directly to syntheticexperiment2.py
#
# ==============================================================================

EXTRA_ARGS="${@}"

# Project setup
PROJECT_DIR="/path/to/your/project"
cd "$PROJECT_DIR" || exit 1

# Create logs directory
mkdir -p logs

# Detect available CPUs (use SLURM allocation or fallback to nproc)
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
    TOTAL_CPUS=$SLURM_CPUS_PER_TASK
elif [ -n "$SLURM_CPUS_ON_NODE" ]; then
    TOTAL_CPUS=$SLURM_CPUS_ON_NODE
else
    TOTAL_CPUS=$(nproc)
fi

# Use half of CPUs for parallel workers
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

# Threading configuration
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "============================================================"
echo "SyntheticExperiment2 (Large-Scale Synthetic Benchmark)"
echo "============================================================"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start time: $(date)"
echo "============================================================"
echo "Default scale: n_neurons=80"
echo "Resource Configuration:"
echo "  Total CPUs: $TOTAL_CPUS"
echo "  Max Workers: $MAX_WORKERS"
echo "  GPU Count: $GPU_COUNT"
echo "  GPU Info: $GPU_INFO"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "============================================================"
echo "Extra args: $EXTRA_ARGS"
echo "============================================================"

module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true

# Virtual Environment
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
fi

source env/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

echo ""
echo "Checking dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Ensure torch can see CUDA on GPU jobs.
# On some clusters, a newer torch CUDA build (e.g., cu130) can be incompatible
# with the node driver (e.g., 12.8). We try cu128 wheels automatically.
if [ "$GPU_COUNT" -gt 0 ]; then
    echo ""
    echo "Checking PyTorch CUDA runtime compatibility..."
    python - <<'PY'
import torch
print(f"  torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
PY

    CUDA_OK=$(python - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)

    if [ "$CUDA_OK" != "1" ]; then
        echo "PyTorch cannot access CUDA on a GPU node; trying CUDA 12.8 wheels..."
        pip install --quiet --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio || true

        CUDA_OK=$(python - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)
        python - <<'PY'
import torch
print(f"  after-fix torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
PY
    fi

    if [ "$CUDA_OK" != "1" ]; then
        if [ "${ALLOW_CPU_FALLBACK:-0}" = "1" ]; then
            echo "WARNING: CUDA still unavailable; continuing on CPU because ALLOW_CPU_FALLBACK=1"
        else
            echo "ERROR: CUDA unavailable on GPU allocation; refusing CPU fallback."
            echo "Set ALLOW_CPU_FALLBACK=1 to continue on CPU, or use a torch build compatible with node drivers."
            exit 2
        fi
    fi
fi

# Optional baseline dependencies (best effort).
# Set INSTALL_OPTIONAL_BASELINES=0 to skip.
if [ "${INSTALL_OPTIONAL_BASELINES:-1}" = "1" ]; then
    echo ""
    echo "Installing optional classical baseline packages (best effort)..."
    pip install --quiet lingam || echo "lingam install failed; VAR-LiNGAM may be skipped"
    pip install --quiet tigramite || echo "tigramite install failed; PCMCI+ may be skipped"
    pip install --quiet causalnex || echo "causalnex install failed; DYNOTEARS may be skipped"
    pip install --quiet git+https://github.com/xunzheng/notears.git || echo "notears install failed; NOTEARS may be skipped"
fi

# External DL baselines prep (only when requested).
if [[ "$EXTRA_ARGS" == *"--run-external-baselines"* ]]; then
    echo ""
    echo "Preparing external DL baseline repos (NRI / NetFormer / LINT)..."
    EXT_DIR="merged_results/external_baselines"
    mkdir -p "$EXT_DIR"
    cd "$EXT_DIR" || exit 1

    if [ ! -d "nri" ]; then
        echo "Cloning NRI..."
        git clone https://github.com/ethanfetaya/NRI.git nri || true
    fi
    if [ ! -d "NetFormer" ]; then
        echo "Cloning NetFormer..."
        git clone https://github.com/johnlyzhou/NetFormer.git NetFormer || true
    fi
    if [ ! -d "lowrank_inference" ]; then
        echo "Cloning lowrank_inference..."
        git clone https://github.com/adrian-valente/lowrank_inference.git lowrank_inference || true
    fi

    cd "$PROJECT_DIR" || exit 1
    pip install --quiet einops pytorch-lightning || echo "Optional external deps install had issues"
fi

# Run experiment
python -u pipeline/syntheticexperiment2.py \
    --max-workers "$MAX_WORKERS" \
    --print-deps \
    $EXTRA_ARGS

EXIT_CODE=$?

# Verify external baseline execution when requested.
if [ "$EXIT_CODE" -eq 0 ] && [[ "$EXTRA_ARGS" == *"--run-external-baselines"* ]]; then
    echo ""
    echo "Validating external baseline outputs..."
    python - <<'PY'
from pathlib import Path
import pandas as pd
import sys

csv_path = Path("merged_results/external_baselines/evaluation_synthetic.csv")
if not csv_path.exists():
    print(f"ERROR: missing {csv_path}")
    sys.exit(3)

df = pd.read_csv(csv_path)
if df.empty:
    print(f"ERROR: {csv_path} is empty")
    sys.exit(3)

methods = set(df.get("Method", pd.Series(dtype=str)).dropna().astype(str).unique())
required = {"NRI", "NetFormer", "LINT"}
missing = required - methods
if missing:
    print(f"ERROR: external synthetic output missing methods: {sorted(missing)}")
    print(f"Found methods: {sorted(methods)}")
    sys.exit(3)

print(f"External baselines verified: {sorted(methods)}")
PY
    VERIFY_CODE=$?
    if [ "$VERIFY_CODE" -ne 0 ]; then
        EXIT_CODE=$VERIFY_CODE
    fi
fi

echo "============================================================"
echo "SyntheticExperiment2 completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

RESULT_DIR="sbtg_results_syntheticexp2"
if [[ "$EXTRA_ARGS" == *"--mini"* ]]; then
    RESULT_DIR="sbtg_results_syntheticexp2_mini"
elif [[ "$EXTRA_ARGS" == *"--medium"* ]]; then
    RESULT_DIR="sbtg_results_syntheticexp2_medium"
fi
if [ -d "$RESULT_DIR" ]; then
    echo ""
    echo "============================================================"
    echo "RESULTS SUMMARY"
    echo "============================================================"
    echo "Output directory: $RESULT_DIR"
    echo ""
    echo "Generated files:"
    ls -la "$RESULT_DIR/" 2>/dev/null

    if [ -f "$RESULT_DIR/metrics.csv" ]; then
        echo ""
        echo "--- Metrics Overview ---"
        echo "Total rows: $(wc -l < "$RESULT_DIR/metrics.csv")"
        echo "Methods evaluated:"
        tail -n +2 "$RESULT_DIR/metrics.csv" | cut -d',' -f5 | sort | uniq -c | sort -rn
    fi

    if [ -f "$RESULT_DIR/report.md" ]; then
        echo ""
        echo "--- Report Head ---"
        sed -n '1,80p' "$RESULT_DIR/report.md"
    fi

    if [ -f "$RESULT_DIR/report_complete.md" ]; then
        echo ""
        echo "--- Complete Report Head ---"
        sed -n '1,120p' "$RESULT_DIR/report_complete.md"
    fi

    echo ""
    echo "--- Generated Figures ---"
    ls -la "$RESULT_DIR"/*.png 2>/dev/null || echo "No figures generated"

    if [[ "$EXTRA_ARGS" == *"--run-external-baselines"* ]]; then
        echo ""
        echo "--- External Baseline Outputs ---"
        EXT_CSV_MERGED="merged_results/external_baselines/evaluation_synthetic.csv"
        EXT_CSV_RUN="$RESULT_DIR/external_baselines/evaluation_synthetic.csv"
        if [ -f "$EXT_CSV_MERGED" ]; then
            echo "Merged-results CSV: $EXT_CSV_MERGED"
            python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path("merged_results/external_baselines/evaluation_synthetic.csv")
df = pd.read_csv(p)
print(df.to_string(index=False))
PY
        else
            echo "Merged-results CSV not found: $EXT_CSV_MERGED"
        fi
        if [ -f "$EXT_CSV_RUN" ]; then
            echo "Run-local CSV copy: $EXT_CSV_RUN"
        else
            echo "Run-local CSV copy not found: $EXT_CSV_RUN"
        fi
    fi
fi

exit $EXIT_CODE
