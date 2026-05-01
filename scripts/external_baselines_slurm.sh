#!/bin/bash
#SBATCH --job-name=ext_baselines
#SBATCH --output=logs/external_baselines_%j.log
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

# Configurable parameters
DATASET="${1:-nacl}"
EPOCHS="${2:-50}"

# 1. Setup Project Environment
PROJECT_DIR="/path/to/your/project"
cd "$PROJECT_DIR" || exit 1

echo "============================================================"
echo "External Deep Learning Baselines (NRI / NetFormer / LINT)"
echo "============================================================"
echo "Project:  $PROJECT_DIR"
echo "Dataset:  $DATASET"
echo "Epochs:   $EPOCHS"
echo "============================================================"

# 2. Load modules (adjust for your cluster)
# module load Python/3.x.x  # Adjust to your cluster

# 3. Virtual Environment
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
fi

source env/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# 4. Install Dependencies
echo ""
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 5. Threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 6. Logging
mkdir -p logs

# 7. GPU Status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "nvidia-smi not available"
echo ""

# 8. Clone external repos if not present
cd merged_results/external_baselines || exit 1

if [ ! -d "nri" ]; then
    echo "Cloning NRI ..."
    git clone https://github.com/ethanfetaya/NRI.git nri
fi

if [ ! -d "NetFormer" ]; then
    echo "Cloning NetFormer ..."
    git clone https://github.com/johnlyzhou/NetFormer.git NetFormer
fi

if [ ! -d "lowrank_inference" ]; then
    echo "Cloning lowrank_inference ..."
    git clone https://github.com/adrian-valente/lowrank_inference.git lowrank_inference
fi

cd "$PROJECT_DIR" || exit 1

# 9. Run external baselines on empirical data
echo "============================================================"
echo "Step 1/3: Training external baselines on $DATASET dataset"
echo "============================================================"
python merged_results/external_baselines/external_analysis.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch_size 64 \
    --lint_rank 5 \
    --window_size 10

# 10. Evaluate against Cook/Leifer ground truths
echo ""
echo "============================================================"
echo "Step 2/3: Evaluating against structural/functional ground truths"
echo "============================================================"
python merged_results/external_baselines/evaluate_external.py --dataset "$DATASET"

# 11. Synthetic benchmarks
echo ""
echo "============================================================"
echo "Step 3/3: Running synthetic benchmarks (VAR + Hawkes)"
echo "============================================================"
python merged_results/external_baselines/synthetic_analysis.py

echo ""
echo "============================================================"
echo "External Baselines Complete!"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - merged_results/external_baselines/external_results_${DATASET}.npz"
echo "  - merged_results/external_baselines/evaluation_${DATASET}.csv"
echo "  - merged_results/external_baselines/evaluation_synthetic.csv"
echo "============================================================"

exit 0
