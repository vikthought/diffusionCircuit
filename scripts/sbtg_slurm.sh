#!/bin/bash
#SBATCH --job-name=validation_longer
#SBATCH --output=logs/longer_objective_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

# Parse command line args
N_TRIALS=200

# 1. Setup Project Environment
PROJECT_DIR="/path/to/your/project"
cd "$PROJECT_DIR" || exit 1

echo "============================================================"
echo "SBTG Pipeline with Null Contrast Objective"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "N_TRIALS: $N_TRIALS"
echo "============================================================"

# 2. Load modules (adjust for your cluster)
# module load Python/3.x.x  # Adjust to your cluster

# 3. Virtual Environment (Create if missing)
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

# 5. Environment Variables for Threading
# Pin BLAS threads to 1 to avoid oversubscription when using n_jobs > 1
# Each trial (n_jobs) will run on 1 CPU core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# 6. Create logs directory
mkdir -p logs

# 7. Report GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo "nvidia-smi not available"
echo ""

# 8. Run the full pipeline with null_contrast objective
echo "============================================================"
echo "Running pipeline with null_contrast objective and $N_TRIALS HP trials"
echo "============================================================"

./run_pipeline.sh all --trials $N_TRIALS

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - results/sbtg_training/       (trained models)"
echo "  - results/figures/             (visualizations)"
echo "  - results/tables/              (CSV tables)"
echo "  - results/evaluation/          (AUROC/AUPRC metrics)"
echo "  - results/hp_objective_validation/  (HP validation if run)"
echo "============================================================"

exit 0
