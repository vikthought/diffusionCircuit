#!/bin/bash
#SBATCH --job-name=synthexp2_hawkes
#SBATCH --output=logs/synthexp2_hawkes_%j.log
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-cpu=4G
# ==============================================================================
# Per-family launcher: Hawkes-like DGP at n=80 (reviewer scaling response)
# ==============================================================================
# See syntheticexperiment2_var_slurm.sh for full rationale.
#
# Grid: 1 family x 2 noise x 2 length x 1 seed = 4 variants per family.
#
# Usage:
#   sbatch scripts/syntheticexperiment2_hawkes_slurm.sh
#   sbatch scripts/syntheticexperiment2_hawkes_slurm.sh --n-seeds 2
#
# Output: sbtg_results_syntheticexp2_hawkes/
# ==============================================================================

mkdir -p logs

exec bash "$(dirname "$0")/syntheticexperiment2_slurm.sh" \
    --families hawkes \
    --noise-levels low high \
    --length-types short long \
    --t-short 300 \
    --t-long 3000 \
    --hp-trials 30 \
    --sbtg-epochs 125 \
    --n-seeds 1 \
    --max-workers 2 \
    --output-dir ./sbtg_results_syntheticexp2_hawkes \
    "$@"
