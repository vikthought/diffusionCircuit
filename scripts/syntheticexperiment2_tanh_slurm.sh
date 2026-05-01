#!/bin/bash
#SBATCH --job-name=synthexp2_tanh
#SBATCH --output=logs/synthexp2_tanh_%j.log
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-cpu=4G
# ==============================================================================
# Per-family launcher: Tanh-VAR(2) DGP at n=80 (reviewer scaling response)
# ==============================================================================
# See syntheticexperiment2_var_slurm.sh for full rationale.
#
# Tanh-VAR is the family used for the headline synthetic table in main.tex
# (tab:synthetic). Worth running with extra HP trials if walltime allows.
#
# Grid: 1 family x 2 noise x 2 length x 1 seed = 4 variants per family.
#
# Usage:
#   sbatch scripts/syntheticexperiment2_tanh_slurm.sh
#   sbatch scripts/syntheticexperiment2_tanh_slurm.sh --n-seeds 2
#   sbatch scripts/syntheticexperiment2_tanh_slurm.sh --hp-trials 50
#
# Output: sbtg_results_syntheticexp2_tanh/
# ==============================================================================

mkdir -p logs

exec bash "$(dirname "$0")/syntheticexperiment2_slurm.sh" \
    --families tanh \
    --noise-levels low high \
    --length-types short long \
    --t-short 300 \
    --t-long 3000 \
    --hp-trials 30 \
    --sbtg-epochs 125 \
    --n-seeds 1 \
    --max-workers 2 \
    --output-dir ./sbtg_results_syntheticexp2_tanh \
    "$@"
