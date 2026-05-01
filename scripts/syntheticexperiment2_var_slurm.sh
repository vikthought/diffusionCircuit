#!/bin/bash
#SBATCH --job-name=synthexp2_var
#SBATCH --output=logs/synthexp2_var_%j.log
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --mail-type=ALL
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-cpu=4G
# ==============================================================================
# Per-family launcher: VAR(2) DGP at n=80 (reviewer scaling response)
# ==============================================================================
#
# Reviewer pushback addressed:
#   "Synthetic experiments used n=10, much smaller than the 80-neuron empirical
#   regime; unclear how the method performs when scaled up."
#
# Why the prior --medium n=80 attempt collapsed to chance:
#   - T_long=800, m_stim=3 -> ~2.4k pooled windows for 6,320 directed edges
#     (~0.4 windows/edge; HAC + BY FDR has no signal to work with)
#   - hp_trials=10 (TPE startup is 20, so this is pure random search)
#   - sbtg_epochs=50 with 8x more parameters than n=10 -> undertrained
#
# This wrapper restores statistical power per the reviewer-grade recipe:
#   - T_long=3000 -> ~9k pooled windows (~1.4 windows/edge)
#   - hp_trials=30 (10 Bayesian after 20 random startup)
#   - sbtg_epochs=125 (matches syntheticexperiment2.py default)
#
# Grid: 1 family x 2 noise x 2 length x 1 seed = 4 variants per family.
# Bump --n-seeds 2 (or pass --seeds 0 1) if walltime allows.
#
# Usage:
#   sbatch scripts/syntheticexperiment2_var_slurm.sh
#   sbatch scripts/syntheticexperiment2_var_slurm.sh --n-seeds 2
#   sbatch scripts/syntheticexperiment2_var_slurm.sh --skip-baselines
#   sbatch scripts/syntheticexperiment2_var_slurm.sh --t-long 5000 --hp-trials 50
#
# Output: sbtg_results_syntheticexp2_var/
# ==============================================================================

mkdir -p logs

exec bash "$(dirname "$0")/syntheticexperiment2_slurm.sh" \
    --families var \
    --noise-levels low high \
    --length-types short long \
    --t-short 300 \
    --t-long 3000 \
    --hp-trials 30 \
    --sbtg-epochs 125 \
    --n-seeds 1 \
    --max-workers 2 \
    --output-dir ./sbtg_results_syntheticexp2_var \
    "$@"
