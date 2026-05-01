#!/bin/bash
#SBATCH --job-name=synthetic_extra_run
#SBATCH --output=logs/synthetic_extra_run_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G


# ==============================================================================
# Synthetic Extra Run (fast): PCMCI+, DYNOTEARS, external DL baselines
# ==============================================================================
#
# Purpose:
#   Fast reviewer-facing run that skips SBTG and most classical baselines.
#   It evaluates only:
#     - PCMCI+
#     - DYNOTEARS
#     - External: NRI, NetFormer, LINT
#
# Usage:
#   sbatch scripts/synthetic_extra_run.sh
#   sbatch scripts/synthetic_extra_run.sh --seeds "0 1" --output-dir sbtg_results_synthetic_extra_run
#   sbatch scripts/synthetic_extra_run.sh --families "var poisson" --noise-levels "low high" --length-types "short long"
#
# Optional args:
#   --project-dir PATH
#   --output-dir DIR
#   --n-neurons INT
#   --m-stimuli INT
#   --seeds "0 1"
#   --families "var poisson"
#   --noise-levels "low high"
#   --length-types "short long"
#   --t-short INT
#   --t-long INT
#   --run-external 0|1
#   --external-epochs INT
#   --external-lint-epochs INT
#
# ==============================================================================

set -euo pipefail

# -------- defaults --------
PROJECT_DIR="/path/to/your/project"
OUTPUT_DIR="sbtg_results_synthetic_extra_run"
N_NEURONS=80
M_STIMULI=3
SEEDS="0 1"
FAMILIES="var poisson"
NOISE_LEVELS="low high"
LENGTH_TYPES="short long"
T_SHORT=300
T_LONG=800
RUN_EXTERNAL=1
EXTERNAL_EPOCHS=50
EXTERNAL_LINT_EPOCHS=100

# -------- arg parsing --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --n-neurons) N_NEURONS="$2"; shift 2 ;;
    --m-stimuli) M_STIMULI="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --families) FAMILIES="$2"; shift 2 ;;
    --noise-levels) NOISE_LEVELS="$2"; shift 2 ;;
    --length-types) LENGTH_TYPES="$2"; shift 2 ;;
    --t-short) T_SHORT="$2"; shift 2 ;;
    --t-long) T_LONG="$2"; shift 2 ;;
    --run-external) RUN_EXTERNAL="$2"; shift 2 ;;
    --external-epochs) EXTERNAL_EPOCHS="$2"; shift 2 ;;
    --external-lint-epochs) EXTERNAL_LINT_EPOCHS="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

cd "$PROJECT_DIR" || exit 1
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    TOTAL_CPUS=$SLURM_CPUS_PER_TASK
elif [ -n "${SLURM_CPUS_ON_NODE:-}" ]; then
    TOTAL_CPUS=$SLURM_CPUS_ON_NODE
else
    TOTAL_CPUS=$(nproc)
fi

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | sed -n '1p' || echo "0")
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed -n '1p' || echo "None")
else
    GPU_COUNT=0
    GPU_INFO="None"
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
elif [ "$GPU_COUNT" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "============================================================"
echo "Synthetic Extra Run (PCMCI+, DYNOTEARS, external)"
echo "============================================================"
echo "Running on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start time: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "n_neurons: $N_NEURONS | m_stimuli: $M_STIMULI"
echo "families: $FAMILIES"
echo "noise levels: $NOISE_LEVELS"
echo "length types: $LENGTH_TYPES (short=$T_SHORT, long=$T_LONG)"
echo "seeds: $SEEDS"
echo "run external: $RUN_EXTERNAL"
echo "CPU: $TOTAL_CPUS | GPU count: $GPU_COUNT | GPU: $GPU_INFO"
echo "============================================================"

module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true

if [ ! -d "env" ]; then
    python3 -m venv env
fi
source env/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python --version)"

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Optional packages needed for requested baselines
pip install --quiet tigramite || echo "tigramite install failed; PCMCI+ may be skipped"
pip install --quiet causalnex || echo "causalnex install failed; DYNOTEARS may be skipped"

python - <<'PY'
import torch
print(f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
PY

if [ "$RUN_EXTERNAL" = "1" ]; then
  EXT_DIR="merged_results/external_baselines"
  mkdir -p "$EXT_DIR"
  cd "$EXT_DIR" || exit 1
  if [ ! -d "nri" ]; then
    git clone https://github.com/ethanfetaya/NRI.git nri || true
  fi
  if [ ! -d "NetFormer" ]; then
    git clone https://github.com/johnlyzhou/NetFormer.git NetFormer || true
  fi
  if [ ! -d "lowrank_inference" ]; then
    git clone https://github.com/adrian-valente/lowrank_inference.git lowrank_inference || true
  fi
  cd "$PROJECT_DIR" || exit 1
  pip install --quiet einops pytorch-lightning || echo "optional external deps had issues"
fi

export EXTRA_OUT_DIR="$OUTPUT_DIR"
export EXTRA_N_NEURONS="$N_NEURONS"
export EXTRA_M_STIMULI="$M_STIMULI"
export EXTRA_SEEDS="$SEEDS"
export EXTRA_FAMILIES="$FAMILIES"
export EXTRA_NOISE_LEVELS="$NOISE_LEVELS"
export EXTRA_LENGTH_TYPES="$LENGTH_TYPES"
export EXTRA_T_SHORT="$T_SHORT"
export EXTRA_T_LONG="$T_LONG"

python -u - <<'PY'
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.SyntheticTestingUtils import (
    _generate_dataset,
    pcmci_plus_baseline,
    dynotears_baseline,
    evaluate_binary,
    evaluate_weighted,
    TimeoutError,
)

out_dir = Path(os.environ["EXTRA_OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

n_neurons = int(os.environ["EXTRA_N_NEURONS"])
m_stimuli = int(os.environ["EXTRA_M_STIMULI"])
seeds = [int(s) for s in os.environ["EXTRA_SEEDS"].split() if s.strip()]
families = [s for s in os.environ["EXTRA_FAMILIES"].split() if s.strip()]
noise_levels = [s for s in os.environ["EXTRA_NOISE_LEVELS"].split() if s.strip()]
length_types = [s for s in os.environ["EXTRA_LENGTH_TYPES"].split() if s.strip()]
t_short = int(os.environ["EXTRA_T_SHORT"])
t_long = int(os.environ["EXTRA_T_LONG"])

metrics_rows = []
status_rows = []

def run_one(name, fn, truth_graph, is_binary=False, **kwargs):
    t0 = time.time()
    try:
        result = fn(**kwargs)
        dt = time.time() - t0
        if is_binary:
            m = evaluate_binary(truth_graph, result.astype(int))
        else:
            m = evaluate_weighted(truth_graph, result, threshold=1e-6)
        return {
            "status": "SUCCESS",
            "time_seconds": dt,
            "error": "",
            "metrics": m,
        }
    except TimeoutError as e:
        return {"status": "TIMEOUT", "time_seconds": time.time() - t0, "error": str(e), "metrics": None}
    except Exception as e:
        return {"status": "FAILED", "time_seconds": time.time() - t0, "error": str(e), "metrics": None}

total = len(families) * len(noise_levels) * len(length_types) * len(seeds)
done = 0
print(f"[EXTRA] Running {total} dataset variants for PCMCI+/DYNOTEARS.")

for family in families:
    for noise in noise_levels:
        for length in length_types:
            T = t_short if length == "short" else t_long
            for seed in seeds:
                done += 1
                print(f"[EXTRA] [{done}/{total}] family={family} noise={noise} length={length} seed={seed}", flush=True)
                X_list, truth_dict = _generate_dataset(
                    family=family,
                    n=n_neurons,
                    T=T,
                    m_stim=m_stimuli,
                    noise_level=noise,
                    seed=seed,
                )
                truth_graph = truth_dict[1]

                runs = [
                    ("PCMCI+", run_one(
                        "PCMCI+",
                        pcmci_plus_baseline,
                        truth_graph,
                        is_binary=True,
                        X_list=X_list,
                        tau_max=1,
                        alpha_level=0.05,
                    )),
                    ("DYNOTEARS", run_one(
                        "DYNOTEARS",
                        dynotears_baseline,
                        truth_graph,
                        is_binary=False,
                        X_list=X_list,
                    )),
                ]

                for method, res in runs:
                    status_rows.append({
                        "family": family,
                        "noise": noise,
                        "length": length,
                        "seed": seed,
                        "method": method,
                        "status": res["status"],
                        "time_seconds": res["time_seconds"],
                        "error": res["error"],
                    })

                    if res["metrics"] is not None:
                        m = res["metrics"]
                        metrics_rows.append({
                            "family": family,
                            "noise": noise,
                            "length": length,
                            "seed": seed,
                            "method": method,
                            "stat_cfg_name": "default",
                            "precision": m["precision"],
                            "recall": m["recall"],
                            "f1": m["f1"],
                            "roc_auc": m["roc_auc"],
                            "pr_auc": m["pr_auc"],
                            "pred_edges": m.get("pred_edges", np.nan),
                            "true_edges": m.get("true_edges", np.nan),
                            "pred_density": m.get("pred_density", np.nan),
                            "true_density": m.get("true_density", np.nan),
                        })
                        print(
                            f"  [EXTRA] {method}: status={res['status']} "
                            f"F1={m['f1']:.3f} AUROC={m['roc_auc']:.3f} pred_edges={m.get('pred_edges', 0)}",
                            flush=True,
                        )
                    else:
                        print(
                            f"  [EXTRA] {method}: status={res['status']} error={res['error']}",
                            flush=True,
                        )

metrics_df = pd.DataFrame(metrics_rows)
status_df = pd.DataFrame(status_rows)

metrics_csv = out_dir / "extra_metrics.csv"
status_csv = out_dir / "extra_baseline_status.csv"
metrics_df.to_csv(metrics_csv, index=False)
status_df.to_csv(status_csv, index=False)

summary = {}
if not metrics_df.empty:
    summary = (
        metrics_df.groupby(["family", "method"], as_index=False)[["precision", "recall", "f1", "roc_auc", "pr_auc"]]
        .mean()
        .sort_values(["family", "method"])
        .to_dict(orient="records")
    )

summary_path = out_dir / "extra_summary.json"
summary_path.write_text(json.dumps({"aggregated_mean_metrics": summary}, indent=2))

print(f"[EXTRA] Saved metrics: {metrics_csv}")
print(f"[EXTRA] Saved status: {status_csv}")
print(f"[EXTRA] Saved summary: {summary_path}")
PY

EXIT_CODE=$?

if [ "$RUN_EXTERNAL" = "1" ] && [ "$EXIT_CODE" -eq 0 ]; then
  python -u merged_results/external_baselines/synthetic_analysis.py \
    --n-neurons "$N_NEURONS" \
    --T "$T_LONG" \
    --noise-level low \
    --seed 0 \
    --epochs "$EXTERNAL_EPOCHS" \
    --lint-epochs "$EXTERNAL_LINT_EPOCHS" \
    --families VAR Hawkes

  EXT_EXIT=$?
  if [ "$EXT_EXIT" -ne 0 ]; then
    EXIT_CODE=$EXT_EXIT
  else
    mkdir -p "$OUTPUT_DIR/external_baselines"
    if [ -f "merged_results/external_baselines/evaluation_synthetic.csv" ]; then
      cp "merged_results/external_baselines/evaluation_synthetic.csv" "$OUTPUT_DIR/external_baselines/evaluation_synthetic.csv"
      echo "[EXTRA] Copied external CSV to $OUTPUT_DIR/external_baselines/evaluation_synthetic.csv"
    fi
  fi
fi

echo "============================================================"
echo "Synthetic Extra Run completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

if [ -d "$OUTPUT_DIR" ]; then
  echo "Output directory: $OUTPUT_DIR"
  ls -la "$OUTPUT_DIR"
fi

exit $EXIT_CODE
