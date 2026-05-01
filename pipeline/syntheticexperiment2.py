"""
SyntheticExperiment2 - large-scale synthetic benchmark for reviewer response.

This entrypoint reuses the existing synthetic benchmark implementation while
changing defaults to a larger neuron count (80) and exposing explicit controls
for seed scheduling and scale sweeps.

Usage:
    python pipeline/syntheticexperiment2.py
    python pipeline/syntheticexperiment2.py --n-neurons 80 --n-seeds 4
    python pipeline/syntheticexperiment2.py --mini --skip-baselines --n-neurons 80
    python pipeline/syntheticexperiment2.py --medium --n-neurons 80   # overnight-friendly + baselines
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import itertools
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.SyntheticTesting import run_benchmarks
from pipeline.SyntheticTestingUtils import (
    CPU_COUNT,
    CUDA_AVAILABLE,
    DEVICE,
    GPU_COUNT,
    GPU_NAME,
    N_HP_TRIALS,
    SBTG_A_NAME,
    check_dependencies,
)


def _resolve_seeds(seeds_arg: List[int] | None, n_seeds: int) -> List[int]:
    """Resolve explicit seed list or generate range(n_seeds)."""
    if seeds_arg:
        return seeds_arg
    if n_seeds < 1:
        raise ValueError("--n-seeds must be >= 1 when --seeds is not provided.")
    return list(range(n_seeds))


def _run_external_baselines(
    n_neurons: int,
    t_long: int,
    output_dir: str,
    external_epochs: int,
) -> None:
    """Run optional external DL synthetic baselines."""
    ext_script = PROJECT_ROOT / "merged_results" / "external_baselines" / "synthetic_analysis.py"
    if not ext_script.exists():
        print(f"[EXTERNAL] Skipping: script not found at {ext_script}", flush=True)
        return

    cmd = [
        sys.executable,
        str(ext_script),
        "--n-neurons",
        str(n_neurons),
        "--T",
        str(t_long),
        "--noise-level",
        "low",
        "--seed",
        "0",
        "--epochs",
        str(external_epochs),
        "--lint-epochs",
        str(external_epochs * 2),
        "--families",
        "VAR",
        "Hawkes",
    ]
    print(f"[EXTERNAL] Running: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
        print("[EXTERNAL] Completed external synthetic baselines.", flush=True)
        ext_csv = PROJECT_ROOT / "merged_results" / "external_baselines" / "evaluation_synthetic.csv"
        print(f"[EXTERNAL] Expected output: {ext_csv}", flush=True)

        # Mirror external outputs into this run's output directory for discoverability.
        out_ext_dir = Path(output_dir) / "external_baselines"
        out_ext_dir.mkdir(parents=True, exist_ok=True)
        if ext_csv.exists():
            dst = out_ext_dir / "evaluation_synthetic.csv"
            shutil.copy2(ext_csv, dst)
            print(f"[EXTERNAL] Copied summary to: {dst}", flush=True)
        else:
            print("[EXTERNAL] Warning: evaluation_synthetic.csv not found after run.", flush=True)
    except Exception as exc:
        print(f"[EXTERNAL] Failed to run external baselines: {exc}", flush=True)
        print(
            "[EXTERNAL] This is non-fatal for syntheticexperiment2; "
            "install external baseline deps/repos if needed.",
            flush=True,
        )


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render DataFrame as markdown with csv fallback."""
    if df.empty:
        return "_No rows._"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```csv\n" + df.to_csv(index=False) + "```"


def _write_complete_report(
    output_dir: str,
    n_neurons: int,
    m_stimuli: int,
    families: List[str],
    noise_levels: List[str],
    length_types: List[str],
    seeds: List[int],
    n_hp_trials: int,
    sbtg_epochs: int,
    sbtg_method_names: List[str],
    run_baselines: bool,
    run_external_baselines: bool,
) -> None:
    """Create a complete, reviewer-facing report with all available details."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.csv"
    baseline_status_path = out_dir / "baseline_status.csv"
    best_params_path = out_dir / "best_params.json"
    external_path_local = out_dir / "external_baselines" / "evaluation_synthetic.csv"
    external_path_merged = PROJECT_ROOT / "merged_results" / "external_baselines" / "evaluation_synthetic.csv"
    external_path = external_path_local if external_path_local.exists() else external_path_merged

    lines: List[str] = []
    lines.append("# SyntheticExperiment2 Complete Report\n")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`\n")
    lines.append("## Configuration\n")
    lines.append(f"- n_neurons: `{n_neurons}`")
    lines.append(f"- m_stimuli: `{m_stimuli}`")
    lines.append(f"- families: `{families}`")
    lines.append(f"- noise_levels: `{noise_levels}`")
    lines.append(f"- length_types: `{length_types}`")
    lines.append(f"- seeds: `{seeds}`")
    lines.append(f"- hp_trials: `{n_hp_trials}`")
    lines.append(f"- sbtg_epochs: `{sbtg_epochs}`")
    lines.append(f"- sbtg_methods: `{sbtg_method_names}`")
    lines.append(f"- classical_baselines_enabled: `{run_baselines}`")
    lines.append(f"- external_baselines_requested: `{run_external_baselines}`\n")

    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        metrics_eval_df = metrics_df.copy()
        if "status" in metrics_eval_df.columns:
            is_sbtg = metrics_eval_df["method"].astype(str).str.startswith("SBTG-")
            sbtg_success = (
                metrics_eval_df["status"]
                .fillna("")
                .astype(str)
                .str.lower()
                .eq("success")
            )
            metrics_eval_df = metrics_eval_df[(~is_sbtg) | sbtg_success].copy()
        lines.append("## Metrics Overview (SBTG + classical baselines)\n")
        lines.append(f"- metrics rows: `{len(metrics_df)}`")
        lines.append(f"- metrics columns: `{list(metrics_df.columns)}`\n")
        lines.append(
            f"- summary/evaluation rows used (success-filtered): `{len(metrics_eval_df)}` "
            "(SBTG rows with non-success status are excluded from aggregates)\n"
        )

        expected_num = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
        for col in expected_num:
            if col in metrics_eval_df.columns:
                metrics_eval_df[col] = pd.to_numeric(metrics_eval_df[col], errors="coerce")
            if col in metrics_df.columns:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

        group_cols = [c for c in ["family", "method"] if c in metrics_df.columns]
        if group_cols:
            summary_df = (
                metrics_eval_df.groupby(group_cols, dropna=False)
                .agg(
                    rows=("method", "size"),
                    precision_mean=("precision", "mean"),
                    recall_mean=("recall", "mean"),
                    f1_mean=("f1", "mean"),
                    roc_auc_mean=("roc_auc", "mean"),
                    pr_auc_mean=("pr_auc", "mean"),
                )
                .reset_index()
                .sort_values(group_cols)
            )
            lines.append("### Aggregated Performance by Family/Method (success-filtered)\n")
            lines.append(_df_to_markdown(summary_df))
            lines.append("")

        lines.append("### Full Metrics Table (all rows)\n")
        metrics_detail_cols = [
            c
            for c in [
                "family",
                "noise",
                "length",
                "seed",
                "method",
                "stat_cfg_name",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "status",
                "time_seconds",
                "edges_mean",
                "edges_energy_only",
                "edges_volatility_only",
                "edges_union",
                "auroc_lag1",
                "auroc_lag2",
            ]
            if c in metrics_df.columns
        ]
        metrics_detail_df = metrics_df[metrics_detail_cols].sort_values(
            [c for c in ["family", "noise", "length", "seed", "method", "stat_cfg_name"] if c in metrics_detail_cols]
        )
        lines.append(_df_to_markdown(metrics_detail_df))
        lines.append("")
    else:
        lines.append(f"## Metrics\nMissing file: `{metrics_path}`\n")

    if baseline_status_path.exists():
        status_df = pd.read_csv(baseline_status_path)
        lines.append("## Classical Baseline Status and Failures\n")
        lines.append(f"- baseline status rows: `{len(status_df)}`\n")

        if "time_seconds" in status_df.columns:
            status_df["time_seconds"] = pd.to_numeric(status_df["time_seconds"], errors="coerce")

        status_summary = (
            status_df.groupby(["method", "status"], dropna=False)
            .agg(count=("status", "size"), total_time_seconds=("time_seconds", "sum"))
            .reset_index()
            .sort_values(["method", "status"])
        )
        lines.append("### Status Counts by Method\n")
        lines.append(_df_to_markdown(status_summary))
        lines.append("")

        status_detail_cols = [
            c for c in ["family", "noise", "length", "seed", "method", "status", "time_seconds", "error"] if c in status_df.columns
        ]
        status_detail_df = status_df[status_detail_cols].sort_values(
            [c for c in ["family", "noise", "length", "seed", "method"] if c in status_detail_cols]
        )
        lines.append("### Full Baseline Status Table (all rows)\n")
        lines.append(_df_to_markdown(status_detail_df))
        lines.append("")
    else:
        lines.append(f"## Classical Baseline Status\nMissing file: `{baseline_status_path}`\n")

    # --------------------------
    # Success-rate and common-success analysis
    # --------------------------
    lines.append("## Method Success Rates and Common-Success Subsets\n")
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
        sbtg_status_df = pd.DataFrame(columns=["family", "noise", "length", "seed", "method", "status"])
        if not metrics_df.empty:
            sbtg_rows = metrics_df[
                metrics_df["method"].astype(str).str.startswith("SBTG-")
            ].copy()
            if not sbtg_rows.empty and "status" in sbtg_rows.columns:
                sbtg_status_df = (
                    sbtg_rows.groupby(["family", "noise", "length", "seed", "method"], as_index=False)
                    .agg(status=("status", lambda s: "SUCCESS" if (s.astype(str).str.lower() == "success").any() else "FAILED"))
                )

        baseline_status_df = pd.read_csv(baseline_status_path) if baseline_status_path.exists() else pd.DataFrame(
            columns=["family", "noise", "length", "seed", "method", "status", "error", "time_seconds"]
        )

        status_union = pd.concat(
            [
                sbtg_status_df[["family", "noise", "length", "seed", "method", "status"]],
                baseline_status_df[["family", "noise", "length", "seed", "method", "status"]],
            ],
            ignore_index=True,
        )
        if not status_union.empty:
            status_union["status"] = status_union["status"].astype(str).str.upper()

            all_variants = pd.DataFrame(
                list(itertools.product(families, noise_levels, length_types, seeds)),
                columns=["family", "noise", "length", "seed"],
            )
            methods = sorted(status_union["method"].dropna().astype(str).unique().tolist())
            grid = all_variants.assign(_k=1).merge(
                pd.DataFrame({"method": methods, "_k": 1}), on="_k"
            ).drop(columns="_k")
            status_full = grid.merge(
                status_union,
                on=["family", "noise", "length", "seed", "method"],
                how="left",
            )
            status_full["status"] = status_full["status"].fillna("MISSING")
            status_full["attempted"] = ~status_full["status"].isin(["SKIPPED", "MISSING"])
            status_full["success"] = status_full["status"] == "SUCCESS"

            success_rates = (
                status_full.groupby(["method"], as_index=False)
                .agg(
                    total_variants=("method", "size"),
                    attempted_variants=("attempted", "sum"),
                    successful_variants=("success", "sum"),
                )
            )
            success_rates["coverage"] = (
                success_rates["attempted_variants"] / success_rates["total_variants"]
            )
            success_rates["success_rate_given_attempted"] = success_rates.apply(
                lambda r: (r["successful_variants"] / r["attempted_variants"]) if r["attempted_variants"] > 0 else float("nan"),
                axis=1,
            )
            success_rates_path = out_dir / "method_success_rates.csv"
            success_rates.to_csv(success_rates_path, index=False)
            lines.append(f"- Saved method success rates: `{success_rates_path.name}`\n")
            lines.append("### Method Success Rates\n")
            lines.append(_df_to_markdown(success_rates.sort_values("method")))
            lines.append("")

            # Family-specific common-success subset comparisons.
            common_rows: List[pd.DataFrame] = []
            lines.append("### Common-Success Subset Aggregates (by family)\n")
            for fam in families:
                fam_status = status_full[status_full["family"] == fam].copy()
                fam_metrics = metrics_df[metrics_df["family"] == fam].copy()
                if fam_metrics.empty:
                    lines.append(f"#### {fam}\n_No metrics rows._\n")
                    continue

                fam_methods = sorted(
                    fam_status.groupby("method")["attempted"].sum().loc[lambda x: x > 0].index.tolist()
                )
                if not fam_methods:
                    lines.append(f"#### {fam}\n_No attempted methods for this family._\n")
                    continue

                fam_status = fam_status[fam_status["method"].isin(fam_methods)]
                fam_variant_success = (
                    fam_status.groupby(["family", "noise", "length", "seed"])["success"]
                    .all()
                    .reset_index()
                )
                common_variants = fam_variant_success[fam_variant_success["success"]].drop(columns="success")
                common_count = len(common_variants)
                total_count = len(fam_variant_success)
                lines.append(f"#### {fam}")
                lines.append(f"- common-success variants: `{common_count}/{total_count}`")
                lines.append(f"- methods in common-success gate: `{fam_methods}`\n")

                if common_count == 0:
                    lines.append("_No common-success variants for this family._\n")
                    continue

                fam_common_metrics = fam_metrics.merge(
                    common_variants,
                    on=["family", "noise", "length", "seed"],
                    how="inner",
                )
                fam_common_metrics = fam_common_metrics[fam_common_metrics["method"].isin(fam_methods)]
                if "status" in fam_common_metrics.columns:
                    is_sbtg_rows = fam_common_metrics["method"].astype(str).str.startswith("SBTG-")
                    sbtg_ok = (
                        fam_common_metrics["status"]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        .eq("success")
                    )
                    fam_common_metrics = fam_common_metrics[(~is_sbtg_rows) | sbtg_ok]
                if fam_common_metrics.empty:
                    lines.append("_No metric rows available on common-success subset._\n")
                    continue

                for col in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
                    if col in fam_common_metrics.columns:
                        fam_common_metrics[col] = pd.to_numeric(fam_common_metrics[col], errors="coerce")

                agg = (
                    fam_common_metrics.groupby("method", as_index=False)
                    .agg(
                        rows=("method", "size"),
                        precision_mean=("precision", "mean"),
                        recall_mean=("recall", "mean"),
                        f1_mean=("f1", "mean"),
                        roc_auc_mean=("roc_auc", "mean"),
                        pr_auc_mean=("pr_auc", "mean"),
                    )
                    .sort_values("method")
                )
                lines.append(_df_to_markdown(agg))
                lines.append("")
                common_rows.append(fam_common_metrics)

            if common_rows:
                common_df = pd.concat(common_rows, ignore_index=True)
                common_path = out_dir / "metrics_common_success.csv"
                common_df.to_csv(common_path, index=False)
                lines.append(f"- Saved common-success rows: `{common_path.name}`\n")
        else:
            lines.append("_Status union table is empty; cannot compute success rates._\n")

    lines.append("## External Baselines (NRI / NetFormer / LINT)\n")
    if external_path.exists():
        ext_df = pd.read_csv(external_path)
        lines.append(f"- source: `{external_path}`")
        lines.append(f"- rows: `{len(ext_df)}`")
        lines.append(f"- columns: `{list(ext_df.columns)}`\n")
        lines.append("### External Baseline Metrics (all rows)\n")
        lines.append(_df_to_markdown(ext_df))
        lines.append("")
    else:
        lines.append(
            f"External baseline CSV not found at `{external_path_local}` or `{external_path_merged}`.\n"
        )

    if best_params_path.exists():
        lines.append("## Best SBTG Params JSON\n")
        try:
            best_params = json.loads(best_params_path.read_text())
            lines.append("```json")
            lines.append(json.dumps(best_params, indent=2))
            lines.append("```\n")
        except Exception as exc:
            lines.append(f"Could not parse `{best_params_path}`: `{exc}`\n")

    lines.append("## Artifact Index\n")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            lines.append(f"- `{p.relative_to(out_dir)}`")

    complete_report_path = out_dir / "report_complete.md"
    complete_report_path.write_text("\n".join(lines) + "\n")
    print(f"[INFO] Saved complete report to {complete_report_path}", flush=True)


def _augment_compact_report(
    output_dir: str,
    run_external_baselines: bool,
) -> None:
    """Append success-rate and external baseline sections to report.md."""
    out_dir = Path(output_dir)
    report_path = out_dir / "report.md"
    if not report_path.exists():
        return

    lines: List[str] = []
    success_rates_path = out_dir / "method_success_rates.csv"
    if success_rates_path.exists():
        df_sr = pd.read_csv(success_rates_path)
        lines.append("\n## Method Success Rates\n")
        lines.append(_df_to_markdown(df_sr.sort_values("method")))
        lines.append("")

    if run_external_baselines:
        ext_path_local = out_dir / "external_baselines" / "evaluation_synthetic.csv"
        ext_path_merged = PROJECT_ROOT / "merged_results" / "external_baselines" / "evaluation_synthetic.csv"
        ext_path = ext_path_local if ext_path_local.exists() else ext_path_merged
        lines.append("## External Baselines (NRI / NetFormer / LINT)\n")
        if ext_path.exists():
            ext_df = pd.read_csv(ext_path)
            lines.append(f"- Source: `{ext_path}`")
            lines.append(f"- Rows: `{len(ext_df)}`\n")
            lines.append(_df_to_markdown(ext_df))
            lines.append("")
        else:
            lines.append(
                f"External baseline CSV not found at `{ext_path_local}` or `{ext_path_merged}`.\n"
            )

    if lines:
        with report_path.open("a") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[INFO] Updated compact report with extended sections: {report_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run larger-scale synthetic SBTG experiments (reviewer-focused)."
    )
    parser.add_argument(
        "--n-neurons",
        type=int,
        default=80,
        help="Number of neurons in synthetic data (default: 80).",
    )
    parser.add_argument(
        "--m-stimuli",
        type=int,
        default=3,
        help="Number of synthetic stimuli/conditions (default: 3).",
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=["var", "poisson", "hawkes", "tanh"],
        help="Synthetic DGP families to run.",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        nargs="+",
        default=["low", "high"],
        help="Noise regimes to run.",
    )
    parser.add_argument(
        "--length-types",
        type=str,
        nargs="+",
        choices=["short", "long"],
        default=["short", "long"],
        help="Sequence-length categories to run.",
    )
    parser.add_argument(
        "--t-short",
        type=int,
        default=300,
        help="Sequence length for short setting (default: 300).",
    )
    parser.add_argument(
        "--t-long",
        type=int,
        default=800,
        help="Sequence length for long setting (default: 800).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed list (overrides --n-seeds).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=2,
        help="Number of seeds when --seeds is not set (default: 2 -> [0,1]).",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline methods, only evaluate SBTG variants.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: auto).",
    )
    parser.add_argument(
        "--hp-trials",
        type=int,
        default=None,
        help=(
            f"Optuna trials per SBTG variant (default: {N_HP_TRIALS} full grid; "
            f"10 with --medium if omitted)."
        ),
    )
    parser.add_argument(
        "--sbtg-epochs",
        type=int,
        default=125,
        help=(
            "SBTG training epochs used for both HP tuning objective fits and "
            "final model fitting (default: 125)."
        ),
    )
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument(
        "--mini",
        action="store_true",
        help="Reduced smoke-test configuration (still honors --n-neurons).",
    )
    preset.add_argument(
        "--medium",
        action="store_true",
        help=(
            "Two DGP families (var, poisson), low/high noise, short/long lengths, "
            "baselines on; T_short/T_long stay 300/800. Default seeds [0,1] unless "
            "--seeds is set. Default HP trials=10 if --hp-trials omitted. "
            "Output: ./sbtg_results_syntheticexp2_medium."
        ),
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check baseline availability and exit.",
    )
    parser.add_argument(
        "--print-deps",
        action="store_true",
        help="Print optional baseline dependency status before running.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose output for baseline execution.",
    )
    parser.add_argument(
        "--run-external-baselines",
        action="store_true",
        help=(
            "Run optional external DL baselines (NRI/NetFormer/LINT) after "
            "the main synthetic benchmark."
        ),
    )
    parser.add_argument(
        "--external-epochs",
        type=int,
        default=50,
        help="Epochs for external baselines when --run-external-baselines is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./sbtg_results_syntheticexp2).",
    )
    args = parser.parse_args()

    if args.check_deps:
        check_dependencies()
        sys.exit(0)
    if args.print_deps:
        check_dependencies()

    print("=" * 80, flush=True)
    print(
        f"[RESOURCES] CPU cores: {CPU_COUNT}, GPU available: {CUDA_AVAILABLE}",
        flush=True,
    )
    if CUDA_AVAILABLE:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(
            f"[RESOURCES] GPU: {GPU_NAME} ({gpu_mem:.1f} GB), count: {GPU_COUNT}",
            flush=True,
        )
    print(f"[RESOURCES] PyTorch {torch.__version__}, device: {DEVICE}", flush=True)
    print("=" * 80, flush=True)

    dataset_families = args.families
    n_neurons = args.n_neurons
    m_stimuli = args.m_stimuli
    sbtg_method_names = [SBTG_A_NAME]
    seeds = _resolve_seeds(args.seeds, args.n_seeds)
    noise_levels = args.noise_levels
    length_types = args.length_types
    t_short = args.t_short
    t_long = args.t_long
    output_dir = args.output_dir or "./sbtg_results_syntheticexp2"

    if CUDA_AVAILABLE:
        default_workers = max(1, min(CPU_COUNT - 1, 8))
    else:
        default_workers = max(1, CPU_COUNT // 2)
    max_workers = args.max_workers if args.max_workers is not None else default_workers

    if args.mini:
        dataset_families = ["var"]
        noise_levels = ["low"]
        length_types = ["short"]
        seeds = [0]
        t_short = 120
        t_long = 120
        output_dir = args.output_dir or "./sbtg_results_syntheticexp2_mini"
    elif args.medium:
        dataset_families = ["var", "poisson"]
        noise_levels = ["low", "high"]
        length_types = ["short", "long"]
        if args.seeds is not None:
            seeds = list(args.seeds)
        else:
            seeds = [0, 1]
        t_short = 300
        t_long = 800
        output_dir = args.output_dir or "./sbtg_results_syntheticexp2_medium"

    if args.hp_trials is None:
        if args.medium:
            n_hp_trials = 10
        else:
            n_hp_trials = N_HP_TRIALS
    else:
        n_hp_trials = args.hp_trials

    total_variants = (
        len(dataset_families)
        * len(noise_levels)
        * len(length_types)
        * len(seeds)
    )
    print(
        f"[CONFIG] n_neurons={n_neurons}, families={dataset_families}, "
        f"noise={noise_levels}, lengths={length_types}, seeds={seeds}",
        flush=True,
    )
    print(
        f"[CONFIG] total dataset variants={total_variants}, workers={max_workers}, "
        f"hp_trials={n_hp_trials}, sbtg_epochs={args.sbtg_epochs}, "
        f"baselines={'off' if args.skip_baselines else 'on'}, "
        f"sbtg_methods={sbtg_method_names}",
        flush=True,
    )

    run_benchmarks(
        dataset_families=dataset_families,
        n_neurons=n_neurons,
        m_stimuli=m_stimuli,
        seeds=seeds,
        noise_levels=noise_levels,
        length_types=length_types,
        T_short=t_short,
        T_long=t_long,
        output_dir=output_dir,
        n_hp_trials=n_hp_trials,
        sbtg_epochs=args.sbtg_epochs,
        sbtg_method_names=sbtg_method_names,
        max_workers=max_workers,
        run_baselines=not args.skip_baselines,
        debug=args.debug,
    )

    if args.run_external_baselines:
        _run_external_baselines(
            n_neurons=n_neurons,
            t_long=t_long,
            output_dir=output_dir,
            external_epochs=args.external_epochs,
        )

    _write_complete_report(
        output_dir=output_dir,
        n_neurons=n_neurons,
        m_stimuli=m_stimuli,
        families=dataset_families,
        noise_levels=noise_levels,
        length_types=length_types,
        seeds=seeds,
        n_hp_trials=n_hp_trials,
        sbtg_epochs=args.sbtg_epochs,
        sbtg_method_names=sbtg_method_names,
        run_baselines=not args.skip_baselines,
        run_external_baselines=args.run_external_baselines,
    )
    _augment_compact_report(
        output_dir=output_dir,
        run_external_baselines=args.run_external_baselines,
    )
