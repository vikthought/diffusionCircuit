"""
SyntheticTesting – synthetic benchmark orchestration for SBTG and classical baselines.

This module is the main driver for running SBTG synthetic benchmarks. It uses
utilities from SyntheticTestingUtils.py for data generation, baseline methods,
and evaluation metrics.

The pipeline is organised into four stages:

1. **Synthetic Data Generation** – create multivariate sequences with known
   directed graphs at lag 1 (and optionally lag 2).
2. **Configuration Expansion** – instantiate multiple variants by toggling
   noise regimes, sequence lengths, and random seeds.
3. **Model Fitting** – train SBTG variants (Linear, FeatureBilinear, Minimal)
   via denoising score matching, perform HAC/FDR statistical testing, and run
   a library of baseline learners.
4. **Evaluation & Reporting** – compute edge-level metrics, sweep statistics,
   aggregate across seeds, and emit CSV, plots, and markdown summaries.

Command-Line Options
====================
* ``--skip-baselines`` – only run SBTG variants.
* ``--max-workers N`` – override auto-selected worker pool size.
* ``--mini`` – shrink the benchmark grid for quick smoke checks.
* ``--test-baselines`` – single-dataset baseline diagnostic run.
* ``--hp-trials N`` – number of Optuna trials per SBTG variant (default: 20).
* ``--output-dir PATH`` – custom output directory.
* ``--check-deps`` – print baseline availability and exit.
* ``--debug`` – verbose baseline logging.

Usage
=====
    python pipeline/SyntheticTesting.py
    python pipeline/SyntheticTesting.py --mini --skip-baselines
    python pipeline/SyntheticTesting.py --hp-trials 50
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from pipeline.SyntheticTestingUtils import (
    # Constants
    SBTG_LIN_NAME, SBTG_A_NAME, SBTG_MINIMAL_NAME, SBTG_METHODS,
    BASELINE_DEFAULT_STAT_NAME, N_HP_TRIALS,
    SBTG_TRAINING_GRID_MINIMAL, SBTG_STAT_PARAM_GRID,
    # Device / environment
    CUDA_AVAILABLE, DEVICE, GPU_NAME, GPU_COUNT, CPU_COUNT,
    # Optional baseline flags
    HAS_LINGAM, HAS_NOTEARS, HAS_TIGRAMITE, HAS_DYNOTEARS,
    LINGAM_IMPORT_ERROR, NOTEARS_IMPORT_ERROR, TIGRAMITE_IMPORT_ERROR,
    DYNOTEARS_IMPORT_ERROR,
    # Data generators
    _generate_dataset, generate_var_data, generate_hawkes_like_data,
    # SBTG fitting
    fit_sbtg_with_timeout, fit_minimal_with_timeout,
    TimeoutError,
    # Baselines
    var_lasso_baseline, var_ridge_baseline, var_lingam_baseline,
    poisson_glm_baseline, notears_baseline, pcmci_plus_baseline,
    dynotears_baseline,
    # Evaluation
    evaluate_weighted, evaluate_binary, evaluate_sbtg,
    evaluate_structured_volatility, evaluate_multilag_auroc,
    # HP search
    hyperparam_search_sbtg_null_contrast,
    # Helpers
    check_dependencies,
)


# -------------------------------------------------------------------------
# Parallel evaluation: one dataset variant
# -------------------------------------------------------------------------


def _evaluate_dataset_variant(
    family: str,
    noise: str,
    length_type: str,
    seed: int,
    T: int,
    n_neurons: int,
    m_stimuli: int,
    train_cfg_lin: Optional[Dict[str, Any]],
    train_cfg_a: Optional[Dict[str, Any]],
    train_cfg_minimal: Optional[Dict[str, Any]],
    sbtg_method_names: Optional[List[str]],
    run_baselines: bool,
    debug: bool = False,
) -> Dict[str, Any]:
    """Evaluate all methods on a single dataset variant (family/noise/length/seed)."""
    print(
        f"[TASK] Begin evaluation: family={family}, noise={noise}, "
        f"length={length_type}, seed={seed}",
        flush=True,
    )

    X_list, truth_dict = _generate_dataset(
        family=family, n=n_neurons, T=T, m_stim=m_stimuli,
        noise_level=noise, seed=seed,
    )
    truth_graph = truth_dict[1]
    smooth_sigma_default = 1.0 if family in ["poisson", "hawkes"] else None

    records: List[Dict[str, Any]] = []
    score_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def store_scores(key: str, y_true: np.ndarray, y_score: np.ndarray) -> None:
        score_map[key] = (y_true, y_score)

    # ----- SBTG methods -----
    method_configs: List[Tuple[str, Dict[str, Any]]] = []
    if train_cfg_lin is not None:
        method_configs.append((SBTG_LIN_NAME, train_cfg_lin))
    if train_cfg_a is not None:
        method_configs.append((SBTG_A_NAME, train_cfg_a))
    if train_cfg_minimal is not None:
        method_configs.append((SBTG_MINIMAL_NAME, train_cfg_minimal))
    if sbtg_method_names is not None:
        selected = set(sbtg_method_names)
        method_configs = [mc for mc in method_configs if mc[0] in selected]

    for method_name, train_cfg in method_configs:
        for stat_cfg in SBTG_STAT_PARAM_GRID:
            t_start = time.time()
            try:
                if method_name == SBTG_MINIMAL_NAME:
                    minimal_kwargs = {
                        "tune_hp": train_cfg.get("tune_hp", True),
                        "n_hp_trials": train_cfg.get("n_hp_trials", N_HP_TRIALS),
                        "lags": train_cfg.get("lags", [1, 2]),
                        "epochs": train_cfg.get("dsm_epochs", 100),
                        "n_folds": 5,
                        "hac_max_lag": stat_cfg["hac_max_lag"],
                        "fdr_alpha": stat_cfg["fdr_alpha"],
                        "fdr_method": stat_cfg["fdr_method"],
                        "device": DEVICE,
                        "verbose": False,
                        "random_state": seed,
                    }
                    sbtg_result = fit_minimal_with_timeout(minimal_kwargs, X_list)
                    elapsed = time.time() - t_start

                    pred_dict = {lag: sbtg_result.mu_hat[lag] for lag in sbtg_result.mu_hat}
                    multilag_metrics = evaluate_multilag_auroc(truth_dict, pred_dict)

                    metrics_sbtg = evaluate_binary(
                        truth_graph, sbtg_result.get_adjacency_for_lag(1).astype(int),
                    )
                    metrics_sbtg.update(multilag_metrics)
                    status = "success"
                else:
                    est_kwargs: Dict[str, Any] = {
                        "window_length": train_cfg.get("window_length", 2),
                        "smooth_sigma": train_cfg.get("smooth_sigma", smooth_sigma_default),
                        "dsm_hidden_dim": train_cfg["dsm_hidden_dim"],
                        "dsm_num_layers": train_cfg["dsm_num_layers"],
                        "dsm_noise_std": train_cfg["dsm_noise_std"],
                        "dsm_epochs": train_cfg["dsm_epochs"],
                        "dsm_batch_size": train_cfg.get("dsm_batch_size", 128),
                        "dsm_lr": train_cfg.get("dsm_lr", 1e-3),
                        "train_frac": train_cfg["train_frac"],
                        "hac_max_lag": stat_cfg["hac_max_lag"],
                        "fdr_alpha": stat_cfg["fdr_alpha"],
                        "fdr_method": stat_cfg["fdr_method"],
                        "volatility_test": True,
                        "compute_undirected": True,
                        "random_state": seed,
                        "verbose": False,
                        "model_type": train_cfg.get("model_type", "linear"),
                    }
                    if "structured_l1_lambda" in train_cfg:
                        est_kwargs["structured_l1_lambda"] = train_cfg["structured_l1_lambda"]
                    if "feature_dim" in train_cfg:
                        est_kwargs["feature_dim"] = train_cfg["feature_dim"]

                    sbtg_result = fit_sbtg_with_timeout(est_kwargs, X_list)
                    elapsed = time.time() - t_start

                    metrics_sbtg = evaluate_structured_volatility(
                        truth_graph, sbtg_result, include_volatility_edges=True,
                    )
                    status = "success"
            except TimeoutError:
                elapsed = time.time() - t_start
                print(
                    f"[SBTG] TIMEOUT: {method_name} after {elapsed:.1f}s "
                    f"(family={family}, stat={stat_cfg['name']})",
                    flush=True,
                )
                metrics_sbtg = {
                    "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5,
                    "pr_auc": 0, "y_true": np.array([]), "y_score": np.array([]),
                }
                status = "timeout"
            except Exception as e:
                elapsed = time.time() - t_start
                print(
                    f"[SBTG] ERROR: {method_name} failed after {elapsed:.1f}s: {e} "
                    f"(family={family}, stat={stat_cfg['name']})",
                    flush=True,
                )
                metrics_sbtg = {
                    "precision": 0, "recall": 0, "f1": 0, "roc_auc": 0.5,
                    "pr_auc": 0, "y_true": np.array([]), "y_score": np.array([]),
                }
                status = "error"

            record = {
                "family": family, "noise": noise, "length": length_type,
                "seed": seed, "method": method_name,
                "stat_cfg_name": stat_cfg["name"],
                "precision": metrics_sbtg["precision"],
                "recall": metrics_sbtg["recall"],
                "f1": metrics_sbtg["f1"],
                "roc_auc": metrics_sbtg["roc_auc"],
                "pr_auc": metrics_sbtg["pr_auc"],
                "edges_mean": metrics_sbtg.get("edges_mean", np.nan),
                "edges_energy_only": metrics_sbtg.get("edges_energy_only", np.nan),
                "edges_volatility_only": metrics_sbtg.get("edges_volatility_only", np.nan),
                "edges_union": metrics_sbtg.get("edges_union", np.nan),
                "auroc_lag1": metrics_sbtg.get("auroc_lag1", np.nan),
                "auroc_lag2": metrics_sbtg.get("auroc_lag2", np.nan),
                "status": status,
                "time_seconds": elapsed,
            }
            records.append(record)

            if status == "success":
                print(
                    f"[SBTG] {method_name}: edges_mean={record['edges_mean']}, "
                    f"volatility_only={record['edges_volatility_only']} "
                    f"(family={family}, stat={stat_cfg['name']}, time={elapsed:.1f}s)",
                    flush=True,
                )
                key = f"{family}|{method_name}|{stat_cfg['name']}"
                store_scores(key, metrics_sbtg["y_true"], metrics_sbtg["y_score"])

    if not run_baselines:
        return {"records": records, "scores": score_map}

    # ----- Baselines -----
    baseline_results = {}

    def _run_baseline(name, fn, *fn_args, is_binary=False, **fn_kwargs):
        """Helper to run a baseline with standardized logging and error handling."""
        print(f"[BASELINE] {name} starting...", flush=True)
        t0 = time.time()
        try:
            result = fn(*fn_args, **fn_kwargs)
            dt = time.time() - t0
            baseline_results[name] = {'status': 'SUCCESS', 'time': dt}

            if is_binary:
                metrics = evaluate_binary(truth_graph, result.astype(int))
            else:
                metrics = evaluate_weighted(truth_graph, result, threshold=1e-6)

            baseline_results[name].update(
                {
                    "pred_edges": metrics.get("pred_edges", np.nan),
                    "true_edges": metrics.get("true_edges", np.nan),
                    "pred_density": metrics.get("pred_density", np.nan),
                    "true_density": metrics.get("true_density", np.nan),
                }
            )

            records.append({
                "family": family, "noise": noise, "length": length_type,
                "seed": seed, "method": name,
                "stat_cfg_name": BASELINE_DEFAULT_STAT_NAME,
                "precision": metrics["precision"], "recall": metrics["recall"],
                "f1": metrics["f1"], "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "edges_mean": metrics.get("pred_edges", np.nan),
                "edges_energy_only": np.nan,
                "edges_volatility_only": np.nan,
                "edges_union": metrics.get("pred_edges", np.nan),
            })
            store_scores(
                f"{family}|{name}|{BASELINE_DEFAULT_STAT_NAME}",
                metrics["y_true"], metrics["y_score"],
            )
            print(
                f"[BASELINE] {name} done in {dt:.2f}s: "
                f"F1={metrics['f1']:.3f}, AUROC={metrics['roc_auc']:.3f}, "
                f"pred_edges={metrics.get('pred_edges', 0)}",
                flush=True,
            )
        except TimeoutError as e:
            dt = time.time() - t0
            baseline_results[name] = {'status': 'TIMEOUT', 'time': dt}
            print(f"[BASELINE] {name} TIMEOUT after {dt:.2f}s: {e}", flush=True)
        except Exception as e:
            dt = time.time() - t0
            baseline_results[name] = {'status': 'FAILED', 'time': dt, 'error': str(e)}
            print(f"[BASELINE] {name} FAILED after {dt:.2f}s: {e}", flush=True)

    # Fast baselines (always available)
    _run_baseline("VAR-LASSO", var_lasso_baseline, X_list, alpha=0.05)
    _run_baseline("VAR-Ridge", var_ridge_baseline, X_list, alpha=1.0)

    # Optional baselines
    if HAS_LINGAM:
        _run_baseline("VAR-LiNGAM", var_lingam_baseline, X_list)
    else:
        baseline_results['VAR-LiNGAM'] = {
            'status': 'SKIPPED',
            'time': 0.0,
            'error': LINGAM_IMPORT_ERROR or "Package unavailable",
        }

    if family in ["poisson", "hawkes"]:
        _run_baseline("Poisson-GLM", poisson_glm_baseline, X_list)
    else:
        baseline_results['Poisson-GLM'] = {'status': 'SKIPPED', 'time': 0.0}

    if HAS_NOTEARS:
        _run_baseline("NOTEARS", notears_baseline, X_list)
    else:
        baseline_results['NOTEARS'] = {
            'status': 'SKIPPED',
            'time': 0.0,
            'error': NOTEARS_IMPORT_ERROR or "Package unavailable",
        }

    if HAS_TIGRAMITE:
        _run_baseline("PCMCI+", pcmci_plus_baseline, X_list, tau_max=1,
                       alpha_level=0.05, is_binary=True)
    else:
        baseline_results['PCMCI+'] = {
            'status': 'SKIPPED',
            'time': 0.0,
            'error': TIGRAMITE_IMPORT_ERROR or "Package unavailable",
        }

    if HAS_DYNOTEARS:
        _run_baseline("DYNOTEARS", dynotears_baseline, X_list)
    else:
        baseline_results['DYNOTEARS'] = {
            'status': 'SKIPPED',
            'time': 0.0,
            'error': DYNOTEARS_IMPORT_ERROR or "Package unavailable",
        }

    # Baseline summary
    total_time = sum(r['time'] for r in baseline_results.values())
    print(f"[BASELINE SUMMARY] Total time: {total_time:.2f}s", flush=True)
    for method, result in baseline_results.items():
        print(f"  {method:<15s}: {result['status']:<15s} ({result['time']:>6.2f}s)", flush=True)
        if result.get("error"):
            print(f"    reason: {result['error']}", flush=True)

    return {"records": records, "scores": score_map, "baseline_results": baseline_results}


# -------------------------------------------------------------------------
# Main experiment loop
# -------------------------------------------------------------------------


def run_benchmarks(
    dataset_families: List[str],
    n_neurons: int,
    m_stimuli: int,
    seeds: List[int],
    noise_levels: List[str],
    length_types: List[str],
    T_short: int,
    T_long: int,
    output_dir: str,
    n_hp_trials: int = N_HP_TRIALS,
    sbtg_epochs: int = 100,
    sbtg_method_names: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
    run_baselines: bool = True,
    debug: bool = False,
) -> None:
    """
    Run SBTG and baseline benchmarks on synthetic datasets.

    Results saved to output_dir: metrics.csv, best_params.json,
    ROC/PR/bar plots per family, and report.md.
    """
    os.makedirs(output_dir, exist_ok=True)

    total_variants = (
        len(dataset_families) * len(noise_levels)
        * len(length_types) * len(seeds)
    )
    print(
        f"[INFO] Benchmark: {total_variants} evaluations "
        f"({len(dataset_families)} families x {len(noise_levels)} noise x "
        f"{len(length_types)} lengths x {len(seeds)} seeds).",
        flush=True,
    )
    print(f"[INFO] Results directory: {output_dir}", flush=True)
    print(f"[INFO] Baselines {'enabled' if run_baselines else 'disabled'}.", flush=True)
    print(f"[INFO] HP trials per SBTG variant: {n_hp_trials}", flush=True)
    print(f"[INFO] SBTG epochs (HP tuning + final fit): {sbtg_epochs}", flush=True)
    if sbtg_method_names is None:
        selected_sbtg_methods = list(SBTG_METHODS)
    else:
        selected_sbtg_methods = [m for m in SBTG_METHODS if m in set(sbtg_method_names)]
    if not selected_sbtg_methods:
        raise ValueError("No SBTG methods selected for benchmarking.")
    print(f"[INFO] Selected SBTG methods: {selected_sbtg_methods}", flush=True)

    records: List[Dict[str, Any]] = []
    baseline_status_records: List[Dict[str, Any]] = []
    y_true_all: Dict[str, np.ndarray] = {}
    y_score_all: Dict[str, np.ndarray] = {}

    def append_scores(key: str, y_true: np.ndarray, y_score: np.ndarray):
        if key not in y_true_all:
            y_true_all[key] = y_true.copy()
            y_score_all[key] = y_score.copy()
        else:
            y_true_all[key] = np.concatenate([y_true_all[key], y_true])
            y_score_all[key] = np.concatenate([y_score_all[key], y_score])

    def append_baseline_status(result: Dict[str, Any], task: Tuple[Any, ...]) -> None:
        """Collect per-variant baseline status entries for complete reporting."""
        if "baseline_results" not in result:
            return
        family, noise, length_type, seed = task[0], task[1], task[2], task[3]
        for method, info in result["baseline_results"].items():
            baseline_status_records.append(
                {
                    "family": family,
                    "noise": noise,
                    "length": length_type,
                    "seed": seed,
                    "method": method,
                    "status": info.get("status", "UNKNOWN"),
                    "time_seconds": float(info.get("time", 0.0)),
                    "error": info.get("error", ""),
                    "pred_edges": info.get("pred_edges", np.nan),
                    "true_edges": info.get("true_edges", np.nan),
                    "pred_density": info.get("pred_density", np.nan),
                    "true_density": info.get("true_density", np.nan),
                }
            )

    best_train_params_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    if not seeds:
        raise ValueError("Seeds list must be non-empty.")
    hyperparam_seed = seeds[0]

    # --- HP search per family/noise/length ---
    for family in dataset_families:
        for noise in noise_levels:
            for length_type in length_types:
                T = T_short if length_type == "short" else T_long
                print(
                    f"[INFO] HP search: family={family}, noise={noise}, "
                    f"length={length_type} (T={T})",
                    flush=True,
                )

                X_list_hp, truth_dict_hp = _generate_dataset(
                    family=family, n=n_neurons, T=T, m_stim=m_stimuli,
                    noise_level=noise, seed=hyperparam_seed,
                )
                smooth_sigma = 1.0 if family in ["poisson", "hawkes"] else None

                if SBTG_LIN_NAME in selected_sbtg_methods:
                    best_cfg_lin = hyperparam_search_sbtg_null_contrast(
                        X_list_hp=X_list_hp, smooth_sigma=smooth_sigma,
                        model_type="linear", method_name=SBTG_LIN_NAME,
                        family=family, noise=noise, length_type=length_type,
                        n_trials=n_hp_trials, seed=hyperparam_seed,
                        sbtg_epochs=sbtg_epochs, verbose=True,
                    )
                    best_train_params_map[(family, noise, length_type, SBTG_LIN_NAME)] = best_cfg_lin

                if SBTG_A_NAME in selected_sbtg_methods:
                    best_cfg_a = hyperparam_search_sbtg_null_contrast(
                        X_list_hp=X_list_hp, smooth_sigma=smooth_sigma,
                        model_type="feature_bilinear", method_name=SBTG_A_NAME,
                        family=family, noise=noise, length_type=length_type,
                        n_trials=n_hp_trials, seed=hyperparam_seed + 1,
                        sbtg_epochs=sbtg_epochs, verbose=True,
                    )
                    best_train_params_map[(family, noise, length_type, SBTG_A_NAME)] = best_cfg_a

                if SBTG_MINIMAL_NAME in selected_sbtg_methods:
                    minimal_cfg = dict(SBTG_TRAINING_GRID_MINIMAL[0])
                    minimal_cfg["dsm_epochs"] = sbtg_epochs
                    best_train_params_map[
                        (family, noise, length_type, SBTG_MINIMAL_NAME)
                    ] = minimal_cfg

    # --- Full evaluation ---
    task_args: List[Tuple[Any, ...]] = []
    for family in dataset_families:
        for noise in noise_levels:
            for length_type in length_types:
                T = T_short if length_type == "short" else T_long
                train_cfg_lin = (
                    best_train_params_map[(family, noise, length_type, SBTG_LIN_NAME)]
                    if SBTG_LIN_NAME in selected_sbtg_methods
                    else None
                )
                train_cfg_a = (
                    best_train_params_map[(family, noise, length_type, SBTG_A_NAME)]
                    if SBTG_A_NAME in selected_sbtg_methods
                    else None
                )
                train_cfg_minimal = (
                    best_train_params_map[(family, noise, length_type, SBTG_MINIMAL_NAME)]
                    if SBTG_MINIMAL_NAME in selected_sbtg_methods
                    else None
                )
                for seed in seeds:
                    task_args.append((
                        family, noise, length_type, seed, T, n_neurons, m_stimuli,
                        train_cfg_lin, train_cfg_a, train_cfg_minimal,
                        selected_sbtg_methods,
                        run_baselines, debug,
                    ))

    total_tasks = len(task_args)
    if total_tasks == 0:
        print("[WARN] No evaluation tasks were generated.", flush=True)
    effective_workers = max(1, min(max_workers or 1, total_tasks))

    if effective_workers > 1:
        print(f"[INFO] Running {total_tasks} tasks with {effective_workers} workers.", flush=True)
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(_evaluate_dataset_variant, *args): args
                for args in task_args
            }
            for idx, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                append_baseline_status(result, futures[fut])
                records.extend(result["records"])
                for key, (y_true, y_score) in result["scores"].items():
                    append_scores(key, y_true, y_score)
                if total_tasks >= 4 or idx == total_tasks:
                    print(f"[INFO] Completed {idx}/{total_tasks} tasks.", flush=True)
    else:
        for idx, args in enumerate(task_args, 1):
            result = _evaluate_dataset_variant(*args)
            append_baseline_status(result, args)
            records.extend(result["records"])
            for key, (y_true, y_score) in result["scores"].items():
                append_scores(key, y_true, y_score)
            if total_tasks >= 4 or idx == total_tasks:
                print(f"[INFO] Completed {idx}/{total_tasks} tasks (serial).", flush=True)

    # --- Save metrics ---
    metrics_df = pd.DataFrame.from_records(records)
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[INFO] Saved metrics to {metrics_csv_path}", flush=True)

    # Use success-only rows for summary statistics:
    # - SBTG rows must have status == success
    # - baseline rows are recorded only on success in this pipeline
    metrics_eval_df = metrics_df.copy()
    if "status" in metrics_eval_df.columns:
        is_sbtg = metrics_eval_df["method"].isin(SBTG_METHODS)
        sbtg_success = (
            metrics_eval_df["status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq("success")
        )
        keep_mask = (~is_sbtg) | sbtg_success
        dropped = int((~keep_mask).sum())
        metrics_eval_df = metrics_eval_df[keep_mask].copy()
        if dropped > 0:
            print(
                f"[INFO] Excluding {dropped} non-success SBTG rows from summary/report statistics.",
                flush=True,
            )

    if baseline_status_records:
        baseline_status_df = pd.DataFrame.from_records(baseline_status_records)
        baseline_status_csv_path = os.path.join(output_dir, "baseline_status.csv")
        baseline_status_df.to_csv(baseline_status_csv_path, index=False)
        print(f"[INFO] Saved baseline status to {baseline_status_csv_path}", flush=True)

    # --- Best stat config selection ---
    best_stat_for_plot: Dict[Tuple[str, str], str] = {}
    for family in dataset_families:
        df_family = metrics_eval_df[metrics_eval_df["family"] == family]
        if df_family.empty:
            continue
        for method in df_family["method"].unique():
            df_m = df_family[df_family["method"] == method]
            if method in SBTG_METHODS:
                grouped = df_m.groupby("stat_cfg_name")["f1"].mean().reset_index()
                best_name = grouped.loc[grouped["f1"].idxmax(), "stat_cfg_name"]
            else:
                best_name = BASELINE_DEFAULT_STAT_NAME
            best_stat_for_plot[(family, method)] = best_name

    rows_best = []
    for family in dataset_families:
        df_family = metrics_eval_df[metrics_eval_df["family"] == family]
        if df_family.empty:
            continue
        for method in df_family["method"].unique():
            stat_name = best_stat_for_plot[(family, method)]
            rows_best.append(
                df_family[(df_family["method"] == method) & (df_family["stat_cfg_name"] == stat_name)]
            )
    metrics_beststat_df = pd.concat(rows_best, axis=0) if rows_best else metrics_df.copy()

    # --- Save best SBTG params ---
    best_params: Dict[str, Any] = {}
    for family in dataset_families:
        for noise in noise_levels:
            for length_type in length_types:
                for method_name in selected_sbtg_methods:
                    key_variant = (
                        (metrics_eval_df["family"] == family)
                        & (metrics_eval_df["noise"] == noise)
                        & (metrics_eval_df["length"] == length_type)
                        & (metrics_eval_df["method"] == method_name)
                    )
                    df_variant = metrics_eval_df[key_variant]
                    if df_variant.empty:
                        continue

                    grouped = (
                        df_variant.groupby("stat_cfg_name")["f1"]
                        .agg(["mean", "std"]).reset_index()
                    )
                    best_idx = grouped["mean"].idxmax()
                    best_row = grouped.loc[best_idx]
                    best_stat_name = best_row["stat_cfg_name"]

                    train_cfg = best_train_params_map[(family, noise, length_type, method_name)]
                    stat_cfg_dict = next(
                        cfg for cfg in SBTG_STAT_PARAM_GRID if cfg["name"] == best_stat_name
                    )

                    best_params.setdefault(family, {}).setdefault(noise, {}).setdefault(
                        length_type, {}
                    )[method_name] = {
                        "train_config_name": train_cfg["name"],
                        "train_params": train_cfg,
                        "stat_config_name": best_stat_name,
                        "stat_params": stat_cfg_dict,
                        "f1_mean": float(best_row["mean"]),
                        "f1_std": float(0.0 if np.isnan(best_row["std"]) else best_row["std"]),
                    }

    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"[INFO] Saved best params to {best_params_path}", flush=True)

    # --- Plots ---
    for family in dataset_families:
        df_family = metrics_eval_df[metrics_eval_df["family"] == family]
        if df_family.empty:
            continue
        methods_for_family = sorted(df_family["method"].unique())

        # ROC curves
        plt.figure()
        for method in methods_for_family:
            stat_name = best_stat_for_plot[(family, method)]
            key = f"{family}|{method}|{stat_name}"
            if key not in y_true_all:
                continue
            try:
                fpr, tpr, _ = roc_curve(y_true_all[key], y_score_all[key])
                plt.plot(fpr, tpr, label=f"{method} ({stat_name})")
            except ValueError:
                continue
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves - {family}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"roc_{family}.png"), bbox_inches="tight")
        plt.close()

        # PR curves
        plt.figure()
        for method in methods_for_family:
            stat_name = best_stat_for_plot[(family, method)]
            key = f"{family}|{method}|{stat_name}"
            if key not in y_true_all:
                continue
            try:
                precision, recall, _ = precision_recall_curve(y_true_all[key], y_score_all[key])
                plt.plot(recall, precision, label=f"{method} ({stat_name})")
            except ValueError:
                continue
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curves - {family}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"pr_{family}.png"), bbox_inches="tight")
        plt.close()

        # F1 & ROC AUC barplots
        df_best = metrics_beststat_df[metrics_beststat_df["family"] == family]
        if not df_best.empty:
            grouped = (
                df_best.groupby("method")
                .agg(f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                     roc_mean=("roc_auc", "mean"), roc_std=("roc_auc", "std"))
                .reset_index()
            )
            methods = grouped["method"].tolist()
            x = np.arange(len(methods))

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.bar(x, grouped["f1_mean"], yerr=grouped["f1_std"])
            plt.xticks(x, methods, rotation=45, ha="right")
            plt.ylabel("F1")
            plt.title(f"F1 (mean +/- std) - {family}")

            plt.subplot(1, 2, 2)
            plt.bar(x, grouped["roc_mean"], yerr=grouped["roc_std"])
            plt.xticks(x, methods, rotation=45, ha="right")
            plt.ylabel("ROC AUC")
            plt.title(f"ROC AUC (mean +/- std) - {family}")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"f1_roc_{family}.png"), bbox_inches="tight")
            plt.close()

    # --- Markdown report ---
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("# SBTG Benchmark Report\n\n")
        f.write("## Experimental Setup\n\n")
        f.write(f"- Neurons: {n_neurons}\n")
        f.write(f"- Stimuli: {m_stimuli}\n")
        f.write(f"- Seeds: {seeds}\n")
        f.write(f"- Noise levels: {noise_levels}\n")
        f.write(f"- Lengths: {length_types} (T_short={T_short}, T_long={T_long})\n")
        f.write(f"- HP trials per variant: {n_hp_trials}\n\n")
        f.write("## Reporting Policy\n\n")
        f.write(
            "- Aggregated metrics in this report are computed from success-filtered rows:\n"
        )
        f.write(
            "  - SBTG rows are included only when `status=success`.\n"
        )
        f.write(
            "  - Baseline rows are included when a metric row exists (successful execution path).\n\n"
        )
        f.write(
            f"- Success-filtered metric rows used for aggregation: {len(metrics_eval_df)}\n"
        )
        f.write(
            f"- Raw metric rows in `metrics.csv`: {len(metrics_df)}\n\n"
        )

        f.write("## Average Metrics (Best Stat Config per Method)\n\n")
        for family in dataset_families:
            df_family = metrics_beststat_df[metrics_beststat_df["family"] == family]
            if df_family.empty:
                continue
            f.write(f"### {family}\n\n")
            f.write("| Method | Stat cfg | Prec | Recall | F1 | ROC AUC | PR AUC |\n")
            f.write("|--------|----------|------|--------|----|---------|--------|\n")
            for method in sorted(df_family["method"].unique()):
                df_m = df_family[df_family["method"] == method]
                stat_cfg_name = best_stat_for_plot.get((family, method), BASELINE_DEFAULT_STAT_NAME)
                f.write(
                    f"| {method} | {stat_cfg_name} | "
                    f"{df_m['precision'].mean():.3f} | {df_m['recall'].mean():.3f} | "
                    f"{df_m['f1'].mean():.3f}+/-{df_m['f1'].std():.3f} | "
                    f"{df_m['roc_auc'].mean():.3f}+/-{df_m['roc_auc'].std():.3f} | "
                    f"{df_m['pr_auc'].mean():.3f}+/-{df_m['pr_auc'].std():.3f} |\n"
                )
            f.write("\n")

        f.write("## Best SBTG Configurations\n\n")
        for family, noise_dict in best_params.items():
            f.write(f"### {family}\n\n")
            for noise, length_dict in noise_dict.items():
                for length_type, method_dict in length_dict.items():
                    f.write(f"**{noise} noise, {length_type} length:**\n\n")
                    f.write("| Method | Train cfg | Stat cfg | F1 mean | F1 std |\n")
                    f.write("|--------|-----------|----------|---------|--------|\n")
                    for method_name, cfg in method_dict.items():
                        f.write(
                            f"| {method_name} | {cfg['train_config_name']} | "
                            f"{cfg['stat_config_name']} | "
                            f"{cfg['f1_mean']:.3f} | {cfg['f1_std']:.3f} |\n"
                        )
                    f.write("\n")

    print(f"[INFO] Saved report to {report_path}", flush=True)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SBTG synthetic benchmarks.")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline methods, only evaluate SBTG variants.")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Number of worker processes (default: auto).")
    parser.add_argument("--mini", action="store_true",
                        help="Reduced smoke-test configuration.")
    parser.add_argument("--test-baselines", action="store_true",
                        help="Single-dataset baseline diagnostic run.")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose output for baseline execution.")
    parser.add_argument("--check-deps", action="store_true",
                        help="Check baseline availability and exit.")
    parser.add_argument("--hp-trials", type=int, default=N_HP_TRIALS,
                        help=f"Optuna trials per SBTG variant (default: {N_HP_TRIALS}).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./sbtg_results).")
    args = parser.parse_args()

    if args.check_deps:
        check_dependencies()
        sys.exit(0)

    # Resource summary
    print("=" * 80, flush=True)
    print(f"[RESOURCES] CPU cores: {CPU_COUNT}, GPU: {CUDA_AVAILABLE}", flush=True)
    if CUDA_AVAILABLE:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[RESOURCES] GPU: {GPU_NAME} ({gpu_mem:.1f} GB), count: {GPU_COUNT}", flush=True)
    print(f"[RESOURCES] PyTorch {torch.__version__}, device: {DEVICE}", flush=True)
    print("=" * 80, flush=True)

    # Configuration
    dataset_families = ["var", "poisson", "hawkes", "tanh"]
    n_neurons = 10
    m_stimuli = 3
    seeds = [0, 1]
    noise_levels = ["low", "high"]
    length_types = ["short", "long"]
    T_short = 300
    T_long = 800
    output_dir = args.output_dir or "./sbtg_results"

    if CUDA_AVAILABLE:
        default_workers = max(1, min(CPU_COUNT - 1, 8))
    else:
        default_workers = max(1, CPU_COUNT // 2)
    max_workers = args.max_workers if args.max_workers is not None else default_workers

    n_hp_trials = args.hp_trials

    if args.test_baselines:
        dataset_families = ["var"]
        n_neurons = 6
        m_stimuli = 2
        seeds = [0]
        noise_levels = ["low"]
        length_types = ["short"]
        T_short = 150
        T_long = 150
        output_dir = args.output_dir or "./sbtg_results_baseline_test"
        max_workers = 1
        args.debug = True
        n_hp_trials = 5
        check_dependencies()
    elif args.mini:
        dataset_families = ["var"]
        n_neurons = 6
        m_stimuli = 2
        seeds = [0]
        noise_levels = ["low"]
        length_types = ["short"]
        T_short = 120
        T_long = 120
        output_dir = args.output_dir or "./sbtg_results_mini"

    print(f"[CONFIG] Workers: {max_workers}, HP trials: {n_hp_trials}", flush=True)

    run_benchmarks(
        dataset_families=dataset_families,
        n_neurons=n_neurons,
        m_stimuli=m_stimuli,
        seeds=seeds,
        noise_levels=noise_levels,
        length_types=length_types,
        T_short=T_short,
        T_long=T_long,
        output_dir=output_dir,
        n_hp_trials=n_hp_trials,
        max_workers=max_workers,
        run_baselines=not args.skip_baselines,
        debug=args.debug,
    )
