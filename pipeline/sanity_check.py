#!/usr/bin/env python3
"""
sanity_check.py
===============

Static sanity checks for the SBTG pipeline. Run this before trusting any results!

Checks:
1. All Python files compile without errors
2. Required dependencies are available
3. Data files exist and have expected structure
4. Direction convention is consistent
5. Node order is present in all result files

Usage:
    python pipeline/sanity_check.py
    python pipeline/sanity_check.py --quick    # Skip slow checks
    python pipeline/sanity_check.py --verbose  # Show more details
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def ok(msg: str) -> str:
    return f"{Colors.GREEN}✓{Colors.END} {msg}"

def fail(msg: str) -> str:
    return f"{Colors.RED}✗{Colors.END} {msg}"

def warn(msg: str) -> str:
    return f"{Colors.YELLOW}⚠{Colors.END} {msg}"

def info(msg: str) -> str:
    return f"{Colors.BLUE}ℹ{Colors.END} {msg}"


def check_python_compilation() -> Tuple[int, int, List[str]]:
    """
    Check that all Python files compile without errors.
    
    Uses AST parsing instead of py_compile to avoid cache write issues.
    
    Returns:
        Tuple of (n_passed, n_failed, error_messages)
    """
    import ast
    
    errors = []
    n_passed = 0
    
    def check_file(py_file: Path, prefix: str = "") -> bool:
        nonlocal n_passed
        try:
            with open(py_file, 'r') as f:
                ast.parse(f.read())
            n_passed += 1
            return True
        except SyntaxError as e:
            errors.append(f"{prefix}{py_file.name}: line {e.lineno}: {e.msg}")
            return False
    
    # Check pipeline scripts
    for py_file in sorted((PROJECT_ROOT / "pipeline").glob("*.py")):
        check_file(py_file)
    
    # Check utils
    utils_dir = PROJECT_ROOT / "pipeline" / "utils"
    if utils_dir.exists():
        for py_file in sorted(utils_dir.glob("*.py")):
            check_file(py_file, "utils/")
    
    # Check archive (optional)
    archive_dir = PROJECT_ROOT / "pipeline" / "archive"
    if archive_dir.exists():
        for py_file in sorted(archive_dir.glob("*.py")):
            check_file(py_file, "archive/")
    
    # Check sbtg module
    sbtg_file = PROJECT_ROOT / "pipeline" / "models" / "sbtg.py"
    if sbtg_file.exists():
        check_file(sbtg_file, "pipeline/models/")
    
    return n_passed, len(errors), errors


def check_dependencies() -> Tuple[int, int, List[str]]:
    """Check that required dependencies are available."""
    required = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'matplotlib',
        'wormneuroatlas',
        'tqdm',
        'h5py',
        'openpyxl',
        'statsmodels',
    ]
    
    optional = [
        'torch',
        'optuna',
        'seaborn',
        'networkx',
    ]
    
    errors = []
    n_passed = 0
    
    for pkg in required:
        try:
            __import__(pkg)
            n_passed += 1
        except ImportError:
            errors.append(f"Required package missing: {pkg}")
    
    optional_missing = []
    for pkg in optional:
        try:
            __import__(pkg)
        except ImportError:
            optional_missing.append(pkg)
    
    return n_passed, len(errors), errors, optional_missing


def check_data_files(pre_flight: bool = False) -> Tuple[int, int, List[str]]:
    """Check that expected data files exist."""
    errors = []
    n_passed = 0
    
    # Check raw data
    data_dir = PROJECT_ROOT / "data"
    expected_data = [
        "Head_Activity_OH16230.mat",
        "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",
    ]
    
    for fname in expected_data:
        if (data_dir / fname).exists():
            n_passed += 1
        else:
            errors.append(f"Data file missing: {fname}")
    
    # Check intermediate data (only if not pre-flight)
    if not pre_flight:
        intermediate_dir = PROJECT_ROOT / "results" / "intermediate"
        expected_intermediate = [
            "connectome/A_struct.npy",
            "connectome/nodes.json",
        ]
        
        for fname in expected_intermediate:
            if (intermediate_dir / fname).exists():
                n_passed += 1
            else:
                errors.append(f"Intermediate file missing: {fname} (run 01_prepare_data.py)")
    
    return n_passed, len(errors), errors


def check_node_order_in_results() -> Tuple[int, int, List[str]]:
    """Check that all result files include node_order."""
    errors = []
    n_passed = 0
    
    results_dir = PROJECT_ROOT / "results"
    
    # Find all result.npz files
    for result_file in results_dir.glob("**/result.npz"):
        result_dir = result_file.parent
        
        # Check for node_order.json or neuron_names.json
        has_order = (
            (result_dir / "node_order.json").exists() or
            (result_dir / "neuron_names.json").exists()
        )
        
        if has_order:
            n_passed += 1
        else:
            errors.append(f"Missing node_order: {result_file.relative_to(results_dir)}")
    
    # Check aligned atlas files
    leifer_dir = results_dir / "leifer_evaluation"
    if leifer_dir.exists():
        for atlas_file in leifer_dir.glob("aligned_atlas_*.npz"):
            try:
                import numpy as np
                data = np.load(atlas_file, allow_pickle=True)
                if 'neurons' in data or 'node_order' in data or 'neuron_order' in data:
                    n_passed += 1
                else:
                    errors.append(f"Missing node_order in: {atlas_file.name}")
            except Exception as e:
                errors.append(f"Cannot read: {atlas_file.name}: {e}")
    
    return n_passed, len(errors), errors


def check_connectome_structure() -> Tuple[int, int, List[str]]:
    """Check structural connectome has expected properties."""
    import numpy as np
    
    errors = []
    n_passed = 0
    
    connectome_dir = PROJECT_ROOT / "results" / "intermediate" / "connectome"
    
    struct_file = connectome_dir / "A_struct.npy"
    nodes_file = connectome_dir / "nodes.json"
    
    if not struct_file.exists() or not nodes_file.exists():
        return 0, 1, ["Connectome not found (run 01_prepare_data.py)"]
    
    A = np.load(struct_file)
    with open(nodes_file) as f:
        nodes = json.load(f)
    
    # Check square
    if A.shape[0] == A.shape[1]:
        n_passed += 1
    else:
        errors.append(f"Connectome not square: {A.shape}")
    
    # Check matches node list
    if A.shape[0] == len(nodes):
        n_passed += 1
    else:
        errors.append(f"Connectome size {A.shape[0]} != node list {len(nodes)}")
    
    # Check no NaN
    if not np.isnan(A).any():
        n_passed += 1
    else:
        errors.append(f"Connectome contains {np.isnan(A).sum()} NaN values")
    
    # Check sparse (should be <50% density for biological network)
    density = np.sum(A > 0) / (A.shape[0] * (A.shape[0] - 1))
    if density < 0.5:
        n_passed += 1
    else:
        errors.append(f"Connectome density {density:.2%} seems too high")
    
    return n_passed, len(errors), errors


def check_model_training() -> Tuple[int, int, List[str]]:
    """Test basic model training functionality with synthetic data."""
    import numpy as np
    try:
        from pipeline.models.sbtg import SBTGStructuredVolatilityEstimator
    except ImportError:
        return 0, 1, ["Could not import pipeline.models.sbtg module"]

    errors = []
    n_passed = 0
    
    try:
        # Create synthetic data
        np.random.seed(42)
        n_neurons = 10
        n_timepoints = 50
        X = np.random.randn(n_timepoints, n_neurons)
        
        # Test regime_gated model (current default)
        estimator = SBTGStructuredVolatilityEstimator(
            model_type='regime_gated',
            num_regimes=2,
            gate_hidden_dim=32,
            train_split='prefix',
            window_length=2,
            dsm_epochs=10,  # Very short for testing
            fdr_alpha=0.1,
            verbose=False
        )
        
        result = estimator.fit(X)
        
        # Check outputs
        if result.sign_adj.shape != (n_neurons, n_neurons):
            errors.append(f"Wrong sign_adj shape: {result.sign_adj.shape}")
        
        if result.volatility_adj.shape != (n_neurons, n_neurons):
            errors.append(f"Wrong volatility_adj shape: {result.volatility_adj.shape}")
            
        if not hasattr(result, 'mu_hat'):
             errors.append("Missing mu_hat in result")
             
        if not errors:
            n_passed += 1
            
    except Exception as e:
        errors.append(f"Model training failed: {e}")
        
    return n_passed, len(errors), errors


def run_all_checks(verbose: bool = False, quick: bool = False, pre_flight: bool = False) -> bool:
    """Run all sanity checks and report results."""
    
    print("=" * 70)
    print("SBTG PIPELINE SANITY CHECK")
    if pre_flight:
        print("(Pre-flight mode: Checking inputs only)")
    print("=" * 70)
    
    all_passed = True
    
    # 1. Python compilation
    print("\n[1] Python file compilation...")
    n_pass, n_fail, errors = check_python_compilation()
    if n_fail == 0:
        print(f"    {ok(f'All {n_pass} Python files compile successfully')}")
    else:
        all_passed = False
        print(f"    {fail(f'{n_fail} files failed to compile:')}")
        for e in errors:
            print(f"        - {e}")
    
    # 2. Dependencies
    print("\n[2] Required dependencies...")
    n_pass, n_fail, errors, optional_missing = check_dependencies()
    if n_fail == 0:
        print(f"    {ok(f'All {n_pass} required packages available')}")
    else:
        all_passed = False
        print(f"    {fail(f'{n_fail} required packages missing:')}")
        for e in errors:
            print(f"        - {e}")
    
    if optional_missing:
        missing_str = ", ".join(optional_missing)
        print(f"    {warn(f'Optional packages not installed: {missing_str}')}")
    
    # 3. Data files
    print("\n[3] Data files...")
    n_pass, n_fail, errors = check_data_files(pre_flight=pre_flight)
    if n_fail == 0:
        print(f"    {ok(f'{n_pass} expected files found')}")
    else:
        all_passed = False
        print(f"    {fail(f'{n_fail} files missing:')}")
        for e in errors:
            print(f"        - {e}")
    
    if not quick and not pre_flight:
        # 4. Node order in results
        print("\n[4] Node order in result files...")
        n_pass, n_fail, errors = check_node_order_in_results()
        if n_fail == 0:
            print(f"    {ok(f'{n_pass} result files have node_order')}")
        else:
            print(f"    {warn(f'{n_fail} result files missing node_order:')}")
            for e in errors[:5]:  # Show first 5
                print(f"        - {e}")
            if len(errors) > 5:
                print(f"        - ... and {len(errors) - 5} more")
        
        # 5. Connectome structure
        print("\n[5] Connectome structure...")
        n_pass, n_fail, errors = check_connectome_structure()
        if n_fail == 0:
            print(f"    {ok(f'{n_pass} structure checks passed')}")
        else:
            for e in errors:
                print(f"    {fail(e)}")
            for e in errors:
                print(f"    {fail(e)}")
            all_passed = False
    
    # 6. Model Training (Quick Test)
    if not quick and not pre_flight:
        print("\n[6] Model functionality (quick training)...")
        n_pass, n_fail, errors = check_model_training()
        if n_fail == 0:
            print(f"    {ok('SBTG regime_gated model (synthetic) trained successfully')}")
        else:
            print(f"    {fail(f'Model training failed:')}")
            for e in errors:
                print(f"        - {e}")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print(ok("ALL SANITY CHECKS PASSED"))
    else:
        print(fail("SOME CHECKS FAILED - Review errors above"))
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SBTG pipeline sanity checks")
    parser.add_argument("--quick", action="store_true", help="Skip slow checks")
    parser.add_argument("--pre-flight", action="store_true", help="Run only input checks (for fresh pipeline runs)")
    parser.add_argument("--verbose", action="store_true", help="Show more details")
    args = parser.parse_args()
    
    success = run_all_checks(verbose=args.verbose, quick=args.quick, pre_flight=args.pre_flight)
    sys.exit(0 if success else 1)

