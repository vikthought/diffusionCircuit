"""
Reproducibility utilities: RNG seeds, version pinning, provenance logging.

This module ensures that every run can be reproduced by:
1. Setting all RNG seeds consistently
2. Logging seed state to results
3. Saving pip freeze / dependency versions
4. Recording run metadata (timestamps, git commit, etc.)
"""

import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@dataclass
class RunProvenance:
    """Container for run provenance information."""
    timestamp: str
    python_version: str
    numpy_seed: int
    torch_seed: Optional[int]
    optuna_seed: Optional[int]
    git_commit: Optional[str]
    git_branch: Optional[str]
    git_dirty: Optional[bool]
    pip_freeze: Optional[str]
    hostname: Optional[str]
    working_directory: str
    script_path: Optional[str]
    data_hash: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RunProvenance':
        return cls(**d)


# Global provenance object for current run
RUN_PROVENANCE: Optional[RunProvenance] = None


def set_all_seeds(seed: int = 42) -> Dict[str, int]:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Master seed to use
        
    Returns:
        Dict with seed values used for each library
    """
    seeds = {'numpy': seed}
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # For full determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seeds['torch'] = seed
    
    # Python's random module
    import random
    random.seed(seed)
    seeds['python_random'] = seed
    
    return seeds


def get_seed_state() -> Dict[str, Any]:
    """
    Get current RNG state for logging.
    
    Returns:
        Dict with RNG states (can be used to resume)
    """
    state = {
        'numpy_state': np.random.get_state()[1][:5].tolist(),  # First 5 elements
    }
    
    if HAS_TORCH:
        state['torch_state'] = torch.get_rng_state()[:10].tolist()
    
    return state


def get_git_info() -> Dict[str, Any]:
    """Get current git commit, branch, and dirty status."""
    info = {
        'git_commit': None,
        'git_branch': None, 
        'git_dirty': None,
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['git_commit'] = result.stdout.strip()[:12]  # Short hash
        
        # Get branch name
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['git_branch'] = result.stdout.strip()
        
        # Check if dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info['git_dirty'] = len(result.stdout.strip()) > 0
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return info


def get_pip_freeze() -> Optional[str]:
    """Get pip freeze output for dependency tracking."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def compute_data_hash(data_paths: list) -> str:
    """
    Compute a hash of input data files for provenance.
    
    Args:
        data_paths: List of paths to data files
        
    Returns:
        MD5 hash of file sizes and modification times
    """
    hash_input = []
    for path in data_paths:
        p = Path(path)
        if p.exists():
            stat = p.stat()
            hash_input.append(f"{p.name}:{stat.st_size}:{stat.st_mtime}")
    
    return hashlib.md5('|'.join(sorted(hash_input)).encode()).hexdigest()[:12]


def create_run_provenance(
    seed: int = 42,
    data_paths: Optional[list] = None,
    script_path: Optional[str] = None,
) -> RunProvenance:
    """
    Create a complete provenance record for the current run.
    
    Args:
        seed: RNG seed being used
        data_paths: List of input data file paths
        script_path: Path to the main script being run
        
    Returns:
        RunProvenance object
    """
    global RUN_PROVENANCE
    
    git_info = get_git_info()
    
    hostname = None
    try:
        import socket
        hostname = socket.gethostname()
    except Exception:
        pass
    
    provenance = RunProvenance(
        timestamp=datetime.now().isoformat(),
        python_version=sys.version.split()[0],
        numpy_seed=seed,
        torch_seed=seed if HAS_TORCH else None,
        optuna_seed=seed if HAS_OPTUNA else None,
        git_commit=git_info['git_commit'],
        git_branch=git_info['git_branch'],
        git_dirty=git_info['git_dirty'],
        pip_freeze=get_pip_freeze(),
        hostname=hostname,
        working_directory=os.getcwd(),
        script_path=script_path or (sys.argv[0] if sys.argv else None),
        data_hash=compute_data_hash(data_paths) if data_paths else None,
    )
    
    RUN_PROVENANCE = provenance
    return provenance


def save_run_provenance(
    output_dir: Path,
    seed: int = 42,
    data_paths: Optional[list] = None,
    additional_info: Optional[Dict] = None,
) -> Path:
    """
    Save complete run provenance to output directory.
    
    Args:
        output_dir: Directory to save provenance files
        seed: RNG seed used
        data_paths: Input data file paths
        additional_info: Any additional info to include
        
    Returns:
        Path to provenance JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create provenance record
    provenance = create_run_provenance(seed, data_paths)
    
    # Save provenance JSON (without pip_freeze for readability)
    prov_dict = provenance.to_dict()
    pip_freeze = prov_dict.pop('pip_freeze', None)
    
    if additional_info:
        prov_dict['additional'] = additional_info
    
    prov_path = output_dir / 'provenance.json'
    with open(prov_path, 'w') as f:
        json.dump(prov_dict, f, indent=2)
    
    # Save pip freeze separately
    if pip_freeze:
        freeze_path = output_dir / 'requirements_frozen.txt'
        with open(freeze_path, 'w') as f:
            f.write(pip_freeze)
    
    # Save seed file for quick reference
    seed_path = output_dir / 'seed.txt'
    with open(seed_path, 'w') as f:
        f.write(f"{seed}\n")
    
    print(f"✓ Saved run provenance to {output_dir}")
    
    return prov_path


def log_dropped_data(
    n_neurons_original: int,
    n_neurons_kept: int,
    n_timepoints_original: int,
    n_timepoints_kept: int,
    n_nans_found: int,
    output_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Log information about dropped/filtered data.
    
    Args:
        n_neurons_original: Original neuron count
        n_neurons_kept: Neurons kept after filtering
        n_timepoints_original: Original timepoint count
        n_timepoints_kept: Timepoints kept after filtering
        n_nans_found: Number of NaN values found
        output_file: Optional path to save log
        
    Returns:
        Dict with drop statistics
    """
    stats = {
        'neurons_original': n_neurons_original,
        'neurons_kept': n_neurons_kept,
        'neurons_dropped': n_neurons_original - n_neurons_kept,
        'neurons_dropped_pct': 100 * (1 - n_neurons_kept / max(1, n_neurons_original)),
        'timepoints_original': n_timepoints_original,
        'timepoints_kept': n_timepoints_kept,
        'timepoints_dropped': n_timepoints_original - n_timepoints_kept,
        'timepoints_dropped_pct': 100 * (1 - n_timepoints_kept / max(1, n_timepoints_original)),
        'nans_found': n_nans_found,
    }
    
    print(f"  Data filtering: {n_neurons_kept}/{n_neurons_original} neurons, "
          f"{n_timepoints_kept}/{n_timepoints_original} timepoints, "
          f"{n_nans_found} NaNs")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    return stats

