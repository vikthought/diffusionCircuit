"""
Data loading utilities with provenance tracking.

This module provides canonical loaders for all data sources used in the pipeline.
All loaders:
1. Return data in a consistent format
2. Include node_order with every matrix
3. Log data provenance (shapes, NaN counts, etc.)
4. Respect worm boundaries for time series data
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .align import normalize_neuron_name, DIRECTION_CONVENTION


@dataclass
class NeuroPALData:
    """
    Container for NeuroPAL calcium imaging data.
    
    IMPORTANT: Data is organized per-worm to avoid cross-worm discontinuities.
    Methods that need continuous time series should operate per-worm.
    """
    # Per-worm traces: List of (T_w, n_neurons) arrays
    # Each array is a single worm's recording
    traces_per_worm: List[np.ndarray]
    
    # Neuron names in consistent order
    neuron_names: List[str]
    
    # Worm identifiers
    worm_ids: List[str]
    
    # Stimulus information
    stim_names: List[str]
    stim_times: np.ndarray  # (n_stimuli, 2) start/end times
    stims_per_worm: List[np.ndarray]  # Stimulus order for each worm
    
    # Recording parameters
    fps: float
    
    # Data provenance
    source_file: str
    n_nan_values: int
    n_neurons_dropped: int
    
    @property
    def n_worms(self) -> int:
        return len(self.worm_ids)
    
    @property
    def n_neurons(self) -> int:
        return len(self.neuron_names)
    
    def get_concatenated_traces(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get concatenated traces WITH worm boundary markers.
        
        Returns:
            Tuple of:
            - traces: (total_T, n_neurons) concatenated traces
            - worm_boundaries: Array of indices where each worm starts
            
        WARNING: Only use this for methods that properly handle boundaries!
        """
        traces = np.concatenate(self.traces_per_worm, axis=0)
        boundaries = np.cumsum([0] + [t.shape[0] for t in self.traces_per_worm[:-1]])
        return traces, boundaries
    
    def get_stimulus_windows(
        self,
        stimulus_name: str,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Get stimulus response windows per worm.
        
        Args:
            stimulus_name: Name of stimulus ("nacl", "butanone", "pentanedione")
            
        Returns:
            List of (worm_idx, window_array) tuples
            Each window_array is (n_frames, n_neurons)
        """
        stim_idx = None
        for i, name in enumerate(self.stim_names):
            if name.lower() == stimulus_name.lower():
                stim_idx = i
                break
        
        if stim_idx is None:
            raise ValueError(f"Unknown stimulus: {stimulus_name}")
        
        start_sec, end_sec = self.stim_times[stim_idx]
        start_frame = int(start_sec * self.fps)
        end_frame = int(end_sec * self.fps)
        
        windows = []
        for worm_idx, traces in enumerate(self.traces_per_worm):
            if end_frame <= traces.shape[0]:
                window = traces[start_frame:end_frame, :]
                windows.append((worm_idx, window))
        
        return windows


def load_neuropal_data(
    mat_file: Path,
    min_worms: int = 1,
    normalize_names: bool = True,
) -> NeuroPALData:
    """
    Load NeuroPAL calcium imaging data from MAT file.
    
    Args:
        mat_file: Path to MAT file
        min_worms: Minimum number of worms a neuron must appear in
        normalize_names: If True, normalize neuron names to uppercase
        
    Returns:
        NeuroPALData object with per-worm organization
    """
    if not HAS_SCIPY:
        raise ImportError("scipy required to load MAT files")
    
    mat_file = Path(mat_file)
    mat = loadmat(mat_file, simplify_cells=True)
    
    # Extract neuron names
    raw_names = mat['neurons']
    if normalize_names:
        neuron_names = [normalize_neuron_name(str(n)) for n in raw_names]
    else:
        neuron_names = [str(n) for n in raw_names]
    
    # Extract traces
    norm_traces = mat['norm_traces']
    
    # Determine number of worms
    worm_ids = [str(f) for f in mat['files']]
    n_worms = len(worm_ids)
    
    # Build per-worm traces
    traces_per_worm = []
    n_nan_total = 0
    
    # Get trace length from first valid trace
    trace_length = None
    for neuron_traces in norm_traces:
        if isinstance(neuron_traces, np.ndarray) and len(neuron_traces) > 0:
            if isinstance(neuron_traces[0], np.ndarray):
                trace_length = len(neuron_traces[0])
                break
    
    if trace_length is None:
        raise ValueError("Could not determine trace length from data")
    
    for worm_idx in range(n_worms):
        # Collect traces for this worm
        worm_traces = []
        
        for neuron_idx, neuron_traces in enumerate(norm_traces):
            if isinstance(neuron_traces, np.ndarray) and len(neuron_traces) > worm_idx:
                trace = neuron_traces[worm_idx]
                if isinstance(trace, np.ndarray) and len(trace) >= trace_length:
                    trace = trace[:trace_length].astype(float)
                else:
                    trace = np.full(trace_length, np.nan)
            else:
                trace = np.full(trace_length, np.nan)
            
            n_nan_total += np.sum(np.isnan(trace))
            worm_traces.append(trace)
        
        worm_traces = np.column_stack(worm_traces)
        traces_per_worm.append(worm_traces)
    
    # Stimulus information
    stim_names = [str(s) for s in mat['stim_names']]
    stim_times = np.asarray(mat['stim_times'], dtype=float)
    stims_per_worm = [np.asarray(row, dtype=int) for row in mat['stims']]
    fps = float(mat['fps'])
    
    return NeuroPALData(
        traces_per_worm=traces_per_worm,
        neuron_names=neuron_names,
        worm_ids=worm_ids,
        stim_names=stim_names,
        stim_times=stim_times,
        stims_per_worm=stims_per_worm,
        fps=fps,
        source_file=str(mat_file),
        n_nan_values=n_nan_total,
        n_neurons_dropped=0,
    )


def load_structural_connectome(
    connectome_dir: Path,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Load structural connectome from preprocessed files.
    
    Args:
        connectome_dir: Directory containing A_struct.npy and nodes.json
        
    Returns:
        Tuple of:
        - A_struct: Adjacency matrix (n, n) with convention A[post, pre]
        - node_order: List of neuron names
        - metadata: Dict with connectome statistics
    """
    connectome_dir = Path(connectome_dir)
    
    struct_file = connectome_dir / "A_struct.npy"
    nodes_file = connectome_dir / "nodes.json"
    
    if not struct_file.exists():
        raise FileNotFoundError(f"Structural connectome not found: {struct_file}")
    if not nodes_file.exists():
        raise FileNotFoundError(f"Node order not found: {nodes_file}")
    
    A_struct = np.load(struct_file)
    with open(nodes_file, 'r') as f:
        node_order = json.load(f)
    
    n = len(node_order)
    n_edges = int(np.sum(A_struct > 0))
    
    metadata = {
        'n_neurons': n,
        'n_edges': n_edges,
        'density': n_edges / (n * (n - 1)),
        'direction_convention': DIRECTION_CONVENTION,
        'source_dir': str(connectome_dir),
    }
    
    return A_struct, node_order, metadata


def load_leifer_atlas(
    atlas_dir: Path,
    genotype: str = "wild-type",
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load Leifer functional atlas.
    
    Args:
        atlas_dir: Directory containing aligned_atlas_*.npz files
        genotype: "wild-type" or "unc-31"
        
    Returns:
        Tuple of:
        - data: Dict with 'q', 'q_eq', 'amplitude' matrices
        - node_order: List of neuron names
    """
    atlas_dir = Path(atlas_dir)
    atlas_file = atlas_dir / f"aligned_atlas_{genotype}.npz"
    
    if not atlas_file.exists():
        raise FileNotFoundError(f"Leifer atlas not found: {atlas_file}")
    
    data = dict(np.load(atlas_file, allow_pickle=True))
    
    # Extract node order
    if 'neurons' in data:
        node_order = list(data['neurons'])
    elif 'node_order' in data:
        node_order = list(data['node_order'])
    else:
        raise ValueError(f"No node_order found in {atlas_file}")
    
    return data, node_order


@dataclass
class ResultBundle:
    """
    Container for SBTG result files.
    
    Enforces the separation of continuous scores vs thresholded graphs.
    """
    # Continuous scores (for AUROC/AUPRC)
    mu_hat: np.ndarray  # Coupling estimates
    p_mean: np.ndarray  # P-values for mean test
    p_volatility: Optional[np.ndarray]  # P-values for volatility test
    
    # Thresholded graphs (for binary metrics)
    sign_adj: np.ndarray  # Signed adjacency {-1, 0, +1}
    volatility_adj: np.ndarray  # Volatility adjacency {0, 1}
    
    # Node order (REQUIRED)
    node_order: List[str]
    
    # Configuration
    config: Dict[str, Any]
    
    @property
    def combined_adj(self) -> np.ndarray:
        """Combined adjacency: sign ∪ volatility edges."""
        combined = self.sign_adj.copy().astype(float)
        volatility_only = (self.sign_adj == 0) & (self.volatility_adj != 0)
        combined[volatility_only] = 1.0
        return combined


def save_result_bundle(
    output_dir: Path,
    mu_hat: np.ndarray,
    p_mean: np.ndarray,
    sign_adj: np.ndarray,
    node_order: List[str],
    config: Dict[str, Any],
    p_volatility: Optional[np.ndarray] = None,
    volatility_adj: Optional[np.ndarray] = None,
) -> None:
    """
    Save SBTG results with proper separation of continuous vs binary outputs.
    
    Args:
        output_dir: Directory to save results
        mu_hat: Coupling estimates (continuous)
        p_mean: P-values for mean test (continuous)
        sign_adj: Signed adjacency after FDR (binary/ternary)
        node_order: Neuron names (REQUIRED)
        config: Configuration dict
        p_volatility: Optional volatility p-values
        volatility_adj: Optional volatility adjacency
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save continuous scores
    np.savez(
        output_dir / "continuous_scores.npz",
        mu_hat=mu_hat,
        p_mean=p_mean,
        p_volatility=p_volatility if p_volatility is not None else np.array([]),
        node_order=np.array(node_order, dtype=object),
    )
    
    # Save thresholded graphs
    np.savez(
        output_dir / "thresholded_graphs.npz",
        sign_adj=sign_adj,
        volatility_adj=volatility_adj if volatility_adj is not None else np.zeros_like(sign_adj),
        node_order=np.array(node_order, dtype=object),
    )
    
    # Save legacy result.npz for backward compatibility
    np.savez(
        output_dir / "result.npz",
        sign_adj=sign_adj,
        volatility_adj=volatility_adj if volatility_adj is not None else np.zeros_like(sign_adj),
        mu_hat=mu_hat,
        p_mean=p_mean,
        p_volatility=p_volatility if p_volatility is not None else np.array([]),
    )
    
    # Save node order separately for easy inspection
    with open(output_dir / "node_order.json", 'w') as f:
        json.dump(node_order, f, indent=2)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        # Convert numpy types to Python types
        config_clean = {}
        for k, v in config.items():
            if isinstance(v, np.ndarray):
                config_clean[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                config_clean[k] = float(v)
            else:
                config_clean[k] = v
        json.dump(config_clean, f, indent=2)


def load_result_bundle(
    result_dir: Path,
    expected_node_order: Optional[List[str]] = None,
) -> ResultBundle:
    """
    Load SBTG results and verify node order.
    
    Args:
        result_dir: Directory containing result files
        expected_node_order: If provided, verify node order matches
        
    Returns:
        ResultBundle object
    """
    result_dir = Path(result_dir)
    
    # Try new format first, fall back to legacy
    if (result_dir / "continuous_scores.npz").exists():
        cont = np.load(result_dir / "continuous_scores.npz", allow_pickle=True)
        thresh = np.load(result_dir / "thresholded_graphs.npz", allow_pickle=True)
        
        mu_hat = cont['mu_hat']
        p_mean = cont['p_mean']
        p_volatility = cont['p_volatility'] if cont['p_volatility'].size > 0 else None
        sign_adj = thresh['sign_adj']
        volatility_adj = thresh['volatility_adj']
        node_order = list(cont['node_order'])
    else:
        # Legacy format
        data = np.load(result_dir / "result.npz", allow_pickle=True)
        mu_hat = data.get('mu_hat', data.get('W_param', np.array([])))
        p_mean = data.get('p_mean', np.array([]))
        p_volatility = data.get('p_volatility', None)
        sign_adj = data['sign_adj']
        volatility_adj = data.get('volatility_adj', np.zeros_like(sign_adj))
        
        # Try to load node order
        node_order_file = result_dir / "node_order.json"
        neuron_names_file = result_dir / "neuron_names.json"
        
        if node_order_file.exists():
            with open(node_order_file) as f:
                node_order = json.load(f)
        elif neuron_names_file.exists():
            with open(neuron_names_file) as f:
                node_order = json.load(f)
        else:
            raise FileNotFoundError(
                f"No node_order.json found in {result_dir}. "
                "Results without node order are not valid."
            )
    
    # Validate node order if expected
    if expected_node_order is not None:
        from .align import validate_node_order
        validate_node_order(node_order, expected_node_order, context=str(result_dir))
    
    # Load config
    config_file = result_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}
    
    return ResultBundle(
        mu_hat=mu_hat,
        p_mean=p_mean,
        p_volatility=p_volatility,
        sign_adj=sign_adj,
        volatility_adj=volatility_adj,
        node_order=node_order,
        config=config,
    )

