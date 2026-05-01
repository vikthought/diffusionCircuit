"""
Leifer Atlas Utilities
======================

Utilities for loading and aligning the Leifer functional atlas.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
LEIFER_DIR = PROJECT_ROOT / "results" / "leifer_evaluation"
CONNECTOME_DIR = PROJECT_ROOT / "results" / "intermediate" / "connectome"
DATASETS_DIR = PROJECT_ROOT / "results" / "intermediate" / "datasets"

ALPHA = 0.05

def ensure_aligned_atlas():
    """
    Ensure aligned Leifer atlases exist using wormneuroatlas.
    Creates them if missing by aligning to our neuron order.
    """
    wt_file = LEIFER_DIR / "aligned_atlas_wild-type.npz"
    unc31_file = LEIFER_DIR / "aligned_atlas_unc-31.npz"
    
    if wt_file.exists() and unc31_file.exists():
        return
        
    print(f"  [utils.leifer] Creating aligned Leifer atlases...")
    
    try:
        import wormneuroatlas as wa
    except ImportError:
        print("  ERROR: wormneuroatlas not installed. Cannot create Leifer atlas.")
        return
        
    LEIFER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load our neuron order from datasets
    meta_file = DATASETS_DIR / "full_traces" / "standardization.json"
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            our_neurons = json.load(f).get("node_order", [])
    else:
        # Fallback to connectome or any stimulus
        # Try nacl
        meta_file = DATASETS_DIR / "nacl" / "standardization.json"
        if meta_file.exists():
             with open(meta_file, 'r') as f:
                our_neurons = json.load(f).get("node_order", [])
        else:
             # Fallback to connectome
            nodes_file = CONNECTOME_DIR / "nodes.json"
            if nodes_file.exists():
                with open(nodes_file, 'r') as f:
                    our_neurons = json.load(f)
            else:
                print("  ERROR: No neuron order found (run 01_prepare_data.py first)")
                return
    
    # Check neuron count
    if len(our_neurons) == 0:
        print("  ERROR: Empty neuron list")
        return

    print(f"  Aligning to {len(our_neurons)} neurons")
    
    # Load atlas
    atlas = wa.NeuroAtlas()
    h5 = atlas.funatlas_h5
    atlas_neurons = [n.decode('utf-8') for n in h5['neuron_ids'][:]]
    
    # Import alignment util
    from pipeline.utils.align import merge_bilateral_name
    
    # Group atlas indices
    atlas_groups = {}
    for idx, name in enumerate(atlas_neurons):
        collapsed = merge_bilateral_name(name)
        if collapsed not in atlas_groups:
            atlas_groups[collapsed] = []
        atlas_groups[collapsed].append(idx)
        
    # Find common
    common = [n for n in our_neurons if n in atlas_groups]
    
    # Save aligned atlases
    for genotype, gkey in [("wild-type", "wt"), ("unc-31", "unc31")]:
        if gkey not in h5:
            continue
            
        g = h5[gkey]
        q_full = np.array(g['q'])
        q_eq_full = np.array(g['q_eq'])
        dff_full = np.array(g['dFF'])
        
        n = len(common)
        q = np.full((n, n), np.nan)
        q_eq = np.full((n, n), np.nan)
        dff = np.full((n, n), np.nan)
        
        for i, target_i in enumerate(common):
            for j, target_j in enumerate(common):
                src_indices = atlas_groups[target_j] # pre
                tgt_indices = atlas_groups[target_i] # post
                
                sub_q = q_full[np.ix_(tgt_indices, src_indices)]
                sub_q_eq = q_eq_full[np.ix_(tgt_indices, src_indices)]
                sub_dff = dff_full[np.ix_(tgt_indices, src_indices)]
                
                if sub_q.size > 0:
                    if not np.all(np.isnan(sub_q)):
                        q[i, j] = np.nanmin(sub_q)
                    if not np.all(np.isnan(sub_q_eq)):
                        q_eq[i, j] = np.nanmin(sub_q_eq)
                    if not np.all(np.isnan(sub_dff)):
                        dff[i, j] = np.nanmax(sub_dff)
        
        output_file = LEIFER_DIR / f"aligned_atlas_{genotype}.npz"
        np.savez(
            output_file,
            q=q,
            q_eq=q_eq,
            dff=dff,
            neuron_order=np.array(common, dtype=object)
        )
        print(f"  Saved {output_file.name}")

def load_leifer_atlas_data() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """Load Leifer Wild-Type atlas data."""
    ensure_aligned_atlas()
    
    wt_file = LEIFER_DIR / "aligned_atlas_wild-type.npz"
    if not wt_file.exists():
        return None, None, None
        
    data = np.load(wt_file, allow_pickle=True)
    q = data['q']
    q_eq = data.get('q_eq', None)
    neurons = list(data['neuron_order'])
    
    return q, q_eq, neurons
