"""
Neuron Type Classification Utilities

Parse Cook et al. 2019 SI 6 Cell class lists to extract neuron type classifications:
- Sensory neurons (SN1-SN6)
- Interneurons (IN1-IN4)  
- Motor neurons (head motor, sublateral motor)

This module provides functions to load and query neuron types for analysis.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# CANONICAL NEURON TYPE MAPPINGS (from Cook et al. 2019 SI 6)
# =============================================================================

# Neuron classes from Cook et al. 2019 "SI 6 Cell class lists.xlsx"
# Organized by functional category

# SN = Sensory Neurons (6 tiers based on layer/connectivity)
SENSORY_NEURONS = {
    # SN1 - First layer sensory
    'ASI', 'ASJ', 'AWA', 'ASG', 'AWB', 'ASE', 'ADF', 'AFD', 'AWC', 'ASK', 'ASH', 'ADL',
    # SN2 - Amphid channel, etc.
    'BAG', 'URX',
    # SN3 - Mechanosensory, nociceptive
    'ALN', 'PLN', 'SDQ', 'AQR', 'PQR', 'ALM', 'AVM', 'PVM', 'PLM', 'FLP', 'DVA', 'PVD',
    # SN4 - Dopaminergic sensory
    'ADE', 'PDE',
    # SN5 - Phasmid sensory
    'PHA', 'PHB', 'PHC',
    # SN6 - Other sensory
    'CEP', 'OLQ', 'OLL', 'IL1', 'IL2', 'URY', 'URB', 'URA',
}

# IN = Interneurons (4 tiers)
INTERNEURONS = {
    # IN1 - Layer 1 interneurons (amphid processing)
    'AIA', 'AIB', 'AIY', 'AIZ', 'AIM', 'AIN', 'RIA', 'RIB', 'RIG', 'RIH', 'RIS', 'RIF',
    # IN2 - Layer 2 interneurons
    'AVA', 'AVB', 'AVD', 'AVE', 'AVG', 'AVH', 'AVJ', 'AVK', 'AVF', 'AVL', 'PVP', 'PVQ', 'PVT', 'PVW', 'PVN', 'DVB', 'DVC',
    # IN3 - Integration/command
    'RIM', 'RIR', 'RIC', 'RIP', 'RID',
    # IN4 - Pharyngeal and other
    'ADA', 'ALA', 'BDU', 'HSN', 'LUA', 'PVC', 'PVR', 'RMG',
}

# Motor neurons
MOTOR_NEURONS = {
    # Head motor neurons
    'RIV', 'RMD', 'RME', 'RMF', 'RMH', 'SAA', 'SAB', 'SIA', 'SIB', 'SMB', 'SMD',
    # Sublateral motor neurons (typically ventral cord, but SMD can be head)
    'VC', 'VD', 'VA', 'VB', 'AS', 'DA', 'DB', 'DD'  # These are ventral cord motor neurons
}

# Build lookup dictionary
NEURON_TYPE_MAP = {}
for n in SENSORY_NEURONS:
    NEURON_TYPE_MAP[n] = 'sensory'
for n in INTERNEURONS:
    NEURON_TYPE_MAP[n] = 'interneuron'
for n in MOTOR_NEURONS:
    NEURON_TYPE_MAP[n] = 'motor'


def get_neuron_type(neuron_name: str) -> str:
    """
    Get the functional type of a neuron.
    
    Args:
        neuron_name: Canonical neuron class name (e.g., 'AVA', 'AWC')
        
    Returns:
        'sensory', 'interneuron', 'motor', or 'unknown'
    """
    name = neuron_name.strip().upper()
    
    # First check exact match (e.g. ADL, AQR, VC)
    if name in NEURON_TYPE_MAP:
        return NEURON_TYPE_MAP[name]
        
    # Handle bilateral suffixes (L/R) only if exact match failed
    if len(name) > 1 and name[-1] in ('L', 'R'):
        base_name = name[:-1]
        if base_name in NEURON_TYPE_MAP:
             return NEURON_TYPE_MAP[base_name]
             
    return 'unknown'


def get_neuron_types_for_list(neuron_names: List[str]) -> Dict[str, str]:
    """
    Get neuron types for a list of neurons.
    
    Args:
        neuron_names: List of neuron class names
        
    Returns:
        Dictionary mapping neuron name to type
    """
    return {n: get_neuron_type(n) for n in neuron_names}


def get_neurons_by_type(neuron_names: List[str], neuron_type: str) -> List[str]:
    """
    Filter neurons by type.
    
    Args:
        neuron_names: List of neuron names to filter
        neuron_type: 'sensory', 'interneuron', 'motor', or 'unknown'
        
    Returns:
        List of neurons matching the specified type
    """
    return [n for n in neuron_names if get_neuron_type(n) == neuron_type]


def get_type_counts(neuron_names: List[str]) -> Dict[str, int]:
    """
    Count neurons by type.
    
    Args:
        neuron_names: List of neuron names
        
    Returns:
        Dictionary with counts per type
    """
    counts = {'sensory': 0, 'interneuron': 0, 'motor': 0, 'unknown': 0}
    for n in neuron_names:
        ntype = get_neuron_type(n)
        counts[ntype] = counts.get(ntype, 0) + 1
    return counts


def parse_cook_si6_file(si6_path: Path) -> Dict[str, str]:
    """
    Parse Cook et al. SI 6 Cell class lists.xlsx to extract neuron type mappings.
    
    The file has a specific structure:
    - Column 0: Category codes (SN1-SN6, IN1-IN4, etc.)
    - Column 1: Neuron class names (e.g., ASI, ASJ, AWA)
    
    Args:
        si6_path: Path to SI 6 Cell class lists.xlsx
        
    Returns:
        Dictionary mapping neuron name to type ('sensory', 'interneuron', 'motor')
    """
    if not si6_path.exists():
        print(f"Warning: SI 6 file not found at {si6_path}, using built-in mappings")
        return NEURON_TYPE_MAP.copy()
    
    df = pd.read_excel(si6_path, sheet_name='cell class lists', header=None)
    
    neuron_to_type = {}
    current_type = None
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        col0 = str(row[0]) if pd.notna(row[0]) else ''
        col1 = str(row[1]) if pd.notna(row[1]) else ''
        
        # Check for category headers
        if col0.strip() in ['SN1', 'SN2', 'SN3', 'SN4', 'SN5', 'SN6']:
            current_type = 'sensory'
        elif col0.strip() in ['IN1', 'IN2', 'IN3', 'IN4']:
            current_type = 'interneuron'
        elif 'HEAD MOTOR' in col0.upper() or 'SUBLATERAL MOTOR' in col0.upper():
            current_type = 'motor'
        
        # Extract neuron name from column 1
        if current_type and col1 and len(col1.strip()) > 0 and len(col1.strip()) < 10:
            neuron_name = col1.strip().upper()
            if neuron_name not in ['NAN', 'SHARED NEURONS AND MUSCLES']:
                neuron_to_type[neuron_name] = current_type
    
    return neuron_to_type


def create_neuron_metadata(nodes_json_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Create a comprehensive neuron metadata file.
    
    Args:
        nodes_json_path: Path to nodes.json with neuron list
        output_path: Path to save the metadata CSV
        
    Returns:
        DataFrame with neuron metadata
    """
    with open(nodes_json_path, 'r') as f:
        neurons = json.load(f)
    
    records = []
    for i, neuron in enumerate(neurons):
        ntype = get_neuron_type(neuron)
        records.append({
            'index': i,
            'neuron': neuron,
            'type': ntype,
            'is_sensory': ntype == 'sensory',
            'is_interneuron': ntype == 'interneuron',
            'is_motor': ntype == 'motor',
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    
    print(f"Created neuron metadata with {len(df)} neurons:")
    type_counts = df['type'].value_counts()
    for t, c in type_counts.items():
        print(f"  {t}: {c}")
    
    return df


if __name__ == "__main__":
    # Test the module
    import sys
    project_root = Path(__file__).parent.parent.parent
    nodes_path = project_root / "results" / "intermediate" / "connectome" / "nodes.json"
    
    if nodes_path.exists():
        with open(nodes_path, 'r') as f:
            neurons = json.load(f)
        
        print(f"\nNeuron type classification for {len(neurons)} neurons:\n")
        
        type_counts = get_type_counts(neurons)
        for t, c in type_counts.items():
            print(f"  {t}: {c}")
        
        print("\nSample classifications:")
        for n in neurons[:10]:
            print(f"  {n}: {get_neuron_type(n)}")
        
        # Create metadata file
        output_path = project_root / "results" / "intermediate" / "connectome" / "neuron_metadata.csv"
        create_neuron_metadata(nodes_path, output_path)
    else:
        print(f"nodes.json not found at {nodes_path}")
