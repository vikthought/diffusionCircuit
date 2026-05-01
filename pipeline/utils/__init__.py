"""
Shared utility modules for the SBTG pipeline.

This package provides canonical implementations for:
- io: Data loading with provenance tracking
- align: Node alignment and direction convention enforcement
- labels: Leifer atlas labeling (single source of truth)
- metrics: Evaluation metrics (AUROC, AUPRC, F1@density, specificity)
- reproducibility: RNG seeds, version pinning, logging
"""

from .io import (
    load_neuropal_data,
    load_structural_connectome,
    load_leifer_atlas,
    save_result_bundle,
    load_result_bundle,
)

from .align import (
    normalize_neuron_name,
    find_common_neurons,
    align_matrices,
    align_timeseries_to_connectome,
    validate_node_order,
    assert_direction_convention,
    DIRECTION_CONVENTION,
    collapse_dv_subtypes,
    collapse_all_dv_subtypes,
    DV_COLLAPSE_PATTERNS,
)

from .labels import (
    create_leifer_labels,
    LeiferLabels,
)

from .metrics import (
    compute_binary_metrics,
    compute_auroc_auprc,
    compute_specificity_on_negatives,
    compute_random_baseline_f1,
    compute_metrics_at_density,
)

from .reproducibility import (
    set_all_seeds,
    get_seed_state,
    save_run_provenance,
    RUN_PROVENANCE,
)

from .neuron_types import (
    get_neuron_type,
    get_neuron_types_for_list,
    get_neurons_by_type,
    get_type_counts,
    SENSORY_NEURONS,
    INTERNEURONS,
    MOTOR_NEURONS,
)

__all__ = [
    # io
    'load_neuropal_data',
    'load_structural_connectome', 
    'load_leifer_atlas',
    'save_result_bundle',
    'load_result_bundle',
    # align
    'normalize_neuron_name',
    'find_common_neurons',
    'align_matrices',
    'validate_node_order',
    'assert_direction_convention',
    'DIRECTION_CONVENTION',
    # labels
    'create_leifer_labels',
    'LeiferLabels',
    # metrics
    'compute_binary_metrics',
    'compute_auroc_auprc',
    'compute_specificity_on_negatives',
    'compute_random_baseline_f1',
    'compute_metrics_at_density',
    # reproducibility
    'set_all_seeds',
    'get_seed_state',
    'save_run_provenance',
    'RUN_PROVENANCE',
    # neuron types
    'get_neuron_type',
    'get_neuron_types_for_list',
    'get_neurons_by_type',
    'get_type_counts',
    'SENSORY_NEURONS',
    'INTERNEURONS',
    'MOTOR_NEURONS',]