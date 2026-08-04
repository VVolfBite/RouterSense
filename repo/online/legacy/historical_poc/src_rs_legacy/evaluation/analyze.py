from __future__ import annotations

"""Analysis-stage evaluation helpers.

This is a compatibility-friendly aggregation layer over the existing legacy
modules. It lets new code import by pipeline stage while preserving historical
entry points and behavior.
"""

from .analysis import FAST_ALGORITHMS, compute_effective_makespan, run_pairwise_analysis, write_json
from .cross_layer import (
    analyze_cross_layer_correlation,
    analyze_cross_layer_predictability,
    build_batch_rank_correlation,
    evaluate_gate2,
)
from .dc_asymmetry import run_dc_asymmetry_analysis
from .traffic_matrix import build_owner_by_expert, build_predicted_traffic, build_same_prompt_batches, build_sample_layer_matrices, combine_matrix_from_dispatch

__all__ = [
    "FAST_ALGORITHMS",
    "analyze_cross_layer_correlation",
    "analyze_cross_layer_predictability",
    "build_batch_rank_correlation",
    "build_owner_by_expert",
    "build_predicted_traffic",
    "build_same_prompt_batches",
    "build_sample_layer_matrices",
    "combine_matrix_from_dispatch",
    "compute_effective_makespan",
    "evaluate_gate2",
    "run_dc_asymmetry_analysis",
    "run_pairwise_analysis",
    "write_json",
]
