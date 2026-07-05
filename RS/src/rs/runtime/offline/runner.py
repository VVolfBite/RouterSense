"""Canonical offline runtime exports.

This module intentionally re-exports the formal offline trace, traffic, and
prediction helpers from ``rs.runtime.offline`` without routing back through the
historical offline compatibility namespaces.
"""

from __future__ import annotations

from rs.runtime.offline.prediction.calibration import measure_asymmetry, run_dc_asymmetry_analysis
from rs.runtime.offline.prediction.cross_layer import (
    GatePredictionStat,
    LayerTransitionStat,
    analyze_cross_layer_correlation,
    analyze_cross_layer_predictability,
    build_batch_rank_correlation,
    evaluate_gate2,
    load_gate_weight_bundle,
    load_hidden_state_bundle,
    spearman_rank_correlation,
)
from rs.runtime.offline.trace.olmoe import (
    collect_full_sequence_trace,
    collect_moe_architecture_probe,
    collect_olmoe_router_trace,
    discover_moe_layer_ids,
    extract_gate_weights,
)
from rs.runtime.offline.trace.qwen import (
    collect_qwen_moe_architecture_probe,
    collect_qwen_moe_full_sequence_trace,
    collect_qwen_moe_router_trace,
)
from rs.runtime.offline.traffic.matrix_builder import (
    TraceRecord,
    build_owner_by_expert,
    build_predicted_traffic,
    build_sample_layer_matrices,
    combine_matrix_from_dispatch,
    load_trace_jsonl,
)

__all__ = [name for name in globals() if not name.startswith("_")]
