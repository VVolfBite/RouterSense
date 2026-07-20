from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rs.runtime.offline.prediction.cross_layer import _summary_stats
from rs.runtime.offline.traffic.matrix_builder import (
    build_owner_by_expert,
    build_sample_layer_matrices,
    combine_matrix_from_dispatch,
    load_trace_jsonl,
)


def measure_asymmetry(dispatch_matrix: list[list[int]]) -> dict[str, float | int]:
    num_gpus = len(dispatch_matrix)
    dst_load = [sum(dispatch_matrix[src][dst] for src in range(num_gpus)) for dst in range(num_gpus)]
    src_load = [sum(dispatch_matrix[src][dst] for dst in range(num_gpus)) for src in range(num_gpus)]
    mean_dst = sum(dst_load) / max(len(dst_load), 1)
    mean_src = sum(src_load) / max(len(src_load), 1)
    dispatch_dst_skew = max(dst_load) / max(mean_dst, 1e-9)
    combine_src_skew = max(src_load) / max(mean_src, 1e-9)
    nonzero_dispatch = sum(1 for row in dispatch_matrix for value in row if value > 0)
    total_pairs = num_gpus * num_gpus
    sparsity = 1.0 - (nonzero_dispatch / max(total_pairs, 1))
    asymmetry_ratio = dispatch_dst_skew / max(combine_src_skew, 1e-9)
    return {
        "dispatch_dst_skew": float(dispatch_dst_skew),
        "combine_src_skew": float(combine_src_skew),
        "asymmetry_ratio": float(asymmetry_ratio),
        "sparsity": float(sparsity),
        "nonzero_pairs": int(nonzero_dispatch),
        "total_pairs": int(total_pairs),
    }


def load_pairwise_results_index(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(str(row["sample_id"]), str(row["layer_pair"])): row for row in payload}


def run_dc_asymmetry_analysis(
    *,
    trace_jsonl: str | Path,
    placement: str,
    num_gpus: int,
    pairwise_results_json: str | Path | None = None,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    records = load_trace_jsonl(trace_jsonl)
    owner_by_expert = build_owner_by_expert(records, placement=placement, num_gpus=num_gpus)
    sample_layer_matrices = build_sample_layer_matrices(records, owner_by_expert=owner_by_expert, num_gpus=num_gpus)
    sample_ids = sorted(sample_layer_matrices)
    if sample_limit is not None:
        sample_ids = sample_ids[:sample_limit]
    pairwise_results_index = load_pairwise_results_index(pairwise_results_json) if pairwise_results_json is not None else {}
    asymmetry_records: list[dict[str, Any]] = []
    schedule_records: list[dict[str, Any]] = []
    hotfirst_gains: list[float] = []
    for sample_id in sample_ids:
        layer_map = sample_layer_matrices[sample_id]
        sorted_layers = sorted(layer_map)
        for layer_id in sorted_layers:
            dispatch_matrix = layer_map[layer_id]
            asymmetry_records.append({"sample_id": sample_id, "layer_id": layer_id, **measure_asymmetry(dispatch_matrix)})
        for idx in range(len(sorted_layers) - 1):
            from_layer = sorted_layers[idx]
            to_layer = sorted_layers[idx + 1]
            dispatch_matrix = layer_map[from_layer]
            combine_matrix = combine_matrix_from_dispatch(dispatch_matrix, 1.0)
            layer_pair = f"{from_layer}->{to_layer}"
            oracle_row = pairwise_results_index.get((sample_id, layer_pair), {})
            schedule_records.append(
                {
                    "sample_id": sample_id,
                    "layer_pair": layer_pair,
                    "dispatch_total": sum(sum(row) for row in dispatch_matrix),
                    "combine_total": sum(sum(row) for row in combine_matrix),
                    "oracle_status": str(oracle_row.get("oracle_perfect_solver_status", "missing")),
                }
            )
            hotfirst_gains.append(float(measure_asymmetry(dispatch_matrix)["asymmetry_ratio"]))
    asymmetry_values = [float(row["asymmetry_ratio"]) for row in asymmetry_records]
    return {
        "asymmetry_records": asymmetry_records,
        "schedule_records": schedule_records,
        "overall_asymmetry": _summary_stats(asymmetry_values),
        "schedule_summary": {"a_hotfirst_gain_pct": _summary_stats(hotfirst_gains)},
        "go_no_go": {"asymmetry_significant": (sum(hotfirst_gains) / len(hotfirst_gains) if hotfirst_gains else 0.0) >= 1.3},
    }
