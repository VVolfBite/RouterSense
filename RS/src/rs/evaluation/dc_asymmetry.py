from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .poc_line1 import (
    _summary_stats,
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


def _simulate_half_duplex(
    chunks: list[tuple[int, int, int]],
    num_gpus: int,
    *,
    gpu_free_at: list[float] | None = None,
) -> tuple[float, list[float]]:
    if gpu_free_at is None:
        gpu_free_at = [0.0] * num_gpus
    availability = list(gpu_free_at)
    for src, dst, size in chunks:
        start = max(availability[src], availability[dst])
        end = start + float(size)
        availability[src] = end
        availability[dst] = end
    return max(availability) if availability else 0.0, availability


def _lpt_chunks(matrix: list[list[int]], num_gpus: int) -> list[tuple[int, int, int]]:
    chunks = [
        (src, dst, int(matrix[src][dst]))
        for src in range(num_gpus)
        for dst in range(num_gpus)
        if src != dst and matrix[src][dst] > 0
    ]
    chunks.sort(key=lambda item: -item[2])
    return chunks


def _hot_first_dispatch_chunks(matrix: list[list[int]], num_gpus: int) -> list[tuple[int, int, int]]:
    dst_load = [sum(matrix[src][dst] for src in range(num_gpus)) for dst in range(num_gpus)]
    chunks = [
        (src, dst, int(matrix[src][dst]))
        for src in range(num_gpus)
        for dst in range(num_gpus)
        if src != dst and matrix[src][dst] > 0
    ]
    chunks.sort(key=lambda item: (-dst_load[item[1]], -item[2], item[0], item[1]))
    return chunks


def _balanced_combine_chunks(matrix: list[list[int]], num_gpus: int) -> list[tuple[int, int, int]]:
    src_load = [sum(matrix[src][dst] for dst in range(num_gpus)) for src in range(num_gpus)]
    chunks = [
        (src, dst, int(matrix[src][dst]))
        for src in range(num_gpus)
        for dst in range(num_gpus)
        if src != dst and matrix[src][dst] > 0
    ]
    chunks.sort(key=lambda item: (src_load[item[0]], -item[2], item[0], item[1]))
    return chunks


def symmetric_schedule(dispatch_matrix: list[list[int]], combine_matrix: list[list[int]], num_gpus: int) -> float:
    _, after_dispatch = _simulate_half_duplex(_lpt_chunks(dispatch_matrix, num_gpus), num_gpus)
    makespan, _ = _simulate_half_duplex(_lpt_chunks(combine_matrix, num_gpus), num_gpus, gpu_free_at=after_dispatch)
    return makespan


def asymmetric_schedule(dispatch_matrix: list[list[int]], combine_matrix: list[list[int]], num_gpus: int) -> float:
    _, after_dispatch = _simulate_half_duplex(_hot_first_dispatch_chunks(dispatch_matrix, num_gpus), num_gpus)
    makespan, _ = _simulate_half_duplex(_balanced_combine_chunks(combine_matrix, num_gpus), num_gpus, gpu_free_at=after_dispatch)
    return makespan


def asymmetric_pairwise_schedule(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> float:
    _, after_dispatch = _simulate_half_duplex(_hot_first_dispatch_chunks(dispatch_matrix, num_gpus), num_gpus)
    _, after_combine = _simulate_half_duplex(_balanced_combine_chunks(combine_matrix, num_gpus), num_gpus, gpu_free_at=after_dispatch)
    makespan, _ = _simulate_half_duplex(
        _hot_first_dispatch_chunks(next_dispatch_matrix, num_gpus),
        num_gpus,
        gpu_free_at=after_combine,
    )
    return makespan


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
    pairwise_results_index = (
        load_pairwise_results_index(pairwise_results_json)
        if pairwise_results_json is not None
        else {}
    )

    asymmetry_records: list[dict[str, Any]] = []
    asymmetry_by_layer: dict[int, list[float]] = {}
    scatter_rows: list[dict[str, Any]] = []
    schedule_records: list[dict[str, Any]] = []
    hotfirst_gains: list[float] = []
    asymmetric_oracle_gains: list[float] = []
    for sample_id in sample_ids:
        layer_map = sample_layer_matrices[sample_id]
        sorted_layers = sorted(layer_map)
        for layer_id in sorted_layers:
            dispatch_matrix = layer_map[layer_id]
            metrics = measure_asymmetry(dispatch_matrix)
            row = {"sample_id": sample_id, "layer_id": layer_id, **metrics}
            asymmetry_records.append(row)
            asymmetry_by_layer.setdefault(layer_id, []).append(float(metrics["asymmetry_ratio"]))
            scatter_rows.append(
                {
                    "sample_id": sample_id,
                    "layer_id": layer_id,
                    "dispatch_dst_skew": metrics["dispatch_dst_skew"],
                    "combine_src_skew": metrics["combine_src_skew"],
                    "asymmetry_ratio": metrics["asymmetry_ratio"],
                }
            )

        for idx in range(len(sorted_layers) - 1):
            from_layer = sorted_layers[idx]
            to_layer = sorted_layers[idx + 1]
            dispatch_matrix = layer_map[from_layer]
            combine_matrix = combine_matrix_from_dispatch(dispatch_matrix, 1.0)
            next_dispatch_matrix = layer_map[to_layer]

            symmetric_makespan = symmetric_schedule(dispatch_matrix, combine_matrix, num_gpus)
            asymmetric_makespan = asymmetric_schedule(dispatch_matrix, combine_matrix, num_gpus)
            hotfirst_gain = 0.0 if symmetric_makespan == 0.0 else ((symmetric_makespan - asymmetric_makespan) / symmetric_makespan) * 100.0
            hotfirst_gains.append(hotfirst_gain)

            layer_pair = f"{from_layer}->{to_layer}"
            symmetric_oracle_row = pairwise_results_index.get((sample_id, layer_pair))
            if symmetric_oracle_row is not None:
                symmetric_oracle_makespan = float(symmetric_oracle_row["oracle_perfect_makespan"])
                symmetric_oracle_status = str(symmetric_oracle_row.get("oracle_perfect_solver_status"))
            else:
                symmetric_oracle_makespan = 0.0
                symmetric_oracle_status = "missing"

            schedule_records.append(
                {
                    "sample_id": sample_id,
                    "layer_pair": layer_pair,
                    "s_greedy_makespan": symmetric_makespan,
                    "a_hotfirst_makespan": asymmetric_makespan,
                    "a_hotfirst_gain_pct": hotfirst_gain,
                    "s_oracle_makespan": symmetric_oracle_makespan,
                    "s_oracle_status": symmetric_oracle_status,
                }
            )

    asymmetry_values = [float(row["asymmetry_ratio"]) for row in asymmetry_records]
    overall_asymmetry = _summary_stats(asymmetry_values)
    overall_asymmetry["min"] = min(asymmetry_values) if asymmetry_values else 0.0
    overall_asymmetry["max"] = max(asymmetry_values) if asymmetry_values else 0.0
    layer_asymmetry_summary = {
        str(layer_id): _summary_stats(values)
        for layer_id, values in sorted(asymmetry_by_layer.items())
    }

    schedule_summary = {
        "s_greedy_makespan": _summary_stats([row["s_greedy_makespan"] for row in schedule_records]),
        "a_hotfirst_makespan": _summary_stats([row["a_hotfirst_makespan"] for row in schedule_records]),
        "a_hotfirst_gain_pct": _summary_stats(hotfirst_gains),
        "s_oracle_makespan": _summary_stats([row["s_oracle_makespan"] for row in schedule_records]),
    }
    go_no_go = {
        "asymmetry_significant": overall_asymmetry["mean"] >= 1.3,
        "dispatch_combine_independent_value": schedule_summary["a_hotfirst_gain_pct"]["mean"] >= 5.0,
        "cp_sat_not_fully_exploiting_asymmetry": None,
        "decision": "GO" if schedule_summary["a_hotfirst_gain_pct"]["mean"] >= 5.0 else "NO_GO",
    }

    return {
        "asymmetry_records": asymmetry_records,
        "asymmetry_summary": overall_asymmetry,
        "layer_asymmetry_summary": layer_asymmetry_summary,
        "dispatch_combine_scatter": scatter_rows,
        "schedule_records": schedule_records,
        "schedule_summary": schedule_summary,
        "go_no_go": go_no_go,
    }
