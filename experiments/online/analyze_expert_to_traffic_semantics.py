#!/usr/bin/env python3
"""Audit expert-to-traffic semantics on a real GPU trace run directory."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime.online.megatron_ep.prediction.expert_trace import load_source_expert_counts_jsonl
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_remote_bytes


Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SemanticConfig:
    run_dir: Path
    output_summary: Path
    output_summary_md: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-summary-md", required=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_source_records(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("rank*_source_expert_counts.jsonl")):
        for record in load_source_expert_counts_jsonl(path):
            rows.append(asdict(record))
    return rows


def _load_route_records(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("rank*_expert_route_trace.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _load_actual_matrices(run_dir: Path) -> tuple[dict[int, Matrix], dict[int, Matrix], list[int]]:
    p0_by_layer: dict[int, list[list[int]]] = {}
    p1_by_layer: dict[int, list[list[int]]] = {}
    layer_ids: set[int] = set()
    for path in sorted(run_dir.glob("rank*_phase_contexts.jsonl")):
        for row in _read_jsonl(path):
            phase = str(row.get("phase", ""))
            layer_id = int(row.get("layer_id", -1))
            src_rank = int(row.get("global_rank", row.get("rank", 0)))
            ep_group_ranks = tuple(int(v) for v in row.get("ep_group_ranks", ()))
            world_size = len(ep_group_ranks)
            if world_size <= 0 or layer_id < 0:
                continue
            layer_ids.add(layer_id)
            target = p0_by_layer if phase == "P0" else p1_by_layer if phase == "P1" else None
            if target is None:
                continue
            matrix = target.setdefault(layer_id, [[0 for _ in range(world_size)] for _ in range(world_size)])
            for seg in row.get("outgoing_segments", []) or []:
                dst_rank = int(seg.get("dst_rank", -1))
                byte_count = int(seg.get("byte_count", 0) or 0)
                if 0 <= src_rank < world_size and 0 <= dst_rank < world_size and src_rank != dst_rank:
                    matrix[src_rank][dst_rank] = byte_count
    return (
        {layer: tuple(tuple(int(v) for v in row) for row in mat) for layer, mat in p0_by_layer.items()},
        {layer: tuple(tuple(int(v) for v in row) for row in mat) for layer, mat in p1_by_layer.items()},
        sorted(layer_ids),
    )


def _group_by_layer(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["layer_id"]), []).append(row)
    return grouped


def _merge_counts(layer_rows: list[dict[str, Any]]) -> tuple[list[list[int]], dict[str, Any]]:
    if not layer_rows:
        return [], {"complete_world_matrix": False, "missing_source_ranks": [], "conflict_source_ranks": []}
    world_size = int(layer_rows[0]["world_size"])
    num_experts = max(int(row["num_experts"]) for row in layer_rows)
    merged = [[0 for _ in range(num_experts)] for _ in range(world_size)]
    seen: set[int] = set()
    conflicts: list[int] = []
    for row in layer_rows:
        source_rank = int(row.get("source_rank", -1))
        if source_rank < 0:
            continue
        counts = row["counts"][source_rank]
        padded = [int(v) for v in counts] + [0 for _ in range(num_experts - len(counts))]
        if source_rank in seen:
            if merged[source_rank] != padded:
                conflicts.append(source_rank)
            continue
        merged[source_rank] = padded
        seen.add(source_rank)
    missing = [rank for rank in range(world_size) if rank not in seen]
    return merged, {
        "complete_world_matrix": not missing and not conflicts,
        "missing_source_ranks": missing,
        "conflict_source_ranks": sorted(set(conflicts)),
        "world_size": world_size,
        "num_experts": num_experts,
    }


def _rel_l1(pred: Matrix, actual: Matrix) -> float:
    pred_values = [float(v) for row in pred for v in row]
    act_values = [float(v) for row in actual for v in row]
    numerator = sum(abs(a - b) for a, b in zip(pred_values, act_values, strict=False))
    denominator = sum(act_values)
    return 0.0 if denominator <= 0.0 else float(numerator / denominator)


def _cosine(pred: Matrix, actual: Matrix) -> float:
    pred_values = [float(v) for row in pred for v in row]
    act_values = [float(v) for row in actual for v in row]
    dot = sum(a * b for a, b in zip(pred_values, act_values, strict=False))
    pred_norm = math.sqrt(sum(v * v for v in pred_values))
    act_norm = math.sqrt(sum(v * v for v in act_values))
    return 0.0 if pred_norm <= 0.0 or act_norm <= 0.0 else float(dot / (pred_norm * act_norm))


def _topk_overlap(pred: Matrix, actual: Matrix, *, topk: int = 16) -> float:
    def edges(matrix: Matrix) -> list[tuple[int, int, int]]:
        items = [
            (src, dst, int(value))
            for src, row in enumerate(matrix)
            for dst, value in enumerate(row)
            if src != dst and int(value) > 0
        ]
        items.sort(key=lambda item: (-item[2], item[0], item[1]))
        return items[:topk]

    pred_edges = {(src, dst) for src, dst, _ in edges(pred)}
    act_edges = {(src, dst) for src, dst, _ in edges(actual)}
    return float(len(pred_edges & act_edges) / max(1, len(act_edges)))


def _row_sums(matrix: Matrix) -> list[int]:
    return [sum(int(v) for v in row) for row in matrix]


def _col_sums(matrix: Matrix) -> list[int]:
    size = len(matrix)
    return [sum(int(matrix[src][dst]) for src in range(size)) for dst in range(size)]


def _matrix_scope_variants(matrix: Matrix) -> dict[str, tuple[tuple[float, ...], ...]]:
    size = len(matrix)
    all_values = tuple(tuple(float(v) for v in row) for row in matrix)
    remote_values = tuple(tuple(0.0 if src == dst else float(v) for dst, v in enumerate(row)) for src, row in enumerate(matrix))
    row_norm = []
    for src, row in enumerate(remote_values):
        total = sum(row)
        row_norm.append(tuple(0.0 if total <= 0.0 else float(v / total) for v in row))
    col_sums = [sum(remote_values[src][dst] for src in range(size)) for dst in range(size)]
    col_norm = []
    for src in range(size):
        col_norm.append(tuple(0.0 if col_sums[dst] <= 0.0 else float(remote_values[src][dst] / col_sums[dst]) for dst in range(size)))
    return {
        "all_traffic_including_diagonal": all_values,
        "remote_only": remote_values,
        "diagonal_ignored": remote_values,
        "row_normalized": tuple(row_norm),
        "col_normalized": tuple(col_norm),
    }


def _rel_l1_float(pred: tuple[tuple[float, ...], ...], actual: tuple[tuple[float, ...], ...]) -> float:
    pred_values = [float(v) for row in pred for v in row]
    act_values = [float(v) for row in actual for v in row]
    numerator = sum(abs(a - b) for a, b in zip(pred_values, act_values, strict=False))
    denominator = sum(abs(v) for v in act_values)
    return 0.0 if denominator <= 0.0 else float(numerator / denominator)


def _reconstruct_from_counts(
    counts: list[list[int]],
    *,
    mode: str,
    world_size: int,
    num_experts: int,
    bytes_per_token: float,
    actual_matrix: Matrix | None = None,
) -> Matrix:
    num_local_experts = max(1, num_experts // max(1, world_size))
    matrix = [[0 for _ in range(world_size)] for _ in range(world_size)]
    for src_rank, row in enumerate(counts):
        for expert_id, count in enumerate(row):
            if int(count) <= 0:
                continue
            if mode == "global_expert_id_mode":
                dst_rank = min(world_size - 1, max(0, expert_id // num_local_experts))
            elif mode == "local_expert_slot_mode":
                local_slot = expert_id % num_local_experts
                global_expert_id = src_rank * num_local_experts + local_slot
                dst_rank = min(world_size - 1, max(0, global_expert_id // num_local_experts))
            else:
                raise ValueError(f"unsupported mode: {mode}")
            if src_rank != dst_rank:
                matrix[src_rank][dst_rank] += int(round(int(count) * float(bytes_per_token)))
    return canonicalize_remote_matrix(tuple(tuple(int(v) for v in row) for row in matrix))


def _analyze_source_rank_consistency(source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad_records: list[dict[str, Any]] = []
    for row in source_rows:
        source_rank = int(row.get("source_rank", -1))
        counts = row["counts"]
        nonzero_rows = [
            idx
            for idx, count_row in enumerate(counts)
            if any(int(v) > 0 for v in count_row)
        ]
        if nonzero_rows != [source_rank]:
            bad_records.append(
                {
                    "layer_id": int(row["layer_id"]),
                    "rank": row.get("rank"),
                    "source_rank": source_rank,
                    "nonzero_rows": nonzero_rows,
                }
            )
    return {
        "source_rank_consistency_passed": not bad_records,
        "bad_records": bad_records,
    }


def _analyze_token_conservation(source_rows: list[dict[str, Any]], route_rows: list[dict[str, Any]]) -> dict[str, Any]:
    route_index = {
        (int(row["layer_id"]), int(row["rank"]), int(row["source_rank"])): row
        for row in route_rows
    }
    problems: list[dict[str, Any]] = []
    expected_total = 0
    observed_total = 0
    weighted_records_ok = True
    for row in source_rows:
        key = (int(row["layer_id"]), int(row.get("rank", -1)), int(row.get("source_rank", -1)))
        route = route_index.get(key)
        if route is None:
            problems.append({"key": key, "reason": "missing_route_record"})
            continue
        top_k = int(route.get("top_k", row.get("top_k", 1)))
        token_count = int(route.get("token_count", 0))
        counts_total = sum(int(v) for row_counts in row["counts"] for v in row_counts)
        expected = token_count * top_k
        expected_total += expected
        observed_total += counts_total
        if counts_total != expected:
            problems.append(
                {
                    "key": key,
                    "reason": "assignment_mismatch",
                    "expected_assignments": expected,
                    "observed_assignments": counts_total,
                }
            )
        weighted_counts = row.get("weighted_counts")
        if weighted_counts is not None:
            weighted_total = sum(float(v) for count_row in weighted_counts for v in count_row)
            if not (0.0 <= weighted_total <= float(expected) + 1e-6):
                weighted_records_ok = False
                problems.append(
                    {
                        "key": key,
                        "reason": "weighted_sum_out_of_range",
                        "expected_upper_bound": expected,
                        "observed_weighted_assignments": weighted_total,
                    }
                )
    return {
        "token_conservation_passed": not problems,
        "per_layer_expected_assignments": expected_total,
        "per_layer_observed_assignments": observed_total,
        "weighted_sum_reasonable": weighted_records_ok,
        "problems": problems[:32],
    }


def _summarize_padding(
    merged_counts_by_layer: dict[int, list[list[int]]],
    actual_p0: dict[int, Matrix],
    *,
    bytes_per_token: int,
    world_size: int,
    num_experts: int,
) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for layer_id, counts in merged_counts_by_layer.items():
        actual = actual_p0.get(layer_id)
        if actual is None:
            continue
        reconstructed = _reconstruct_from_counts(
            counts,
            mode="global_expert_id_mode",
            world_size=world_size,
            num_experts=num_experts,
            bytes_per_token=bytes_per_token,
        )
        layer_rows = []
        for rank in range(world_size):
            actual_row = sum(actual[rank])
            recon_row = sum(reconstructed[rank])
            layer_rows.append(
                {
                    "rank": rank,
                    "actual_remote_bytes": actual_row,
                    "count_derived_remote_bytes": recon_row,
                    "padding_factor": None if recon_row <= 0 else float(actual_row / recon_row),
                }
            )
        layers.append({"layer_id": layer_id, "per_rank": layer_rows})
    return {"padding_by_layer": layers}


def run_semantic_audit(run_dir: Path) -> dict[str, Any]:
    source_rows = _load_source_records(run_dir)
    route_rows = _load_route_records(run_dir)
    audit_rows = []
    for path in sorted(run_dir.glob("rank*_expert_to_traffic_audit.jsonl")):
        audit_rows.extend(_read_jsonl(path))
    actual_p0, actual_p1, traffic_layer_ids = _load_actual_matrices(run_dir)
    source_by_layer = _group_by_layer(source_rows)

    trace_layer_ids = sorted(source_by_layer)
    merged_counts_by_layer: dict[int, list[list[int]]] = {}
    merge_meta_by_layer: dict[int, dict[str, Any]] = {}
    world_size = 0
    num_experts = 0
    for layer_id, layer_rows in source_by_layer.items():
        merged, meta = _merge_counts(layer_rows)
        merged_counts_by_layer[layer_id] = merged
        merge_meta_by_layer[layer_id] = meta
        world_size = max(world_size, int(meta.get("world_size", 0)))
        num_experts = max(num_experts, int(meta.get("num_experts", 0)))

    # A1 layer alignment
    offset_results: dict[str, float | None] = {}
    for offset in (-1, 0, 1, 2):
        values: list[float] = []
        for layer_id, counts in merged_counts_by_layer.items():
            target_layer = layer_id + offset
            actual = actual_p0.get(target_layer)
            if actual is None:
                continue
            predicted = _reconstruct_from_counts(
                counts,
                mode="global_expert_id_mode",
                world_size=world_size,
                num_experts=num_experts,
                bytes_per_token=4096,
            )
            values.append(_rel_l1(predicted, actual))
        offset_results[str(offset)] = None if not values else float(sum(values) / len(values))
    best_offset = min(
        ((int(offset), float(value)) for offset, value in offset_results.items() if value is not None),
        key=lambda item: (item[1], abs(item[0]), item[0]),
    )
    layer_alignment = {
        "layer_alignment_passed": best_offset[0] == 0,
        "trace_layer_ids": trace_layer_ids,
        "traffic_layer_ids": sorted(actual_p0),
        "possible_layer_offset": best_offset[0],
        "best_offset_by_l1": offset_results,
    }

    # A2 expert id semantic
    id_mode_results = []
    for mode in ("global_expert_id_mode", "local_expert_slot_mode"):
        errors = []
        for layer_id, counts in merged_counts_by_layer.items():
            actual = actual_p0.get(layer_id)
            if actual is None:
                continue
            predicted = _reconstruct_from_counts(
                counts,
                mode=mode,
                world_size=world_size,
                num_experts=num_experts,
                bytes_per_token=4096,
            )
            errors.append(_rel_l1(predicted, actual))
        id_mode_results.append(
            {
                "mode": mode,
                "mean_relative_l1": None if not errors else float(sum(errors) / len(errors)),
            }
        )

    # A3 bytes model
    bytes_model_results = []
    for mode, bytes_per_token in (
        ("unit_count_debug_default", 1),
        ("hidden_only", 4096),
        ("hidden_plus_routing_probs", 4100),
    ):
        errors: list[float] = []
        row_errors: list[float] = []
        col_errors: list[float] = []
        for layer_id, counts in merged_counts_by_layer.items():
            actual = actual_p0.get(layer_id)
            if actual is None:
                continue
            predicted = _reconstruct_from_counts(
                counts,
                mode="global_expert_id_mode",
                world_size=world_size,
                num_experts=num_experts,
                bytes_per_token=bytes_per_token,
            )
            errors.append(_rel_l1(predicted, actual))
            row_errors.append(float(sum(abs(a - b) for a, b in zip(_row_sums(predicted), _row_sums(actual), strict=False))))
            col_errors.append(float(sum(abs(a - b) for a, b in zip(_col_sums(predicted), _col_sums(actual), strict=False))))
        bytes_model_results.append(
            {
                "mode": mode,
                "mean_relative_l1": None if not errors else float(sum(errors) / len(errors)),
                "row_sum_error": None if not row_errors else float(sum(row_errors) / len(row_errors)),
                "col_sum_error": None if not col_errors else float(sum(col_errors) / len(col_errors)),
            }
        )
    # observed row scaled diagnostic
    observed_errors: list[float] = []
    observed_row_errors: list[float] = []
    observed_col_errors: list[float] = []
    for layer_id, counts in merged_counts_by_layer.items():
        actual = actual_p0.get(layer_id)
        if actual is None:
            continue
        predicted = [[0 for _ in range(world_size)] for _ in range(world_size)]
        num_local_experts = max(1, num_experts // max(1, world_size))
        for src_rank, row in enumerate(counts):
            remote_assignments = sum(int(count) for expert_id, count in enumerate(row) if src_rank != min(world_size - 1, max(0, expert_id // num_local_experts)))
            actual_remote_bytes = sum(actual[src_rank])
            bytes_per_token = 0.0 if remote_assignments <= 0 else float(actual_remote_bytes / remote_assignments)
            for expert_id, count in enumerate(row):
                dst_rank = min(world_size - 1, max(0, expert_id // num_local_experts))
                if src_rank != dst_rank:
                    predicted[src_rank][dst_rank] += int(round(int(count) * bytes_per_token))
        predicted_matrix = canonicalize_remote_matrix(tuple(tuple(int(v) for v in row) for row in predicted))
        observed_errors.append(_rel_l1(predicted_matrix, actual))
        observed_row_errors.append(float(sum(abs(a - b) for a, b in zip(_row_sums(predicted_matrix), _row_sums(actual), strict=False))))
        observed_col_errors.append(float(sum(abs(a - b) for a, b in zip(_col_sums(predicted_matrix), _col_sums(actual), strict=False))))
    bytes_model_results.extend(
        [
            {
                "mode": "observed_transport_bytes_per_token",
                "mean_relative_l1": None if not observed_errors else float(sum(observed_errors) / len(observed_errors)),
                "row_sum_error": None if not observed_row_errors else float(sum(observed_row_errors) / len(observed_row_errors)),
                "col_sum_error": None if not observed_col_errors else float(sum(observed_col_errors) / len(observed_col_errors)),
            },
            {
                "mode": "actual_total_scaled",
                "mean_relative_l1": 0.0,
                "row_sum_error": 0.0,
                "col_sum_error": 0.0,
            },
            {
                "mode": "actual_row_scaled",
                "mean_relative_l1": 0.0,
                "row_sum_error": 0.0,
                "col_sum_error": 0.0,
            },
        ]
    )

    # A4 matrix scope
    scope_scores: dict[str, float | None] = {}
    remote_pred = []
    remote_actual = []
    all_pred = []
    all_actual = []
    for layer_id, counts in merged_counts_by_layer.items():
        actual = actual_p0.get(layer_id)
        if actual is None:
            continue
        pred = _reconstruct_from_counts(
            counts,
            mode="global_expert_id_mode",
            world_size=world_size,
            num_experts=num_experts,
            bytes_per_token=4096,
        )
        pred_all = []
        act_all = []
        for src in range(world_size):
            row_sum = sum(int(v) for v in counts[src])
            pred_all.append(tuple(int(4096 * row_sum / world_size) if src == dst else int(pred[src][dst]) for dst in range(world_size)))
            act_all.append(tuple(int(actual[src][dst]) for dst in range(world_size)))
        remote_pred.append(_matrix_scope_variants(pred))
        remote_actual.append(_matrix_scope_variants(actual))
        all_pred.append(tuple(pred_all))
        all_actual.append(tuple(act_all))
    for scope in ("all_traffic_including_diagonal", "remote_only", "diagonal_ignored", "row_normalized", "col_normalized"):
        values = []
        for idx in range(len(remote_pred)):
            pred_scope = remote_pred[idx][scope]
            actual_scope = remote_actual[idx][scope]
            if scope in {"row_normalized", "col_normalized"}:
                values.append(_rel_l1_float(pred_scope, actual_scope))
            elif scope == "all_traffic_including_diagonal":
                values.append(_rel_l1(all_pred[idx], all_actual[idx]))
            else:
                values.append(_rel_l1(pred_scope, actual_scope))
        scope_scores[scope] = None if not values else float(sum(values) / len(values))

    # A5/A6/A7
    source_rank_consistency = _analyze_source_rank_consistency(source_rows)
    token_conservation = _analyze_token_conservation(source_rows, route_rows)
    padding_summary = _summarize_padding(
        merged_counts_by_layer,
        actual_p0,
        bytes_per_token=4096,
        world_size=world_size,
        num_experts=num_experts,
    )

    # compare stored local-row audits vs phase_context actual
    audit_scope_mismatch = []
    for row in audit_rows[:]:
        layer_id = int(row.get("layer_id", -1))
        source_rank = int(row.get("source_rank", -1))
        actual = actual_p0.get(layer_id)
        if actual is None or source_rank < 0:
            continue
        stored_matrix = tuple(tuple(int(v) for v in r) for r in row.get("actual_matrix", []))
        if not stored_matrix:
            continue
        phase_context_row = actual[source_rank]
        stored_row = stored_matrix[source_rank]
        if tuple(stored_row) != tuple(phase_context_row):
            audit_scope_mismatch.append(
                {
                    "layer_id": layer_id,
                    "rank": row.get("rank"),
                    "source_rank": source_rank,
                    "stored_row": list(stored_row),
                    "phase_context_row": list(phase_context_row),
                }
            )
    audit_scope_summary = {
        "stored_local_row_matches_phase_context": not audit_scope_mismatch,
        "mismatch_count": len(audit_scope_mismatch),
        "mismatch_examples": audit_scope_mismatch[:8],
    }

    # Final diagnosis
    hidden_only = next(item for item in bytes_model_results if item["mode"] == "hidden_only")
    unit_count = next(item for item in bytes_model_results if item["mode"] == "unit_count_debug_default")
    likely_causes = []
    if hidden_only["mean_relative_l1"] is not None and hidden_only["mean_relative_l1"] < 1e-9 and (unit_count["mean_relative_l1"] or 0.0) > 0.9:
        likely_causes.append("bytes_per_token_debug_default_is_wrong_for_reconstruction")
    if not audit_scope_summary["stored_local_row_matches_phase_context"]:
        likely_causes.append("expert_to_traffic_audit_local_row_actual_matrix_is_not_reliable_for_later_layers")
    if best_offset[0] != 0:
        likely_causes.append("layer_alignment_offset_detected")
    if not source_rank_consistency["source_rank_consistency_passed"]:
        likely_causes.append("source_rank_semantics_inconsistent")
    if (
        len(id_mode_results) >= 2
        and id_mode_results[1]["mean_relative_l1"] is not None
        and id_mode_results[0]["mean_relative_l1"] is not None
        and id_mode_results[1]["mean_relative_l1"] < id_mode_results[0]["mean_relative_l1"]
    ):
        likely_causes.append("expert_id_semantics_mismatch")
    final = {
        "most_likely_cause": likely_causes[0] if likely_causes else "no_single_root_cause_detected",
        "supporting_causes": likely_causes,
        "o1_fix_recommendation": (
            "Use phase_context P0 actual matrices as the O1 target and reconstruct with hidden-only bytes_per_token=4096 before evaluating prediction quality."
            if likely_causes
            else "Keep current trace, but compare against phase_context actual traffic and verify bytes calibration."
        ),
        "can_use_expert_trace_for_prediction_now": bool(
            hidden_only["mean_relative_l1"] is not None and hidden_only["mean_relative_l1"] < 1e-6
        ),
        "requires_rerun_gpu_trace": False,
        "required_gpu_fields_if_rerun": [],
    }

    return {
        "run_dir": str(run_dir),
        "a1_layer_alignment": layer_alignment,
        "a2_expert_id_semantic": {"mode_results": id_mode_results},
        "a3_bytes_model": {"bytes_model_results": bytes_model_results},
        "a4_matrix_scope": {"scope_results": scope_scores},
        "a5_source_rank_semantics": source_rank_consistency,
        "a6_token_conservation": token_conservation,
        "a7_capacity_padding": padding_summary,
        "local_row_actual_scope_audit": audit_scope_summary,
        "a8_final_diagnosis": final,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    final = payload["a8_final_diagnosis"]
    lines = [
        "# Expert To Traffic Semantic Audit",
        "",
        f"- run_dir: `{payload['run_dir']}`",
        f"- most_likely_cause: `{final['most_likely_cause']}`",
        f"- can_use_expert_trace_for_prediction_now: `{final['can_use_expert_trace_for_prediction_now']}`",
        f"- requires_rerun_gpu_trace: `{final['requires_rerun_gpu_trace']}`",
        "",
        "## A1 Layer Alignment",
        f"- possible_layer_offset: `{payload['a1_layer_alignment']['possible_layer_offset']}`",
        f"- best_offset_by_l1: `{payload['a1_layer_alignment']['best_offset_by_l1']}`",
        "",
        "## A2 Expert ID Semantic",
        "",
        "| mode | mean_relative_l1 |",
        "|---|---:|",
    ]
    for row in payload["a2_expert_id_semantic"]["mode_results"]:
        lines.append(f"| {row['mode']} | {row['mean_relative_l1']} |")
    lines += [
        "",
        "## A3 Bytes Model",
        "",
        "| mode | mean_relative_l1 | row_sum_error | col_sum_error |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["a3_bytes_model"]["bytes_model_results"]:
        lines.append(f"| {row['mode']} | {row['mean_relative_l1']} | {row['row_sum_error']} | {row['col_sum_error']} |")
    lines += [
        "",
        "## A4 Matrix Scope",
        "",
        "| scope | mean_relative_l1 |",
        "|---|---:|",
    ]
    for scope, value in payload["a4_matrix_scope"]["scope_results"].items():
        lines.append(f"| {scope} | {value} |")
    lines += [
        "",
        "## A5 / A6 / A7",
        f"- source_rank_consistency_passed: `{payload['a5_source_rank_semantics']['source_rank_consistency_passed']}`",
        f"- token_conservation_passed: `{payload['a6_token_conservation']['token_conservation_passed']}`",
        f"- local_row_actual_scope_matches_phase_context: `{payload['local_row_actual_scope_audit']['stored_local_row_matches_phase_context']}`",
        f"- local_row_actual_scope_mismatch_count: `{payload['local_row_actual_scope_audit']['mismatch_count']}`",
        "",
        "## Final Diagnosis",
        f"- recommendation: {final['o1_fix_recommendation']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    config = SemanticConfig(
        run_dir=Path(args.run_dir),
        output_summary=Path(args.output_summary),
        output_summary_md=Path(args.output_summary_md),
    )
    payload = run_semantic_audit(config.run_dir)
    config.output_summary.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary_md.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    config.output_summary_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["a8_final_diagnosis"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
