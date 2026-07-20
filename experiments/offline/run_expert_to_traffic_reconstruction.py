#!/usr/bin/env python3
"""Audit expert->traffic reconstruction from expert-route traces when available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.runtime.online.megatron_ep.prediction.expert_to_traffic import (
    DEFAULT_HIDDEN_ONLY_BYTES_PER_TOKEN,
    compare_reconstructed_traffic,
    source_expert_counts_to_traffic_matrix,
)
from rs.runtime.online.megatron_ep.prediction.expert_trace import SourceExpertCountMatrix, load_source_expert_counts_jsonl
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-summary-md", required=True)
    parser.add_argument("--bytes-per-token", type=int, default=DEFAULT_HIDDEN_ONLY_BYTES_PER_TOKEN)
    return parser.parse_args()


def merge_source_expert_counts_by_layer_and_source_rank(
    records,
    *,
    layer_id: int,
    world_size: int,
    num_experts: int,
) -> tuple[SourceExpertCountMatrix, dict[str, Any]]:
    filtered = [record for record in records if int(record.layer_id) == int(layer_id)]
    merged_rows = [[0 for _ in range(num_experts)] for _ in range(world_size)]
    merged_weighted = None
    if any(record.weighted_counts is not None for record in filtered):
        merged_weighted = [[0.0 for _ in range(num_experts)] for _ in range(world_size)]
    seen_source_ranks: set[int] = set()
    missing_source_ranks: list[int] = []
    conflict_source_ranks: list[int] = []
    expert_to_rank_map: tuple[int, ...] | None = None
    bytes_per_token = 1
    selected_experts_available = False
    routing_weights_available = False
    for record in filtered:
        source_rank = record.source_rank
        if source_rank is None:
            nonzero_rows = [idx for idx, row in enumerate(record.counts) if any(int(value) > 0 for value in row)]
            if len(nonzero_rows) == 1:
                source_rank = int(nonzero_rows[0])
            else:
                raise ValueError(f"cannot infer unique source_rank for layer_id={layer_id}")
        source_rank = int(source_rank)
        if source_rank < 0 or source_rank >= world_size:
            raise ValueError(f"invalid source_rank={source_rank} for layer_id={layer_id}")
        raw_row = tuple(int(value) for value in record.counts[source_rank])
        if len(raw_row) > num_experts:
            raise ValueError(f"source_expert_counts row wider than num_experts for layer_id={layer_id}")
        row_values = raw_row + tuple(0 for _ in range(num_experts - len(raw_row)))
        if source_rank in seen_source_ranks:
            if tuple(merged_rows[source_rank]) != row_values:
                conflict_source_ranks.append(source_rank)
            continue
        merged_rows[source_rank] = list(row_values)
        if merged_weighted is not None and record.weighted_counts is not None:
            raw_weighted_row = tuple(float(value) for value in record.weighted_counts[source_rank])
            if len(raw_weighted_row) > num_experts:
                raise ValueError(f"weighted_source_expert_counts row wider than num_experts for layer_id={layer_id}")
            merged_weighted[source_rank] = list(raw_weighted_row + tuple(0.0 for _ in range(num_experts - len(raw_weighted_row))))
        seen_source_ranks.add(source_rank)
        if record.expert_to_rank_map is not None and (
            expert_to_rank_map is None or len(record.expert_to_rank_map) > len(expert_to_rank_map)
        ):
            expert_to_rank_map = tuple(int(v) for v in record.expert_to_rank_map)
        bytes_per_token = int(record.bytes_per_token or bytes_per_token)
        selected_experts_available = selected_experts_available or bool(record.selected_experts_available)
        routing_weights_available = routing_weights_available or bool(record.routing_weights_available)
    for source_rank in range(world_size):
        if source_rank not in seen_source_ranks:
            missing_source_ranks.append(source_rank)
    diagnostics = {
        "layer_id": int(layer_id),
        "source_expert_records_count": len(filtered),
        "seen_source_ranks": sorted(seen_source_ranks),
        "missing_source_ranks": missing_source_ranks,
        "conflict_source_ranks": sorted(set(conflict_source_ranks)),
        "complete_world_matrix": not missing_source_ranks and not conflict_source_ranks,
    }
    if conflict_source_ranks:
        raise ValueError(f"conflicting source_expert_counts for layer_id={layer_id}: {sorted(set(conflict_source_ranks))}")
    merged = SourceExpertCountMatrix(
        layer_id=int(layer_id),
        world_size=int(world_size),
        num_experts=int(num_experts),
        counts=tuple(tuple(int(value) for value in row) for row in merged_rows),
        weighted_counts=None
        if merged_weighted is None
        else tuple(tuple(float(value) for value in row) for row in merged_weighted),
        expert_to_rank_map=expert_to_rank_map,
        bytes_per_token=int(bytes_per_token),
        selected_experts_available=bool(selected_experts_available),
        routing_weights_available=bool(routing_weights_available),
    )
    return merged, diagnostics


def merge_actual_traffic_by_layer_and_source_rank(
    audit_rows: list[dict[str, Any]],
    *,
    layer_id: int,
    world_size: int,
) -> tuple[tuple[tuple[int, ...], ...] | None, dict[str, Any]]:
    filtered = [row for row in audit_rows if int(row.get("layer_id", -1)) == int(layer_id)]
    merged_rows = [[0 for _ in range(world_size)] for _ in range(world_size)]
    seen_source_ranks: set[int] = set()
    missing_source_ranks: list[int] = []
    conflict_source_ranks: list[int] = []
    for row in filtered:
        source_rank = row.get("source_rank")
        if source_rank is None:
            raise ValueError(f"missing source_rank in expert_to_traffic_audit for layer_id={layer_id}")
        source_rank = int(source_rank)
        actual_matrix = row.get("actual_matrix")
        if not isinstance(actual_matrix, list) or source_rank >= len(actual_matrix):
            raise ValueError(f"invalid actual_matrix for layer_id={layer_id} source_rank={source_rank}")
        local_row = tuple(int(v) for v in actual_matrix[source_rank])
        if source_rank in seen_source_ranks:
            if tuple(merged_rows[source_rank]) != local_row:
                conflict_source_ranks.append(source_rank)
            continue
        merged_rows[source_rank] = list(local_row)
        seen_source_ranks.add(source_rank)
    for source_rank in range(world_size):
        if source_rank not in seen_source_ranks:
            missing_source_ranks.append(source_rank)
    diagnostics = {
        "layer_id": int(layer_id),
        "actual_audit_row_count": len(filtered),
        "seen_source_ranks": sorted(seen_source_ranks),
        "missing_source_ranks": missing_source_ranks,
        "conflict_source_ranks": sorted(set(conflict_source_ranks)),
        "complete_world_matrix": not missing_source_ranks and not conflict_source_ranks,
    }
    if conflict_source_ranks:
        raise ValueError(f"conflicting actual_matrix rows for layer_id={layer_id}: {sorted(set(conflict_source_ranks))}")
    if not filtered:
        return None, diagnostics
    matrix = tuple(tuple(int(v) for v in row) for row in merged_rows)
    return matrix, diagnostics


def merge_phase_context_dispatch_matrix_by_layer(
    phase_context_rows: list[dict[str, Any]],
    *,
    layer_id: int,
    world_size: int,
) -> tuple[tuple[tuple[int, ...], ...] | None, dict[str, Any]]:
    filtered = [
        row
        for row in phase_context_rows
        if str(row.get("phase", "")).upper() == "P0" and int(row.get("layer_id", -1)) == int(layer_id)
    ]
    merged_rows = [[0 for _ in range(world_size)] for _ in range(world_size)]
    seen_source_ranks: set[int] = set()
    missing_source_ranks: list[int] = []
    conflict_source_ranks: list[int] = []
    for row in filtered:
        source_rank = row.get("global_rank")
        if source_rank is None:
            source_rank = row.get("rank")
        if source_rank is None:
            source_rank = (row.get("topology", {}) or {}).get("ep_group_rank")
        if source_rank is None:
            raise ValueError(f"missing source rank in phase context for layer_id={layer_id}")
        source_rank = int(source_rank)
        peer_bytes = tuple(int(value) for value in (row.get("per_peer_bytes") or ()))
        if len(peer_bytes) != world_size:
            raise ValueError(
                f"per_peer_bytes width mismatch for layer_id={layer_id}: got {len(peer_bytes)} expected {world_size}"
            )
        if source_rank in seen_source_ranks:
            if tuple(merged_rows[source_rank]) != peer_bytes:
                conflict_source_ranks.append(source_rank)
            continue
        merged_rows[source_rank] = list(peer_bytes)
        seen_source_ranks.add(source_rank)
    for source_rank in range(world_size):
        if source_rank not in seen_source_ranks:
            missing_source_ranks.append(source_rank)
    diagnostics = {
        "layer_id": int(layer_id),
        "phase_context_row_count": len(filtered),
        "seen_source_ranks": sorted(seen_source_ranks),
        "missing_source_ranks": missing_source_ranks,
        "conflict_source_ranks": sorted(set(conflict_source_ranks)),
        "complete_world_matrix": not missing_source_ranks and not conflict_source_ranks,
    }
    if conflict_source_ranks:
        raise ValueError(f"conflicting phase_context per_peer_bytes for layer_id={layer_id}: {sorted(set(conflict_source_ranks))}")
    if not filtered:
        return None, diagnostics
    return canonicalize_remote_matrix(tuple(tuple(int(v) for v in row) for row in merged_rows)), diagnostics


def _load_phase_context_rows(fixture_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(fixture_dir.glob("*phase_contexts*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_expert_to_traffic_reconstruction(*, fixture_dir: Path, bytes_per_token: int = DEFAULT_HIDDEN_ONLY_BYTES_PER_TOKEN) -> dict[str, Any]:
    source_count_files = sorted(fixture_dir.glob("*source_expert_counts*.jsonl"))
    audit_files = sorted(fixture_dir.glob("*expert_to_traffic_audit*.jsonl"))
    phase_context_rows = _load_phase_context_rows(fixture_dir)
    if not source_count_files:
        return {
            "fixture_dir": str(fixture_dir),
            "expert_trace_available": False,
            "expert_trace_unavailable_for_real_fixture": True,
            "gpu_collection_required": True,
            "required_gpu_fields": [
                "selected_experts",
                "routing_weights",
                "source_rank",
                "layer_id",
                "expert_to_rank_map",
                "source_expert_counts",
                "bytes_per_token",
            ],
            "rows": [],
            "summary": {
                "record_count": 0,
                "source_expert_records_count": 0,
                "merged_layer_count": 0,
                "complete_world_matrix_layer_count": 0,
                "incomplete_world_matrix_layer_count": 0,
                "missing_source_ranks_by_layer": {},
                "conflict_source_ranks_by_layer": {},
                "mean_relative_l1_error": None,
                "o1_corrected_relative_l1": None,
                "o1_corrected_cosine": None,
                "o1_corrected_row_sum_error": None,
                "o1_corrected_col_sum_error": None,
                "o1_legacy_debug_relative_l1": None,
                "bytes_model_used": "hidden_only",
                "actual_matrix_source": None,
                "matrix_scope": "remote_only",
                "expert_to_traffic_mapping_valid": False,
                "source_rank_granularity_required": None,
            },
        }
    all_records = []
    for count_path in source_count_files:
        all_records.extend(load_source_expert_counts_jsonl(count_path))
    audit_rows: list[dict[str, Any]] = []
    for audit_path in audit_files:
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                audit_rows.append(json.loads(line))
    by_layer_ids = sorted({int(record.layer_id) for record in all_records})
    rows: list[dict[str, Any]] = []
    o1_errors: list[float] = []
    o1_cosines: list[float] = []
    o1_row_errors: list[float] = []
    o1_col_errors: list[float] = []
    legacy_o1_errors: list[float] = []
    o2_errors: list[float] = []
    o3_errors: list[float] = []
    o4_errors: list[float] = []
    merged_layer_count = 0
    complete_world_matrix_layer_count = 0
    incomplete_world_matrix_layer_count = 0
    missing_source_ranks_by_layer: dict[str, Any] = {}
    conflict_source_ranks_by_layer: dict[str, Any] = {}
    for layer_id in by_layer_ids:
        fixture_path = fixture_dir / f"replay_layer_{layer_id}.json"
        layer_records = [record for record in all_records if int(record.layer_id) == int(layer_id)]
        if not layer_records:
            continue
        num_experts = max(int(record.num_experts) for record in layer_records)
        world_size = max(int(record.world_size) for record in layer_records)
        try:
            merged_counts, merge_diag = merge_source_expert_counts_by_layer_and_source_rank(
                layer_records,
                layer_id=int(layer_id),
                world_size=int(world_size),
                num_experts=int(num_experts),
            )
        except ValueError as exc:
            merged_layer_count += 1
            incomplete_world_matrix_layer_count += 1
            conflict_source_ranks_by_layer[str(layer_id)] = str(exc)
            rows.append(
                {
                    "layer_id": int(layer_id),
                    "complete_world_matrix": False,
                    "o1_valid": False,
                    "merge_error": str(exc),
                }
            )
            continue
        merged_layer_count += 1
        missing_source_ranks_by_layer[str(layer_id)] = merge_diag["missing_source_ranks"]
        conflict_source_ranks_by_layer[str(layer_id)] = merge_diag["conflict_source_ranks"]
        if merge_diag["complete_world_matrix"]:
            complete_world_matrix_layer_count += 1
        else:
            incomplete_world_matrix_layer_count += 1
        expert_to_rank_map = merged_counts.expert_to_rank_map
        if expert_to_rank_map is None:
            raise ValueError(f"missing expert_to_rank_map for layer_id={layer_id}")
        expert_to_rank = {expert_id: int(expert_to_rank_map[expert_id]) for expert_id in range(num_experts)}
        actual = None
        actual_matrix_source = None
        actual, phase_context_diag = merge_phase_context_dispatch_matrix_by_layer(
            phase_context_rows,
            layer_id=int(layer_id),
            world_size=int(world_size),
        )
        if actual is not None and phase_context_diag["complete_world_matrix"]:
            actual_matrix_source = "phase_context_aggregated_p0_dispatch"
        elif fixture_path.exists():
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            actual = canonicalize_remote_matrix(tuple(tuple(int(v) for v in row) for row in fixture["p0_dispatch_matrix"]))
            actual_matrix_source = "fixture_p0_dispatch_matrix"
        else:
            actual, actual_diag = merge_actual_traffic_by_layer_and_source_rank(
                audit_rows,
                layer_id=int(layer_id),
                world_size=int(world_size),
            )
            if actual is None:
                rows.append(
                    {
                        "layer_id": int(layer_id),
                        "source_expert_records_count": int(merge_diag["source_expert_records_count"]),
                        "complete_world_matrix": False,
                        "o1_valid": False,
                        "merge_error": "actual_dispatch_matrix_unavailable",
                    }
                )
                incomplete_world_matrix_layer_count += 1
                continue
            actual = canonicalize_remote_matrix(actual)
            actual_matrix_source = "legacy_local_row_audit_merged"
        reconstructed = source_expert_counts_to_traffic_matrix(
            merged_counts,
            expert_to_rank,
            bytes_per_token=int(bytes_per_token or merged_counts.bytes_per_token or DEFAULT_HIDDEN_ONLY_BYTES_PER_TOKEN),
        )
        audit = compare_reconstructed_traffic(reconstructed, actual)
        if merge_diag["complete_world_matrix"]:
            o1_errors.append(float(audit.relative_l1_error))
            o1_cosines.append(float(audit.cosine_similarity))
            o1_row_errors.append(float(audit.row_sum_error))
            o1_col_errors.append(float(audit.col_sum_error))
        legacy_actual, _legacy_diag = merge_actual_traffic_by_layer_and_source_rank(
            audit_rows,
            layer_id=int(layer_id),
            world_size=int(world_size),
        )
        if legacy_actual is not None:
            legacy_o1_errors.append(float(compare_reconstructed_traffic(reconstructed, canonicalize_remote_matrix(legacy_actual)).relative_l1_error))
        global_counts = SourceExpertCountMatrix(
            layer_id=int(layer_id),
            world_size=world_size,
            num_experts=num_experts,
            counts=tuple(
                tuple(
                    int(sum(merged_counts.counts[src_rank][expert_id] for src_rank in range(world_size)))
                    if source_rank == 0
                    else 0
                    for expert_id in range(num_experts)
                )
                for source_rank in range(world_size)
            ),
            expert_to_rank_map=tuple(int(v) for v in expert_to_rank_map),
            bytes_per_token=int(bytes_per_token or merged_counts.bytes_per_token or DEFAULT_HIDDEN_ONLY_BYTES_PER_TOKEN),
            selected_experts_available=bool(merged_counts.selected_experts_available),
            routing_weights_available=bool(merged_counts.routing_weights_available),
        )
        global_reconstructed = source_expert_counts_to_traffic_matrix(
            global_counts,
            expert_to_rank,
            bytes_per_token=int(bytes_per_token or merged_counts.bytes_per_token or DEFAULT_HIDDEN_ONLY_BYTES_PER_TOKEN),
        )
        global_audit = compare_reconstructed_traffic(global_reconstructed, actual)
        o2_errors.append(float(global_audit.relative_l1_error))
        current_expert_copy_audit = None
        current_traffic_copy_audit = None
        next_fixture_path = fixture_dir / f"replay_layer_{int(layer_id) + 1}.json"
        next_actual = None
        if next_fixture_path.exists():
            next_fixture = json.loads(next_fixture_path.read_text(encoding="utf-8"))
            next_actual = canonicalize_remote_matrix(tuple(tuple(int(v) for v in row) for row in next_fixture["p0_dispatch_matrix"]))
        else:
            next_actual, _ = merge_actual_traffic_by_layer_and_source_rank(
                audit_rows,
                layer_id=int(layer_id) + 1,
                world_size=int(world_size),
            )
            if next_actual is not None:
                next_actual = canonicalize_remote_matrix(next_actual)
        if next_actual is not None:
            current_expert_copy_audit = compare_reconstructed_traffic(reconstructed, next_actual)
            current_traffic_copy_audit = compare_reconstructed_traffic(actual, next_actual)
            o3_errors.append(float(current_expert_copy_audit.relative_l1_error))
            o4_errors.append(float(current_traffic_copy_audit.relative_l1_error))
        rows.append(
            {
                "layer_id": int(layer_id),
                "source_expert_records_count": int(merge_diag["source_expert_records_count"]),
                "complete_world_matrix": bool(merge_diag["complete_world_matrix"]),
                "missing_source_ranks": list(merge_diag["missing_source_ranks"]),
                "conflict_source_ranks": list(merge_diag["conflict_source_ranks"]),
                "o1_valid": bool(merge_diag["complete_world_matrix"]),
                "actual_matrix_source": actual_matrix_source,
                "bytes_model_used": "hidden_only",
                "matrix_scope": "remote_only",
                "actual_source_expert_to_traffic_relative_l1": audit.relative_l1_error,
                "traffic_cosine": audit.cosine_similarity,
                "topk_edge_overlap": audit.topk_edge_overlap,
                "row_sum_error": audit.row_sum_error,
                "col_sum_error": audit.col_sum_error,
                "bottleneck_src_match": None,
                "bottleneck_dst_match": None,
                "global_expert_count_to_traffic_relative_l1": global_audit.relative_l1_error,
                "gap_vs_source_rank_expert_counts": float(global_audit.relative_l1_error - audit.relative_l1_error),
                "expert_count_copy_baseline_l1": None if current_expert_copy_audit is None else current_expert_copy_audit.relative_l1_error,
                "traffic_copy_baseline_l1": None if current_traffic_copy_audit is None else current_traffic_copy_audit.relative_l1_error,
                "self_bytes_ignored": audit.self_bytes_ignored,
            }
        )
    valid_o1_rows = [row for row in rows if row.get("o1_valid") and row.get("actual_source_expert_to_traffic_relative_l1") is not None]
    mean_relative = None if not valid_o1_rows else sum(float(row["actual_source_expert_to_traffic_relative_l1"]) for row in valid_o1_rows) / len(valid_o1_rows)
    return {
        "fixture_dir": str(fixture_dir),
        "expert_trace_available": True,
        "gpu_collection_required": False,
        "rows": rows,
        "summary": {
            "record_count": len(rows),
            "source_expert_records_count": len(all_records),
            "merged_layer_count": int(merged_layer_count),
            "complete_world_matrix_layer_count": int(complete_world_matrix_layer_count),
            "incomplete_world_matrix_layer_count": int(incomplete_world_matrix_layer_count),
            "missing_source_ranks_by_layer": missing_source_ranks_by_layer,
            "conflict_source_ranks_by_layer": conflict_source_ranks_by_layer,
            "mean_relative_l1_error": mean_relative,
            "o1_corrected_relative_l1": mean_relative,
            "o1_corrected_cosine": (sum(o1_cosines) / len(o1_cosines)) if o1_cosines else None,
            "o1_corrected_row_sum_error": (sum(o1_row_errors) / len(o1_row_errors)) if o1_row_errors else None,
            "o1_corrected_col_sum_error": (sum(o1_col_errors) / len(o1_col_errors)) if o1_col_errors else None,
            "o1_legacy_debug_relative_l1": (sum(legacy_o1_errors) / len(legacy_o1_errors)) if legacy_o1_errors else None,
            "bytes_model_used": "hidden_only",
            "actual_matrix_source": (
                "phase_context_aggregated_p0_dispatch"
                if any(row.get("actual_matrix_source") == "phase_context_aggregated_p0_dispatch" for row in rows)
                else ("fixture_p0_dispatch_matrix" if any(row.get("actual_matrix_source") == "fixture_p0_dispatch_matrix" for row in rows) else "legacy_local_row_audit_merged")
            ),
            "matrix_scope": "remote_only",
            "expert_to_traffic_mapping_valid": mean_relative is not None and complete_world_matrix_layer_count > 0,
            "source_rank_granularity_required": None if not o1_errors or not o2_errors else (sum(o1_errors) / len(o1_errors)) <= (sum(o2_errors) / len(o2_errors)),
            "expert_count_copy_baseline_l1": None if not o3_errors else sum(o3_errors) / len(o3_errors),
            "traffic_copy_baseline_l1": None if not o4_errors else sum(o4_errors) / len(o4_errors),
            "best_non_oracle_expert_to_traffic_l1": min(
                [value for value in [mean_relative, None if not o3_errors else sum(o3_errors) / len(o3_errors)] if value is not None],
                default=None,
            ),
            "recommended_next_predictor_direction": (
                "collect_complete_source_rank_expert_trace"
                if complete_world_matrix_layer_count <= 0
                else "source_rank_expert_prediction"
            ),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Expert To Traffic Reconstruction", ""]
    lines.append(f"- fixture_dir: `{payload['fixture_dir']}`")
    lines.append(f"- expert_trace_available: `{payload['expert_trace_available']}`")
    if not payload["expert_trace_available"]:
        lines.append("- GPU collection required: collect selected_experts, routing_weights, source_rank, layer_id, expert_to_rank")
        return "\n".join(lines) + "\n"
        lines.extend(
        [
            "",
            "| Layer | O1 source-expert->traffic L1 | O2 global-expert->traffic L1 | O3 current-expert-copy->next L1 | O4 current-traffic-copy->next L1 | cosine | self bytes ignored |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        if not row.get("o1_valid", False):
            lines.append(
                f"| {row['layer_id']} | incomplete | incomplete | - | - | - | - |"
            )
            continue
        expert_copy_text = "-" if row["expert_count_copy_baseline_l1"] is None else f"{row['expert_count_copy_baseline_l1']:.4f}"
        traffic_copy_text = "-" if row["traffic_copy_baseline_l1"] is None else f"{row['traffic_copy_baseline_l1']:.4f}"
        lines.append(
            f"| {row['layer_id']} | {row['actual_source_expert_to_traffic_relative_l1']:.4f} | {row['global_expert_count_to_traffic_relative_l1']:.4f} | "
            f"{expert_copy_text} | "
            f"{traffic_copy_text} | "
            f"{row['traffic_cosine']:.4f} | {row['self_bytes_ignored']} |"
        )
    summary = payload["summary"]
    lines.extend(
        [
            "",
            "## Corrected O1",
            f"- o1_corrected_relative_l1: `{summary.get('o1_corrected_relative_l1')}`",
            f"- o1_corrected_cosine: `{summary.get('o1_corrected_cosine')}`",
            f"- o1_corrected_row_sum_error: `{summary.get('o1_corrected_row_sum_error')}`",
            f"- o1_corrected_col_sum_error: `{summary.get('o1_corrected_col_sum_error')}`",
            f"- o1_legacy_debug_relative_l1: `{summary.get('o1_legacy_debug_relative_l1')}`",
            f"- bytes_model_used: `{summary.get('bytes_model_used')}`",
            f"- actual_matrix_source: `{summary.get('actual_matrix_source')}`",
            f"- matrix_scope: `{summary.get('matrix_scope')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    payload = run_expert_to_traffic_reconstruction(
        fixture_dir=Path(args.fixture_dir),
        bytes_per_token=int(args.bytes_per_token),
    )
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_summary_md).write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
