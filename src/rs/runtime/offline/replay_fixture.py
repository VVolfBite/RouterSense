"""Replay fixture builders derived from online control traces."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rs.scheduling.traffic_matrix import (
    canonicalize_remote_matrix,
    matrix_diagonal_report,
    matrix_nonzero_remote_edge_count,
    matrix_remote_bytes,
)


def build_replay_fixture_bundle(rows: list[dict[str, Any]], *, policy_name: str | None = None) -> dict[str, Any]:
    grouped = _group_rows(rows, policy_name=policy_name)
    phases_by_layer: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    run_id_digest = ""
    selected_policy = policy_name or ""
    for (run_digest, current_policy, layer_id, phase), group_rows in grouped.items():
        if not run_id_digest:
            run_id_digest = run_digest
        if not selected_policy:
            selected_policy = current_policy
        raw_matrix, matrix, stats = _matrix_from_group(group_rows)
        layer_name = str(group_rows[0].get("layer_name", layer_id))
        phases_by_layer[layer_id][phase] = {
            "layer_name": layer_name,
            "matrix": matrix,
            "raw_matrix": raw_matrix,
            "stats": stats,
        }
    ordered_layers = sorted(phases_by_layer.keys(), key=lambda value: int(value) if str(value).isdigit() else str(value))
    fixtures: list[dict[str, Any]] = []
    for index, layer_id in enumerate(ordered_layers):
        phase_map = phases_by_layer[layer_id]
        if "P0" not in phase_map or "P1" not in phase_map:
            continue
        p0_entry = phase_map["P0"]
        p1_entry = phase_map["P1"]
        num_gpus = int(p0_entry["stats"]["num_gpus"])
        next_p0 = _zero_matrix(num_gpus)
        next_layer_id = ""
        if index + 1 < len(ordered_layers):
            next_layer_id = ordered_layers[index + 1]
            next_phase_map = phases_by_layer[next_layer_id]
            if "P0" in next_phase_map:
                next_p0 = [[int(value) for value in row] for row in next_phase_map["P0"]["matrix"]]
                next_p0_raw = [[int(value) for value in row] for row in next_phase_map["P0"]["raw_matrix"]]
            else:
                next_p0_raw = _zero_matrix(num_gpus)
        else:
            next_p0_raw = _zero_matrix(num_gpus)
        fixtures.append(
            {
                "fixture_name": f"replay_layer_{layer_id}",
                "num_gpus": num_gpus,
                "p0_dispatch_matrix": p0_entry["matrix"],
                "p1_return_matrix": p1_entry["matrix"],
                "p2_next_dispatch_forecast_matrix": next_p0,
                "p2_next_dispatch_matrix": next_p0,
                "metadata": {
                    "source": "control_replay_trace",
                    "run_id_digest": run_id_digest,
                    "policy_name": selected_policy,
                    "layer_id": str(layer_id),
                    "layer_name": str(p0_entry["layer_name"]),
                    "next_layer_id": str(next_layer_id),
                    "p0_seen_ranks": p0_entry["stats"]["seen_ranks"],
                    "p1_seen_ranks": p1_entry["stats"]["seen_ranks"],
                    "p0_missing_ranks": p0_entry["stats"]["missing_ranks"],
                    "p1_missing_ranks": p1_entry["stats"]["missing_ranks"],
                    "p0_total_bytes": int(p0_entry["stats"]["remote_total_bytes"]),
                    "p1_total_bytes": int(p1_entry["stats"]["remote_total_bytes"]),
                    "p2_total_bytes": int(matrix_remote_bytes(next_p0)),
                    "p0_raw_total_bytes": int(p0_entry["stats"]["raw_total_bytes"]),
                    "p1_raw_total_bytes": int(p1_entry["stats"]["raw_total_bytes"]),
                    "p2_raw_total_bytes": int(sum(sum(row) for row in next_p0_raw)),
                    "p0_self_bytes": int(p0_entry["stats"]["self_bytes"]),
                    "p1_self_bytes": int(p1_entry["stats"]["self_bytes"]),
                    "p2_self_bytes": int(matrix_diagonal_report(next_p0_raw)["self_bytes"]),
                    "p0_self_byte_ratio": float(p0_entry["stats"]["self_byte_ratio"]),
                    "p1_self_byte_ratio": float(p1_entry["stats"]["self_byte_ratio"]),
                    "p2_self_byte_ratio": float(matrix_diagonal_report(next_p0_raw)["self_byte_ratio"]),
                    "p0_diagonal_nonzero_count": int(p0_entry["stats"]["diagonal_nonzero_count"]),
                    "p1_diagonal_nonzero_count": int(p1_entry["stats"]["diagonal_nonzero_count"]),
                    "p2_diagonal_nonzero_count": int(matrix_diagonal_report(next_p0_raw)["diagonal_nonzero_count"]),
                    "p2_source": "zero_for_last_layer" if not next_layer_id else "next_layer_p0_actual",
                    "p0_nonzero_edge_count": int(matrix_nonzero_remote_edge_count(p0_entry["matrix"])),
                    "p1_nonzero_edge_count": int(matrix_nonzero_remote_edge_count(p1_entry["matrix"])),
                    "p2_nonzero_edge_count": int(matrix_nonzero_remote_edge_count(next_p0)),
                },
            }
        )
    return {
        "run_id_digest": run_id_digest,
        "policy_name": selected_policy,
        "layer_count": len(ordered_layers),
        "fixture_count": len(fixtures),
        "fixtures": fixtures,
    }


def build_replay_fixture_audit_summary(
    bundle: dict[str, Any],
    *,
    source_kind: str,
    trace_file_count: int,
) -> dict[str, Any]:
    fixtures = list(bundle.get("fixtures", []))
    expected_rank_count = int(fixtures[0]["num_gpus"]) if fixtures else 0
    per_layer = []
    total_p0_bytes = 0
    total_p1_bytes = 0
    total_p2_bytes = 0
    total_p0_raw_bytes = 0
    total_p1_raw_bytes = 0
    total_p2_raw_bytes = 0
    total_p0_self_bytes = 0
    total_p1_self_bytes = 0
    total_p2_self_bytes = 0
    layer_count_with_complete_p0p1 = 0
    layer_count_with_missing_rank = 0
    max_p0 = ("", 0)
    max_p1 = ("", 0)
    max_p2 = ("", 0)
    for fixture in fixtures:
        meta = dict(fixture.get("metadata", {}))
        p0_missing = list(meta.get("p0_missing_ranks", []))
        p1_missing = list(meta.get("p1_missing_ranks", []))
        p0_bytes = int(meta.get("p0_total_bytes", 0))
        p1_bytes = int(meta.get("p1_total_bytes", 0))
        p2_bytes = int(meta.get("p2_total_bytes", 0))
        p0_self_bytes = int(meta.get("p0_self_bytes", 0))
        p1_self_bytes = int(meta.get("p1_self_bytes", 0))
        p2_self_bytes = int(meta.get("p2_self_bytes", 0))
        p0_raw_bytes = int(meta.get("p0_raw_total_bytes", p0_bytes + p0_self_bytes))
        p1_raw_bytes = int(meta.get("p1_raw_total_bytes", p1_bytes + p1_self_bytes))
        p2_raw_bytes = int(meta.get("p2_raw_total_bytes", p2_bytes + p2_self_bytes))
        total_p0_bytes += p0_bytes
        total_p1_bytes += p1_bytes
        total_p2_bytes += p2_bytes
        total_p0_raw_bytes += p0_raw_bytes
        total_p1_raw_bytes += p1_raw_bytes
        total_p2_raw_bytes += p2_raw_bytes
        total_p0_self_bytes += p0_self_bytes
        total_p1_self_bytes += p1_self_bytes
        total_p2_self_bytes += p2_self_bytes
        if not p0_missing and not p1_missing:
            layer_count_with_complete_p0p1 += 1
        else:
            layer_count_with_missing_rank += 1
        if p0_bytes > max_p0[1]:
            max_p0 = (str(fixture["fixture_name"]), p0_bytes)
        if p1_bytes > max_p1[1]:
            max_p1 = (str(fixture["fixture_name"]), p1_bytes)
        if p2_bytes > max_p2[1]:
            max_p2 = (str(fixture["fixture_name"]), p2_bytes)
        per_layer.append(
            {
                "fixture_name": str(fixture["fixture_name"]),
                "layer_id": str(meta.get("layer_id", "")),
                "next_layer_id": str(meta.get("next_layer_id", "")),
                "p0_seen_ranks": list(meta.get("p0_seen_ranks", [])),
                "p1_seen_ranks": list(meta.get("p1_seen_ranks", [])),
                "p0_missing_ranks": p0_missing,
                "p1_missing_ranks": p1_missing,
                "p0_total_bytes": p0_bytes,
                "p1_total_bytes": p1_bytes,
                "p2_total_bytes": p2_bytes,
                "p0_self_bytes": p0_self_bytes,
                "p1_self_bytes": p1_self_bytes,
                "p2_self_bytes": p2_self_bytes,
                "p0_raw_total_bytes": p0_raw_bytes,
                "p1_raw_total_bytes": p1_raw_bytes,
                "p2_raw_total_bytes": p2_raw_bytes,
                "p0_self_byte_ratio": float(meta.get("p0_self_byte_ratio", 0.0)),
                "p1_self_byte_ratio": float(meta.get("p1_self_byte_ratio", 0.0)),
                "p2_self_byte_ratio": float(meta.get("p2_self_byte_ratio", 0.0)),
                "p0_diagonal_nonzero_count": int(meta.get("p0_diagonal_nonzero_count", 0)),
                "p1_diagonal_nonzero_count": int(meta.get("p1_diagonal_nonzero_count", 0)),
                "p2_diagonal_nonzero_count": int(meta.get("p2_diagonal_nonzero_count", 0)),
                "p2_source": str(meta.get("p2_source", "")),
                "p0_nonzero_edge_count": int(meta.get("p0_nonzero_edge_count", 0)),
                "p1_nonzero_edge_count": int(meta.get("p1_nonzero_edge_count", 0)),
                "p2_nonzero_edge_count": int(meta.get("p2_nonzero_edge_count", 0)),
            }
        )
    layer_count = len(per_layer)
    return {
        "source_kind": source_kind,
        "trace_file_count": int(trace_file_count),
        "policy_name": str(bundle.get("policy_name", "")),
        "run_id_digest": str(bundle.get("run_id_digest", "")),
        "layer_count": int(bundle.get("layer_count", 0)),
        "fixture_count": int(bundle.get("fixture_count", 0)),
        "num_gpus": expected_rank_count,
        "expected_rank_count": expected_rank_count,
        "layers": per_layer,
        "layer_count_with_complete_p0p1": layer_count_with_complete_p0p1,
        "layer_count_with_missing_rank": layer_count_with_missing_rank,
        "total_p0_bytes": total_p0_bytes,
        "total_p1_bytes": total_p1_bytes,
        "total_p2_bytes": total_p2_bytes,
        "total_p0_raw_bytes": total_p0_raw_bytes,
        "total_p1_raw_bytes": total_p1_raw_bytes,
        "total_p2_raw_bytes": total_p2_raw_bytes,
        "total_p0_self_bytes": total_p0_self_bytes,
        "total_p1_self_bytes": total_p1_self_bytes,
        "total_p2_self_bytes": total_p2_self_bytes,
        "avg_p0_bytes_per_layer": (total_p0_bytes / layer_count) if layer_count else 0.0,
        "avg_p1_bytes_per_layer": (total_p1_bytes / layer_count) if layer_count else 0.0,
        "avg_p2_bytes_per_layer": (total_p2_bytes / layer_count) if layer_count else 0.0,
        "max_p0_bytes_layer": {"fixture_name": max_p0[0], "bytes": max_p0[1]},
        "max_p1_bytes_layer": {"fixture_name": max_p1[0], "bytes": max_p1[1]},
        "max_p2_bytes_layer": {"fixture_name": max_p2[0], "bytes": max_p2[1]},
    }


def _group_rows(rows: list[dict[str, Any]], *, policy_name: str | None) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        current_policy = str(row.get("policy_name", ""))
        if policy_name and current_policy != policy_name:
            continue
        key = (
            str(row.get("run_id_digest", "")),
            current_policy,
            str(row.get("layer_id", "")),
            str(row.get("phase", "")),
        )
        grouped[key].append(row)
    return grouped


def _matrix_from_group(group_rows: list[dict[str, Any]]) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    if not group_rows:
        raise ValueError("group_rows must not be empty")
    num_gpus = int(group_rows[0].get("ep_group_size", 0))
    matrix = [[0 for _ in range(num_gpus)] for _ in range(num_gpus)]
    seen_ranks: set[int] = set()
    for row in group_rows:
        src_rank = _infer_source_rank(row)
        per_peer = [int(value) for value in row.get("per_rank_peer_bytes", []) or []]
        if len(per_peer) != num_gpus:
            raise ValueError(f"rank {src_rank} has per_rank_peer_bytes len {len(per_peer)} != ep_group_size {num_gpus}")
        matrix[src_rank] = per_peer
        seen_ranks.add(src_rank)
    remote_matrix = [list(row) for row in canonicalize_remote_matrix(matrix)]
    diag = matrix_diagonal_report(matrix)
    return matrix, remote_matrix, {
        "num_gpus": num_gpus,
        "seen_ranks": sorted(seen_ranks),
        "missing_ranks": [rank for rank in range(num_gpus) if rank not in seen_ranks],
        "raw_total_bytes": int(diag["total_bytes"]),
        "remote_total_bytes": int(diag["remote_bytes"]),
        "self_bytes": int(diag["self_bytes"]),
        "self_byte_ratio": float(diag["self_byte_ratio"]),
        "diagonal_nonzero_count": int(diag["diagonal_nonzero_count"]),
    }


def _infer_source_rank(row: dict[str, Any]) -> int:
    if "global_rank" in row:
        return int(row["global_rank"])
    if "local_rank" in row:
        return int(row["local_rank"])
    nonzero_edges = row.get("nonzero_edges", []) or []
    src_ranks = {int(edge.get("src_rank", -1)) for edge in nonzero_edges}
    src_ranks.discard(-1)
    if len(src_ranks) == 1:
        return next(iter(src_ranks))
    raise ValueError(f"cannot infer source rank from replay trace row: {row}")


def _zero_matrix(num_gpus: int) -> list[list[int]]:
    return [[0 for _ in range(num_gpus)] for _ in range(num_gpus)]

