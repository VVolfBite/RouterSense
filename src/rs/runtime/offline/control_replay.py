"""Control replay trace summarization helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def summarize_control_replay_trace(
    rows: list[dict[str, Any]],
    *,
    rank_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    per_policy: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_phase: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_rank: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_summary_elements = 0
    total_plan_elements = 0
    total_task_refs = 0
    wave_counts: list[int] = []
    bucket_counts: list[int] = []
    nonzero_edge_counts: list[int] = []
    total_byte_counts: list[int] = []
    unique_layer_phase: set[tuple[str, str]] = set()
    for row in rows:
        policy = str(row.get("policy_name", "unknown"))
        phase = str(row.get("phase", "unknown"))
        layer_id = str(row.get("layer_id", row.get("layer", "")))
        transport = dict(row.get("transport_summary", {}) or {})
        timing = dict(row.get("timing_summary", {}) or {})
        summary_len = int(transport.get("planning_summary_tensor_len", 0) or 0)
        plan_len = int(transport.get("abstract_plan_tensor_len", 0) or 0)
        task_refs = int(row.get("abstract_plan_summary", {}).get("task_ref_count", 0) or 0)
        wave_count = int(row.get("abstract_plan_summary", {}).get("wave_count", 0) or 0)
        bucket_count = int(transport.get("bucket_count", 0) or 0)
        nonzero_edge_count = int(row.get("nonzero_edge_count", 0) or 0)
        total_byte_count = int(transport.get("total_byte_count", 0) or 0)
        total_summary_elements += summary_len
        total_plan_elements += plan_len
        total_task_refs += task_refs
        wave_counts.append(wave_count)
        bucket_counts.append(bucket_count)
        nonzero_edge_counts.append(nonzero_edge_count)
        total_byte_counts.append(total_byte_count)
        unique_layer_phase.add((layer_id, phase))
        per_policy[policy]["phase_count"] += 1
        per_policy[policy]["summary_elements"] += summary_len
        per_policy[policy]["plan_elements"] += plan_len
        per_policy[policy]["task_refs"] += task_refs
        per_policy[policy]["wave_count"] += wave_count
        per_policy[policy]["bucket_count"] += bucket_count
        per_policy[policy]["nonzero_edge_count"] += nonzero_edge_count
        per_policy[policy]["total_byte_count"] += total_byte_count
        per_policy[policy]["all_gather_time_us"] += float(timing.get("all_gather_time_us", 0.0) or 0.0)
        per_policy[policy]["build_plan_time_us"] += float(timing.get("build_plan_time_us", 0.0) or 0.0)
        per_policy[policy]["broadcast_time_us"] += float(timing.get("broadcast_time_us", 0.0) or 0.0)
        per_phase[phase]["phase_count"] += 1
        per_phase[phase]["summary_elements"] += summary_len
        per_phase[phase]["plan_elements"] += plan_len
        per_phase[phase]["task_refs"] += task_refs
        per_phase[phase]["wave_count"] += wave_count
        per_phase[phase]["bucket_count"] += bucket_count
        per_phase[phase]["nonzero_edge_count"] += nonzero_edge_count
        per_phase[phase]["total_byte_count"] += total_byte_count
        rank_key = _infer_rank_key(row)
        per_rank[rank_key]["phase_count"] += 1
        per_rank[rank_key]["summary_elements"] += summary_len
        per_rank[rank_key]["plan_elements"] += plan_len
        per_rank[rank_key]["task_refs"] += task_refs
        per_rank[rank_key]["wave_count"] += wave_count
        per_rank[rank_key]["bucket_count"] += bucket_count
        per_rank[rank_key]["nonzero_edge_count"] += nonzero_edge_count
        per_rank[rank_key]["total_byte_count"] += total_byte_count
    rows_per_rank = {key: len(value) for key, value in (rank_rows or {}).items()}
    if not rows_per_rank and per_rank:
        rows_per_rank = {key: int(value.get("phase_count", 0)) for key, value in per_rank.items()}
    return {
        "trace_file_count": len(rank_rows or {}) if rank_rows is not None else 1,
        "rank_count": len(rows_per_rank or per_rank),
        "rows_per_rank": rows_per_rank,
        "unique_layer_phase_count": len(unique_layer_phase),
        "total_phase_count": len(rows),
        "total_all_gather_calls": len(rows),
        "total_broadcast_calls": len(rows),
        "total_summary_elements": total_summary_elements,
        "total_plan_elements": total_plan_elements,
        "total_task_refs": total_task_refs,
        "avg_summary_elements_per_phase": (total_summary_elements / len(rows)) if rows else 0.0,
        "avg_plan_elements_per_phase": (total_plan_elements / len(rows)) if rows else 0.0,
        "avg_task_refs_per_phase": (total_task_refs / len(rows)) if rows else 0.0,
        "avg_wave_count": (sum(wave_counts) / len(wave_counts)) if wave_counts else 0.0,
        "max_wave_count": max(wave_counts) if wave_counts else 0,
        "avg_bucket_count": (sum(bucket_counts) / len(bucket_counts)) if bucket_counts else 0.0,
        "max_bucket_count": max(bucket_counts) if bucket_counts else 0,
        "avg_nonzero_edge_count": (sum(nonzero_edge_counts) / len(nonzero_edge_counts)) if nonzero_edge_counts else 0.0,
        "max_nonzero_edge_count": max(nonzero_edge_counts) if nonzero_edge_counts else 0,
        "avg_total_byte_count": (sum(total_byte_counts) / len(total_byte_counts)) if total_byte_counts else 0.0,
        "max_total_byte_count": max(total_byte_counts) if total_byte_counts else 0,
        "per_policy": {key: dict(value) for key, value in per_policy.items()},
        "per_phase": {key: dict(value) for key, value in per_phase.items()},
        "per_rank": {key: dict(value) for key, value in per_rank.items()},
    }


def collect_trace_rows(*, trace_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    rank_rows: dict[str, list[dict[str, Any]]] = {}
    for path in trace_paths:
        file_rows = read_jsonl(path)
        fallback_rank = _rank_from_filename(path)
        for row in file_rows:
            row.setdefault("rank", fallback_rank)
        rank_rows[fallback_rank] = file_rows
        rows.extend(file_rows)
    return rows, rank_rows


def trace_paths_from_args(*, trace_values: list[str], trace_dir: str) -> list[Path]:
    paths = [Path(value) for value in trace_values if value]
    if trace_dir:
        paths.extend(sorted(Path(trace_dir).glob("rank*_control_replay_trace.jsonl")))
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def _infer_rank_key(row: dict[str, Any], *, fallback: str = "unknown") -> str:
    if "global_rank" in row:
        return str(int(row["global_rank"]))
    if "rank" in row:
        try:
            return str(int(row["rank"]))
        except (TypeError, ValueError):
            return str(row["rank"] or fallback)
    return fallback


def _rank_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.startswith("rank"):
        digits = "".join(ch for ch in stem[4:] if ch.isdigit())
        if digits:
            return digits
    return "unknown"

