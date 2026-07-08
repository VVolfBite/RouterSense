"""Prepared shadow plan alignment analysis for online strategy artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _phase_key(row: dict[str, Any]) -> tuple[str, str]:
    layer_name = str(row.get("layer_name") or row.get("layer") or "")
    phase = str(row.get("phase") or row.get("phase_name") or "")
    return (layer_name, phase.upper())


def _scheduled_plan_phase_key(
    row: dict[str, Any],
    *,
    plan_hash_to_phase: dict[str, tuple[str, str]],
) -> tuple[str, str]:
    direct = _phase_key(row)
    if direct[0]:
        return direct
    plan_hash = str(row.get("plan_hash") or "")
    if plan_hash and plan_hash in plan_hash_to_phase:
        return plan_hash_to_phase[plan_hash]
    plan_key = row.get("plan_key", {}) or {}
    layer_id = str(plan_key.get("layer_id") or "")
    phase = str(row.get("phase") or plan_key.get("phase") or row.get("phase_name") or "").upper()
    return (layer_id, phase)


def _latest_by_phase(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _phase_key(row)
        previous = latest.get(key)
        current_ts = int(row.get("ts_us", 0) or 0)
        previous_ts = int(previous.get("ts_us", 0) or 0) if previous is not None else -1
        if previous is None or current_ts >= previous_ts:
            latest[key] = row
    return latest


def _extract_plan_bucket_order(plan_row: dict[str, Any]) -> list[str]:
    if "compiled_bucket_order" in plan_row:
        return [str(item) for item in plan_row.get("compiled_bucket_order", []) or []]
    metrics = plan_row.get("metrics", {}) or {}
    return [str(item) for item in metrics.get("bucket_order", []) or []]


def _extract_transport_order(events: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in events:
        task_id = str(row.get("task_id") or row.get("bucket_id") or "")
        if task_id and task_id not in seen:
            ordered.append(task_id)
            seen.add(task_id)
    return ordered


def _prefix_match_count(expected: list[str], actual: list[str]) -> int:
    count = 0
    for index, expected_item in enumerate(expected):
        if index >= len(actual) or actual[index] != expected_item:
            break
        count += 1
    return count


def _overlap_rate(expected: list[str], actual: list[str]) -> float:
    if not expected:
        return 0.0
    actual_set = set(actual)
    return float(sum(1 for item in expected if item in actual_set)) / float(len(expected))


def build_shadow_plan_alignment(
    *,
    prepared_phase_plan_shadow: list[dict[str, Any]],
    scheduled_phase_plans: list[dict[str, Any]],
    transport_execution: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared_by_phase = _latest_by_phase(prepared_phase_plan_shadow)
    transport_by_phase: dict[tuple[str, str], list[dict[str, Any]]] = {}
    plan_hash_to_phase: dict[str, tuple[str, str]] = {}
    for row in transport_execution:
        layer_name = str(row.get("layer_name") or row.get("layer") or "")
        if not layer_name:
            continue
        phase = str(row.get("phase") or row.get("phase_name") or "").upper()
        key = (layer_name, phase)
        transport_by_phase.setdefault(key, []).append(row)
        plan_hash = str(row.get("plan_hash") or "")
        if plan_hash:
            plan_hash_to_phase[plan_hash] = key

    actual_by_phase: dict[tuple[str, str], dict[str, Any]] = {}
    for row in scheduled_phase_plans:
        key = _scheduled_plan_phase_key(row, plan_hash_to_phase=plan_hash_to_phase)
        previous = actual_by_phase.get(key)
        current_ts = int(row.get("ts_us", 0) or 0)
        previous_ts = int(previous.get("ts_us", 0) or 0) if previous is not None else -1
        if previous is None or current_ts >= previous_ts:
            actual_by_phase[key] = row

    keys = sorted(set(prepared_by_phase) | set(actual_by_phase) | set(transport_by_phase))
    rows: list[dict[str, Any]] = []
    for key in keys:
        layer_name, phase = key
        prepared = prepared_by_phase.get(key)
        actual = actual_by_phase.get(key)
        transport_rows = transport_by_phase.get(key, [])
        prepared_compile_status = str((prepared or {}).get("compile_status", "ok"))
        prepared_order = _extract_plan_bucket_order(prepared or {})
        actual_order = _extract_plan_bucket_order(actual or {})
        execution_order = _extract_transport_order(transport_rows)
        prepared_prefix = _prefix_match_count(prepared_order, actual_order)
        execution_prefix = _prefix_match_count(actual_order, execution_order)
        rows.append(
            {
                "layer_name": layer_name,
                "phase": phase,
                "has_prepared_phase_plan_shadow": prepared is not None,
                "has_actual_scheduled_plan": actual is not None,
                "has_transport_execution": bool(transport_rows),
                "prepared_compile_status": prepared_compile_status,
                "prepared_window_key": str((prepared or {}).get("prepared_window_key", "")),
                "prepared_plan_hash": str((prepared or {}).get("compiled_plan_hash", "")),
                "actual_plan_hash": str((actual or {}).get("plan_hash", "")),
                "plan_hash_match": bool(
                    prepared is not None
                    and actual is not None
                    and str((prepared or {}).get("compiled_plan_hash", "")) == str((actual or {}).get("plan_hash", ""))
                ),
                "prepared_bucket_order": prepared_order,
                "actual_bucket_order": actual_order,
                "actual_execution_order": execution_order,
                "prepared_to_actual_prefix_match_count": prepared_prefix,
                "prepared_to_actual_exact_match": bool(prepared_order == actual_order and prepared_order),
                "prepared_to_actual_overlap_rate": _overlap_rate(prepared_order, actual_order),
                "actual_plan_to_execution_prefix_match_count": execution_prefix,
                "actual_plan_to_execution_exact_match": bool(actual_order == execution_order and actual_order),
                "actual_plan_to_execution_overlap_rate": _overlap_rate(actual_order, execution_order),
                "prepared_plan_order_preserved": bool((prepared or {}).get("prepared_plan_order_preserved", False)),
                "actual_ordered_by_prepared_plan": bool(((actual or {}).get("metrics", {}) or {}).get("ordered_by_prepared_plan", False)),
                "prepared_hint_edges_consumed": int((prepared or {}).get("hint_edges_consumed", 0) or 0),
                "actual_hint_edges_consumed": int((((actual or {}).get("metrics", {}) or {}).get("hint_edges_consumed", 0) or 0)),
                "prepared_hint_match_rate": float((prepared or {}).get("hint_match_rate", 0.0) or 0.0),
                "actual_hint_match_rate": float((((actual or {}).get("metrics", {}) or {}).get("hint_match_rate", 0.0) or 0.0)),
            }
        )
    return rows


def summarize_shadow_plan_alignment(rows: list[dict[str, Any]]) -> dict[str, float]:
    considered = [
        row
        for row in rows
        if row.get("has_prepared_phase_plan_shadow")
        and row.get("has_actual_scheduled_plan")
        and row.get("prepared_compile_status") == "ok"
    ]
    compile_failed_count = float(
        sum(1 for row in rows if row.get("has_prepared_phase_plan_shadow") and row.get("prepared_compile_status") != "ok")
    )
    if not considered:
        return {
            "prepared_shadow_phase_count": 0.0,
            "prepared_shadow_compile_failed_count": compile_failed_count,
            "prepared_shadow_plan_hash_match_count": 0.0,
            "prepared_shadow_exact_order_match_count": 0.0,
            "prepared_shadow_execution_exact_match_count": 0.0,
            "prepared_shadow_avg_actual_overlap_rate": 0.0,
            "prepared_shadow_avg_execution_overlap_rate": 0.0,
        }
    return {
        "prepared_shadow_phase_count": float(len(considered)),
        "prepared_shadow_compile_failed_count": compile_failed_count,
        "prepared_shadow_plan_hash_match_count": float(sum(1 for row in considered if row.get("plan_hash_match"))),
        "prepared_shadow_exact_order_match_count": float(sum(1 for row in considered if row.get("prepared_to_actual_exact_match"))),
        "prepared_shadow_execution_exact_match_count": float(sum(1 for row in considered if row.get("actual_plan_to_execution_exact_match"))),
        "prepared_shadow_avg_actual_overlap_rate": float(
            sum(float(row.get("prepared_to_actual_overlap_rate", 0.0) or 0.0) for row in considered) / len(considered)
        ),
        "prepared_shadow_avg_execution_overlap_rate": float(
            sum(float(row.get("actual_plan_to_execution_overlap_rate", 0.0) or 0.0) for row in considered) / len(considered)
        ),
    }


def analyze_rank_artifacts(run_dir: Path, *, rank: int = 0) -> dict[str, Any]:
    rows = build_shadow_plan_alignment(
        prepared_phase_plan_shadow=_read_jsonl(run_dir / f"rank{rank}_prepared_phase_plan_shadow.jsonl"),
        scheduled_phase_plans=_read_jsonl(run_dir / f"rank{rank}_scheduled_phase_plans.jsonl"),
        transport_execution=_read_jsonl(run_dir / f"rank{rank}_transport_execution.jsonl"),
    )
    return {
        "rows": rows,
        "summary": summarize_shadow_plan_alignment(rows),
    }


__all__ = [
    "analyze_rank_artifacts",
    "build_shadow_plan_alignment",
    "summarize_shadow_plan_alignment",
]
