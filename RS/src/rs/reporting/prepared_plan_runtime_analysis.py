from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rs.core.contracts.result import ResultBundle


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_result_details(run_dir: Path) -> dict[str, Any]:
    bundle_path = run_dir / "result_bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing canonical result bundle: {bundle_path}")
    bundle = ResultBundle.from_dict(_read_json(bundle_path))
    return dict(bundle.details)


def _plan_row_summary(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {}) or {}
    waves = row.get("waves", []) or []
    first_wave = waves[0] if waves else {}
    first_wave_tasks = first_wave.get("bucket_tasks", []) or []
    return {
        "layer_id": str((row.get("plan_key", {}) or {}).get("layer_id", "")),
        "phase": str(row.get("phase", "")),
        "plan_hash": str(row.get("plan_hash", "")),
        "prepared_window_key": str(metrics.get("prepared_window_key", "")),
        "source_logical_plan_hash": str(metrics.get("source_logical_plan_hash", "")),
        "ordered_by_prepared_plan": bool(metrics.get("ordered_by_prepared_plan", False)),
        "hint_edges_available": int(metrics.get("hint_edges_available", 0) or 0),
        "hint_edges_matched": int(metrics.get("hint_edges_matched", 0) or 0),
        "hint_edges_consumed": int(metrics.get("hint_edges_consumed", 0) or 0),
        "hint_match_rate": float(metrics.get("hint_match_rate", 0.0) or 0.0),
        "wave_count": len(waves),
        "bucket_count": int(metrics.get("bucket_count", 0) or 0),
        "first_wave_bucket_tasks": [
            {
                "task_id": str(task.get("task_id", "")),
                "src_rank": int(task.get("src_rank", 0)),
                "dst_rank": int(task.get("dst_rank", 0)),
                "row_count": int(task.get("row_count", 0)),
                "byte_count": int(task.get("byte_count", 0)),
            }
            for task in first_wave_tasks
        ],
    }


def analyze_prepared_plan_runtime(run_dir: Path, *, rank: int = 0) -> dict[str, Any]:
    details = _load_result_details(run_dir)
    arrivals = _read_jsonl(run_dir / f"rank{rank}_plan_arrival_records.jsonl")
    bindings = _read_jsonl(run_dir / f"rank{rank}_prepared_plan_bindings.jsonl")
    plans = _read_jsonl(run_dir / f"rank{rank}_scheduled_phase_plans.jsonl")
    audits = _read_json(run_dir / f"rank{rank}_execution_audit.json").get("audits", []) or []

    prepared_arrivals = [row for row in arrivals if bool(row.get("has_prepared_plan", False))]
    before_commit = [row for row in prepared_arrivals if row.get("arrival_status") == "before_commit"]
    inflight = [row for row in prepared_arrivals if row.get("arrival_status") == "in_flight"]
    none_rows = [row for row in arrivals if row.get("arrival_status") == "none"]

    return {
        "run_dir": str(run_dir),
        "rank": int(rank),
        "policy_name": str(details.get("policy_name", "")),
        "p2_hint_mode": str(details.get("p2_hint_mode", "")),
        "execution_audit_status": str(details.get("execution_audit_status", "")),
        "plan_arrival_summary": {
            "record_count": len(arrivals),
            "prepared_plan_arrival_count": len(prepared_arrivals),
            "before_commit_count": len(before_commit),
            "in_flight_count": len(inflight),
            "none_count": len(none_rows),
            "avg_plan_age_us": (
                float(sum(int(row.get("plan_age_us", 0) or 0) for row in prepared_arrivals) / len(prepared_arrivals))
                if prepared_arrivals
                else 0.0
            ),
            "first_prepared_arrival": prepared_arrivals[0] if prepared_arrivals else None,
        },
        "prepared_plan_bindings": bindings,
        "scheduled_phase_plan_summaries": [_plan_row_summary(row) for row in plans],
        "execution_audit_summaries": [
            {
                "layer_id": str(row.get("layer_id", "")),
                "phase": str(row.get("phase", "")),
                "status": str(row.get("status", "")),
                "planned_wave_count": int(row.get("planned_wave_count", 0) or 0),
                "executed_wave_count": int(row.get("executed_wave_count", 0) or 0),
                "hint_edges_consumed": int(((row.get("details", {}) or {}).get("hint_edges_consumed", 0) or 0)),
                "hint_match_rate": float(((row.get("details", {}) or {}).get("hint_match_rate", 0.0) or 0.0)),
                "prepared_window_key": str(((row.get("details", {}) or {}).get("prepared_window_key", ""))),
                "source_logical_plan_hash": str(((row.get("details", {}) or {}).get("source_logical_plan_hash", ""))),
                "prepared_plan_order_preserved": bool(((row.get("details", {}) or {}).get("prepared_plan_order_preserved", False))),
                "p0_bundle_atomicity_preserved": bool(row.get("p0_bundle_atomicity_preserved", True)),
            }
            for row in audits
        ],
    }


__all__ = ["analyze_prepared_plan_runtime"]
