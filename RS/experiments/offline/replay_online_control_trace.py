from __future__ import annotations

import argparse
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


def summarize_control_replay_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_policy: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_phase: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    total_summary_elements = 0
    total_plan_elements = 0
    total_task_refs = 0
    wave_counts: list[int] = []
    bucket_counts: list[int] = []
    nonzero_edge_counts: list[int] = []
    total_byte_counts: list[int] = []
    for row in rows:
        policy = str(row.get("policy_name", "unknown"))
        phase = str(row.get("phase", "unknown"))
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
    return {
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize lightweight online control replay traces.")
    parser.add_argument("--trace", required=True, help="Path to rank*_control_replay_trace.jsonl")
    args = parser.parse_args()
    rows = read_jsonl(Path(args.trace))
    print(json.dumps(summarize_control_replay_trace(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
