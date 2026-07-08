#!/usr/bin/env python3
"""CPU-only replay of online phase planner artifacts."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.core.artifact import write_json
from rs.scheduling.phase_execution import PhaseReadyContext
from rs.scheduling.phase_local.common import estimate_planning_quantum_rows_from_contexts
from rs.scheduling.registry import resolve_phase_policy
from rs.runtime.online.megatron_ep.pending_window import MultiphasePendingWindowAdapter


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to a per-strategy/repN directory")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--bucket-rows", type=int, default=1024)
    parser.add_argument("--phase", default="all", choices=("all", "P0", "P1"))
    parser.add_argument("--layer-id", default="all")
    parser.add_argument("--mode", default="phase_sync_wave", choices=("phase_sync_wave", "multiphase_pending_window"))
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_grouped_contexts(run_dir: Path) -> dict[tuple[str, str], tuple[PhaseReadyContext, ...]]:
    grouped: dict[tuple[str, str], list[PhaseReadyContext]] = {}
    for path in sorted(run_dir.glob("rank*_phase_contexts.jsonl")):
        for row in _read_jsonl(path):
            key = (str(row["layer_id"]), str(row["phase"]))
            grouped.setdefault(key, []).append(PhaseReadyContext.from_dict(row))
    return {
        key: tuple(sorted(contexts, key=lambda ctx: int(ctx.global_rank)))
        for key, contexts in grouped.items()
    }


def _shared_state_from_run_dir(run_dir: Path) -> dict[str, Any]:
    bindings = _read_jsonl(run_dir / "rank0_prepared_plan_bindings.jsonl")
    if not bindings:
        return {"prepared_plan": None, "plan_created_at_us": 0, "plan_source_layer": ""}
    latest = bindings[-1]
    # Replay only needs presence metadata; binding content is consumed via phase contexts and stored hints.
    return {"prepared_plan": None, "plan_created_at_us": int(latest.get("ts_us", 0) or 0), "plan_source_layer": str(latest.get("source_layer_name", ""))}


def _select_keys(grouped: dict[tuple[str, str], tuple[PhaseReadyContext, ...]], *, phase: str, layer_id: str) -> list[tuple[str, str]]:
    keys = sorted(grouped.keys(), key=lambda item: (int(item[0]), item[1]))
    selected: list[tuple[str, str]] = []
    for key in keys:
        if phase != "all" and key[1] != phase:
            continue
        if layer_id != "all" and key[0] != layer_id:
            continue
        selected.append(key)
    return selected


def _bucket_statistics(plan: Any) -> dict[str, Any]:
    row_hist = Counter()
    flow_bucket_counts = Counter()
    total_tasks = 0
    for wave in plan.waves:
        for task in wave.bucket_tasks:
            total_tasks += 1
            row_hist[int(task.row_count)] += 1
            flow_key = f"{task.phase}:{int(task.src_rank)}->{int(task.dst_rank)}:{int(task.segment_ordinal)}"
            flow_bucket_counts[flow_key] += 1
    if flow_bucket_counts:
        counts = list(flow_bucket_counts.values())
        min_buckets = min(counts)
        max_buckets = max(counts)
        mean_buckets = sum(counts) / len(counts)
    else:
        min_buckets = 0
        max_buckets = 0
        mean_buckets = 0.0
    top_split_flows = [
        {"flow_key": key, "bucket_count": int(count)}
        for key, count in sorted(flow_bucket_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]
    return {
        "task_count": int(total_tasks),
        "bucket_row_histogram": {str(rows): int(count) for rows, count in sorted(row_hist.items())},
        "flow_bucket_count_summary": {
            "flow_count": int(len(flow_bucket_counts)),
            "min_bucket_count_per_flow": int(min_buckets),
            "max_bucket_count_per_flow": int(max_buckets),
            "mean_bucket_count_per_flow": float(mean_buckets),
            "top_split_flows": top_split_flows,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    grouped = _load_grouped_contexts(run_dir)
    selected = _select_keys(grouped, phase=args.phase, layer_id=args.layer_id)
    if not selected:
        raise SystemExit("no matching phase contexts found")

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    shared_state = _shared_state_from_run_dir(run_dir)
    if args.mode == "multiphase_pending_window":
        policy = MultiphasePendingWindowAdapter(
            shared_state=shared_state,
            phase_policy_name=args.policy,
            bucket_rows=args.bucket_rows,
            p0_weight=1.0,
            p1_reservation_weight=1.0,
            p2_hint_weight=1.0,
        )
    else:
        policy = resolve_phase_policy(policy_name=args.policy, bucket_rows=args.bucket_rows)

    for layer, phase in selected:
        contexts = grouped[(layer, phase)]
        local_context = contexts[0]
        suggested_quantum_rows = estimate_planning_quantum_rows_from_contexts(
            global_contexts=contexts,
            phase=local_context.phase,
        )
        start_ns = time.perf_counter_ns()
        try:
            plan = policy.build_plan(local_context=local_context, global_contexts=contexts)
        except Exception as exc:  # pragma: no cover - exercised via CLI integration
            end_ns = time.perf_counter_ns()
            if not args.skip_errors:
                raise
            skipped.append(
                {
                    "layer_id": layer,
                    "phase": phase,
                    "elapsed_us": (end_ns - start_ns) / 1000.0,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        end_ns = time.perf_counter_ns()
        records.append(
            {
                "layer_id": layer,
                "phase": phase,
                "policy_name": str(plan.policy_name),
                "wave_count": len(plan.waves),
                "bucket_count": int(plan.metrics.get("bucket_count", 0) or 0),
                "suggested_quantum_rows": int(suggested_quantum_rows),
                "elapsed_us": (end_ns - start_ns) / 1000.0,
                "plan_hash": str(plan.plan_hash),
                **_bucket_statistics(plan),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "policy": args.policy,
        "execution_mode": args.mode,
        "bucket_rows": int(args.bucket_rows),
        "phase_count": len(records),
        "skipped_count": len(skipped),
        "total_elapsed_us": sum(float(row["elapsed_us"]) for row in records),
        "max_elapsed_us": max(float(row["elapsed_us"]) for row in records),
        "records": records,
        "skipped": skipped,
    }
    if args.output:
        write_json(Path(args.output), summary)
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
