#!/usr/bin/env python3
"""Audit 4GPU online strategy overhead and hook-path timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-summary-md", required=True)
    parser.add_argument("--run-c-dir")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sum_stage(rows: list[dict[str, Any]], stage: str) -> float:
    return float(sum(float(row.get("duration_us", 0.0) or 0.0) for row in rows if row.get("stage") == stage))


def _transport_makespan(rows: list[dict[str, Any]]) -> tuple[float | None, str | None, dict[str, float | None]]:
    interval_rows: list[tuple[float, float, str]] = []
    for row in rows:
        start = row.get("start_timestamp_us", row.get("start_us", row.get("start_timestamp")))
        end = row.get("end_timestamp_us", row.get("end_us", row.get("end_timestamp")))
        phase = str(row.get("phase", "")).lower()
        if start is None or end is None:
            continue
        interval_rows.append((float(start), float(end), phase))
    if not interval_rows:
        return None, "transport_execution rows do not expose reliable start/end timestamps", {
            "actual_dispatch_transport_makespan_us": None,
            "actual_return_transport_makespan_us": None,
        }
    overall = max(end for _start, end, _phase in interval_rows) - min(start for start, _end, _phase in interval_rows)
    dispatch_rows = [(start, end) for start, end, phase in interval_rows if "dispatch" in phase or phase == "p0"]
    return_rows = [(start, end) for start, end, phase in interval_rows if "return" in phase or phase == "p1"]
    return overall, None, {
        "actual_dispatch_transport_makespan_us": None if not dispatch_rows else max(end for _start, end in dispatch_rows) - min(start for start, _end in dispatch_rows),
        "actual_return_transport_makespan_us": None if not return_rows else max(end for _start, end in return_rows) - min(start for start, _end in return_rows),
    }


def _remote_ratio_from_phase_contexts(rows: list[dict[str, Any]]) -> float | None:
    remote = 0
    total = 0
    for row in rows:
        for seg in row.get("outgoing_segments", []) or []:
            byte_count = int(seg.get("byte_count", 0) or 0)
            total += byte_count
            if not seg.get("is_local", False):
                remote += byte_count
    return None if total <= 0 else float(remote / total)


def _collect_strategy(rep_dir: Path) -> dict[str, Any]:
    summary = _read_json(rep_dir / "summary.json")
    planning_rows = _read_jsonl(rep_dir / "rank0_planning_timing.jsonl")
    phase_context_rows = _read_jsonl(rep_dir / "rank0_phase_contexts.jsonl")
    transport_rows = _read_jsonl(rep_dir / "rank0_transport_execution.jsonl")
    watchdog_path = rep_dir / "watchdog_report.json"
    watchdog = _read_json(watchdog_path) if watchdog_path.exists() else {"status": "not_available"}

    details = summary.get("details", {})
    total_forward_us = summary.get("total_forward_us", details.get("total_forward_us"))
    policy_name = summary.get("policy_name", details.get("policy_name"))
    execution_audit_status = summary.get("execution_audit_status", details.get("execution_audit_status"))

    stage_names = [
        "build_runtime_observation",
        "predict_next_dispatch",
        "build_p2_hint",
        "record_window_state",
        "prepared_phase_plan_shadow",
        "store_prepared_plan",
        "run_phase_plan_agreement",
        "activate_transport",
        "clear_transport",
        "hook_before_token_dispatch_total",
        "hook_after_token_dispatch_total",
        "hook_before_token_combine_total",
        "hook_after_token_combine_total",
    ]
    stage_sums = {stage: _sum_stage(planning_rows, stage) for stage in stage_names}

    dispatch_hook_path_us = stage_sums["hook_before_token_dispatch_total"] + stage_sums["hook_after_token_dispatch_total"]
    combine_hook_path_us = stage_sums["hook_before_token_combine_total"] + stage_sums["hook_after_token_combine_total"]
    transport_hook_path_total_us = dispatch_hook_path_us + combine_hook_path_us
    sync_makespan_us = stage_sums["run_phase_plan_agreement"]
    actual_transport_makespan_us, actual_transport_reason, actual_transport_breakdown = _transport_makespan(transport_rows)
    named_dispatch_substage_sum = (
        stage_sums["predict_next_dispatch"]
        + stage_sums["build_p2_hint"]
        + stage_sums["record_window_state"]
        + stage_sums["prepared_phase_plan_shadow"]
        + stage_sums["store_prepared_plan"]
        + stage_sums["run_phase_plan_agreement"]
    )
    unattributed_dispatch_hook_us = max(0.0, dispatch_hook_path_us - named_dispatch_substage_sum)
    unattributed_combine_hook_us = combine_hook_path_us

    overhead_us = (
        stage_sums["build_runtime_observation"]
        + stage_sums["predict_next_dispatch"]
        + stage_sums["build_p2_hint"]
        + stage_sums["record_window_state"]
        + stage_sums["prepared_phase_plan_shadow"]
        + stage_sums["store_prepared_plan"]
        + stage_sums["run_phase_plan_agreement"]
        + stage_sums["activate_transport"]
        + stage_sums["clear_transport"]
    )

    return {
        "strategy": policy_name,
        "status": summary.get("status"),
        "reason": summary.get("reason"),
        "success": summary.get("status") == "ready" and execution_audit_status in {None, "passed"},
        "watchdog_status": watchdog.get("status"),
        "execution_audit_status": execution_audit_status,
        "total_forward_us": total_forward_us,
        "dispatch_hook_path_us": dispatch_hook_path_us,
        "combine_hook_path_us": combine_hook_path_us,
        "transport_hook_path_total_us": transport_hook_path_total_us,
        "inclusive_dispatch_hook_us": dispatch_hook_path_us,
        "inclusive_combine_hook_us": combine_hook_path_us,
        "actual_transport_makespan_us": actual_transport_makespan_us,
        "actual_transport_makespan_unavailable_reason": actual_transport_reason,
        "actual_dispatch_transport_makespan_us": actual_transport_breakdown["actual_dispatch_transport_makespan_us"],
        "actual_return_transport_makespan_us": actual_transport_breakdown["actual_return_transport_makespan_us"],
        "sync_makespan_us": sync_makespan_us,
        "compute_proxy_us": None if total_forward_us is None else max(
            0.0,
            float(total_forward_us) - transport_hook_path_total_us,
        ),
        "plan_build_time_us": float(sum(float(row.get("build_plan_time_us", 0.0) or 0.0) for row in planning_rows if row.get("stage") == "run_phase_plan_agreement")),
        "plan_agreement_time_us": stage_sums["run_phase_plan_agreement"],
        "policy_select_time_us": float(sum(float(row.get("prediction_time_us", 0.0) or 0.0) for row in planning_rows if row.get("stage") == "predict_next_dispatch")),
        "predict_next_dispatch_us": stage_sums["predict_next_dispatch"],
        "build_p2_hint_us": stage_sums["build_p2_hint"],
        "record_window_state_us": stage_sums["record_window_state"],
        "prepared_phase_plan_shadow_us": stage_sums["prepared_phase_plan_shadow"],
        "store_prepared_plan_us": stage_sums["store_prepared_plan"],
        "hook_before_token_dispatch_total_us": stage_sums["hook_before_token_dispatch_total"],
        "hook_before_token_combine_total_us": stage_sums["hook_before_token_combine_total"],
        "unattributed_dispatch_hook_us": unattributed_dispatch_hook_us,
        "unattributed_combine_hook_us": unattributed_combine_hook_us,
        "artifact_recording_us": 0.0,
        "scheduling_overhead_us": overhead_us,
        "remote_byte_ratio": _remote_ratio_from_phase_contexts(phase_context_rows),
        "transport_execution_count_rank0": len(transport_rows),
        "phase_context_count_rank0": len(phase_context_rows),
        "planning_stage_sums_us": stage_sums,
        "run_dir": str(rep_dir),
    }


def _collect_run_c(run_c_dir: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"run_c_dir": str(run_c_dir), "strategies": {}}
    for strategy_dir in sorted(run_c_dir.iterdir()):
        rep_dir = strategy_dir / "rep0"
        if not rep_dir.is_dir() or not (rep_dir / "summary.json").exists():
            continue
        payload["strategies"][strategy_dir.name] = _collect_strategy(rep_dir)
    return payload


def run_overhead_audit(run_a_dir: Path, run_c_dir: Path | None) -> dict[str, Any]:
    strategies = {}
    for strategy_dir in sorted(run_a_dir.iterdir()):
        rep_dir = strategy_dir / "rep0"
        if rep_dir.is_dir() and (rep_dir / "summary.json").exists():
            strategies[strategy_dir.name] = _collect_strategy(rep_dir)

    birkhoff = strategies.get("birkhoff_phase_local")
    for row in strategies.values():
        if birkhoff is None or row["strategy"] == "birkhoff_phase_local":
            row["benefit_vs_overhead"] = {
                "transport_delta_vs_birkhoff_us": 0.0 if row["strategy"] == "birkhoff_phase_local" else None,
                "transport_delta_reason": None,
                "scheduling_overhead_delta_vs_birkhoff_us": 0.0 if row["strategy"] == "birkhoff_phase_local" else None,
                "total_delta_vs_birkhoff_us": 0.0 if row["strategy"] == "birkhoff_phase_local" else None,
                "overhead_explains_slowdown": None,
                "slowdown_classification": "baseline",
            }
            continue
        if row["actual_transport_makespan_us"] is not None and birkhoff["actual_transport_makespan_us"] is not None:
            transport_delta = float(row["actual_transport_makespan_us"] - birkhoff["actual_transport_makespan_us"])
            transport_delta_reason = None
        else:
            transport_delta = float(row["transport_hook_path_total_us"] - birkhoff["transport_hook_path_total_us"])
            transport_delta_reason = "actual transport makespan unavailable; using hook-path proxy only"
        overhead_delta = float(row["scheduling_overhead_us"] - birkhoff["scheduling_overhead_us"])
        total_delta = float(row["total_forward_us"] - birkhoff["total_forward_us"])
        if transport_delta > 0 and overhead_delta > 0:
            classification = "mixed"
        elif transport_delta > 0:
            classification = "comm_worsens"
        elif overhead_delta > 0:
            classification = "overhead_eats_gain"
        else:
            classification = "no_comm_gain"
        row["benefit_vs_overhead"] = {
            "transport_delta_vs_birkhoff_us": transport_delta,
            "transport_delta_reason": transport_delta_reason,
            "scheduling_overhead_delta_vs_birkhoff_us": overhead_delta,
            "total_delta_vs_birkhoff_us": total_delta,
            "overhead_explains_slowdown": bool(overhead_delta > 0 and total_delta > 0),
            "slowdown_classification": classification,
        }

    run_c_payload = None if run_c_dir is None else _collect_run_c(run_c_dir)
    run_c_vs_run_a = {
        "run_c_run_a_comparable": False,
        "differences": [
            "single-run smoke only",
            "different launch moments and no repeated trials",
            "run_c was a bridge probe, not a comparison harness",
            "run_a includes disabled native baseline and regenerated configs",
            "run_c should not be used for performance claims",
        ],
        "use_run_c_for_performance": False,
    }

    top_overhead_sources = []
    joint = strategies.get("routersense_joint_priority_phase_sync")
    if joint is not None:
        stage_sums = joint["planning_stage_sums_us"]
        top_overhead_sources = [
            stage
            for stage, _ in sorted(stage_sums.items(), key=lambda item: item[1], reverse=True)[:5]
        ]

    return {
        "run_a_dir": str(run_a_dir),
        "strategies": strategies,
        "run_c_comparison": run_c_vs_run_a,
        "run_c_payload": run_c_payload,
        "next_optimization_target": {
            "top_overhead_sources": top_overhead_sources,
            "recommended_fix_order": [
                "reduce_or_cache_predict_next_dispatch",
                "remove_prepared_phase_plan_shadow_from_hot_path",
                "defer_or_compact_record_window_state",
                "compact_store_prepared_plan",
                "re-measure_actual_transport_makespan_after_control_cost_reduction",
            ],
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 4GPU Strategy Overhead Audit",
        "",
        f"- run_a_dir: `{payload['run_a_dir']}`",
        "",
        "| strategy | total_forward_us | transport_hook_path_total_us | actual_transport_makespan_us | scheduling_overhead_us | transport_delta_vs_birkhoff_us | overhead_delta_vs_birkhoff_us | slowdown_classification |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for strategy_name, row in payload["strategies"].items():
        benefit = row["benefit_vs_overhead"]
        lines.append(
            f"| {strategy_name} | {row['total_forward_us']} | {row['transport_hook_path_total_us']} | {row['actual_transport_makespan_us']} | "
            f"{row['scheduling_overhead_us']} | {benefit['transport_delta_vs_birkhoff_us']} | "
            f"{benefit['scheduling_overhead_delta_vs_birkhoff_us']} | {benefit['slowdown_classification']} |"
        )
    lines += [
        "",
        "## Run C vs Run A",
        f"- comparable: `{payload['run_c_comparison']['run_c_run_a_comparable']}`",
        f"- use_run_c_for_performance: `{payload['run_c_comparison']['use_run_c_for_performance']}`",
        "",
        "## Next Optimization Target",
        f"- top_overhead_sources: `{payload['next_optimization_target']['top_overhead_sources']}`",
        f"- recommended_fix_order: `{payload['next_optimization_target']['recommended_fix_order']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    payload = run_overhead_audit(
        run_a_dir=Path(args.run_a_dir),
        run_c_dir=None if not args.run_c_dir else Path(args.run_c_dir),
    )
    output_summary = Path(args.output_summary)
    output_summary_md = Path(args.output_summary_md)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary_md.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_summary_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["next_optimization_target"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
