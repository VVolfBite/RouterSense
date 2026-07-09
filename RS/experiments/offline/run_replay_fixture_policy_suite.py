#!/usr/bin/env python3
"""Batch policy study over replay-derived fixture directories."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline.replay_fixture_policy_study import _build_problem, _expected_flows
from rs.runtime.offline.runner import replay_and_audit_logical_plan, summarize_schedule_tail_metrics
from rs.scheduling import resolve_policy
from rs.scheduling.validation import validate_logical_plan


TABLE_A_POLICIES = (
    "phase_barrier_fifo",
    "greedy_ready_set",
    "birkhoff_phase_local",
    "aurora_order_fixed",
    "fast_bvn_single_tier",
    "routersense_multiphase_lookahead:p0_p1_p2",
)

TABLE_B_POLICIES = (
    "B_birkhoff_wave",
    "U_gated_maxweight_matching",
    "U_barrier_criticality_global_matching",
)

TABLE_C_POLICIES = (
    "birkhoff_phase_local",
    "routersense_multiphase_lookahead:p0_p1_p2",
    "U_gated_maxweight_matching",
    "U_barrier_criticality_global_matching",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-summary-md", required=True)
    return parser.parse_args()


def run_policy_suite(
    *,
    fixture_dir: Path,
    policies: tuple[str, ...],
    mode: str,
    p2_source: str,
    expert_compute_delay: float,
    baseline_policy: str,
    relative_key: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    for fixture_path in fixture_paths:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        problem = _build_problem(
            fixture,
            mode=mode,
            p2_source=p2_source,
            expert_compute_delay=expert_compute_delay,
        )
        expected = _expected_flows(problem)
        for policy_name in policies:
            policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
            plan = policy.build_logical_plan(problem)
            validation = validate_logical_plan(
                plan,
                expected_flows=expected,
                mode=mode,
                expert_compute_delay=expert_compute_delay,
            )
            audit = replay_and_audit_logical_plan(problem, plan)
            tail = summarize_schedule_tail_metrics(problem=problem, plan=plan, audit=audit)
            rows.append(
                {
                    "fixture_name": fixture_path.name,
                    "layer_id": str(fixture.get("metadata", {}).get("layer_id", "")),
                    "policy_name": policy_name,
                    "future_information_mode": str(plan.diagnostics.get("future_information_mode", "")),
                    "evaluation_eligible": bool(plan.diagnostics.get("evaluation_eligible", True)),
                    "valid": bool(validation["valid"]) and bool(audit.get("valid", False)),
                    "makespan": float(plan.diagnostics.get("makespan", audit.get("makespan", 0.0))),
                    "wave_count": int(len(plan.waves)),
                    "tail_completion": float(tail.get("wave_duration_max", 0.0) or 0.0),
                    "p0_completion": float(tail.get("p0_inbound_completion_max", 0.0) or 0.0),
                    "p1_completion": float(tail.get("p1_inbound_completion_max", 0.0) or 0.0),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    by_policy: dict[str, list[dict[str, Any]]] = {name: [row for row in rows if row["policy_name"] == name] for name in policies}
    baseline_rows = by_policy[baseline_policy]
    baseline_mean = statistics.mean([row["makespan"] for row in baseline_rows]) if baseline_rows else 0.0
    for policy_name in policies:
        policy_rows = by_policy[policy_name]
        valid_rows = [row for row in policy_rows if row["valid"]]
        invalid_rows = [row for row in policy_rows if not row["valid"]]
        makespans = [row["makespan"] for row in valid_rows]
        wave_counts = [row["wave_count"] for row in valid_rows]
        tail_completion = [row["tail_completion"] for row in valid_rows]
        p0_completion = [row["p0_completion"] for row in valid_rows]
        p1_completion = [row["p1_completion"] for row in valid_rows]
        mean_makespan = statistics.mean(makespans) if makespans else None
        relative = None if mean_makespan is None or baseline_mean == 0.0 else float((mean_makespan - baseline_mean) / baseline_mean)
        summary_rows.append(
            {
                "policy_name": policy_name,
                "valid_layer_count": len(valid_rows),
                "invalid_layer_count": len(invalid_rows),
                "mean_makespan": mean_makespan,
                "median_makespan": statistics.median(makespans) if makespans else None,
                "min_makespan": min(makespans) if makespans else None,
                "max_makespan": max(makespans) if makespans else None,
                "mean_wave_count": statistics.mean(wave_counts) if wave_counts else None,
                "mean_tail_completion": statistics.mean(tail_completion) if tail_completion else None,
                "mean_p0_completion": statistics.mean(p0_completion) if p0_completion else None,
                "mean_p1_completion": statistics.mean(p1_completion) if p1_completion else None,
                relative_key: relative,
                "evaluation_eligible": bool(policy_rows[0]["evaluation_eligible"]) if policy_rows else False,
                "future_information_mode": str(policy_rows[0]["future_information_mode"]) if policy_rows else "",
            }
        )
    return {
        "mode": mode,
        "p2_source": p2_source,
        "expert_compute_delay": expert_compute_delay,
        "baseline_policy": baseline_policy,
        "rows": rows,
        "summary": summary_rows,
    }


def run_prediction_suite(
    *,
    fixture_dir: Path,
    policies: tuple[str, ...],
    p2_sources: tuple[str, ...],
    expert_compute_delay: float,
) -> dict[str, Any]:
    fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    rows: list[dict[str, Any]] = []
    for fixture_path in fixture_paths:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        for p2_source in p2_sources:
            mode = "runtime_lookahead"
            if p2_source == "actual_trace":
                mode = "runtime_lookahead"
            problem = _build_problem(
                fixture,
                mode=mode,
                p2_source=p2_source,
                expert_compute_delay=expert_compute_delay,
            )
            expected = _expected_flows(problem)
            for policy_name in policies:
                policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
                plan = policy.build_logical_plan(problem)
                validation = validate_logical_plan(
                    plan,
                    expected_flows=expected,
                    mode=mode,
                    expert_compute_delay=expert_compute_delay,
                )
                audit = replay_and_audit_logical_plan(problem, plan)
                rows.append(
                    {
                        "fixture_name": fixture_path.name,
                        "layer_id": str(fixture.get("metadata", {}).get("layer_id", "")),
                        "policy_name": policy_name,
                        "p2_source": p2_source,
                        "future_information_mode": str(plan.diagnostics.get("future_information_mode", "")),
                        "evaluation_eligible": bool(plan.diagnostics.get("evaluation_eligible", True)),
                        "valid": bool(validation["valid"]) and bool(audit.get("valid", False)),
                        "makespan": float(plan.diagnostics.get("makespan", audit.get("makespan", 0.0))),
                        "forecast_matrix_total_bytes": int(problem.forecast.matrix_total_bytes),
                    }
                )
    summary_rows: list[dict[str, Any]] = []
    for policy_name in policies:
        policy_rows = [row for row in rows if row["policy_name"] == policy_name]
        by_source = {source: [row for row in policy_rows if row["p2_source"] == source and row["valid"]] for source in p2_sources}
        zero_mean = statistics.mean([row["makespan"] for row in by_source["zero_hint"]]) if by_source["zero_hint"] else None
        perfect_mean = statistics.mean([row["makespan"] for row in by_source["perfect_trace"]]) if by_source["perfect_trace"] else None
        for p2_source in p2_sources:
            current_rows = by_source[p2_source]
            makespans = [row["makespan"] for row in current_rows]
            mean_makespan = statistics.mean(makespans) if makespans else None
            relative_zero = None
            relative_perfect = None
            if mean_makespan is not None and zero_mean not in (None, 0.0):
                relative_zero = float((mean_makespan - zero_mean) / zero_mean)
            if mean_makespan is not None and perfect_mean not in (None, 0.0):
                relative_perfect = float((mean_makespan - perfect_mean) / perfect_mean)
            seed_row = next((row for row in policy_rows if row["p2_source"] == p2_source), None)
            summary_rows.append(
                {
                    "policy_name": policy_name,
                    "p2_source": p2_source,
                    "mean_makespan": mean_makespan,
                    "relative_to_zero_hint": relative_zero,
                    "relative_to_perfect_trace": relative_perfect,
                    "forecast_matrix_total_bytes": (
                        statistics.mean([row["forecast_matrix_total_bytes"] for row in current_rows]) if current_rows else 0.0
                    ),
                    "future_information_mode": str(seed_row["future_information_mode"]) if seed_row else "",
                    "evaluation_eligible": bool(seed_row["evaluation_eligible"]) if seed_row else False,
                }
            )
    return {
        "mode": "runtime_lookahead",
        "expert_compute_delay": expert_compute_delay,
        "p2_sources": list(p2_sources),
        "rows": rows,
        "summary": summary_rows,
    }


def _render_md(payload: dict[str, Any], audit_summary: dict[str, Any]) -> str:
    lines = ["# Real Trace Replay Summary", "", "## Data source and fixture audit", ""]
    lines.extend(
        [
            f"- fixture_dir: `{payload['fixture_dir']}`",
            f"- source_kind: `{audit_summary['source_kind']}`",
            f"- layer_count: {audit_summary['layer_count']}",
            f"- fixture_count: {audit_summary['fixture_count']}",
            f"- layer_count_with_missing_rank: {audit_summary['layer_count_with_missing_rank']}",
            f"- total_p0_bytes: {audit_summary['total_p0_bytes']}",
            f"- total_p1_bytes: {audit_summary['total_p1_bytes']}",
            f"- total_p2_bytes: {audit_summary['total_p2_bytes']}",
            "- P2 source semantics: next layer uses actual next-layer P0; last layer uses zero matrix.",
            "",
            "## Phase-sync-compatible result",
            "",
            "| Policy | Valid | Mean Makespan | Median | Relative to Birkhoff | Future Mode |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    table_a = payload["table_a"]["summary"]
    for row in table_a:
        rel = row["relative_to_birkhoff_phase_local"]
        rel_text = "-" if rel is None else f"{rel * 100:.2f}%"
        lines.append(
            f"| {row['policy_name']} | {row['valid_layer_count']} | {row['mean_makespan']:.0f} | "
            f"{row['median_makespan']:.0f} | {rel_text} | {row['future_information_mode']} |"
        )
    lines.extend(
        [
            "",
            "当前 `birkhoff_phase_local` 仍然是这组真实自然 trace 上的强 baseline。",
            "当前 `routersense_multiphase_lookahead:p0_p1_p2` 在这组 runtime_lookahead replay 上没有赢过 `birkhoff_phase_local`。",
            "",
            "## Execution-window joint result",
            "",
            "| Policy | Valid | Mean Makespan | Median | Relative to B_birkhoff_wave | Future Mode |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    table_b = payload["table_b"]["summary"]
    for row in table_b:
        rel = row["relative_to_B_birkhoff_wave"]
        rel_text = "-" if rel is None else f"{rel * 100:.2f}%"
        lines.append(
            f"| {row['policy_name']} | {row['valid_layer_count']} | {row['mean_makespan']:.0f} | "
            f"{row['median_makespan']:.0f} | {rel_text} | {row['future_information_mode']} |"
        )
    lines.extend(
        [
            "",
            "在 execution_window 语义下，`U_gated_maxweight_matching` 和 `U_barrier_criticality_global_matching` 都优于 `B_birkhoff_wave`。",
            "这说明多 phase joint scheduling 的空间在真实 trace 上仍然存在。",
            "",
            "## Interpretation for paper",
            "",
            "- 当前 online RouterSense hint policy 还不是 full joint execution-window scheduler。",
            "- offline U_* 结果说明多 phase joint scheduling 仍有空间，但这不等于当前 online RouterSense 已经拿到了这部分收益。",
            "- 下一步需要 transport-stress / EP replay 或 async_release 风格执行语义，才能把 U_* 的空间转成在线系统收益。",
            "- prepared-plan 的 P2 矩阵在真实分布式 EP group 可用时已经可以来自 gathered_global_matrix；但这只是修正全局矩阵来源，不等于真实 next-layer predictor 已经接入。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    fixture_dir = Path(args.fixture_dir)
    audit_path = fixture_dir.parent / "replay_fixture_audit_summary.json"
    if not audit_path.exists():
        raise SystemExit(f"missing fixture audit summary: {audit_path}")
    audit_summary = json.loads(audit_path.read_text(encoding="utf-8"))
    table_a = run_policy_suite(
        fixture_dir=fixture_dir,
        policies=TABLE_A_POLICIES,
        mode="runtime_lookahead",
        p2_source="copy_current_dispatch",
        expert_compute_delay=0.0,
        baseline_policy="birkhoff_phase_local",
        relative_key="relative_to_birkhoff_phase_local",
    )
    table_b = run_policy_suite(
        fixture_dir=fixture_dir,
        policies=TABLE_B_POLICIES,
        mode="execution_window",
        p2_source="actual_trace",
        expert_compute_delay=0.0,
        baseline_policy="B_birkhoff_wave",
        relative_key="relative_to_B_birkhoff_wave",
    )
    payload = {
        "fixture_dir": str(fixture_dir),
        "audit_summary_path": str(audit_path),
        "table_a": table_a,
        "table_b": table_b,
    }
    Path(args.output_summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_summary_md).write_text(_render_md(payload, audit_summary), encoding="utf-8")


if __name__ == "__main__":
    main()
