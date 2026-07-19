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
from rs.runtime.offline.prediction import rolling_predictor_records, summarize_prediction_records
from rs.runtime.offline.runner import replay_and_audit_logical_plan, summarize_schedule_tail_metrics
from rs.runtime.online.megatron_ep.async_release import simulate_async_release
from rs.scheduling.algorithm_catalog import get_algorithm_metadata, is_paired_comparison_ready, joint_oracle_reference, local_oracle_reference, pair_status_summary
from rs.scheduling.online_adapters import build_priority_artifact_from_plan
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

TABLE_D_PHASE_SYNC_POLICIES = (
    "birkhoff_phase_local",
    "routersense_multiphase_lookahead:p0_p1_p2",
    "routersense_joint_priority_phase_sync",
)

TABLE_D_JOINT_REPLAY_POLICIES = (
    "U_gated_maxweight_matching",
    "U_barrier_criticality_global_matching",
)

PAIRED_FAMILY_ROWS = (
    ("birkhoff_bvn", "B_birkhoff", "U_ibbr"),
    ("gated_greedy", "B_gated_greedy_maximal", "U_gated_greedy_maximal"),
    ("gated_maxweight_matching", "B_gated_maxweight_matching", "U_gated_maxweight_matching"),
    ("barrier_criticality_matching", "B_barrier_criticality_matching", "U_barrier_criticality_global_matching"),
    ("barrier_price_adaptive_matching", "B_barrier_price_adaptive_matching", "U_barrier_price_adaptive_matching"),
    ("lagrangian_cross_phase", "B_lagrangian_phase_local", "U_lagrangian"),
)

SAFE_POLICY_BY_FAMILY = {
    "birkhoff_bvn": "RS_safe_ibbr",
    "gated_greedy": "RS_safe_gated_greedy",
    "gated_maxweight_matching": "RS_safe_gated_maxweight",
    "barrier_criticality_matching": "RS_safe_barrier_criticality",
    "barrier_price_adaptive_matching": "RS_safe_barrier_price",
    "lagrangian_cross_phase": "RS_safe_lagrangian",
}

PREDICTION_SOURCE_LABELS = {
    "zero_hint": "zero_hint",
    "copy_current_dispatch": "copy_current_dispatch",
    "perfect_trace": "perfect_trace_oracle",
    "actual_trace": "actual_trace_oracle",
    "fate_style_history": "fate_style_history",
    "fate_style_linear": "fate_style_linear",
}

PREDICTION_U_POLICIES = (
    "U_gated_greedy_maximal",
    "RS_safe_gated_greedy",
    "U_barrier_criticality_global_matching",
    "RS_safe_barrier_criticality",
    "U_gated_maxweight_matching",
    "RS_safe_gated_maxweight",
)


def _predicted_matrices_by_source(fixture_dir: Path) -> dict[str, dict[str, tuple[tuple[int, ...], ...]]]:
    result: dict[str, dict[str, tuple[tuple[int, ...], ...]]] = {}
    for predictor_name in ("fate_style_history", "fate_style_linear"):
        rows = rolling_predictor_records(fixture_dir=fixture_dir, predictor_name=predictor_name)
        result[predictor_name] = {str(row.layer_id): row.predicted_matrix for row in rows}
    return result


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
                    "segment_count": int(sum(len(wave.flows) for wave in plan.waves)),
                    "future_information_mode": str(plan.diagnostics.get("future_information_mode", "")),
                    "evaluation_eligible": bool(plan.diagnostics.get("evaluation_eligible", True)),
                    "valid": bool(validation["valid"]) and bool(audit.get("valid", False)),
                    "makespan": float(plan.diagnostics.get("makespan", audit.get("makespan", 0.0))),
                    "wave_count": int(len(plan.waves)),
                    "tail_completion": float(tail.get("wave_duration_max", 0.0) or 0.0),
                    "p0_completion": float(tail.get("p0_inbound_completion_max", 0.0) or 0.0),
                    "p1_completion": float(tail.get("p1_inbound_completion_max", 0.0) or 0.0),
                    "fallback_to_paired_b": bool(plan.diagnostics.get("fallback_to_paired_b", False)),
                    "selected_policy": str(plan.diagnostics.get("selected_policy", plan.policy_name)),
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
        segment_counts = [row["segment_count"] for row in valid_rows]
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
                "mean_segment_count": statistics.mean(segment_counts) if segment_counts else None,
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


def run_paired_suite(
    *,
    fixture_dir: Path,
    expert_compute_delay: float,
) -> dict[str, Any]:
    policies = tuple(
        sorted(
            {
                name
                for family, b_name, u_name in PAIRED_FAMILY_ROWS
                for name in (b_name, u_name, SAFE_POLICY_BY_FAMILY[family])
            }
        )
    )
    phase_sync = run_policy_suite(
        fixture_dir=fixture_dir,
        policies=policies,
        mode="runtime_lookahead",
        p2_source="copy_current_dispatch",
        expert_compute_delay=expert_compute_delay,
        baseline_policy="B_gated_maxweight_matching",
        relative_key="relative_to_phase_local_pair_baseline",
    )
    summary_by_policy = {row["policy_name"]: row for row in phase_sync["summary"]}
    rows: list[dict[str, Any]] = []
    for family, b_name, u_name in PAIRED_FAMILY_ROWS:
        safe_name = SAFE_POLICY_BY_FAMILY[family]
        b_row = summary_by_policy.get(b_name)
        u_row = summary_by_policy.get(u_name)
        safe_row = summary_by_policy.get(safe_name)
        b_meta = get_algorithm_metadata(b_name)
        u_meta = get_algorithm_metadata(u_name)
        safe_meta = get_algorithm_metadata(safe_name)
        b_mean = None if b_row is None else b_row.get("mean_makespan")
        u_mean = None if u_row is None else u_row.get("mean_makespan")
        safe_mean = None if safe_row is None else safe_row.get("mean_makespan")
        raw_improvement = None
        safe_improvement = None
        if b_mean not in (None, 0.0) and u_mean is not None:
            raw_improvement = float((float(u_mean) - float(b_mean)) / float(b_mean))
        if b_mean not in (None, 0.0) and safe_mean is not None:
            safe_improvement = float((float(safe_mean) - float(b_mean)) / float(b_mean))
        rows.append(
            {
                "heuristic_family": family,
                "B_algorithm": b_name,
                "raw_U_algorithm": u_name,
                "safe_U_algorithm": safe_name,
                "B_display_name": b_meta["display_name"],
                "raw_U_display_name": u_meta["display_name"],
                "safe_U_display_name": safe_meta["display_name"],
                "B_granularity_mode": b_meta["granularity_mode"],
                "raw_U_granularity_mode": u_meta["granularity_mode"],
                "safe_U_granularity_mode": safe_meta["granularity_mode"],
                "B_mean_makespan": b_mean,
                "raw_U_mean_makespan": u_mean,
                "safe_U_mean_makespan": safe_mean,
                "raw_U_vs_B_improvement_pct": None if raw_improvement is None else float(-100.0 * raw_improvement),
                "safe_U_vs_B_improvement_pct": None if safe_improvement is None else float(-100.0 * safe_improvement),
                "B_valid_layer_count": 0 if b_row is None else int(b_row["valid_layer_count"]),
                "raw_U_valid_layer_count": 0 if u_row is None else int(u_row["valid_layer_count"]),
                "safe_U_valid_layer_count": 0 if safe_row is None else int(safe_row["valid_layer_count"]),
                "B_segment_count": None if b_row is None else b_row.get("mean_segment_count"),
                "raw_U_segment_count": None if u_row is None else u_row.get("mean_segment_count"),
                "safe_U_segment_count": None if safe_row is None else safe_row.get("mean_segment_count"),
                "uses_p2_forecast": bool(u_meta["planning_scope"] in {"multiphase_joint", "execution_window"}),
                "uses_dependency": True,
                "paired_comparison_ready": bool(is_paired_comparison_ready(family)),
                "evaluation_eligible": bool((safe_row or u_row or {}).get("evaluation_eligible", False)),
                "safe_selected_U_ratio": 0.0
                if safe_row is None or int(safe_row["valid_layer_count"]) <= 0
                else float(
                    sum(1 for item in phase_sync["rows"] if item["policy_name"] == safe_name and not bool(item.get("fallback_to_paired_b", False)))
                    / max(1, sum(1 for item in phase_sync["rows"] if item["policy_name"] == safe_name))
                ),
                "safe_fallback_to_B_ratio": 0.0
                if safe_row is None or int(safe_row["valid_layer_count"]) <= 0
                else float(
                    sum(1 for item in phase_sync["rows"] if item["policy_name"] == safe_name and bool(item.get("fallback_to_paired_b", False)))
                    / max(1, sum(1 for item in phase_sync["rows"] if item["policy_name"] == safe_name))
                ),
                "notes": str(u_meta.get("notes", "")),
            }
        )
    return {
        "families": list(PAIRED_FAMILY_ROWS),
        "phase_sync_policy_suite": phase_sync,
        "summary": rows,
    }


def build_oracle_table() -> dict[str, Any]:
    local_ref = local_oracle_reference()
    joint_ref = joint_oracle_reference()
    rows = [
        {
            "oracle_name": local_ref["algorithm_id"],
            "oracle_type": "phase_local",
            "implementation": "rs.scheduling.reference.exact_small_instance::solve_problem_exact_with_scope(scope=local)",
            "objective": "runtime_bucket_wave_exact_phase_barrier",
            "deterministic_solver": bool(local_ref["deterministic_solver"]),
            "heavy_solver": bool(local_ref["heavy_solver"]),
            "best_bound": "exact",
            "optimality_gap": 0.0,
            "gap_from_best_U": None,
            "notes": local_ref["notes"],
        },
        {
            "oracle_name": joint_ref["algorithm_id"],
            "oracle_type": "joint",
            "implementation": "rs.scheduling.reference.exact_small_instance::solve_problem_exact_with_scope(scope=joint)",
            "objective": "runtime_bucket_wave_exact_rank_release_joint",
            "deterministic_solver": bool(joint_ref["deterministic_solver"]),
            "heavy_solver": bool(joint_ref["heavy_solver"]),
            "best_bound": "exact",
            "optimality_gap": 0.0,
            "gap_from_best_U": None,
            "notes": joint_ref["notes"],
        },
        {
            "oracle_name": "exact_small_instance_reference",
            "oracle_type": "joint",
            "implementation": "formal exact_small_instance_reference",
            "objective": "small_exact_joint_reference",
            "deterministic_solver": True,
            "heavy_solver": True,
            "best_bound": "exact",
            "optimality_gap": 0.0,
            "gap_from_best_U": None,
            "notes": "Small-instance exact reference only.",
        },
    ]
    return {"summary": rows}


def summarize_best_pair(paired_summary: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in paired_summary.get("summary", []) if row.get("safe_U_vs_B_improvement_pct") is not None]
    if not rows:
        return {
            "best_family": "",
            "best_improvement_pct": None,
            "u_beats_birkhoff_phase_oracle": False,
        }
    best = max(rows, key=lambda row: float(row["safe_U_vs_B_improvement_pct"]))
    return {
        "best_family": str(best["heuristic_family"]),
        "best_B_algorithm": str(best["B_algorithm"]),
        "best_U_algorithm": str(best["safe_U_algorithm"]),
        "best_improvement_pct": float(best["safe_U_vs_B_improvement_pct"]),
        "u_beats_birkhoff_phase_oracle": False,
    }


def summarize_best_u_frontier(
    *,
    paired_summary: dict[str, Any],
    execution_window_summary: dict[str, Any],
) -> dict[str, Any]:
    paired_rows = [row for row in paired_summary.get("summary", []) if row.get("safe_U_mean_makespan") is not None]
    exec_rows = [
        row
        for row in execution_window_summary.get("summary", [])
        if str(row.get("policy_name", "")).startswith("U_") and row.get("mean_makespan") is not None
    ]
    best_phase_sync = min(paired_rows, key=lambda row: float(row["safe_U_mean_makespan"])) if paired_rows else None
    best_exec = min(exec_rows, key=lambda row: float(row["mean_makespan"])) if exec_rows else None
    birkhoff_wave_row = next(
        (row for row in execution_window_summary.get("summary", []) if row.get("policy_name") == "B_birkhoff_wave"),
        None,
    )
    return {
        "best_phase_sync_u_family": None if best_phase_sync is None else str(best_phase_sync["heuristic_family"]),
        "best_phase_sync_u_algorithm": None if best_phase_sync is None else str(best_phase_sync["safe_U_algorithm"]),
        "best_phase_sync_u_makespan": None if best_phase_sync is None else float(best_phase_sync["safe_U_mean_makespan"]),
        "best_execution_window_u_algorithm": None if best_exec is None else str(best_exec["policy_name"]),
        "best_execution_window_u_makespan": None if best_exec is None else float(best_exec["mean_makespan"]),
        "best_execution_window_gap_to_B_birkhoff_wave_pct": (
            None
            if best_exec is None or birkhoff_wave_row is None or birkhoff_wave_row.get("mean_makespan") in (None, 0.0)
            else float(-100.0 * float(best_exec["relative_to_B_birkhoff_wave"]))
        ),
    }


def run_prediction_suite(
    *,
    fixture_dir: Path,
    policies: tuple[str, ...],
    p2_sources: tuple[str, ...],
    expert_compute_delay: float,
) -> dict[str, Any]:
    fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    predicted_matrices = _predicted_matrices_by_source(fixture_dir)
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
                predicted_p2_matrix=predicted_matrices.get(p2_source, {}).get(str(fixture.get("metadata", {}).get("layer_id", ""))),
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
                        "predictor_name": p2_source if p2_source.startswith("fate_style_") else "",
                        "safe_policy": bool(policy_name.startswith("RS_safe_")),
                        "fallback_to_paired_b": bool(plan.diagnostics.get("fallback_to_paired_b", False)),
                        "selected_policy": str(plan.diagnostics.get("selected_policy", plan.policy_name)),
                    }
                )
    summary_rows: list[dict[str, Any]] = []
    for policy_name in policies:
        try:
            meta = get_algorithm_metadata(policy_name)
        except KeyError:
            meta = None
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
                    "heuristic_family": "" if meta is None else str(meta["heuristic_family"]),
                    "U_algorithm": policy_name,
                    "policy_name": policy_name,
                    "safe_policy": bool(policy_name.startswith("RS_safe_")),
                    "p2_source": str(PREDICTION_SOURCE_LABELS.get(p2_source, p2_source)),
                    "mean_makespan": mean_makespan,
                    "relative_to_zero_hint": relative_zero,
                    "relative_to_perfect_trace": relative_perfect,
                    "forecast_matrix_total_bytes": (
                        statistics.mean([row["forecast_matrix_total_bytes"] for row in current_rows]) if current_rows else 0.0
                    ),
                    "future_information_mode": str(seed_row["future_information_mode"]) if seed_row else "",
                    "evaluation_eligible": bool(seed_row["evaluation_eligible"]) if seed_row else False,
                    "predictor_name": str(seed_row["predictor_name"]) if seed_row else "",
                    "fallback_to_B_ratio": 0.0
                    if not current_rows or not policy_name.startswith("RS_safe_")
                    else float(sum(1 for row in current_rows if bool(row["fallback_to_paired_b"])) / max(1, len(current_rows))),
                    "selected_U_ratio": 0.0
                    if not current_rows or not policy_name.startswith("RS_safe_")
                    else float(sum(1 for row in current_rows if not bool(row["fallback_to_paired_b"])) / max(1, len(current_rows))),
                    "online_adapter_ready": bool(policy_name.startswith("RS_safe_")),
                    "priority_artifact_ready": bool(policy_name.startswith("RS_safe_")),
                    "planning_time_ms": None,
                }
            )
    return {
        "mode": "runtime_lookahead",
        "expert_compute_delay": expert_compute_delay,
        "p2_sources": list(p2_sources),
        "rows": rows,
        "summary": summary_rows,
        "predictor_summaries": {
            predictor_name: summarize_prediction_records(rolling_predictor_records(fixture_dir=fixture_dir, predictor_name=predictor_name))
            for predictor_name in ("fate_style_history", "fate_style_linear")
        },
    }


def run_prediction_u_suite(
    *,
    fixture_dir: Path,
    p2_sources: tuple[str, ...],
    expert_compute_delay: float,
) -> dict[str, Any]:
    payload = run_prediction_suite(
        fixture_dir=fixture_dir,
        policies=PREDICTION_U_POLICIES,
        p2_sources=p2_sources,
        expert_compute_delay=expert_compute_delay,
    )
    payload["u_policies"] = list(PREDICTION_U_POLICIES)
    return payload


def run_bridge_suite(
    *,
    fixture_dir: Path,
    expert_compute_delay: float,
) -> dict[str, Any]:
    phase_sync = run_policy_suite(
        fixture_dir=fixture_dir,
        policies=TABLE_D_PHASE_SYNC_POLICIES,
        mode="runtime_lookahead",
        p2_source="copy_current_dispatch",
        expert_compute_delay=expert_compute_delay,
        baseline_policy="birkhoff_phase_local",
        relative_key="relative_to_birkhoff_phase_local",
    )
    joint_replay = run_policy_suite(
        fixture_dir=fixture_dir,
        policies=TABLE_D_JOINT_REPLAY_POLICIES,
        mode="execution_window",
        p2_source="actual_trace",
        expert_compute_delay=expert_compute_delay,
        baseline_policy="U_gated_maxweight_matching",
        relative_key="relative_to_best_joint_replay",
    )
    predictor_records = rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="fate_style_linear")
    predicted_by_layer = {str(record.layer_id): record.predicted_matrix for record in predictor_records}
    async_rows: list[dict[str, Any]] = []
    fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    for fixture_path in fixture_paths:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        layer_id = str(fixture.get("metadata", {}).get("layer_id", ""))
        problem = _build_problem(
            fixture,
            mode="runtime_lookahead",
            p2_source="fate_style_linear" if layer_id in predicted_by_layer else "copy_current_dispatch",
            expert_compute_delay=expert_compute_delay,
            predicted_p2_matrix=predicted_by_layer.get(layer_id),
        )
        safe_plan = resolve_policy(policy_name="RS_safe_barrier_criticality", bucket_rows=0).build_logical_plan(problem)
        artifact = build_priority_artifact_from_plan(
            problem=problem,
            plan=safe_plan,
            heuristic_family="barrier_criticality_matching",
            predictor_name="fate_style_linear",
            p2_source="fate_style_linear" if layer_id in predicted_by_layer else "copy_current_dispatch",
        )
        sim = simulate_async_release(
            p0_dispatch_matrix=fixture["p0_dispatch_matrix"],
            p1_return_matrix=fixture["p1_return_matrix"],
            predicted_p2_matrix=predicted_by_layer.get(layer_id, tuple(tuple(int(v) for v in row) for row in fixture["p0_dispatch_matrix"])),
            compute_delay=float(expert_compute_delay),
            planning_time_us=0.0,
            prediction_time_us=0.0,
            control_delay_us=0.0,
            prediction_lead_time_us=0.0,
            policy_name="routersense_joint_async_release",
            priority_artifact=artifact,
        )
        async_rows.append({"fixture_name": fixture_path.name, "layer_id": layer_id, **sim})
    baseline_map = {row["policy_name"]: row for row in phase_sync["summary"]}
    birkhoff_mean = float(baseline_map["birkhoff_phase_local"]["mean_makespan"] or 0.0)
    current_mean = float(baseline_map["routersense_multiphase_lookahead:p0_p1_p2"]["mean_makespan"] or 0.0)
    upper_best = min((float(row["mean_makespan"]) for row in joint_replay["summary"] if row["mean_makespan"] is not None), default=0.0)
    async_mean = statistics.mean([float(row["completion_time"]) for row in async_rows]) if async_rows else None
    summary = []
    for row in phase_sync["summary"]:
        mean = row["mean_makespan"]
        summary.append(
            {
                **row,
                "relative_to_current_routersense": None if mean is None or current_mean == 0.0 else float((float(mean) - current_mean) / current_mean),
                "gap_to_best_joint_raw_u": None if mean is None or upper_best == 0.0 else float((float(mean) - upper_best) / upper_best),
                "p2_source": "copy_current_dispatch",
                "predictor_name": "",
                "online_eligible": row["policy_name"] != "routersense_joint_async_release_sim",
                "async_release_required": False,
                "evaluation_mode": "runtime_lookahead",
            }
        )
    summary.append(
        {
            "policy_name": "routersense_joint_async_release_sim",
            "valid_layer_count": len(async_rows),
            "invalid_layer_count": sum(1 for row in async_rows if int(row["dependency_violations"]) > 0),
            "mean_makespan": async_mean,
            "median_makespan": statistics.median([float(row["completion_time"]) for row in async_rows]) if async_rows else None,
            "min_makespan": min((float(row["completion_time"]) for row in async_rows), default=None),
            "max_makespan": max((float(row["completion_time"]) for row in async_rows), default=None),
            "mean_wave_count": statistics.mean([len(row["task_release_timeline"]) for row in async_rows]) if async_rows else None,
            "mean_tail_completion": async_mean,
            "mean_p0_completion": statistics.mean([float(row["p0_inbound_completion_max"]) for row in async_rows]) if async_rows else None,
            "mean_p1_completion": async_mean,
            "relative_to_birkhoff_phase_local": None if async_mean is None or birkhoff_mean == 0.0 else float((float(async_mean) - birkhoff_mean) / birkhoff_mean),
            "relative_to_current_routersense": None if async_mean is None or current_mean == 0.0 else float((float(async_mean) - current_mean) / current_mean),
            "gap_to_best_joint_raw_u": None if async_mean is None or upper_best == 0.0 else float((float(async_mean) - upper_best) / upper_best),
            "p2_source": "fate_style_linear",
            "predictor_name": "fate_style_linear",
            "online_eligible": False,
            "async_release_required": True,
            "evaluation_mode": "async_release_sim",
        }
    )
    for row in joint_replay["summary"]:
        summary.append(
            {
                **row,
                "relative_to_current_routersense": None if row["mean_makespan"] is None or current_mean == 0.0 else float((float(row["mean_makespan"]) - current_mean) / current_mean),
                "gap_to_best_joint_raw_u": None if row["mean_makespan"] is None or upper_best == 0.0 else float((float(row["mean_makespan"]) - upper_best) / upper_best),
                "p2_source": "actual_trace_oracle",
                "predictor_name": "oracle",
                "online_eligible": False,
                "async_release_required": False,
                "evaluation_mode": "execution_window",
            }
        )
    return {"phase_sync": phase_sync, "joint_replay": joint_replay, "async_rows": async_rows, "summary": summary}


def _render_md(payload: dict[str, Any], audit_summary: dict[str, Any]) -> str:
    def _fmt(value: Any, digits: int = 0) -> str:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"

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
            f"| {row['policy_name']} | {row['valid_layer_count']} | {_fmt(row['mean_makespan'])} | "
            f"{_fmt(row['median_makespan'])} | {rel_text} | {row['future_information_mode']} |"
        )
    lines.extend(
        [
            "",
            "## Paired safe-U mainline",
            "",
            f"- main_safe_u_family: `{payload.get('main_safe_u_family')}`",
            f"- main_safe_u_policy: `{payload.get('main_safe_u_policy')}`",
            f"- main_safe_u_improvement_pct: {payload.get('main_safe_u_improvement_pct')}",
            f"- main_raw_u_policy: `{payload.get('main_raw_u_policy')}`",
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
            f"| {row['policy_name']} | {row['valid_layer_count']} | {_fmt(row['mean_makespan'])} | "
            f"{_fmt(row['median_makespan'])} | {rel_text} | {row['future_information_mode']} |"
        )
    lines.extend(
        [
            "",
            "在 execution_window 语义下，`U_gated_maxweight_matching` 和 `U_barrier_criticality_global_matching` 都优于 `B_birkhoff_wave`。",
            "这说明多 phase joint scheduling 的空间在真实 trace 上仍然存在。",
            f"当前 execution-window best U: `{payload.get('execution_window_best_u')}` "
            f"({payload.get('execution_window_u_vs_b_birkhoff_wave_pct')}% vs `B_birkhoff_wave`).",
            "",
            "## Interpretation for paper",
            "",
            "- 当前 online RouterSense hint policy 还不是 full joint execution-window scheduler。",
            "- offline U_* 结果说明多 phase joint scheduling 仍有空间，但这不等于当前 online RouterSense 已经拿到了这部分收益。",
            "- 下一步需要 transport-stress / EP replay 或 async_release 风格执行语义，才能把 U_* 的空间转成在线系统收益。",
            "- prepared-plan 的 gathered_global_matrix 只是 traffic matrix construction，不是 predictor；真实 predictor 仍待接入。",
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
    table_c = run_prediction_u_suite(
        fixture_dir=fixture_dir,
        p2_sources=("zero_hint", "copy_current_dispatch", "fate_style_history", "fate_style_linear", "perfect_trace", "actual_trace"),
        expert_compute_delay=0.0,
    )
    paired = run_paired_suite(
        fixture_dir=fixture_dir,
        expert_compute_delay=0.0,
    )
    table_d = run_bridge_suite(
        fixture_dir=fixture_dir,
        expert_compute_delay=0.0,
    )
    paired_rows = list(paired.get("summary", []))
    safe_ready_rows = [row for row in paired_rows if row.get("safe_U_vs_B_improvement_pct") is not None]
    best_safe = max(safe_ready_rows, key=lambda row: float(row["safe_U_vs_B_improvement_pct"])) if safe_ready_rows else None
    raw_ready_rows = [row for row in paired_rows if row.get("raw_U_vs_B_improvement_pct") is not None]
    best_raw = max(raw_ready_rows, key=lambda row: float(row["raw_U_vs_B_improvement_pct"])) if raw_ready_rows else None
    execution_window_rows = [row for row in table_b.get("summary", []) if row.get("relative_to_B_birkhoff_wave") is not None]
    best_exec_u = min(execution_window_rows, key=lambda row: float(row["relative_to_B_birkhoff_wave"])) if execution_window_rows else None
    oracle_table = build_oracle_table()
    payload = {
        "fixture_dir": str(fixture_dir),
        "audit_summary_path": str(audit_path),
        "main_safe_u_family": None if best_safe is None else best_safe["heuristic_family"],
        "main_safe_u_policy": None if best_safe is None else best_safe["safe_U_algorithm"],
        "main_safe_u_improvement_pct": None if best_safe is None else best_safe["safe_U_vs_B_improvement_pct"],
        "main_raw_u_policy": None if best_raw is None else best_raw["raw_U_algorithm"],
        "execution_window_best_u": None if best_exec_u is None else best_exec_u["policy_name"],
        "execution_window_u_vs_b_birkhoff_wave_pct": None if best_exec_u is None else float(-100.0 * float(best_exec_u["relative_to_B_birkhoff_wave"])),
        "O_local_definition": "exact_runtime_bucket_wave_scope=local",
        "O_joint_definition": "exact_runtime_bucket_wave_scope=joint",
        "O_joint_real_fixture_status": "proxy_only",
        "oracle_gap_small_fixture_summary": {
            "available_via": "experiments.offline.run_oracle_gap_replay",
            "formal_O_local": "exact_runtime_bucket_wave_scope=local",
            "formal_O_joint_small": "exact_runtime_bucket_wave_scope=joint",
        },
        "table_a": table_a,
        "table_b": table_b,
        "table_c": table_c,
        "table_d": table_d,
        "paired_b_vs_u": paired,
        "oracle_table": oracle_table,
    }
    Path(args.output_summary).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_summary_md).write_text(_render_md(payload, audit_summary), encoding="utf-8")


if __name__ == "__main__":
    main()
