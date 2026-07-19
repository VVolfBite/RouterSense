"""Experiment-support helpers for replay prediction policy studies."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from rs.runtime.offline.policy_study import build_replay_problem, expected_replay_flows
from rs.runtime.offline.prediction import rolling_predictor_records, summarize_prediction_records
from rs.runtime.offline.runner import replay_and_audit_logical_plan, summarize_schedule_tail_metrics
from rs.runtime.online.megatron_ep.async_release import simulate_async_release
from rs.scheduling import resolve_policy
from rs.scheduling.algorithm_catalog import (
    get_algorithm_metadata,
    is_paired_comparison_ready,
    joint_oracle_reference,
    local_oracle_reference,
    pair_status_summary,
)
from rs.scheduling.online_adapters import build_priority_artifact_from_plan
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
        problem = build_replay_problem(
            fixture,
            mode=mode,
            p2_source=p2_source,
            expert_compute_delay=expert_compute_delay,
        )
        expected = expected_replay_flows(problem)
        for policy_name in policies:
            policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
            plan = policy.build_logical_plan(problem)
            validation = validate_logical_plan(plan, expected_flows=expected, mode=mode, expert_compute_delay=expert_compute_delay)
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
    by_policy = {name: [row for row in rows if row["policy_name"] == name] for name in policies}
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


def run_paired_suite(*, fixture_dir: Path, expert_compute_delay: float) -> dict[str, Any]:
    policies = tuple(sorted({name for family, b_name, u_name in PAIRED_FAMILY_ROWS for name in (b_name, u_name, SAFE_POLICY_BY_FAMILY[family])}))
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
        raw_improvement = None if b_mean in (None, 0.0) or u_mean is None else float((float(u_mean) - float(b_mean)) / float(b_mean))
        safe_improvement = None if b_mean in (None, 0.0) or safe_mean is None else float((float(safe_mean) - float(b_mean)) / float(b_mean))
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
                "safe_selected_U_ratio": 0.0 if safe_row is None else float(
                    sum(1 for item in phase_sync["rows"] if item["policy_name"] == safe_name and not bool(item.get("fallback_to_paired_b", False)))
                    / max(1, sum(1 for item in phase_sync["rows"] if item["policy_name"] == safe_name))
                ),
                "safe_fallback_to_B_ratio": 0.0 if safe_row is None else float(
                    sum(1 for item in phase_sync["rows"] if item["policy_name"] == safe_name and bool(item.get("fallback_to_paired_b", False)))
                    / max(1, sum(1 for item in phase_sync["rows"] if item["policy_name"] == safe_name))
                ),
                "notes": str(u_meta.get("notes", "")),
            }
        )
    return {"families": list(PAIRED_FAMILY_ROWS), "phase_sync_policy_suite": phase_sync, "summary": rows}


def build_oracle_table() -> dict[str, Any]:
    local_ref = local_oracle_reference()
    joint_ref = joint_oracle_reference()
    return {
        "summary": [
            {
                "oracle_name": local_ref["algorithm_id"],
                "oracle_type": "phase_local",
                "implementation": "birkhoff_von_neumann_fluid",
                "objective": "single_phase_fluid_makespan",
                "deterministic_solver": bool(local_ref["deterministic_solver"]),
                "heavy_solver": bool(local_ref["heavy_solver"]),
                "best_bound": None,
                "optimality_gap": None,
                "gap_from_best_U": None,
                "notes": local_ref["notes"],
            },
            {
                "oracle_name": joint_ref["algorithm_id"],
                "oracle_type": "joint",
                "implementation": "legacy/historical_poc/src_rs_legacy/scheduler/oracle.py::pairwise_oracle",
                "objective": "joint_p0_p1_p2_cp_sat",
                "deterministic_solver": bool(joint_ref["deterministic_solver"]),
                "heavy_solver": bool(joint_ref["heavy_solver"]),
                "best_bound": "legacy_exposes_best_bound",
                "optimality_gap": "legacy_exposes_optimality_gap",
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
    }


def run_prediction_suite(*, fixture_dir: Path, policies: tuple[str, ...], p2_sources: tuple[str, ...], expert_compute_delay: float) -> dict[str, Any]:
    fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    predicted_matrices = _predicted_matrices_by_source(fixture_dir)
    rows: list[dict[str, Any]] = []
    for fixture_path in fixture_paths:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        for p2_source in p2_sources:
            problem = build_replay_problem(
                fixture,
                mode="runtime_lookahead",
                p2_source=p2_source,
                expert_compute_delay=expert_compute_delay,
                predicted_p2_matrix=predicted_matrices.get(p2_source, {}).get(str(fixture.get("metadata", {}).get("layer_id", ""))),
            )
            expected = expected_replay_flows(problem)
            for policy_name in policies:
                policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
                plan = policy.build_logical_plan(problem)
                validation = validate_logical_plan(plan, expected_flows=expected, mode="runtime_lookahead", expert_compute_delay=expert_compute_delay)
                audit = replay_and_audit_logical_plan(problem, plan)
                rows.append(
                    {
                        "fixture_name": fixture_path.name,
                        "layer_id": str(fixture.get("metadata", {}).get("layer_id", "")),
                        "policy_name": policy_name,
                        "p2_source": p2_source,
                        "predictor_name": PREDICTION_SOURCE_LABELS.get(p2_source, p2_source),
                        "forecast_matrix_total_bytes": float(sum(sum(int(value) for value in row) for row in problem.p2_next_dispatch_forecast_matrix)),
                        "future_information_mode": str(plan.diagnostics.get("future_information_mode", "")),
                        "evaluation_eligible": bool(plan.diagnostics.get("evaluation_eligible", True)),
                        "valid": bool(validation["valid"]) and bool(audit.get("valid", False)),
                        "makespan": float(plan.diagnostics.get("makespan", audit.get("makespan", 0.0))),
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
            relative_zero = None if mean_makespan is None or zero_mean in (None, 0.0) else float((mean_makespan - zero_mean) / zero_mean)
            relative_perfect = None if mean_makespan is None or perfect_mean in (None, 0.0) else float((mean_makespan - perfect_mean) / perfect_mean)
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
                    "forecast_matrix_total_bytes": statistics.mean([row["forecast_matrix_total_bytes"] for row in current_rows]) if current_rows else 0.0,
                    "future_information_mode": str(seed_row["future_information_mode"]) if seed_row else "",
                    "evaluation_eligible": bool(seed_row["evaluation_eligible"]) if seed_row else False,
                    "predictor_name": str(seed_row["predictor_name"]) if seed_row else "",
                    "fallback_to_B_ratio": 0.0 if not current_rows or not policy_name.startswith("RS_safe_") else float(sum(1 for row in current_rows if bool(row["fallback_to_paired_b"])) / max(1, len(current_rows))),
                    "selected_U_ratio": 0.0 if not current_rows or not policy_name.startswith("RS_safe_") else float(sum(1 for row in current_rows if not bool(row["fallback_to_paired_b"])) / max(1, len(current_rows))),
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
        "predictor_summaries": {name: summarize_prediction_records(rolling_predictor_records(fixture_dir=fixture_dir, predictor_name=name)) for name in ("fate_style_history", "fate_style_linear")},
    }


def run_prediction_u_suite(*, fixture_dir: Path, p2_sources: tuple[str, ...], expert_compute_delay: float) -> dict[str, Any]:
    payload = run_prediction_suite(
        fixture_dir=fixture_dir,
        policies=PREDICTION_U_POLICIES,
        p2_sources=p2_sources,
        expert_compute_delay=expert_compute_delay,
    )
    payload["u_policies"] = list(PREDICTION_U_POLICIES)
    return payload


def run_bridge_suite(*, fixture_dir: Path, expert_compute_delay: float) -> dict[str, Any]:
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
        problem = build_replay_problem(
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


__all__ = [
    "PREDICTION_SOURCE_LABELS",
    "PREDICTION_U_POLICIES",
    "SAFE_POLICY_BY_FAMILY",
    "TABLE_A_POLICIES",
    "TABLE_B_POLICIES",
    "TABLE_C_POLICIES",
    "TABLE_D_JOINT_REPLAY_POLICIES",
    "TABLE_D_PHASE_SYNC_POLICIES",
    "build_oracle_table",
    "pair_status_summary",
    "run_bridge_suite",
    "run_paired_suite",
    "run_policy_suite",
    "run_prediction_suite",
    "run_prediction_u_suite",
]
