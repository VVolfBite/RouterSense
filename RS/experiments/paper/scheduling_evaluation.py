from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rs.runtime.offline.replay_unified import PlanningHint, build_execution_truth, build_multiphase_problem, build_planning_problem, ReplayWindow
from rs.scheduling.algorithm_catalog import get_algorithm_metadata
from rs.scheduling.catalog import resolve_algorithm_id
from rs.scheduling.reference.exact_small_instance import solve_problem_exact

from .adapters.scheduling_adapter import execute_policy, replay_window_from_matrices
from .contracts import RecordMetadata, ScheduleEvaluationRecord
from .trace_dataset import discover_replay_fixtures, load_replay_fixture, replay_fixture_to_trace_sample
from .traffic_builder import build_traffic_instance


def _policy_meta(policy_id: str) -> dict[str, Any]:
    try:
        return get_algorithm_metadata(policy_id)
    except KeyError:
        resolved = resolve_algorithm_id(policy_id)
        try:
            return get_algorithm_metadata(str(resolved.canonical_name))
        except KeyError:
            return {
                "algorithm_id": str(resolved.canonical_name),
                "heuristic_family": "unclassified",
                "role": "unclassified",
                "oracle_like": False,
            }


def _planning_problem(window: ReplayWindow) -> tuple[Any, Any, Any]:
    hint = PlanningHint(
        hint_type="perfect_trace_hint",
        p2_hint_rows=window.p2_truth_rows,
        confidence=1.0,
        source_layer=int(window.layer_id),
        target_layer=int(window.layer_id) + 1,
    )
    planning_problem = build_planning_problem(replay_window=window, planning_hint=hint)
    execution_truth = build_execution_truth(window)
    problem = build_multiphase_problem(
        planning_problem=planning_problem,
        execution_truth=execution_truth,
        scheduling_mode="execution_window",
        expert_compute_delay=0.0,
        max_waves=256,
    )
    return hint, execution_truth, problem


def _exact_record(*, instance_id: str, policy_id: str, meta: dict[str, Any], problem, metadata: RecordMetadata, cost_model_id: str) -> ScheduleEvaluationRecord:
    started = time.perf_counter_ns()
    result = solve_problem_exact(problem)
    ended = time.perf_counter_ns()
    nonempty_demand = bool(problem.flow_window.ready_flows or problem.flow_window.blocked_flows)
    empty_plan = not bool(result.get("schedule"))
    supported = bool(result.get("supported", False))
    solver_status = str(result.get("solver_status", "unknown"))
    certified_optimal = bool(result.get("certified_optimal", False))
    objective_logical_makespan = None if result.get("objective_logical_makespan") is None else float(result["objective_logical_makespan"])
    comparable = supported and solver_status == "optimal" and certified_optimal and objective_logical_makespan is not None and not (nonempty_demand and empty_plan)
    validation_errors: list[str] = []
    if nonempty_demand and empty_plan:
        validation_errors.append("empty_exact_plan_with_nonempty_demand")
    if not supported:
        validation_errors.append("exact_solver_unsupported")
    if solver_status != "optimal":
        validation_errors.append(f"solver_status={solver_status}")
    if not certified_optimal:
        validation_errors.append("certified_optimal=false")
    return ScheduleEvaluationRecord(
        instance_id=instance_id,
        policy_id=policy_id,
        policy_family=str(meta.get("heuristic_family", "oracle")),
        scope="joint",
        is_exact=True,
        oracle_like=bool(meta.get("oracle_like", False)),
        reference_model=str(result.get("reference_model")) if result.get("reference_model") is not None else None,
        heuristic=False,
        solver_supported=supported,
        solver_status=solver_status,
        certified_optimal=certified_optimal,
        objective_logical_makespan=objective_logical_makespan if comparable else None,
        best_bound=float(result["best_bound"]) if comparable and result.get("best_bound") is not None else None,
        optimality_gap=float(result["optimality_gap"]) if comparable and result.get("optimality_gap") is not None else None,
        planner_runtime_ms=None,
        validation_runtime_ms=None,
        replay_evaluation_runtime_ms=None,
        record_construction_runtime_ms=None,
        evaluation_total_runtime_ms=(ended - started) / 1_000_000.0,
        objective=objective_logical_makespan if comparable else None,
        coverage_valid=comparable,
        plan_digest=None,
        fallback_count=None,
        comparable=comparable,
        comparable_reason="exact_joint_supported" if comparable else "exact_joint_not_comparable",
        validation_errors=tuple(validation_errors),
        cost_model_id=cost_model_id,
        runtime_info={
            "supported": supported,
            "solver_status": solver_status,
            "certified_optimal": certified_optimal,
            "best_bound": result.get("best_bound"),
            "optimality_gap": result.get("optimality_gap"),
            "search_nodes": result.get("search_nodes"),
            "time_limit_ms": result.get("time_limit_ms"),
        },
        metadata=metadata,
    )


def _heuristic_record(*, instance_id: str, policy_id: str, meta: dict[str, Any], replay_window: ReplayWindow, metadata: RecordMetadata, cost_model_id: str) -> ScheduleEvaluationRecord:
    result = execute_policy(
        replay_window=replay_window,
        policy_name=policy_id,
        hint_type="perfect_trace_hint",
        p2_hint_rows=replay_window.p2_truth_rows,
        confidence=1.0,
        bucket_rows=1,
    )
    validation_errors = tuple(str(item) for item in result.get("validation_errors", ()) or ())
    audit_valid = bool(result.get("audit_valid", False))
    objective = float(result["makespan"]) if audit_valid else None
    comparable = audit_valid
    comparable_reason = "same_cost_model_valid_plan"
    if policy_id == "birkhoff_von_neumann_fluid":
        comparable = False
        comparable_reason = "semantic_mismatch_local_reference_not_exact"
    status = str(result.get("status", "VALID"))
    if not audit_valid:
        comparable = False
        comparable_reason = "invalid_plan"
        if not validation_errors:
            validation_errors = ("audit_valid=false",)
    return ScheduleEvaluationRecord(
        instance_id=instance_id,
        policy_id=policy_id,
        policy_family=str(meta.get("heuristic_family", "unknown")),
        scope="joint" if "joint" in str(meta.get("role", "")) or policy_id.startswith("U_") else "local",
        is_exact=False,
        oracle_like=bool(meta.get("oracle_like", False)),
        reference_model=None,
        heuristic=True,
        solver_supported=None,
        solver_status="NOT_APPLICABLE",
        certified_optimal=None,
        objective_logical_makespan=None,
        best_bound=None,
        optimality_gap=None,
        planner_runtime_ms=None,
        validation_runtime_ms=None,
        replay_evaluation_runtime_ms=None,
        record_construction_runtime_ms=None,
        evaluation_total_runtime_ms=float(result.get("planning_runtime_ms_wall", 0.0) or 0.0),
        objective=objective,
        coverage_valid=audit_valid,
        plan_digest=str(result["logical_plan_digest"]) if audit_valid else None,
        fallback_count=None,
        comparable=comparable,
        comparable_reason=comparable_reason,
        validation_errors=validation_errors,
        cost_model_id=cost_model_id,
        runtime_info={"status": status, "audit": result.get("audit")},
        metadata=metadata,
    )


def evaluate_scheduling(
    *,
    fixture_dir: Path,
    metadata: RecordMetadata,
    model_id: str,
    model_revision: str,
    policy_ids: tuple[str, ...],
    cost_model_id: str = "formal_replay_makespan",
) -> dict[str, Any]:
    records: list[ScheduleEvaluationRecord] = []
    invalid_policy_seen = False
    for path in discover_replay_fixtures(fixture_dir):
        fixture = load_replay_fixture(path)
        trace_sample = replay_fixture_to_trace_sample(
            fixture,
            model_id=model_id,
            model_revision=model_revision,
            metadata=metadata,
        )
        traffic = build_traffic_instance(
            trace_sample=trace_sample,
            p0_matrix=fixture["p0_dispatch_matrix"],
            p1_matrix=fixture["p1_return_matrix"],
            p2_matrix=fixture["p2_next_dispatch_matrix"],
            virtual_ep_size=len(fixture["p0_dispatch_matrix"]),
            metadata=metadata,
            cost_model_id=cost_model_id,
        )
        replay_window = replay_window_from_matrices(
            fixture_id=trace_sample.trace_sample_id,
            layer_id=int(fixture["metadata"].get("layer_id", 0) or 0),
            p0_matrix=traffic.P0_matrix,
            p1_matrix=traffic.P1_matrix,
            p2_matrix=traffic.P2_truth_matrix,
        )
        _hint, _truth, problem = _planning_problem(replay_window)
        for policy_id in policy_ids:
            meta = _policy_meta(policy_id)
            if policy_id == "exact_small_instance_reference":
                record = _exact_record(
                    instance_id=traffic.instance_id,
                    policy_id=policy_id,
                    meta=meta,
                    problem=problem,
                    metadata=metadata,
                    cost_model_id=cost_model_id,
                )
            else:
                record = _heuristic_record(
                    instance_id=traffic.instance_id,
                    policy_id=policy_id,
                    meta=meta,
                    replay_window=replay_window,
                    metadata=metadata,
                    cost_model_id=cost_model_id,
                )
            records.append(record)
            if not record.coverage_valid:
                invalid_policy_seen = True
    status = "OK"
    if invalid_policy_seen:
        status = "PARTIAL_INVALID_POLICY"
    return {
        "records": [record.to_dict() for record in records],
        "status": status,
        "exact_oracle_comparable_count": len([record for record in records if record.is_exact and record.comparable]),
        "o_local_status": "SEMANTICALLY_INVALID",
        "o_local_reason": "no phase-local exact solver under same discrete objective as O_joint",
    }
