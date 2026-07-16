from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rs.core.contracts import PlanningConstraints, PlanningIdentity, PlanningTopology, PlanningWeights
from rs.planning import PlannerRegistry
from rs.planning.request_builder import build_window_planning_request
from rs.runtime.offline.replay_unified import PlanningHint, build_execution_truth, build_multiphase_problem, build_planning_problem, ReplayEngine, ReplayWindow
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.runtime.online.megatron_ep.target_planning.contracts import _compat_logical_plan_from_window_plan
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


def _common_core_metadata(result: dict[str, Any]) -> dict[str, Any]:
    common = dict(result.get("plan_metadata", {}).get("common_core", {}))
    return {
        "matching_core_id": common.get("matching_core_id"),
        "task_contract_digest": common.get("task_contract_digest"),
        "bucket_contract_digest": common.get("bucket_contract_digest"),
        "cost_contract_digest": common.get("cost_contract_digest"),
        "service_model_id": common.get("service_model_id"),
        "solver_budget_digest": common.get("solver_budget_digest"),
    }


def _phase2_served_volume(audit: dict[str, Any]) -> float:
    total = 0.0
    for row in audit.get("raw_schedule", ()) or ():
        phase = row.get("phase")
        flow_id = str(row.get("flow_id", ""))
        chunk_id = str(row.get("chunk_id", ""))
        if phase in {2, "p2_next_dispatch", "phase2"} or "phase2" in flow_id or "p2_next_dispatch" in flow_id or "phase2" in chunk_id:
            total += float(row.get("served_volume", 0.0) or 0.0)
    return total


def _phase2_truth_volume(window: ReplayWindow) -> float:
    return float(sum(sum(int(value) for value in row) for row in window.p2_truth_rows))


def _execution_window_request(*, replay_window: ReplayWindow, planning_hint: PlanningHint, bucket_rows: int = 1) -> Any:
    return build_window_planning_request(
        identity=PlanningIdentity(
            request_id=f"{replay_window.fixture_id}:{replay_window.window_id}:direct",
            run_id=str(replay_window.fixture_id),
            window_id=str(replay_window.window_id),
            source_layer_id=str(replay_window.layer_id),
            target_layer_id=str(planning_hint.target_layer),
        ),
        p0_dispatch_rows=replay_window.p0_truth_rows,
        p1_return_rows=replay_window.p1_truth_rows,
        p2_hint_rows=planning_hint.p2_hint_rows,
        predictor_id=str(planning_hint.hint_type),
        confidence=float(planning_hint.confidence),
        topology=PlanningTopology(world_size=int(replay_window.group_size)),
        constraints=PlanningConstraints(
            bucket_rows=int(bucket_rows),
            max_waves=256,
            expert_compute_delay=0.0,
            phase_release_model="p1_return",
        ),
        weights=PlanningWeights(
            p0_weight=1.0,
            p1_weight=1.0,
            p2_weight=float(planning_hint.confidence),
        ),
        information_mode="p0_p1_p2",
        hint_type=str(planning_hint.hint_type),
        oracle=bool(planning_hint.hint_type == "perfect_trace_hint"),
        planning_track="execution_window",
        p2_semantics="executable_actual",
    )


def _execution_window_bridge_summary(*, replay_window: ReplayWindow, planning_hint: PlanningHint, policy_name: str) -> dict[str, Any]:
    _hint, execution_truth, problem = _planning_problem(replay_window)
    request = _execution_window_request(replay_window=replay_window, planning_hint=planning_hint)
    planner = PlannerRegistry.create(str(policy_name), None)
    direct_plan = planner.plan(request)
    direct_audit = replay_and_audit_logical_plan(problem, _compat_logical_plan_from_window_plan(direct_plan))
    replay_result = ReplayEngine(
        scheduling_mode="execution_window",
        expert_compute_delay=0.0,
        bucket_rows=1,
    ).execute(
        replay_window=replay_window,
        planning_hint=planning_hint,
        policy_name=str(policy_name),
    )
    truth_volume = _phase2_truth_volume(replay_window)
    return {
        "policy_id": str(policy_name),
        "truth_p2_volume": truth_volume,
        "direct_global_scheduler": {
            "audit_valid": bool(direct_audit.get("valid", False)),
            "phase2_served_volume": _phase2_served_volume(dict(direct_plan.metadata)),
            "validation_errors": list(direct_audit.get("validation_errors", ()) or ()),
        },
        "replay_engine": {
            "audit_valid": bool(replay_result.get("audit_valid", False)),
            "planning_track": replay_result.get("planning_track"),
            "p2_semantics": replay_result.get("p2_semantics"),
            "phase2_served_volume": _phase2_served_volume(dict(replay_result.get("plan_metadata", {}))),
            "validation_errors": list(dict(replay_result.get("audit", {})).get("validation_errors", ()) or ()),
        },
    }


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
        matching_core_id=None,
        task_contract_digest=None,
        bucket_contract_digest=None,
        cost_contract_digest=None,
        service_model_id=str(result.get("reference_model")) if result.get("reference_model") is not None else None,
        solver_budget_digest=None,
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
    common_core = _common_core_metadata(result)
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
        matching_core_id=common_core["matching_core_id"],
        task_contract_digest=common_core["task_contract_digest"],
        bucket_contract_digest=common_core["bucket_contract_digest"],
        cost_contract_digest=common_core["cost_contract_digest"],
        service_model_id=common_core["service_model_id"],
        solver_budget_digest=common_core["solver_budget_digest"],
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
    execution_window_bridge_summary: dict[str, Any] | None = None
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
        if execution_window_bridge_summary is None and "U_barrier_criticality_global_matching" in policy_ids:
            execution_window_bridge_summary = _execution_window_bridge_summary(
                replay_window=replay_window,
                planning_hint=PlanningHint(
                    hint_type="perfect_trace_hint",
                    p2_hint_rows=replay_window.p2_truth_rows,
                    confidence=1.0,
                    source_layer=int(replay_window.layer_id),
                    target_layer=int(replay_window.layer_id) + 1,
                ),
                policy_name="U_barrier_criticality_global_matching",
            )
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
    strict_pair = [record for record in records if record.policy_id in {"B_barrier_criticality_core_independent", "U_barrier_criticality_global_matching"}]
    same_core_pair_summary = {
        "status": "NOT_AVAILABLE",
        "records": [],
    }
    paired_by_instance: dict[str, dict[str, ScheduleEvaluationRecord]] = {}
    for record in strict_pair:
        paired_by_instance.setdefault(record.instance_id, {})[record.policy_id] = record
    pair_records: tuple[ScheduleEvaluationRecord, ScheduleEvaluationRecord] | None = None
    for instance_id in sorted(paired_by_instance):
        group = paired_by_instance[instance_id]
        if "B_barrier_criticality_core_independent" in group and "U_barrier_criticality_global_matching" in group:
            pair_records = (
                group["B_barrier_criticality_core_independent"],
                group["U_barrier_criticality_global_matching"],
            )
            break
    if pair_records is not None:
        b_record, u_record = pair_records
        same_core_pair_summary = {
            "status": "READY",
            "family_pair": {
                "b_policy_id": "B_barrier_criticality_matching",
                "u_policy_id": "U_barrier_criticality_global_matching",
            },
            "strict_same_core_pair": {
                "b_policy_id": b_record.policy_id,
                "u_policy_id": u_record.policy_id,
            },
            "safe_fallback_pair": {
                "policy_id": "RS_safe_barrier_criticality",
                "raw_u_policy_id": "U_barrier_criticality_global_matching",
                "paired_b_policy_id": "B_barrier_criticality_matching",
            },
            "records": [b_record.to_dict(), u_record.to_dict()],
            "contract_match": {
                "matching_core_id": b_record.matching_core_id == u_record.matching_core_id,
                "task_contract_digest": b_record.task_contract_digest == u_record.task_contract_digest,
                "bucket_contract_digest": b_record.bucket_contract_digest == u_record.bucket_contract_digest,
                "cost_contract_digest": b_record.cost_contract_digest == u_record.cost_contract_digest,
                "service_model_id": b_record.service_model_id == u_record.service_model_id,
                "solver_budget_digest": b_record.solver_budget_digest == u_record.solver_budget_digest,
            },
        }
    return {
        "records": [record.to_dict() for record in records],
        "status": status,
        "exact_oracle_comparable_count": len([record for record in records if record.is_exact and record.comparable]),
        "o_local_status": "SEMANTICALLY_INVALID",
        "o_local_reason": "no phase-local exact solver under same discrete objective as O_joint",
        "execution_window_bridge_summary": execution_window_bridge_summary,
        "same_core_pair_summary": same_core_pair_summary,
    }
