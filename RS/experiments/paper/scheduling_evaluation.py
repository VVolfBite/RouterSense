from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from rs.core.contracts import PlanningConstraints, PlanningIdentity, PlanningTopology, PlanningWeights
from rs.planning import PlannerRegistry
from rs.planning.request_builder import build_window_planning_request
from rs.runtime.offline.replay_unified import PlanningHint, ReplayEngine, ReplayWindow, build_execution_truth, build_multiphase_problem, build_planning_problem
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.runtime.online.megatron_ep.target_planning.contracts import _compat_logical_plan_from_window_plan
from rs.scheduling.algorithm_catalog import get_algorithm_metadata, list_pair_families
from rs.scheduling.catalog import resolve_algorithm_id
from rs.scheduling.reference.exact_small_instance import exact_result_to_logical_plan, solve_problem_exact_with_scope

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


def build_paper_execution_window_problem(
    *,
    fixture_id: str,
    layer_id: int,
    p0_matrix: tuple[tuple[int, ...], ...],
    p1_matrix: tuple[tuple[int, ...], ...],
    p2_matrix: tuple[tuple[int, ...], ...],
    bucket_rows: int = 1,
    expert_compute_delay: float = 0.0,
) -> tuple[ReplayWindow, PlanningHint, Any]:
    replay_window = replay_window_from_matrices(
        fixture_id=fixture_id,
        layer_id=int(layer_id),
        p0_matrix=p0_matrix,
        p1_matrix=p1_matrix,
        p2_matrix=p2_matrix,
    )
    hint = PlanningHint(
        hint_type="perfect_trace_hint",
        p2_hint_rows=replay_window.p2_truth_rows,
        confidence=1.0,
        source_layer=int(replay_window.layer_id),
        target_layer=int(replay_window.layer_id) + 1,
    )
    planning_problem = build_planning_problem(replay_window=replay_window, planning_hint=hint)
    execution_truth = build_execution_truth(replay_window)
    problem = build_multiphase_problem(
        planning_problem=planning_problem,
        execution_truth=execution_truth,
        scheduling_mode="execution_window",
        expert_compute_delay=float(expert_compute_delay),
        max_waves=max(256, int(bucket_rows)),
    )
    return replay_window, hint, problem


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
    served = dict(audit.get("served_volume_by_phase", {}) or {})
    return float(served.get(2, served.get("2", 0.0)) or 0.0)


def _phase2_truth_report(window: ReplayWindow) -> dict[str, float]:
    total = 0.0
    self_volume = 0.0
    remote = 0.0
    for src, row in enumerate(window.p2_truth_rows):
        for dst, value in enumerate(row):
            volume = float(int(value))
            total += volume
            if int(src) == int(dst):
                self_volume += volume
            else:
                remote += volume
    return {
        "truth_p2_total_volume": total,
        "truth_p2_self_volume": self_volume,
        "truth_p2_remote_volume": remote,
    }


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
    _hint, _execution_truth, problem = _planning_problem(replay_window)
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
    truth = _phase2_truth_report(replay_window)
    return {
        "policy_id": str(policy_name),
        **truth,
        "direct_global_scheduler": {
            "audit_valid": bool(direct_audit.get("valid", False)),
            "served_p2_remote_volume": _phase2_served_volume(direct_audit),
            "validation_errors": list(direct_audit.get("validation_errors", ()) or ()),
        },
        "replay_engine": {
            "audit_valid": bool(replay_result.get("audit_valid", False)),
            "planning_track": replay_result.get("planning_track"),
            "p2_semantics": replay_result.get("p2_semantics"),
            "served_p2_remote_volume": _phase2_served_volume(dict(replay_result.get("audit", {}))),
            "validation_errors": list(dict(replay_result.get("audit", {})).get("validation_errors", ()) or ()),
        },
    }


def _exact_record(
    *,
    instance_id: str,
    policy_id: str,
    meta: dict[str, Any],
    problem,
    metadata: RecordMetadata,
    cost_model_id: str,
    scope: str,
) -> ScheduleEvaluationRecord:
    started = time.perf_counter_ns()
    result = solve_problem_exact_with_scope(problem, scope=scope)
    replay_started = time.perf_counter_ns()
    nonempty_demand = bool(problem.flow_window.ready_flows or problem.flow_window.blocked_flows)
    empty_plan = not bool(result.get("schedule"))
    supported = bool(result.get("supported", False))
    solver_status = str(result.get("solver_status", "unknown"))
    certified_optimal = bool(result.get("certified_optimal", False))
    objective_logical_makespan = None if result.get("objective_logical_makespan") is None else float(result["objective_logical_makespan"])
    comparable = supported and solver_status == "optimal" and certified_optimal and objective_logical_makespan is not None and not (nonempty_demand and empty_plan)
    validation_errors: list[str] = []
    replay_audit: dict[str, Any] = {}
    validation_runtime_ms = None
    replay_evaluation_runtime_ms = None
    plan_digest = None
    if nonempty_demand and empty_plan:
        validation_errors.append("empty_exact_plan_with_nonempty_demand")
    if not supported:
        validation_errors.append("exact_solver_unsupported")
    if solver_status != "optimal":
        validation_errors.append(f"solver_status={solver_status}")
    if not certified_optimal:
        validation_errors.append("certified_optimal=false")
    if comparable:
        logical_plan = exact_result_to_logical_plan(result, policy_name=policy_id)
        validation_started = time.perf_counter_ns()
        replay_audit = replay_and_audit_logical_plan(problem, logical_plan)
        validation_ended = time.perf_counter_ns()
        validation_runtime_ms = (validation_ended - validation_started) / 1_000_000.0
        replay_evaluation_runtime_ms = validation_runtime_ms
        plan_digest = hashlib.sha256(json.dumps(logical_plan.to_dict(), ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()
        if not bool(replay_audit.get("valid", False)):
            comparable = False
            objective_logical_makespan = None
            validation_errors.extend(str(item) for item in replay_audit.get("validation_errors", ()) or ())
        else:
            objective_logical_makespan = float(replay_audit.get("replay_makespan", replay_audit.get("makespan", objective_logical_makespan)))
    ended = time.perf_counter_ns()
    reference_model = str(result.get("reference_model")) if result.get("reference_model") is not None else None
    return ScheduleEvaluationRecord(
        instance_id=instance_id,
        policy_id=policy_id,
        policy_family=str(meta.get("heuristic_family", "oracle")),
        scope="local" if scope == "local" else "joint",
        is_exact=True,
        oracle_like=bool(meta.get("oracle_like", False)),
        reference_model=reference_model,
        heuristic=False,
        solver_supported=supported,
        solver_status=solver_status,
        certified_optimal=certified_optimal,
        objective_logical_makespan=objective_logical_makespan if comparable else None,
        best_bound=float(result["best_bound"]) if comparable and result.get("best_bound") is not None else None,
        optimality_gap=float(result["optimality_gap"]) if comparable and result.get("optimality_gap") is not None else None,
        planner_runtime_ms=float(result.get("solver_runtime_ms_wall")) if result.get("solver_runtime_ms_wall") is not None else None,
        validation_runtime_ms=validation_runtime_ms,
        replay_evaluation_runtime_ms=replay_evaluation_runtime_ms,
        record_construction_runtime_ms=None,
        evaluation_total_runtime_ms=(ended - started) / 1_000_000.0,
        objective=objective_logical_makespan if comparable else None,
        coverage_valid=bool(replay_audit.get("valid", comparable)),
        plan_digest=plan_digest if comparable else None,
        fallback_count=None,
        comparable=comparable,
        comparable_reason=f"exact_{scope}_supported" if comparable else f"exact_{scope}_not_comparable",
        validation_errors=tuple(validation_errors),
        cost_model_id=cost_model_id,
        matching_core_id=None,
        task_contract_digest=None,
        bucket_contract_digest=None,
        cost_contract_digest=None,
        service_model_id=reference_model,
        solver_budget_digest=None,
        runtime_info={
            "supported": supported,
            "solver_status": solver_status,
            "certified_optimal": certified_optimal,
            "best_bound": result.get("best_bound"),
            "optimality_gap": result.get("optimality_gap"),
            "search_nodes": result.get("search_nodes"),
            "solver_runtime_ms_wall": result.get("solver_runtime_ms_wall"),
            "solver_time_limit_ms": result.get("time_limit_ms"),
            "phase_solver_results": result.get("phase_solver_results"),
            "combined_validation": replay_audit if replay_audit else result.get("combined_validation"),
            "served_p2_volume": _phase2_served_volume(replay_audit),
        },
        metadata=metadata,
    )


def _heuristic_record(
    *,
    instance_id: str,
    policy_id: str,
    meta: dict[str, Any],
    replay_window: ReplayWindow,
    metadata: RecordMetadata,
    cost_model_id: str,
) -> ScheduleEvaluationRecord:
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
        planner_runtime_ms=float(result.get("planning_runtime_ms_wall", 0.0) or 0.0),
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


def _strict_pair_summary(records: list[ScheduleEvaluationRecord]) -> dict[str, Any]:
    same_core = [record for record in records if record.policy_id in {"B_barrier_criticality_core_independent", "U_barrier_criticality_global_matching"}]
    paired_by_instance: dict[str, dict[str, ScheduleEvaluationRecord]] = {}
    for record in same_core:
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
    if pair_records is None:
        return {
            "status": "NOT_AVAILABLE",
            "pair_kind": "strict_same_core",
            "pair_status": "NOT_AVAILABLE",
            "comparable": False,
            "records": [],
        }
    b_record, u_record = pair_records
    contract_match = {
        "matching_core_id_equal": b_record.matching_core_id == u_record.matching_core_id,
        "task_contract_digest_equal": b_record.task_contract_digest == u_record.task_contract_digest,
        "bucket_contract_digest_equal": b_record.bucket_contract_digest == u_record.bucket_contract_digest,
        "cost_contract_digest_equal": b_record.cost_contract_digest == u_record.cost_contract_digest,
        "service_model_id_equal": b_record.service_model_id == u_record.service_model_id,
        "solver_budget_digest_equal": b_record.solver_budget_digest == u_record.solver_budget_digest,
    }
    comparable = all(contract_match.values()) and b_record.coverage_valid and u_record.coverage_valid
    return {
        "status": "READY" if comparable else "PARTIAL_INVALID_POLICY",
        "pair_kind": "strict_same_core",
        "pair_status": "VALID_PAIR" if comparable else "INVALID_PAIR",
        "comparable": comparable,
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
        "contract_match": contract_match,
        "metadata_contract": {
            "matching_core_id": b_record.matching_core_id,
            "task_contract_digest": b_record.task_contract_digest,
            "bucket_contract_digest": b_record.bucket_contract_digest,
            "cost_contract_digest": b_record.cost_contract_digest,
            "service_model_id": b_record.service_model_id,
            "solver_budget_digest": b_record.solver_budget_digest,
        },
        "b_objective": b_record.objective,
        "u_objective": u_record.objective,
        "paired_delta": None if b_record.objective is None or u_record.objective is None else float(u_record.objective - b_record.objective),
    }



def _family_pair_summaries(records: list[ScheduleEvaluationRecord]) -> list[dict[str, Any]]:
    by_policy_instance: dict[tuple[str, str], ScheduleEvaluationRecord] = {
        (record.policy_id, record.instance_id): record for record in records
    }
    summaries: list[dict[str, Any]] = []
    contract_fields = (
        "matching_core_id",
        "task_contract_digest",
        "bucket_contract_digest",
        "cost_contract_digest",
        "service_model_id",
        "solver_budget_digest",
    )
    for pair in list_pair_families():
        family_id = str(pair["heuristic_family"])
        local_id = pair.get("B_algorithm")
        joint_id = pair.get("U_algorithm")
        if not local_id or not joint_id or not bool(pair.get("paired_comparison_ready", False)):
            continue
        instance_ids = sorted(
            {
                instance_id
                for policy_id, instance_id in by_policy_instance
                if policy_id in {str(local_id), str(joint_id)}
            }
        )
        paired_rows: list[dict[str, Any]] = []
        for instance_id in instance_ids:
            local = by_policy_instance.get((str(local_id), instance_id))
            joint = by_policy_instance.get((str(joint_id), instance_id))
            if local is None or joint is None:
                continue
            contract_match = {
                f"{field}_equal": getattr(local, field) == getattr(joint, field)
                for field in contract_fields
            }
            comparable = bool(local.comparable and joint.comparable and all(contract_match.values()))
            delta = None if local.objective is None or joint.objective is None else float(joint.objective - local.objective)
            improvement = None
            if local.objective is not None and joint.objective is not None and float(local.objective) > 0.0:
                improvement = (float(local.objective) - float(joint.objective)) / float(local.objective) * 100.0
            local_runtime = local.planner_runtime_ms
            joint_runtime = joint.planner_runtime_ms
            paired_rows.append(
                {
                    "instance_id": instance_id,
                    "comparable": comparable,
                    "contract_match": contract_match,
                    "local_objective": local.objective,
                    "joint_objective": joint.objective,
                    "joint_minus_local_makespan": delta,
                    "joint_improvement_pct": improvement,
                    "outcome": None if delta is None else "win" if delta < -1e-9 else "loss" if delta > 1e-9 else "tie",
                    "local_planner_runtime_ms": local_runtime,
                    "joint_planner_runtime_ms": joint_runtime,
                    "joint_minus_local_runtime_ms": None if local_runtime is None or joint_runtime is None else float(joint_runtime - local_runtime),
                }
            )
        if not paired_rows:
            continue
        comparable_rows = [row for row in paired_rows if row["comparable"]]
        summaries.append(
            {
                "family_id": family_id,
                "local_policy_id": local_id,
                "joint_policy_id": joint_id,
                "status": "READY" if len(comparable_rows) == len(paired_rows) else "PARTIAL_INVALID_PAIR",
                "pair_count": len(paired_rows),
                "comparable_count": len(comparable_rows),
                "win_count": sum(row["outcome"] == "win" for row in comparable_rows),
                "tie_count": sum(row["outcome"] == "tie" for row in comparable_rows),
                "loss_count": sum(row["outcome"] == "loss" for row in comparable_rows),
                "records": paired_rows,
            }
        )
    return summaries

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
    o_local_record: ScheduleEvaluationRecord | None = None
    o_joint_record: ScheduleEvaluationRecord | None = None
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
        planning_hint, _truth, problem = _planning_problem(replay_window)
        if execution_window_bridge_summary is None and "U_barrier_criticality_global_matching" in policy_ids:
            execution_window_bridge_summary = _execution_window_bridge_summary(
                replay_window=replay_window,
                planning_hint=planning_hint,
                policy_name="U_barrier_criticality_global_matching",
            )
        if o_local_record is None:
            o_local_record = _exact_record(
                instance_id=traffic.instance_id,
                policy_id="O_local",
                meta={"heuristic_family": "oracle", "oracle_like": False},
                problem=problem,
                metadata=metadata,
                cost_model_id=cost_model_id,
                scope="local",
            )
        if o_joint_record is None:
            o_joint_record = _exact_record(
                instance_id=traffic.instance_id,
                policy_id="O_joint",
                meta={"heuristic_family": "oracle", "oracle_like": False},
                problem=problem,
                metadata=metadata,
                cost_model_id=cost_model_id,
                scope="joint",
            )
        for policy_id in policy_ids:
            meta = _policy_meta(policy_id)
            if policy_id in {"exact_small_instance_reference", "O_joint"}:
                record = _exact_record(
                    instance_id=traffic.instance_id,
                    policy_id=policy_id,
                    meta=meta,
                    problem=problem,
                    metadata=metadata,
                    cost_model_id=cost_model_id,
                    scope="joint",
                )
            elif policy_id == "O_local":
                record = _exact_record(
                    instance_id=traffic.instance_id,
                    policy_id=policy_id,
                    meta=meta,
                    problem=problem,
                    metadata=metadata,
                    cost_model_id=cost_model_id,
                    scope="local",
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
    same_core_pair_summary = _strict_pair_summary(records)
    family_pair_summaries = _family_pair_summaries(records)
    exact_oracle_comparable = bool(o_local_record and o_local_record.comparable and o_joint_record and o_joint_record.comparable)
    dominance_result = None
    if exact_oracle_comparable and o_local_record is not None and o_joint_record is not None:
        tolerance = 1e-9
        dominance_ok = float(o_joint_record.objective or 0.0) <= float(o_local_record.objective or 0.0) + tolerance
        dominance_result = {
            "status": "OK" if dominance_ok else "ORACLE_DOMINANCE_VIOLATION",
            "o_local_objective": o_local_record.objective,
            "o_joint_objective": o_joint_record.objective,
            "tolerance": tolerance,
        }
    status = "OK"
    if invalid_policy_seen or same_core_pair_summary["pair_status"] == "INVALID_PAIR":
        status = "PARTIAL_INVALID_POLICY"
    if any(row["status"] != "READY" for row in family_pair_summaries):
        status = "PARTIAL_INVALID_POLICY"
    if dominance_result is not None and dominance_result["status"] != "OK":
        status = "ORACLE_DOMINANCE_VIOLATION"
    return {
        "records": [record.to_dict() for record in records],
        "status": status,
        "exact_oracle_comparable_count": int(exact_oracle_comparable),
        "o_local_status": "READY_FOR_SUPPORTED_TINY" if o_local_record and o_local_record.comparable else "PARTIAL",
        "o_joint_status": "READY_FOR_SUPPORTED_TINY" if o_joint_record and o_joint_record.comparable else "PARTIAL",
        "o_local_record": None if o_local_record is None else o_local_record.to_dict(),
        "o_joint_record": None if o_joint_record is None else o_joint_record.to_dict(),
        "oracle_dominance": dominance_result,
        "execution_window_bridge_summary": execution_window_bridge_summary,
        "same_core_pair_summary": same_core_pair_summary,
        "family_pair_summaries": family_pair_summaries,
    }
