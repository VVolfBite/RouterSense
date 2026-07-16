from __future__ import annotations

from pathlib import Path
from typing import Any

from rs.scheduling.algorithm_catalog import get_algorithm_metadata
from rs.scheduling.catalog import resolve_algorithm_id

from .adapters.scheduling_adapter import execute_policy, replay_window_from_matrices
from .contracts import RecordMetadata, ScheduleEvaluationRecord
from .trace_dataset import discover_replay_fixtures, load_replay_fixture, replay_fixture_to_trace_sample
from .traffic_builder import build_traffic_instance


def evaluate_scheduling(
    *,
    fixture_dir: Path,
    metadata: RecordMetadata,
    model_id: str,
    model_revision: str,
    policy_ids: tuple[str, ...],
) -> dict[str, Any]:
    records: list[ScheduleEvaluationRecord] = []
    dominance_violations: list[str] = []
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
        )
        replay_window = replay_window_from_matrices(
            fixture_id=trace_sample.trace_sample_id,
            layer_id=int(fixture["metadata"].get("layer_id", 0) or 0),
            p0_matrix=traffic.P0_matrix,
            p1_matrix=traffic.P1_matrix,
            p2_matrix=traffic.P2_truth_matrix,
        )
        results_by_policy: dict[str, dict[str, Any]] = {}
        for policy_id in policy_ids:
            hint_rows = traffic.P2_truth_matrix
            result = execute_policy(
                replay_window=replay_window,
                policy_name=policy_id,
                hint_type="perfect_trace_hint",
                p2_hint_rows=hint_rows,
                confidence=1.0,
                bucket_rows=1,
            )
            results_by_policy[policy_id] = result
            meta = _policy_meta(policy_id)
            planner_family = str(meta["role"])
            scope = "joint" if "joint" in planner_family or policy_id.startswith("U_") or policy_id.startswith("RS_safe") or policy_id.startswith("exact_") else "local"
            objective = float(result["makespan"]) if result.get("audit_valid") else None
            records.append(
                ScheduleEvaluationRecord(
                    instance_id=traffic.instance_id,
                    policy_id=policy_id,
                    policy_family=str(meta["heuristic_family"]),
                    scope=scope,
                    is_exact=bool(meta["oracle_like"] or policy_id in {"birkhoff_von_neumann_fluid", "exact_small_instance_reference"}),
                    solver_status="VALID" if result.get("audit_valid") else "INVALID",
                    solver_runtime_ms=None,
                    planning_runtime_ms=float(result["planning_runtime_ms_wall"]),
                    objective=objective,
                    best_bound=objective,
                    optimality_gap=0.0 if objective is not None else None,
                    coverage_valid=bool(result.get("audit_valid")),
                    plan_digest=str(result["logical_plan_digest"]),
                    fallback_count=0,
                    comparable=True,
                    comparable_reason="same_fixture_same_truth",
                    metadata=metadata,
                )
            )
        o_local = results_by_policy.get("birkhoff_von_neumann_fluid", {}).get("makespan")
        o_joint = results_by_policy.get("exact_small_instance_reference", {}).get("makespan")
        if o_local is not None and o_joint is not None and float(o_joint) > float(o_local) + 1e-9:
            dominance_violations.append(traffic.instance_id)
    return {
        "records": [record.to_dict() for record in records],
        "oracle_dominance_ok": not dominance_violations,
        "oracle_dominance_violations": dominance_violations,
        "status": "OK" if not dominance_violations else "ORACLE_DOMINANCE_VIOLATION",
    }
