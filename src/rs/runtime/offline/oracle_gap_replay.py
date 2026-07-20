#!/usr/bin/env python3
"""Unified tiny-instance O_local/O_joint gap and heuristic-gap replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from rs.core.contracts import PlanningConstraints, PlanningIdentity, PlanningTopology, PlanningWeights
from rs.planning import PlannerRegistry
from rs.planning.request_builder import build_window_planning_request
from rs.runtime.offline.exact_oracle_suite import ExactInstance, build_exact_problem, solve_exact_instance
from rs.runtime.offline.replay_unified import PlanningHint, ReplayEngine, ReplayWindow
from rs.runtime.online.megatron_ep.target_planning.contracts import _compat_logical_plan_from_window_plan
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.scheduling import resolve_policy


def _small_instance() -> ExactInstance:
    # Frozen release-sensitive witness also used by the formal oracle controls.
    return ExactInstance(
        instance_id="oracle_joint_advantage_v1",
        rank_count=3,
        p0=((0, 1, 1), (0, 0, 0), (0, 0, 0)),
        p1=((0, 0, 0), (1, 0, 0), (1, 0, 0)),
        p2=((0, 1, 1), (0, 0, 0), (0, 0, 0)),
        sparsity_regime="sparse",
        skew_regime="hotspot",
        correlation_regime="high",
        p2_strength_regime="medium",
        seed=0,
    )


def _run_policy(problem, policy_name: str, instance: ExactInstance) -> dict[str, Any]:
    """Evaluate one planner through the canonical offline truth-binding path."""
    replay_window = ReplayWindow(
        fixture_id=str(instance.instance_id),
        window_id=f"{instance.instance_id}:0",
        layer_id=0,
        p0_truth_rows=instance.p0,
        p1_truth_rows=instance.p1,
        p2_truth_rows=instance.p2,
        matrix_unit="rows",
        group_size=int(instance.rank_count),
        payload_row_bytes_by_phase={"P0": 1, "P1": 1, "P2": 1},
        metadata={"source": "oracle_gap_replay"},
    )
    result = ReplayEngine(
        scheduling_mode="execution_window",
        expert_compute_delay=0.0,
        bucket_rows=1,
        max_waves=256,
    ).execute(
        replay_window=replay_window,
        planning_hint=PlanningHint(
            hint_type="perfect_trace_hint",
            p2_hint_rows=instance.p2,
            confidence=1.0,
            source_layer=0,
            target_layer=1,
        ),
        policy_name=str(policy_name),
    )
    audit = dict(result.get("audit", {}) or {})
    return {
        "policy_name": policy_name,
        "makespan": float(result.get("makespan", 0.0)),
        "valid": bool(result.get("audit_valid", False)),
        "validation_errors": list(audit.get("validation_errors", ()) or ()),
    }


def _relative_gap(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or float(reference) == 0.0:
        return None
    return (float(value) - float(reference)) / float(reference)


def run_oracle_gap_replay(
    *,
    fixture_dir: Path,
    small_only: bool = False,
    policies: Iterable[str] | None = None,
) -> dict[str, Any]:
    instance = _small_instance()
    problem = build_exact_problem(instance)
    local = solve_exact_instance(instance, scope="local")
    joint = solve_exact_instance(instance, scope="joint")
    o_local = None if local.get("objective") is None else float(local["objective"])
    o_joint = None if joint.get("objective") is None else float(joint["objective"])

    selected_policies = list(
        policies
        or [
            "fifo_bucket",
            "birkhoff_bucket_phase_local",
            "current:p012:local:global:rscf",
            "current:p012:joint:global:rscf",
        ]
    )
    heuristic_rows = [_run_policy(problem, policy_name, instance) for policy_name in selected_policies]
    for row in heuristic_rows:
        row["gap_to_O_local"] = _relative_gap(float(row["makespan"]), o_local)
        row["gap_to_O_joint"] = _relative_gap(float(row["makespan"]), o_joint)

    exact_comparable = (
        local.get("solver_status") == "OPTIMAL"
        and joint.get("solver_status") == "OPTIMAL"
        and o_local is not None
        and o_joint is not None
    )
    dominance_violation = bool(exact_comparable and float(o_joint) > float(o_local) + 1.0e-9)
    payload = {
        "schema_version": "routersense_unified_oracle_gap_replay.v2",
        "reference_model": local.get("reference_model"),
        "task_model_id": local.get("task_model_id"),
        "cost_model_id": local.get("cost_model_id"),
        "release_model_id": local.get("release_model_id"),
        "scope_comparison_contract": {
            "same_tasks": local.get("task_model_id") == joint.get("task_model_id"),
            "same_cost_model": local.get("cost_model_id") == joint.get("cost_model_id"),
            "same_release_model": local.get("release_model_id") == joint.get("release_model_id"),
            "only_scope_changes": True,
            "local_scope": "phase_barrier",
            "joint_scope": "rank_local_release",
        },
        "O_local_definition": "exact_runtime_bucket_wave_scope=local",
        "O_joint_definition": "exact_runtime_bucket_wave_scope=joint",
        "legacy_atomic_cp_sat_status": "historical_sensitivity_only_not_formal_oracle",
        "O_joint_small_fixture_available": bool(exact_comparable),
        "O_joint_real_fixture_available": False,
        "real_fixture_joint_proxy": None,
        "selected_policies": selected_policies,
        "exact_rows": [
            {"policy_name": "O_local", **local},
            {"policy_name": "O_joint", **joint},
        ],
        "small_fixture_rows": heuristic_rows,
        "oracle_gap_small_fixture_summary": {
            "O_local": o_local,
            "O_joint": o_joint,
            "O_joint_vs_O_local_gap": _relative_gap(o_joint, o_local),
            "O_joint_improvement_vs_O_local": None
            if o_local is None or o_joint is None or o_local == 0.0
            else (o_local - o_joint) / o_local,
            "dominance_violation": dominance_violation,
        },
    }
    if not small_only:
        fixture_paths = sorted(fixture_dir.glob("replay_layer_*.json"))
        payload["real_fixture_joint_proxy_status"] = "available" if fixture_paths else "missing_fixture_dir"
    return payload


__all__ = ["run_oracle_gap_replay"]
