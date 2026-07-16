"""Controlled Local(f) versus Joint(f) scheduling-family evaluation."""

from __future__ import annotations

import statistics
from typing import Any, Iterable

from rs.runtime.offline.replay_unified import ReplayWindow
from rs.scheduling.families import STRICT_FAMILY_IDS, canonical_family_policy_id, get_family_kernel_spec
from rs.scheduling.families.core import FamilyScope

from .adapters.scheduling_adapter import execute_policy


_CONTRACT_FIELDS = (
    "matching_core_id",
    "task_contract_digest",
    "bucket_contract_digest",
    "cost_contract_digest",
    "service_model_id",
    "solver_budget_digest",
    "kernel_parameters",
)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _measure_policy(
    *,
    replay_window: ReplayWindow,
    policy_id: str,
    repeats: int,
    warmups: int,
    expert_compute_delay: float,
    bucket_rows: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for run_index in range(max(0, int(warmups)) + max(1, int(repeats))):
        result = execute_policy(
            replay_window=replay_window,
            policy_name=policy_id,
            hint_type="perfect_trace_hint",
            p2_hint_rows=replay_window.p2_truth_rows,
            confidence=1.0,
            expert_compute_delay=float(expert_compute_delay),
            bucket_rows=int(bucket_rows),
        )
        if run_index >= max(0, int(warmups)):
            results.append(result)
    runtimes = [float(row.get("planning_runtime_ms_wall", 0.0) or 0.0) for row in results]
    objectives = [float(row.get("makespan", 0.0) or 0.0) for row in results]
    digests = sorted({str(row.get("logical_plan_digest")) for row in results})
    first = results[0]
    plan_metadata = dict(first.get("plan_metadata", {}) or {})
    audit = dict(first.get("audit", {}) or {})
    return {
        "policy_id": policy_id,
        "valid": all(bool(row.get("audit_valid", False)) for row in results),
        "objective": statistics.median(objectives),
        "objective_samples": objectives,
        "planning_runtime_ms": {
            "samples": runtimes,
            "median": statistics.median(runtimes),
            "p95": _percentile(runtimes, 0.95),
            "max": max(runtimes),
        },
        "plan_digests": digests,
        "deterministic_plan": len(digests) == 1,
        "family_id": plan_metadata.get("family_id"),
        "family_scope": plan_metadata.get("family_scope"),
        "common_core": dict(plan_metadata.get("common_core", {}) or {}),
        "kernel_call_count": plan_metadata.get("kernel_call_count"),
        "phase_kernel_runtime_ms": plan_metadata.get("phase_kernel_runtime_ms"),
        "wave_count": audit.get("wave_count"),
        "served_volume_by_phase": audit.get("served_volume_by_phase"),
        "validation_errors": list(audit.get("validation_errors", ()) or ()),
    }


def evaluate_family_pairs(
    *,
    replay_window: ReplayWindow,
    family_ids: Iterable[str] = STRICT_FAMILY_IDS,
    repeats: int = 5,
    warmups: int = 1,
    expert_compute_delay: float = 0.0,
    bucket_rows: int = 1,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for family_id in tuple(str(item) for item in family_ids):
        local_id = canonical_family_policy_id(family_id, FamilyScope.LOCAL)
        joint_id = canonical_family_policy_id(family_id, FamilyScope.JOINT)
        local = _measure_policy(
            replay_window=replay_window,
            policy_id=local_id,
            repeats=repeats,
            warmups=warmups,
            expert_compute_delay=expert_compute_delay,
            bucket_rows=bucket_rows,
        )
        joint = _measure_policy(
            replay_window=replay_window,
            policy_id=joint_id,
            repeats=repeats,
            warmups=warmups,
            expert_compute_delay=expert_compute_delay,
            bucket_rows=bucket_rows,
        )
        local_core = dict(local.get("common_core", {}) or {})
        joint_core = dict(joint.get("common_core", {}) or {})
        contract_equal = all(local_core.get(key) == joint_core.get(key) for key in _CONTRACT_FIELDS)
        local_objective = float(local["objective"])
        joint_objective = float(joint["objective"])
        delta = joint_objective - local_objective
        improvement = None if local_objective <= 0.0 else (local_objective - joint_objective) / local_objective * 100.0
        local_runtime = float(local["planning_runtime_ms"]["median"])
        joint_runtime = float(joint["planning_runtime_ms"]["median"])
        spec = get_family_kernel_spec(family_id)
        records.append(
            {
                "family_id": family_id,
                "display_name": spec.display_name,
                "paper_label": spec.literature.paper_label,
                "literature_mapping_level": spec.literature.mapping_level,
                "literature_citation_key": spec.literature.citation_key,
                "primary_for_paper": bool(spec.primary_for_paper),
                "status": "READY" if local["valid"] and joint["valid"] and contract_equal else "INVALID",
                "contract_equal": contract_equal,
                "contract_mismatches": [
                    key for key in _CONTRACT_FIELDS if local_core.get(key) != joint_core.get(key)
                ],
                "local": local,
                "joint": joint,
                "effect": {
                    "joint_minus_local_makespan": delta,
                    "joint_improvement_pct": improvement,
                    "outcome": "win" if delta < -1e-9 else "loss" if delta > 1e-9 else "tie",
                },
                "overhead": {
                    "joint_minus_local_runtime_ms": joint_runtime - local_runtime,
                    "joint_over_local_runtime_ratio": None if local_runtime <= 0.0 else joint_runtime / local_runtime,
                },
            }
        )
    return {
        "schema_version": "family_pair_evaluation.v1",
        "fixture_id": replay_window.fixture_id,
        "window_id": replay_window.window_id,
        "repeats": int(repeats),
        "warmups": int(warmups),
        "records": records,
        "status": "READY" if records and all(row["status"] == "READY" for row in records) else "PARTIAL",
    }


__all__ = ["evaluate_family_pairs"]
