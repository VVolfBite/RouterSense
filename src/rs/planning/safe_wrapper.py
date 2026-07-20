from __future__ import annotations

"""Generic same-request Safe wrapper for paired Joint/Local planners.

The wrapper is deliberately independent of U/B naming and of any specific
algorithm family.  It runs a configured Joint planner and its paired Local
planner on the *same canonical PlanningRequest*, audits both returned plans,
and publishes the lower-cost valid plan under one formal WindowPlan contract.

This is an ahead-of-time selection component.  It may run in the existing
asynchronous target-planning service, but MUST NOT be invoked from the target
layer's visible bind/materialization path.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Protocol

from rs.core.contracts import PlanningRequest, WindowPlan
from rs.scheduling.safe_pair import (
    SafeCandidate, SafePairSelectionError, select_safe_pair,
)


class PlannerLike(Protocol):
    @property
    def planner_id(self) -> str: ...
    @property
    def planner_family(self) -> str: ...
    def plan(self, request: PlanningRequest) -> WindowPlan: ...


class SafeSelectionError(SafePairSelectionError):
    """Formal-planning compatibility name for SafePairSelectionError."""


@dataclass(frozen=True)
class SafePlannerConfig:
    runtime_usage: bool = True
    tie_break: str = "local"
    selection_metric: str = "audited_makespan"
    minimum_joint_gain: float = 0.0
    joint_visible_overhead: float = 0.0
    local_visible_overhead: float = 0.0
    pairing_key: str | None = None
    require_same_core: bool = True

    def validate(self) -> None:
        if self.tie_break not in {"local", "joint"}:
            raise ValueError("tie_break must be 'local' or 'joint'")
        if self.selection_metric not in {"audited_makespan", "net_cost"}:
            raise ValueError("unsupported safe selection metric")
        if self.pairing_key is not None and not str(self.pairing_key):
            raise ValueError("pairing_key must be non-empty when provided")
        for name, value in (
            ("minimum_joint_gain", self.minimum_joint_gain),
            ("joint_visible_overhead", self.joint_visible_overhead),
            ("local_visible_overhead", self.local_visible_overhead),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class _Candidate:
    role: str
    planner_id: str
    plan: WindowPlan | None
    valid: bool
    objective: float
    score: float
    planning_ms: float
    error: str | None = None


def _objective(plan: WindowPlan) -> float:
    metadata = dict(getattr(plan, "metadata", {}) or {})
    for key in (
        "audited_makespan",
        "kernel_makespan",
        "makespan",
        "objective_logical_makespan",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0.0:
            return number
    return float(sum(float(wave.estimated_duration) for wave in plan.waves))


def _family_allowed(role: str, family: str) -> bool:
    if role == "joint":
        return family == "joint"
    return family in {"local", "baseline"}


def _infer_core_key(planner_id: str) -> str | None:
    parts = str(planner_id).lower().split(":")
    if len(parts) == 5 and parts[0] in {"current", "future"}:
        core = parts[4]
        return core if core in {"gmwd", "rsbc", "rscf"} else None
    if str(planner_id) == "birkhoff_bucket_phase_local":
        return "birkhoff"
    return None


class SafePlannerWrapper:
    """Select the best valid plan from one paired Joint/Local planner.

    The wrapper is family-agnostic.  Pairing is explicit at construction time;
    callers are responsible for choosing planners with the same task partition,
    topology/cost profile, horizon, and information contract.
    """

    def __init__(
        self,
        *,
        joint_planner: PlannerLike,
        local_planner: PlannerLike,
        config: SafePlannerConfig | None = None,
        wrapper_id: str = "safe_pair",
    ) -> None:
        self.joint_planner = joint_planner
        self.local_planner = local_planner
        self.config = config or SafePlannerConfig()
        self.config.validate()
        self._wrapper_id = str(wrapper_id)
        if not self._wrapper_id:
            raise ValueError("wrapper_id must be non-empty")
        if str(joint_planner.planner_id) == str(local_planner.planner_id):
            raise ValueError("safe paired planners must be distinct")
        if not _family_allowed("joint", str(joint_planner.planner_family)):
            raise ValueError("joint_planner must expose planner_family='joint'")
        if not _family_allowed("local", str(local_planner.planner_family)):
            raise ValueError("local_planner must expose planner_family local/baseline")
        if self.config.runtime_usage:
            forbidden = {"reference_local", "reference_joint", "exact_local", "exact_joint"}
            if str(joint_planner.planner_family) in forbidden or str(local_planner.planner_family) in forbidden:
                raise ValueError("reference/exact planners cannot enter runtime Safe selection")
        joint_core = str(getattr(joint_planner, "safe_pairing_key", "") or "") or _infer_core_key(str(joint_planner.planner_id))
        local_core = str(getattr(local_planner, "safe_pairing_key", "") or "") or _infer_core_key(str(local_planner.planner_id))
        explicit = None if self.config.pairing_key is None else str(self.config.pairing_key)
        if self.config.require_same_core:
            if joint_core is not None and local_core is not None and joint_core != local_core:
                raise ValueError(f"Safe pair core mismatch: joint={joint_core!r}, local={local_core!r}")
            if explicit is None and (joint_core is None or local_core is None):
                raise ValueError("unknown planner IDs require an explicit pairing_key")
            if explicit is not None:
                for role, inferred in (("joint", joint_core), ("local", local_core)):
                    if inferred is not None and inferred != explicit:
                        raise ValueError(f"{role} planner core {inferred!r} != pairing_key {explicit!r}")
        self._pairing_key = explicit or joint_core or local_core or "unspecified"

    @property
    def planner_id(self) -> str:
        return self._wrapper_id

    @property
    def planner_family(self) -> str:
        # The returned plan truthfully carries the selected child's family.
        return "joint"

    def _run(self, *, role: str, planner: PlannerLike, request: PlanningRequest) -> _Candidate:
        start = time.perf_counter_ns()
        try:
            plan = planner.plan(request)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            plan.validate()
            expected_digest = request.semantic_digest()
            if str(plan.request_digest) != str(expected_digest):
                raise ValueError(
                    f"{role} plan request digest mismatch: {plan.request_digest} != {expected_digest}"
                )
            if not _family_allowed(role, str(plan.planner_family)):
                raise ValueError(f"{role} child returned incompatible planner_family={plan.planner_family!r}")
            objective = _objective(plan)
            overhead = (
                float(self.config.joint_visible_overhead)
                if role == "joint"
                else float(self.config.local_visible_overhead)
            )
            score = objective if self.config.selection_metric == "audited_makespan" else objective + overhead
            return _Candidate(role, str(planner.planner_id), plan, True, objective, score, elapsed)
        except Exception as exc:  # fail closed; the paired candidate may still be valid
            elapsed = (time.perf_counter_ns() - start) / 1e6
            return _Candidate(
                role=role,
                planner_id=str(getattr(planner, "planner_id", role)),
                plan=None,
                valid=False,
                objective=float("inf"),
                score=float("inf"),
                planning_ms=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _candidate_from_precomputed(
        self, *, role: str, planner_id: str, plan: WindowPlan | None,
        request: PlanningRequest, planning_ms: float = 0.0, error: str | None = None,
    ) -> _Candidate:
        if plan is None:
            return _Candidate(role, str(planner_id), None, False, float("inf"), float("inf"), float(planning_ms), error or "candidate missing")
        try:
            plan.validate()
            expected_digest = request.semantic_digest()
            if str(plan.request_digest) != str(expected_digest):
                raise ValueError(f"{role} plan request digest mismatch: {plan.request_digest} != {expected_digest}")
            if not _family_allowed(role, str(plan.planner_family)):
                raise ValueError(f"{role} child returned incompatible planner_family={plan.planner_family!r}")
            objective = _objective(plan)
            overhead = float(self.config.joint_visible_overhead if role == "joint" else self.config.local_visible_overhead)
            score = objective if self.config.selection_metric == "audited_makespan" else objective + overhead
            return _Candidate(role, str(planner_id), plan, True, objective, score, float(planning_ms), None)
        except Exception as exc:
            return _Candidate(role, str(planner_id), None, False, float("inf"), float("inf"), float(planning_ms), f"{type(exc).__name__}: {exc}")

    def _select_candidates(self, *, request: PlanningRequest, joint: _Candidate, local: _Candidate) -> WindowPlan:
        request_digest = request.semantic_digest()
        try:
            decision = select_safe_pair(
                joint=SafeCandidate(
                    role="joint", payload=joint if joint.valid else None, valid=joint.valid,
                    objective=joint.objective, score=joint.score, error=joint.error,
                ),
                local=SafeCandidate(
                    role="local", payload=local if local.valid else None, valid=local.valid,
                    objective=local.objective, score=local.score, error=local.error,
                ),
                tie_break=self.config.tie_break,
                minimum_joint_gain=self.config.minimum_joint_gain,
            )
        except SafePairSelectionError as exc:
            raise SafeSelectionError(str(exc)) from exc
        selected = decision.selected.payload
        reason = decision.reason
        assert selected is not None and selected.plan is not None
        metadata = dict(selected.plan.metadata)
        metadata.update(
            {
                "safe_wrapper_semantic_version": "safe_pair_v2",
                "safe_wrapper_id": self._wrapper_id,
                "safe_selection_location": "ahead_of_time_only",
                "same_request_object": True,
                "same_request_digest": request_digest,
                "joint_planner_id": joint.planner_id,
                "local_planner_id": local.planner_id,
                "joint_valid": bool(joint.valid),
                "local_valid": bool(local.valid),
                "joint_error": joint.error,
                "local_error": local.error,
                "joint_objective": None if not joint.valid else float(joint.objective),
                "local_objective": None if not local.valid else float(local.objective),
                "joint_score": None if not joint.valid else float(joint.score),
                "local_score": None if not local.valid else float(local.score),
                "joint_planning_ms": float(joint.planning_ms),
                "local_planning_ms": float(local.planning_ms),
                "safe_total_planning_ms": float(joint.planning_ms + local.planning_ms),
                "selected_role": selected.role,
                "selected_child_planner_id": selected.planner_id,
                "selection_reason": reason,
                "tie_break": self.config.tie_break,
                "selection_metric": self.config.selection_metric,
                "minimum_joint_gain": float(self.config.minimum_joint_gain),
                "safe_pairing_key": self._pairing_key,
                "fallback_to_local": selected.role == "local",
                "selected_policy": selected.planner_id,
            }
        )
        result = WindowPlan(
            planner_id=self._wrapper_id,
            planner_family=str(selected.plan.planner_family),
            request_digest=request_digest,
            waves=selected.plan.waves,
            metadata=metadata,
        )
        result.validate()
        return result

    def select_precomputed(
        self, *, request: PlanningRequest, joint_plan: WindowPlan | None, local_plan: WindowPlan | None,
        joint_planning_ms: float = 0.0, local_planning_ms: float = 0.0,
        joint_error: str | None = None, local_error: str | None = None,
    ) -> WindowPlan:
        """Select already-prepared candidates without re-running either planner."""
        request.validate()
        joint = self._candidate_from_precomputed(
            role="joint", planner_id=self.joint_planner.planner_id, plan=joint_plan,
            request=request, planning_ms=joint_planning_ms, error=joint_error,
        )
        local = self._candidate_from_precomputed(
            role="local", planner_id=self.local_planner.planner_id, plan=local_plan,
            request=request, planning_ms=local_planning_ms, error=local_error,
        )
        return self._select_candidates(request=request, joint=joint, local=local)

    def plan(self, request: PlanningRequest) -> WindowPlan:
        request.validate()
        joint = self._run(role="joint", planner=self.joint_planner, request=request)
        local = self._run(role="local", planner=self.local_planner, request=request)
        return self._select_candidates(request=request, joint=joint, local=local)


__all__ = [
    "SafePlannerConfig",
    "SafePlannerWrapper",
    "SafeSelectionError",
]
