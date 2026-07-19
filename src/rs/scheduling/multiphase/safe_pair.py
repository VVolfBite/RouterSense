"""Generic Safe Joint/Local wrapper with compatibility aliases.

The historical ``SafeJointPolicy`` / ``Safe-U`` names are deprecated.  The
canonical implementation is :class:`SafePlannerWrapper`, which can wrap any
same-core Joint/Local pair.  This algorithm-layer wrapper is intended for
offline audits and ahead-of-time selection; it is not a target-bind fast path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, MultiPhaseSchedulingProblem
from rs.scheduling.multiphase.replay import replay_and_audit_schedule
from rs.scheduling.phase_local.common import include_real_p2_phase
from rs.scheduling.validation import validate_logical_plan
from rs.scheduling.safe_pair import SafeCandidate, SafePairSelectionError, select_safe_pair


@dataclass(frozen=True)
class _EvaluatedPlan:
    plan: LogicalSchedulePlan
    valid: bool
    makespan: float
    validation_errors: tuple[str, ...]


def _expected_flows(problem: MultiPhaseSchedulingProblem) -> tuple[FlowDemand, ...]:
    flows: list[FlowDemand] = []
    flows.extend(problem.flow_window.ready_flows)
    flows.extend(problem.flow_window.blocked_flows)
    if include_real_p2_phase(problem):
        for src_rank, row in enumerate(problem.p2_next_dispatch_forecast_matrix):
            for dst_rank, byte_count in enumerate(row):
                if src_rank == dst_rank or int(byte_count) <= 0:
                    continue
                flows.append(FlowDemand(
                    flow_id=f"p2_next_dispatch:{src_rank}->{dst_rank}",
                    phase="p2_next_dispatch", src_rank=int(src_rank), dst_rank=int(dst_rank),
                    byte_count=int(byte_count), release_state="ready", is_executable=True,
                ))
    return tuple(flows)


def _evaluate_plan(problem: MultiPhaseSchedulingProblem, plan: LogicalSchedulePlan) -> _EvaluatedPlan:
    validation = validate_logical_plan(
        plan, expected_flows=_expected_flows(problem), mode=problem.options.scheduling_mode,
        expert_compute_delay=float(problem.release_model.expert_compute_delay),
    )
    validation_errors = tuple(str(item) for item in validation.get("errors", ()) or ())
    raw_schedule = list(plan.diagnostics.get("raw_schedule", ()))
    if raw_schedule:
        audit = replay_and_audit_schedule(
            schedule=raw_schedule,
            dispatch_matrix=[list(row) for row in problem.p0_dispatch_matrix],
            combine_matrix=[list(row) for row in problem.p1_return_matrix],
            next_dispatch_matrix=[list(row) for row in problem.p2_next_dispatch_forecast_matrix],
            num_gpus=int(problem.topology.num_gpus),
            expert_compute_delay=float(problem.release_model.expert_compute_delay),
            mode=problem.options.scheduling_mode,
            scheduler_name=plan.policy_name,
            planning_time_ms=float(plan.diagnostics.get("solve_time_ms", 0.0)),
            reported_makespan=float(plan.diagnostics.get("makespan", 0.0)),
            prediction_used=bool(plan.diagnostics.get("prediction_used", False)),
        )
        valid = bool(validation.get("valid", False)) and bool(audit.get("valid", False))
        return _EvaluatedPlan(
            plan=plan, valid=valid,
            makespan=float(audit.get("makespan", float("inf"))) if valid else float("inf"),
            validation_errors=validation_errors + tuple(str(item) for item in audit.get("validation_errors", ()) or ()),
        )
    makespan = float(plan.diagnostics.get("makespan", sum(float(w.duration) for w in plan.waves)))
    valid = bool(validation.get("valid", False))
    return _EvaluatedPlan(plan, valid, makespan if valid else float("inf"), validation_errors)


class SafePlannerWrapper:
    policy_version = "v2"
    capabilities = PolicyCapabilities(
        supports_offline=True,
        supports_online_phase_local_execution=False,
        supports_online_multiphase_execution=False,
        uses_current_ready_flows=True,
        uses_blocked_p1_dependency=True,
        uses_p2_forecast=True,
        requires_fixed_placement=True,
        evaluation_eligible=True,
    )

    def __init__(
        self, *, policy_name: str, joint_policy_name: str, local_policy_name: str,
        joint_policy: Any, local_policy: Any, tie_break: str = "local",
    ) -> None:
        if tie_break not in {"local", "joint"}:
            raise ValueError("tie_break must be local or joint")
        self.policy_name = str(policy_name)
        self.joint_policy_name = str(joint_policy_name)
        self.local_policy_name = str(local_policy_name)
        self._joint_policy = joint_policy
        self._local_policy = local_policy
        self.tie_break = tie_break

    def selection_diagnostics(
        self, *, joint_eval: _EvaluatedPlan, local_eval: _EvaluatedPlan,
        selected: _EvaluatedPlan, reason: str,
    ) -> dict[str, Any]:
        """Build the canonical Safe(Joint, Local) decision diagnostics."""
        diagnostics = dict(selected.plan.diagnostics)
        diagnostics.update({
            "safe_wrapper_semantic_version": "safe_pair_v2",
            "joint_policy": self.joint_policy_name,
            "local_policy": self.local_policy_name,
            "selected_policy": selected.plan.policy_name,
            "selected_role": "joint" if selected is joint_eval else "local",
            "joint_makespan": float(joint_eval.makespan),
            "local_makespan": float(local_eval.makespan),
            "safe_makespan": float(selected.makespan),
            "joint_valid": bool(joint_eval.valid),
            "local_valid": bool(local_eval.valid),
            "selection_reason": str(reason),
            "same_information_guard": True,
            "safe_policy": self.policy_name,
            # Compatibility aliases only.
            "raw_u_policy": self.joint_policy_name,
            "paired_b_policy": self.local_policy_name,
            "raw_u_makespan": float(joint_eval.makespan),
            "paired_b_makespan": float(local_eval.makespan),
            "fallback_to_paired_b": selected is local_eval,
        })
        return diagnostics

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        joint_eval, local_eval, selected, reason = self.evaluate_components(problem)
        diagnostics = self.selection_diagnostics(
            joint_eval=joint_eval, local_eval=local_eval, selected=selected, reason=reason,
        )
        return LogicalSchedulePlan(policy_name=self.policy_name, waves=selected.plan.waves, diagnostics=diagnostics)

    def evaluate_components(
        self, problem: MultiPhaseSchedulingProblem, *,
        joint_plan: LogicalSchedulePlan | None = None,
        local_plan: LogicalSchedulePlan | None = None,
    ) -> tuple[_EvaluatedPlan, _EvaluatedPlan, _EvaluatedPlan, str]:
        joint_eval = _evaluate_plan(problem, joint_plan if joint_plan is not None else self._joint_policy.build_logical_plan(problem))
        local_eval = _evaluate_plan(problem, local_plan if local_plan is not None else self._local_policy.build_logical_plan(problem))
        try:
            decision = select_safe_pair(
                joint=SafeCandidate(
                    role="joint", payload=joint_eval if joint_eval.valid else None, valid=joint_eval.valid,
                    objective=joint_eval.makespan, score=joint_eval.makespan,
                    error="; ".join(joint_eval.validation_errors) or None,
                ),
                local=SafeCandidate(
                    role="local", payload=local_eval if local_eval.valid else None, valid=local_eval.valid,
                    objective=local_eval.makespan, score=local_eval.makespan,
                    error="; ".join(local_eval.validation_errors) or None,
                ),
                tie_break=self.tie_break, minimum_joint_gain=0.0,
            )
        except SafePairSelectionError as exc:
            raise ValueError(str(exc)) from exc
        assert decision.selected.payload is not None
        return joint_eval, local_eval, decision.selected.payload, decision.reason


class SafeJointPolicy(SafePlannerWrapper):
    """Deprecated source-compatible alias for older registry/configs."""
    def __init__(
        self, *, policy_name: str, raw_u_policy_name: str, paired_b_policy_name: str,
        raw_u_policy: Any, paired_b_policy: Any,
    ) -> None:
        warnings.warn(
            "SafeJointPolicy/Safe-U is deprecated; use SafePlannerWrapper(Joint, Local)",
            DeprecationWarning, stacklevel=2,
        )
        super().__init__(
            policy_name=policy_name,
            joint_policy_name=raw_u_policy_name,
            local_policy_name=paired_b_policy_name,
            joint_policy=raw_u_policy,
            local_policy=paired_b_policy,
            tie_break="local",
        )


__all__ = ["SafePlannerWrapper", "SafeJointPolicy"]
