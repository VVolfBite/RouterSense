"""Guarded safe joint wrappers for paired RouterSense U-family policies.

Safe U does not change the underlying heuristic family. It compares the raw
joint U candidate against its paired phase-local B reference under the same
information set, then picks the non-regressing plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, MultiPhaseSchedulingProblem
from rs.scheduling.multiphase.replay import replay_and_audit_schedule
from rs.scheduling.phase_local.common import include_real_p2_phase
from rs.scheduling.validation import validate_logical_plan


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
                flows.append(
                    FlowDemand(
                        flow_id=f"p2_next_dispatch:{src_rank}->{dst_rank}",
                        phase="p2_next_dispatch",
                        src_rank=int(src_rank),
                        dst_rank=int(dst_rank),
                        byte_count=int(byte_count),
                        release_state="ready",
                        is_executable=True,
                    )
                )
    return tuple(flows)


def _evaluate_plan(problem: MultiPhaseSchedulingProblem, plan: LogicalSchedulePlan) -> _EvaluatedPlan:
    validation = validate_logical_plan(
        plan,
        expected_flows=_expected_flows(problem),
        mode=problem.options.scheduling_mode,
        expert_compute_delay=float(problem.release_model.expert_compute_delay),
    )
    validation_errors = tuple(str(item) for item in validation.get("errors", ()) or validation.get("validation_errors", ()) or ())
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
            plan=plan,
            valid=valid,
            makespan=float(audit.get("makespan", float("inf"))),
            validation_errors=validation_errors + tuple(str(item) for item in audit.get("validation_errors", ()) or ()),
        )
    makespan = float(plan.diagnostics.get("makespan", sum(float(wave.duration) for wave in plan.waves)))
    return _EvaluatedPlan(
        plan=plan,
        valid=bool(validation.get("valid", False)),
        makespan=makespan if bool(validation.get("valid", False)) else float("inf"),
        validation_errors=validation_errors,
    )


class SafeJointPolicy:
    policy_version = "v1"
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
        self,
        *,
        policy_name: str,
        raw_u_policy_name: str,
        paired_b_policy_name: str,
        raw_u_policy: Any,
        paired_b_policy: Any,
    ) -> None:
        self.policy_name = policy_name
        self.raw_u_policy_name = raw_u_policy_name
        self.paired_b_policy_name = paired_b_policy_name
        self._raw_u_policy = raw_u_policy
        self._paired_b_policy = paired_b_policy

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        raw_eval, b_eval, selected, fallback, reason = self.evaluate_components(problem)
        diagnostics = self._build_safe_diagnostics(
            problem=problem,
            raw_eval=raw_eval,
            b_eval=b_eval,
            selected=selected,
            fallback=fallback,
            reason=reason,
        )
        return LogicalSchedulePlan(
            policy_name=self.policy_name,
            waves=selected.plan.waves,
            diagnostics=diagnostics,
        )

    def evaluate_components(
        self,
        problem: MultiPhaseSchedulingProblem,
    ) -> tuple[_EvaluatedPlan, _EvaluatedPlan, _EvaluatedPlan, bool, str]:
        raw_eval = _evaluate_plan(problem, self._raw_u_policy.build_logical_plan(problem))
        b_eval = _evaluate_plan(problem, self._paired_b_policy.build_logical_plan(problem))
        selected = raw_eval
        fallback = False
        reason = "selected_raw_u"
        if (not raw_eval.valid) or raw_eval.makespan > b_eval.makespan:
            selected = b_eval
            fallback = True
            reason = "fallback_to_paired_b" if b_eval.valid else "raw_u_invalid_and_b_invalid"
        return raw_eval, b_eval, selected, fallback, reason

    def _build_safe_diagnostics(
        self,
        *,
        problem: MultiPhaseSchedulingProblem,
        raw_eval: _EvaluatedPlan,
        b_eval: _EvaluatedPlan,
        selected: _EvaluatedPlan,
        fallback: bool,
        reason: str,
    ) -> dict[str, Any]:
        diagnostics = dict(selected.plan.diagnostics)
        diagnostics.update(
            {
                "raw_u_policy": self.raw_u_policy_name,
                "paired_b_policy": self.paired_b_policy_name,
                "selected_policy": selected.plan.policy_name,
                "fallback_to_paired_b": fallback,
                "raw_u_makespan": float(raw_eval.makespan),
                "paired_b_makespan": float(b_eval.makespan),
                "safe_makespan": float(selected.makespan),
                "selection_reason": reason,
                "same_information_guard": True,
                "future_information_mode": str(selected.plan.diagnostics.get("future_information_mode", "")),
                "p2_source": str(problem.forecast.source) if problem.forecast is not None else "",
                "oracle_input_used": bool(problem.forecast.oracle) if problem.forecast is not None else False,
                "raw_u_valid": bool(raw_eval.valid),
                "paired_b_valid": bool(b_eval.valid),
                "safe_policy": self.policy_name,
            }
        )
        return diagnostics


__all__ = ["SafeJointPolicy"]
