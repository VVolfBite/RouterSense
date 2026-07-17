"""Scope adapters for strict same-core scheduling-family comparisons."""

from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Any

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import LogicalSchedulePlan, MultiPhaseSchedulingProblem
from rs.scheduling.multiphase.flow_model import EXECUTION_WINDOW_MODE
from rs.scheduling.multiphase.replay import replay_and_audit_schedule
from rs.scheduling.multiphase.scheduler_state import run_global_matching_scheduler
from rs.scheduling.multiphase.tier1 import (
    _base_score_lookup_from_phase_orders,
    _birkhoff_phase_orders,
    _offset_schedule,
    _raw_schedule_to_plan,
)
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix

from .core import FamilyKernelSpec, FamilyScope, get_family_kernel_spec, normalize_family_id


_EXPRESSION = re.compile(r"^(Local|Joint)\(([A-Za-z0-9_\-]+)\)$", re.IGNORECASE)

# Old names remain accepted, but resolve to literature-grounded canonical
# families rather than independent ad-hoc B/U implementations.
_LEGACY_NAMES: dict[str, tuple[str, FamilyScope]] = {
    "B_gated_greedy_maximal": ("greedy_control", FamilyScope.LOCAL),
    "U_gated_greedy_maximal": ("greedy_control", FamilyScope.JOINT),
    "B_gated_maxweight_matching": ("gmwd", FamilyScope.LOCAL),
    "U_gated_maxweight_matching": ("gmwd", FamilyScope.JOINT),
    "B_barrier_criticality_core_independent": ("rsbc", FamilyScope.LOCAL),
    "U_barrier_criticality_global_matching": ("rsbc", FamilyScope.JOINT),
    "B_barrier_price_adaptive_matching": ("adaptive_price", FamilyScope.LOCAL),
    "U_barrier_price_adaptive_matching": ("adaptive_price", FamilyScope.JOINT),
    # Canonical IDs from the first scope-layer prototype.
    "gated_greedy_local": ("greedy_control", FamilyScope.LOCAL),
    "gated_greedy_joint": ("greedy_control", FamilyScope.JOINT),
    "gated_maxweight_local": ("gmwd", FamilyScope.LOCAL),
    "gated_maxweight_joint": ("gmwd", FamilyScope.JOINT),
    "barrier_criticality_core_independent": ("rsbc", FamilyScope.LOCAL),
    "barrier_criticality_joint": ("rsbc", FamilyScope.JOINT),
    "birkhoff_ranked_local": ("fast_stage", FamilyScope.LOCAL),
    "birkhoff_ranked_joint": ("fast_stage", FamilyScope.JOINT),
}

_CANONICAL_NAMES: dict[str, tuple[str, FamilyScope]] = {
    f"{family_id}_{scope.value}": (family_id, scope)
    for family_id in (
        "greedy_control",
        "gmwd",
        "rsbc",
        "fast_stage",
        "aurora_order",
        "adaptive_price",
        "rscf",
    )
    for scope in (FamilyScope.LOCAL, FamilyScope.JOINT)
}


def parse_scoped_family_policy(policy_name: str) -> tuple[str, FamilyScope] | None:
    normalized = str(policy_name)
    if normalized in _CANONICAL_NAMES:
        return _CANONICAL_NAMES[normalized]
    if normalized in _LEGACY_NAMES:
        return _LEGACY_NAMES[normalized]
    match = _EXPRESSION.fullmatch(normalized)
    if match is None:
        return None
    scope = FamilyScope.LOCAL if match.group(1).lower() == "local" else FamilyScope.JOINT
    family_id = normalize_family_id(match.group(2))
    get_family_kernel_spec(family_id)
    return family_id, scope


def is_scoped_family_policy(policy_name: str) -> bool:
    return parse_scoped_family_policy(policy_name) is not None


def canonical_family_policy_id(family_id: str, scope: FamilyScope) -> str:
    normalized = normalize_family_id(family_id)
    get_family_kernel_spec(normalized)
    return f"{normalized}_{scope.value}"


class ScopedFamilyPolicy:
    """Apply one immutable algorithm kernel under Local or Joint visibility.

    Local and Joint receive the same kernel specification.  The local adapter
    invokes it once per executable phase and serializes the resulting plans;
    the joint adapter invokes it once on the global release-aware ready set.
    """

    policy_version = "v1"

    def __init__(
        self,
        *,
        family_id: str,
        scope: FamilyScope,
        reported_policy_name: str | None = None,
        residual_weight: float | None = None,
        barrier_weight: float | None = None,
        age_weight: float | None = None,
        prediction_weight: float | None = None,
        endpoint_pressure_weight: float | None = None,
        release_gain_weight: float | None = None,
    ) -> None:
        self.family_id = str(family_id)
        self.scope = FamilyScope(scope)
        self.spec = get_family_kernel_spec(self.family_id).with_weight_overrides(
            residual_weight=residual_weight,
            barrier_weight=barrier_weight,
            age_weight=age_weight,
            prediction_weight=prediction_weight,
            endpoint_pressure_weight=endpoint_pressure_weight,
            release_gain_weight=release_gain_weight,
        )
        self.policy_name = str(reported_policy_name or canonical_family_policy_id(self.family_id, self.scope))
        self.algorithm_id = self.policy_name
        self.collect_debug_trace = False
        self.capabilities = PolicyCapabilities(
            supports_offline=True,
            supports_online_phase_local_execution=False,
            supports_online_multiphase_execution=True,
            uses_current_ready_flows=True,
            uses_blocked_p1_dependency=self.scope is FamilyScope.JOINT,
            uses_p2_forecast=self.scope is FamilyScope.JOINT,
            requires_fixed_placement=True,
            evaluation_eligible=True,
        )

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        if self.scope is FamilyScope.LOCAL:
            return self._build_local(problem)
        return self._build_joint(problem)

    def _base_priority_lookup(self, matrices: list[list[list[int]]]) -> dict[str, float] | None:
        if self.spec.base_priority_model != "birkhoff_round_rank":
            return None
        return _base_score_lookup_from_phase_orders(_birkhoff_phase_orders(matrices))

    def _prediction_matrix(self, problem: MultiPhaseSchedulingProblem) -> list[list[int]]:
        if problem.forecast is None:
            return [list(row) for row in problem.p2_next_dispatch_forecast_matrix]
        metadata = dict(problem.forecast.metadata or {})
        hint = metadata.get("planning_hint_matrix")
        if hint is None:
            return [list(row) for row in problem.forecast.matrix]
        return [list(row) for row in canonicalize_remote_matrix(hint)]

    def _run_kernel(
        self,
        *,
        problem: MultiPhaseSchedulingProblem,
        dispatch: list[list[int]],
        combine: list[list[int]],
        next_dispatch: list[list[int]],
        prediction_confidence: float,
        prediction_matrix: list[list[int]],
        base_score_lookup: dict[str, float] | None,
    ) -> dict[str, Any]:
        return run_global_matching_scheduler(
            dispatch,
            combine,
            next_dispatch,
            int(problem.topology.num_gpus),
            strategy=f"{self.scope.value}:{self.family_id}",
            mode=problem.options.scheduling_mode,
            prediction_confidence=float(prediction_confidence),
            expert_compute_delay=float(problem.release_model.expert_compute_delay),
            exact_matching=bool(self.spec.exact_matching),
            wave_quantum=None,
            max_waves=int(problem.options.max_waves),
            residual_weight=float(self.spec.residual_weight),
            barrier_weight=float(self.spec.barrier_weight),
            age_weight=float(self.spec.age_weight),
            prediction_weight=float(self.spec.prediction_weight),
            endpoint_pressure_weight=float(self.spec.endpoint_pressure_weight),
            release_gain_weight=float(self.spec.release_gain_weight),
            adaptive_prices=bool(self.spec.adaptive_prices),
            price_step=float(self.spec.price_step),
            price_decay=float(self.spec.price_decay),
            price_clip=float(self.spec.price_clip),
            iteration_budget=int(self.spec.iteration_budget),
            atomic=bool(self.spec.atomic),
            prediction_matrix=prediction_matrix,
            base_score_lookup=base_score_lookup,
            base_priority_weight=(float(self.spec.base_priority_weight) if base_score_lookup else 0.0),
            scoring_model=str(self.spec.scoring_model),
            critical_path_weight=float(self.spec.critical_path_weight),
            transitive_unlock_weight=float(self.spec.transitive_unlock_weight),
            endpoint_dual_weight=float(self.spec.endpoint_dual_weight),
            duplex_pair_weight=float(self.spec.duplex_pair_weight),
            dual_temperature=float(self.spec.dual_temperature),
            transitive_tail_weight=float(self.spec.transitive_tail_weight),
            destination_hotspot_weight=float(self.spec.destination_hotspot_weight),
            size_bias_power=float(self.spec.size_bias_power),
            collect_debug_trace=bool(self.collect_debug_trace),
        )

    def _common_diagnostics(self, *, phase_independent: bool) -> dict[str, Any]:
        contract = self.spec.contract()
        return {
            "family_id": self.family_id,
            "family_scope": self.scope.value,
            "family_expression": f"{self.scope.value.title()}({self.family_id})",
            "strict_same_core": True,
            "display_name": self.spec.display_name,
            "paper_label": self.spec.literature.paper_label,
            "literature_mapping_level": self.spec.literature.mapping_level,
            "literature_citation_key": self.spec.literature.citation_key,
            "phase_independent": bool(phase_independent),
            "visibility_contract": (
                "per_phase_only_no_cross_phase_ready_set"
                if phase_independent
                else "global_release_aware_p0_p1_p2_ready_set"
            ),
            "common_core": {
                **contract,
                "algorithm_id": self.policy_name,
                "phase_independent": bool(phase_independent),
                "uses_p2_prediction": self.scope is FamilyScope.JOINT and (
                    self.spec.prediction_weight > 0.0 or self.spec.scoring_model == "critical_frontier"
                ),
            },
        }

    def _build_joint(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        started = time.perf_counter()
        p0 = [list(row) for row in problem.p0_dispatch_matrix]
        p1 = [list(row) for row in problem.p1_return_matrix]
        p2 = [list(row) for row in problem.p2_next_dispatch_forecast_matrix]
        matrices = [p0, p1]
        if problem.options.scheduling_mode == EXECUTION_WINDOW_MODE:
            matrices.append(p2)
        result = self._run_kernel(
            problem=problem,
            dispatch=p0,
            combine=p1,
            next_dispatch=p2,
            prediction_confidence=float(problem.options.prediction_confidence),
            prediction_matrix=self._prediction_matrix(problem),
            base_score_lookup=self._base_priority_lookup(matrices),
        )
        wrapper_runtime_ms = (time.perf_counter() - started) * 1000.0
        kernel_runtime_ms = float(result.get("solve_time_ms", wrapper_runtime_ms))
        return _raw_schedule_to_plan(
            problem=problem,
            algorithm_id=self.policy_name,
            service_model=self.spec.service_model,
            raw_schedule=list(result.get("schedule", [])),
            audit=dict(result.get("audit", {})),
            makespan=float(result.get("makespan", 0.0)),
            planning_time_ms=kernel_runtime_ms,
            solver_status=str(result.get("solver_status", "completed")),
            selection_model=f"family_{self.family_id}_joint",
            extra_diagnostics={
                **self._common_diagnostics(phase_independent=False),
                "kernel_call_count": 1,
                "kernel_runtime_ms": kernel_runtime_ms,
                "phase_kernel_runtime_ms": {"joint": kernel_runtime_ms},
                "wrapper_runtime_ms": wrapper_runtime_ms,
                "scheduler_debug_trace": list(result.get("debug_trace", [])),
            },
        )

    def _phase_inputs(
        self,
        *,
        phase: int,
        matrix: tuple[tuple[int, ...], ...],
        zero: list[list[int]],
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        rows = [list(row) for row in matrix]
        if phase == 0:
            return rows, [list(row) for row in zero], [list(row) for row in zero]
        if phase == 1:
            return [list(row) for row in zero], rows, [list(row) for row in zero]
        return [list(row) for row in zero], [list(row) for row in zero], rows

    def _build_local(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        started = time.perf_counter()
        zero = [[0 for _ in row] for row in problem.p0_dispatch_matrix]
        phase_matrices: list[tuple[int, tuple[tuple[int, ...], ...]]] = [
            (0, problem.p0_dispatch_matrix),
            (1, problem.p1_return_matrix),
        ]
        if problem.options.scheduling_mode == EXECUTION_WINDOW_MODE:
            phase_matrices.append((2, problem.p2_next_dispatch_forecast_matrix))

        raw_schedule: list[dict[str, Any]] = []
        phase_runtime: dict[str, float] = {}
        phase_status: dict[str, str] = {}
        debug_trace: dict[str, list[dict[str, Any]]] = {}
        start_offset = 0.0
        wave_offset = 0
        for phase, matrix in phase_matrices:
            dispatch, combine, next_dispatch = self._phase_inputs(phase=phase, matrix=matrix, zero=zero)
            # Preserve the original phase index in Birkhoff flow IDs.  A
            # phase-1-only kernel call still emits ``phase1_*`` flows, so the
            # priority lookup must include an empty phase-0 placeholder (and
            # analogously for phase 2).
            base_matrices = [
                [list(row) for row in zero]
                for _ in range(phase)
            ] + [[list(row) for row in matrix]]
            base_lookup = self._base_priority_lookup(base_matrices)
            result = self._run_kernel(
                problem=problem,
                dispatch=dispatch,
                combine=combine,
                next_dispatch=next_dispatch,
                prediction_confidence=0.0,
                prediction_matrix=[list(row) for row in zero],
                base_score_lookup=base_lookup,
            )
            phase_name = f"p{phase}"
            phase_runtime[phase_name] = float(result.get("solve_time_ms", 0.0))
            phase_status[phase_name] = str(result.get("solver_status", "completed"))
            debug_trace[phase_name] = list(result.get("debug_trace", []))
            shifted = _offset_schedule(
                list(result.get("schedule", [])),
                start_offset=start_offset,
                wave_offset=wave_offset,
            )
            raw_schedule.extend(shifted)
            start_offset = max((float(row.get("end", start_offset)) for row in shifted), default=start_offset)
            wave_offset = max((int(row.get("wave_id", wave_offset - 1)) for row in shifted), default=wave_offset - 1) + 1

        wrapper_runtime_ms = (time.perf_counter() - started) * 1000.0
        kernel_runtime_ms = float(sum(phase_runtime.values()))
        audit = replay_and_audit_schedule(
            schedule=raw_schedule,
            dispatch_matrix=[list(row) for row in problem.p0_dispatch_matrix],
            combine_matrix=[list(row) for row in problem.p1_return_matrix],
            next_dispatch_matrix=(
                [list(row) for row in problem.p2_next_dispatch_forecast_matrix]
                if problem.options.scheduling_mode == EXECUTION_WINDOW_MODE
                else [list(row) for row in zero]
            ),
            num_gpus=int(problem.topology.num_gpus),
            expert_compute_delay=float(problem.release_model.expert_compute_delay),
            mode=problem.options.scheduling_mode,
            scheduler_name=self.policy_name,
            planning_time_ms=kernel_runtime_ms,
            reported_makespan=max((float(row.get("end", 0.0)) for row in raw_schedule), default=0.0),
            prediction_used=False,
        )
        return _raw_schedule_to_plan(
            problem=problem,
            algorithm_id=self.policy_name,
            service_model=self.spec.service_model,
            raw_schedule=raw_schedule,
            audit=audit,
            makespan=float(audit.get("replay_makespan", 0.0)),
            planning_time_ms=kernel_runtime_ms,
            solver_status="completed" if all(value == "completed" for value in phase_status.values()) else "partial",
            selection_model=f"family_{self.family_id}_local",
            extra_diagnostics={
                **self._common_diagnostics(phase_independent=True),
                "future_information_mode": (
                    "oracle_execution_window"
                    if problem.options.scheduling_mode == EXECUTION_WINDOW_MODE
                    else "none"
                ),
                "forecast_consumed": False,
                "prediction_used": False,
                "evaluation_eligible": (
                    problem.options.scheduling_mode != EXECUTION_WINDOW_MODE
                ),
                "kernel_call_count": len(phase_matrices),
                "kernel_runtime_ms": kernel_runtime_ms,
                "phase_kernel_runtime_ms": phase_runtime,
                "phase_solver_status": phase_status,
                "wrapper_runtime_ms": wrapper_runtime_ms,
                "scheduler_debug_trace_by_phase": debug_trace,
            },
        )


def resolve_scoped_family_policy(
    policy_name: str,
    *,
    residual_weight: float | None = None,
    barrier_weight: float | None = None,
    age_weight: float | None = None,
    prediction_weight: float | None = None,
    endpoint_pressure_weight: float | None = None,
    release_gain_weight: float | None = None,
) -> ScopedFamilyPolicy:
    parsed = parse_scoped_family_policy(policy_name)
    if parsed is None:
        raise ValueError(f"not a scoped family policy: {policy_name!r}")
    family_id, scope = parsed
    # Use stable canonical IDs for expression aliases; preserve historical IDs
    # when callers explicitly request them so old evidence readers keep working.
    reported = str(policy_name)
    if _EXPRESSION.fullmatch(reported):
        reported = canonical_family_policy_id(family_id, scope)
    return ScopedFamilyPolicy(
        family_id=family_id,
        scope=scope,
        reported_policy_name=reported,
        residual_weight=residual_weight,
        barrier_weight=barrier_weight,
        age_weight=age_weight,
        prediction_weight=prediction_weight,
        endpoint_pressure_weight=endpoint_pressure_weight,
        release_gain_weight=release_gain_weight,
    )


__all__ = [
    "ScopedFamilyPolicy",
    "canonical_family_policy_id",
    "is_scoped_family_policy",
    "parse_scoped_family_policy",
    "resolve_scoped_family_policy",
]
