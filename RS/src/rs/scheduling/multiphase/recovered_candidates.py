"""Recovered or lightly wrapped historical multiphase candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import MultiPhaseSchedulingProblem
from rs.scheduling.multiphase.scheduler_state import run_global_matching_scheduler

from .tier1 import _raw_schedule_to_plan


@dataclass(frozen=True)
class RecoveredCandidateSpec:
    algorithm_id: str
    exact_matching: bool
    atomic: bool
    residual_weight: float
    barrier_weight: float
    age_weight: float
    prediction_weight: float
    adaptive_prices: bool = False
    price_step: float = 0.0
    price_decay: float = 0.0
    price_clip: float = 0.0
    iteration_budget: int = 1
    service_model: str = "fluid_wave"


RECOVERED_CANDIDATE_SPECS: dict[str, RecoveredCandidateSpec] = {
    "U_gated_greedy_maximal": RecoveredCandidateSpec(
        algorithm_id="U_gated_greedy_maximal",
        exact_matching=False,
        atomic=False,
        residual_weight=1.0,
        barrier_weight=1.0,
        age_weight=0.1,
        prediction_weight=0.25,
        service_model="fluid_wave",
    ),
    "U_gated_greedy_maximal_atomic": RecoveredCandidateSpec(
        algorithm_id="U_gated_greedy_maximal_atomic",
        exact_matching=False,
        atomic=True,
        residual_weight=1.0,
        barrier_weight=1.0,
        age_weight=0.1,
        prediction_weight=0.25,
        service_model="atomic_chunk",
    ),
    "U_barrier_price_adaptive_matching": RecoveredCandidateSpec(
        algorithm_id="U_barrier_price_adaptive_matching",
        exact_matching=True,
        atomic=False,
        residual_weight=0.75,
        barrier_weight=1.75,
        age_weight=0.15,
        prediction_weight=0.35,
        adaptive_prices=True,
        price_step=0.2,
        price_decay=0.1,
        price_clip=8.0,
        iteration_budget=2,
        service_model="fluid_wave",
    ),
    "U_barrier_price_adaptive_matching_atomic": RecoveredCandidateSpec(
        algorithm_id="U_barrier_price_adaptive_matching_atomic",
        exact_matching=True,
        atomic=True,
        residual_weight=0.75,
        barrier_weight=1.75,
        age_weight=0.15,
        prediction_weight=0.35,
        adaptive_prices=True,
        price_step=0.2,
        price_decay=0.1,
        price_clip=8.0,
        iteration_budget=2,
        service_model="atomic_chunk",
    ),
}


class RecoveredMultiphaseCandidatePolicy:
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

    def __init__(self, algorithm_id: str) -> None:
        if algorithm_id not in RECOVERED_CANDIDATE_SPECS:
            raise ValueError(f"unknown recovered candidate {algorithm_id!r}")
        self.algorithm_id = algorithm_id
        self.policy_name = algorithm_id

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem):  # type: ignore[no-untyped-def]
        spec = RECOVERED_CANDIDATE_SPECS[self.algorithm_id]
        started = time.perf_counter()
        result = run_global_matching_scheduler(
            [list(row) for row in problem.p0_dispatch_matrix],
            [list(row) for row in problem.p1_return_matrix],
            [list(row) for row in problem.p2_next_dispatch_forecast_matrix],
            int(problem.topology.num_gpus),
            strategy=spec.algorithm_id,
            mode=problem.options.scheduling_mode,
            prediction_confidence=float(problem.options.prediction_confidence),
            expert_compute_delay=float(problem.release_model.expert_compute_delay),
            exact_matching=spec.exact_matching,
            wave_quantum=None,
            max_waves=int(problem.options.max_waves),
            residual_weight=spec.residual_weight,
            barrier_weight=spec.barrier_weight,
            age_weight=spec.age_weight,
            prediction_weight=spec.prediction_weight,
            adaptive_prices=spec.adaptive_prices,
            price_step=spec.price_step,
            price_decay=spec.price_decay,
            price_clip=spec.price_clip,
            iteration_budget=spec.iteration_budget,
            atomic=spec.atomic,
        )
        planning_time_ms = float(result.get("solve_time_ms", (time.perf_counter() - started) * 1000.0))
        return _raw_schedule_to_plan(
            problem=problem,
            algorithm_id=spec.algorithm_id,
            service_model=spec.service_model,
            raw_schedule=list(result.get("schedule", [])),
            audit=dict(result.get("audit", {})),
            makespan=float(result.get("makespan", 0.0)),
            planning_time_ms=planning_time_ms,
            solver_status=str(result.get("solver_status", "completed")),
            selection_model="historical_global_ready_set_recovered",
            extra_diagnostics={
                "historical_parameters": {
                    "exact_matching": spec.exact_matching,
                    "atomic": spec.atomic,
                    "residual_weight": spec.residual_weight,
                    "barrier_weight": spec.barrier_weight,
                    "age_weight": spec.age_weight,
                    "prediction_weight": spec.prediction_weight,
                    "adaptive_prices": spec.adaptive_prices,
                }
            },
        )


def is_recovered_candidate(policy_name: str) -> bool:
    return policy_name in RECOVERED_CANDIDATE_SPECS


def resolve_recovered_candidate(policy_name: str) -> RecoveredMultiphaseCandidatePolicy:
    return RecoveredMultiphaseCandidatePolicy(policy_name)
