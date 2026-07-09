"""Recovered or lightly wrapped historical multiphase candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import MultiPhaseSchedulingProblem
from rs.scheduling.multiphase.scheduler_state import run_global_matching_scheduler

from .tier1 import _audit_raw_schedule, _birkhoff_phase_orders, _raw_schedule_to_plan, _real_matrices, _schedule_ordered_chunks


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
    "U_ibbr": RecoveredCandidateSpec(
        algorithm_id="U_ibbr",
        exact_matching=True,
        atomic=False,
        residual_weight=0.25,
        barrier_weight=0.0,
        age_weight=0.05,
        prediction_weight=0.0,
        service_model="fluid_wave",
    ),
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
        if spec.algorithm_id == "U_ibbr":
            return _build_u_ibbr(problem)
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


def _build_u_ibbr(problem: MultiPhaseSchedulingProblem):  # type: ignore[no-untyped-def]
    started = time.perf_counter()
    phase_orders = _birkhoff_phase_orders(_real_matrices(problem))
    best_orders = _clone_phase_orders(phase_orders)
    best_schedule = _schedule_ordered_chunks(best_orders, float(problem.release_model.expert_compute_delay))
    best_makespan = max((float(entry["end"]) for entry in best_schedule), default=0.0)
    no_improve_count = 0
    prev_best = best_makespan
    deadline = started + 0.003
    for _ in range(4):
        if time.perf_counter() > deadline:
            break
        g_star = max(_gpu_completion(best_orders, float(problem.release_model.expert_compute_delay)).items(), key=lambda item: item[1])[0]
        improved = False
        for phase in range(len(best_orders)):
            indices = [
                idx
                for idx, chunk in enumerate(best_orders[phase])
                if int(chunk["src_gpu"]) == g_star or int(chunk["dst_gpu"]) == g_star
            ]
            for left, right in zip(indices, indices[1:]):
                candidate = _clone_phase_orders(best_orders)
                candidate[phase][left], candidate[phase][right] = candidate[phase][right], candidate[phase][left]
                candidate_schedule = _schedule_ordered_chunks(candidate, float(problem.release_model.expert_compute_delay))
                candidate_makespan = max((float(entry["end"]) for entry in candidate_schedule), default=0.0)
                if candidate_makespan < best_makespan:
                    relative = (prev_best - candidate_makespan) / max(prev_best, 1e-9)
                    no_improve_count = 0 if relative >= 1e-4 else no_improve_count + 1
                    prev_best = candidate_makespan
                    best_orders = candidate
                    best_schedule = candidate_schedule
                    best_makespan = candidate_makespan
                    improved = True
                    break
            if improved:
                break
        if not improved:
            no_improve_count += 1
        if not improved or no_improve_count >= 2:
            break
    planning_time_ms = (time.perf_counter() - started) * 1000.0
    audit = _audit_raw_schedule(problem, best_schedule, "U_ibbr", best_makespan, planning_time_ms)
    return _raw_schedule_to_plan(
        problem=problem,
        algorithm_id="U_ibbr",
        service_model="fluid_wave",
        raw_schedule=best_schedule,
        audit=audit,
        makespan=best_makespan,
        planning_time_ms=planning_time_ms,
        solver_status="valid" if audit.get("valid", False) else "invalid",
        selection_model="iterated_birkhoff_barrier_repair",
        extra_diagnostics={
            "historical_parameters": {
                "seed_from_birkhoff": True,
                "local_swap_repair": True,
                "prediction_aware": False,
            }
        },
    )


def _clone_phase_orders(phase_orders: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    return [list(chunks) for chunks in phase_orders]


def _gpu_completion(phase_orders: list[list[dict[str, Any]]], expert_compute_delay: float) -> dict[int, float]:
    schedule = _schedule_ordered_chunks(phase_orders, expert_compute_delay)
    num_gpus = 0
    for phase_chunks in phase_orders:
        for chunk in phase_chunks:
            num_gpus = max(num_gpus, int(chunk["src_gpu"]) + 1, int(chunk["dst_gpu"]) + 1)
    completion = {gpu: 0.0 for gpu in range(num_gpus)}
    for entry in schedule:
        src = int(entry["src_gpu"])
        dst = int(entry["dst_gpu"])
        end = float(entry["end"])
        completion[src] = max(completion[src], end)
        completion[dst] = max(completion[dst], end)
    return completion
