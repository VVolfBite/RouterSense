"""Recovered POC-line Tier 1 multiphase scheduling candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave, MultiPhaseSchedulingProblem
from rs.scheduling.diagnostics import PolicyDiagnostics, WaveDiagnostics
from rs.scheduling.multiphase.flow_model import EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE
from rs.scheduling.multiphase.matching import linear_sum_assignment
from rs.scheduling.multiphase.replay import replay_and_audit_schedule
from rs.scheduling.multiphase.scheduler_state import run_global_matching_scheduler
from rs.scheduling.reference.birkhoff_von_neumann_fluid import decompose_fluid_matrix


TIER1_ALGORITHM_IDS = (
    "B_birkhoff",
    "B_birkhoff_wave",
    "U_gated_maxweight_matching",
    "U_barrier_criticality_global_matching",
    "U_gated_maxweight_matching_atomic",
    "U_barrier_criticality_global_matching_atomic",
    "U_lagrangian",
)

ATOMIC_SERVICE_MODEL = "atomic_chunk"
FLUID_SERVICE_MODEL = "fluid_wave"
LAGRANGIAN_SERVICE_MODEL = "lagrangian_atomic_chunk"

_PHASE_NAMES = {0: "p0_dispatch", 1: "p1_return", 2: "p2_next_dispatch"}


@dataclass(frozen=True)
class Tier1PolicySpec:
    algorithm_id: str
    service_model: str
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


_U_SPECS = {
    "U_gated_maxweight_matching": Tier1PolicySpec(
        algorithm_id="U_gated_maxweight_matching",
        service_model=FLUID_SERVICE_MODEL,
        exact_matching=True,
        atomic=False,
        residual_weight=1.0,
        barrier_weight=1.0,
        age_weight=0.1,
        prediction_weight=0.25,
    ),
    "U_barrier_criticality_global_matching": Tier1PolicySpec(
        algorithm_id="U_barrier_criticality_global_matching",
        service_model=FLUID_SERVICE_MODEL,
        exact_matching=True,
        atomic=False,
        residual_weight=0.75,
        barrier_weight=1.75,
        age_weight=0.15,
        prediction_weight=0.35,
    ),
    "U_gated_maxweight_matching_atomic": Tier1PolicySpec(
        algorithm_id="U_gated_maxweight_matching_atomic",
        service_model=ATOMIC_SERVICE_MODEL,
        exact_matching=True,
        atomic=True,
        residual_weight=1.0,
        barrier_weight=1.0,
        age_weight=0.1,
        prediction_weight=0.25,
    ),
    "U_barrier_criticality_global_matching_atomic": Tier1PolicySpec(
        algorithm_id="U_barrier_criticality_global_matching_atomic",
        service_model=ATOMIC_SERVICE_MODEL,
        exact_matching=True,
        atomic=True,
        residual_weight=0.75,
        barrier_weight=1.75,
        age_weight=0.15,
        prediction_weight=0.35,
    ),
}


class Tier1MultiphasePolicy:
    """Offline-only wrapper for recovered POC-line Tier 1 candidates."""

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
        if algorithm_id not in TIER1_ALGORITHM_IDS:
            raise ValueError(f"unknown Tier 1 algorithm {algorithm_id!r}")
        self.algorithm_id = algorithm_id
        self.policy_name = algorithm_id

    def build_logical_plan(self, problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
        if self.algorithm_id == "B_birkhoff":
            return _build_b_birkhoff_atomic(problem)
        if self.algorithm_id == "B_birkhoff_wave":
            return _build_b_birkhoff_wave(problem)
        if self.algorithm_id == "U_lagrangian":
            return _build_lagrangian(problem)
        return _build_u_policy(problem, _U_SPECS[self.algorithm_id])


def is_tier1_algorithm(policy_name: str) -> bool:
    return policy_name in TIER1_ALGORITHM_IDS


def resolve_tier1_policy(policy_name: str) -> Tier1MultiphasePolicy:
    return Tier1MultiphasePolicy(policy_name)


def tier1_inventory() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "algorithm_id": algorithm_id,
            "policy_name": algorithm_id,
            "supports_offline": True,
            "supports_online_phase_local_execution": False,
            "supports_online_multiphase_execution": False,
            "service_model": _service_model_for_algorithm(algorithm_id),
            "recovery_status": "recovered_exactly",
        }
        for algorithm_id in TIER1_ALGORITHM_IDS
    )


def _build_u_policy(problem: MultiPhaseSchedulingProblem, spec: Tier1PolicySpec) -> LogicalSchedulePlan:
    if spec.exact_matching and linear_sum_assignment is None:
        return _unsupported_exact_backend_plan(problem, spec)
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
        max_waves=256,
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
        solver_status="valid" if result.get("audit", {}).get("valid", False) else "invalid",
        selection_model="global_ready_set_exact_maxweight_matching",
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


def _build_b_birkhoff_atomic(problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
    started = time.perf_counter()
    raw_schedule = _schedule_phase_serial_atomic(
        matrices=_real_matrices(problem),
        expert_compute_delay=float(problem.release_model.expert_compute_delay),
        order_kind="birkhoff",
    )
    makespan = max((float(entry["end"]) for entry in raw_schedule), default=0.0)
    planning_time_ms = (time.perf_counter() - started) * 1000.0
    audit = _audit_raw_schedule(problem, raw_schedule, "B_birkhoff", makespan, planning_time_ms)
    return _raw_schedule_to_plan(
        problem=problem,
        algorithm_id="B_birkhoff",
        service_model=ATOMIC_SERVICE_MODEL,
        raw_schedule=raw_schedule,
        audit=audit,
        makespan=makespan,
        planning_time_ms=planning_time_ms,
        solver_status="valid" if audit.get("valid", False) else "invalid",
        selection_model="phase_serial_birkhoff_chunk_order",
        extra_diagnostics={"phase_serial": True, "fluid_split": False},
    )


def _build_b_birkhoff_wave(problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
    started = time.perf_counter()
    raw_schedule: list[dict[str, Any]] = []
    current_time = 0.0
    wave_id = 0
    phase_certificates: dict[str, Any] = {}
    for phase, matrix in enumerate(_real_matrices(problem)):
        if phase == 1 and raw_schedule:
            current_time += float(problem.release_model.expert_compute_delay)
        waves, certificate = decompose_fluid_matrix(_tuple_matrix(matrix), phase=_PHASE_NAMES[phase], start_wave_id=wave_id)
        phase_certificates[_PHASE_NAMES[phase]] = certificate.to_dict()
        for wave in waves:
            duration = float(wave.duration)
            for flow in wave.flows:
                raw_schedule.append(
                    {
                        "chunk_id": flow.flow_id,
                        "flow_id": f"phase{phase}_src{flow.src_rank}_dst{flow.dst_rank}",
                        "phase": phase,
                        "size": float(flow.byte_count),
                        "served_volume": float(flow.byte_count),
                        "src": int(flow.src_rank),
                        "dst": int(flow.dst_rank),
                        "src_gpu": int(flow.src_rank),
                        "dst_gpu": int(flow.dst_rank),
                        "start": current_time,
                        "end": current_time + duration,
                        "wave_id": wave_id,
                        "priority": [float(flow.byte_count), 0.0, 0.0, 0.0, 0.0],
                    }
                )
            current_time += duration
            wave_id += 1
    makespan = max((float(entry["end"]) for entry in raw_schedule), default=0.0)
    planning_time_ms = (time.perf_counter() - started) * 1000.0
    audit = _audit_raw_schedule(problem, raw_schedule, "B_birkhoff_wave", makespan, planning_time_ms)
    return _raw_schedule_to_plan(
        problem=problem,
        algorithm_id="B_birkhoff_wave",
        service_model=FLUID_SERVICE_MODEL,
        raw_schedule=raw_schedule,
        audit=audit,
        makespan=makespan,
        planning_time_ms=planning_time_ms,
        solver_status="valid" if audit.get("valid", False) else "invalid",
        selection_model="phase_serial_birkhoff_fluid_decomposition",
        extra_diagnostics={"phase_certificates": phase_certificates, "phase_serial": True, "fluid_split": True},
    )


def _build_lagrangian(problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
    started = time.perf_counter()
    matrices = _real_matrices(problem)
    lambda_by_gpu = [0.0 for _ in range(problem.topology.num_gpus)]
    best_order: list[list[dict[str, Any]]] | None = None
    best_makespan = float("inf")
    iterations: list[dict[str, Any]] = []
    for iteration in range(6):
        phase_orders: list[list[dict[str, Any]]] = []
        for phase, matrix in enumerate(matrices):
            chunks = _phase_chunks(matrix, phase)
            row_sums = [sum(max(0, int(value)) for value in row) for row in matrix]
            col_sums = [sum(max(0, int(matrix[src][dst])) for src in range(len(matrix)) if src != dst) for dst in range(len(matrix))]
            ranks = _birkhoff_round_ranks(matrix)
            chunks.sort(
                key=lambda chunk: (
                    ranks.get((chunk["src_gpu"], chunk["dst_gpu"]), 10**9),
                    -(lambda_by_gpu[chunk["src_gpu"]] * row_sums[chunk["src_gpu"]] * 0.01),
                    -(lambda_by_gpu[chunk["dst_gpu"]] * col_sums[chunk["dst_gpu"]] * 0.01),
                    -float(chunk["served_volume"]),
                    int(chunk["src_gpu"]),
                    int(chunk["dst_gpu"]),
                )
            )
            phase_orders.append(chunks)
        raw_schedule = _schedule_ordered_chunks(phase_orders, float(problem.release_model.expert_compute_delay))
        makespan = max((float(entry["end"]) for entry in raw_schedule), default=0.0)
        if makespan < best_makespan:
            best_makespan = makespan
            best_order = phase_orders
        phase0_completion = _phase_completion_by_dst(raw_schedule, 0, problem.topology.num_gpus)
        phase1_completion = _phase_completion_by_src(raw_schedule, 1, problem.topology.num_gpus)
        max_violation = 0.0
        for gpu in range(problem.topology.num_gpus):
            violation = phase1_completion[gpu] - phase0_completion[gpu] - float(problem.release_model.expert_compute_delay)
            max_violation = max(max_violation, violation)
            lambda_by_gpu[gpu] = max(0.0, lambda_by_gpu[gpu] + 0.2 * violation)
        iterations.append({"iteration": iteration, "makespan": makespan, "max_barrier_violation": max_violation, "lambda": list(lambda_by_gpu)})
        if max_violation <= 1e-9:
            break
    raw_schedule = _schedule_ordered_chunks(best_order or [], float(problem.release_model.expert_compute_delay))
    makespan = max((float(entry["end"]) for entry in raw_schedule), default=0.0)
    planning_time_ms = (time.perf_counter() - started) * 1000.0
    audit = _audit_raw_schedule(problem, raw_schedule, "U_lagrangian", makespan, planning_time_ms)
    return _raw_schedule_to_plan(
        problem=problem,
        algorithm_id="U_lagrangian",
        service_model=LAGRANGIAN_SERVICE_MODEL,
        raw_schedule=raw_schedule,
        audit=audit,
        makespan=makespan,
        planning_time_ms=planning_time_ms,
        solver_status="valid" if audit.get("valid", False) else "invalid",
        selection_model="historical_lagrangian_phase_order",
        extra_diagnostics={"lagrangian_iterations": iterations, "lambda_final": lambda_by_gpu},
    )


def _unsupported_exact_backend_plan(problem: MultiPhaseSchedulingProblem, spec: Tier1PolicySpec) -> LogicalSchedulePlan:
    diagnostics = _base_diagnostics(
        problem=problem,
        algorithm_id=spec.algorithm_id,
        service_model=spec.service_model,
        wave_count=0,
        logical_flow_count=0,
        planning_time_ms=0.0,
        makespan=0.0,
        solver_status="exact_matching_backend_unavailable",
        selection_model="global_ready_set_exact_maxweight_matching",
        audit={"valid": False, "validation_errors": ["exact matching backend unavailable"]},
        extra={},
        per_wave=(),
    )
    return LogicalSchedulePlan(policy_name=spec.algorithm_id, waves=(), diagnostics=diagnostics)


def _real_matrices(problem: MultiPhaseSchedulingProblem) -> list[list[list[int]]]:
    matrices = [
        [list(row) for row in problem.p0_dispatch_matrix],
        [list(row) for row in problem.p1_return_matrix],
    ]
    if problem.options.scheduling_mode == EXECUTION_WINDOW_MODE:
        matrices.append([list(row) for row in problem.p2_next_dispatch_forecast_matrix])
    elif problem.options.scheduling_mode != RUNTIME_LOOKAHEAD_MODE:
        raise ValueError(f"unsupported scheduling_mode {problem.options.scheduling_mode!r}")
    return matrices


def _schedule_phase_serial_atomic(
    *,
    matrices: list[list[list[int]]],
    expert_compute_delay: float,
    order_kind: str,
) -> list[dict[str, Any]]:
    phase_orders = []
    for phase, matrix in enumerate(matrices):
        chunks = _phase_chunks(matrix, phase)
        if order_kind == "birkhoff":
            ranks = _birkhoff_round_ranks(matrix)
            chunks.sort(key=lambda chunk: (ranks.get((chunk["src_gpu"], chunk["dst_gpu"]), 10**9), chunk["src_gpu"], chunk["dst_gpu"]))
        phase_orders.append(chunks)
    return _schedule_ordered_chunks(phase_orders, expert_compute_delay)


def _schedule_ordered_chunks(phase_orders: list[list[dict[str, Any]]], expert_compute_delay: float) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    current_time = 0.0
    wave_id = 0
    for phase, chunks in enumerate(phase_orders):
        if phase == 1 and schedule:
            current_time += expert_compute_delay
        pending = list(chunks)
        while pending:
            used_src: set[int] = set()
            used_dst: set[int] = set()
            selected: list[dict[str, Any]] = []
            remaining: list[dict[str, Any]] = []
            for chunk in pending:
                src = int(chunk["src_gpu"])
                dst = int(chunk["dst_gpu"])
                if src not in used_src and dst not in used_dst:
                    selected.append(chunk)
                    used_src.add(src)
                    used_dst.add(dst)
                else:
                    remaining.append(chunk)
            if not selected:
                raise ValueError("phase-serial atomic scheduler made no progress")
            wave_end = current_time
            for chunk in selected:
                served = float(chunk["served_volume"])
                entry = dict(chunk)
                entry.update(
                    {
                        "chunk_id": f"{chunk['flow_id']}_wave{wave_id}",
                        "start": current_time,
                        "end": current_time + served,
                        "wave_id": wave_id,
                        "priority": [served, 0.0, 0.0, 0.0, 0.0],
                    }
                )
                schedule.append(entry)
                wave_end = max(wave_end, current_time + served)
            current_time = wave_end
            wave_id += 1
            pending = remaining
    return schedule


def _phase_chunks(matrix: list[list[int]], phase: int) -> list[dict[str, Any]]:
    chunks = []
    for src, row in enumerate(matrix):
        for dst, value in enumerate(row):
            volume = float(value)
            if src == dst or volume <= 0.0:
                continue
            flow_id = f"phase{phase}_src{src}_dst{dst}"
            chunks.append(
                {
                    "chunk_id": flow_id,
                    "flow_id": flow_id,
                    "phase": int(phase),
                    "size": volume,
                    "served_volume": volume,
                    "src": int(src),
                    "dst": int(dst),
                    "src_gpu": int(src),
                    "dst_gpu": int(dst),
                }
            )
    return chunks


def _birkhoff_round_ranks(matrix: list[list[int]]) -> dict[tuple[int, int], int]:
    residual = {
        (src, dst): int(value)
        for src, row in enumerate(matrix)
        for dst, value in enumerate(row)
        if src != dst and int(value) > 0
    }
    ranks: dict[tuple[int, int], int] = {}
    round_id = 0
    ranks_tuple = tuple(range(len(matrix)))
    from rs.scheduling.matching import maximum_weight_bipartite_matching

    while residual:
        matching = tuple(
            edge
            for edge in maximum_weight_bipartite_matching(
                sources=ranks_tuple,
                destinations=ranks_tuple,
                edge_weight=lambda src, dst: float(residual.get((src, dst), 0)) if src != dst else 0.0,
            )
            if edge in residual
        )
        if not matching:
            raise ValueError("Birkhoff support ordering made no progress")
        quantum = min(residual[edge] for edge in matching)
        for edge in matching:
            ranks.setdefault(edge, round_id)
            residual[edge] -= quantum
            if residual[edge] <= 0:
                residual.pop(edge, None)
        round_id += 1
    return ranks


def _raw_schedule_to_plan(
    *,
    problem: MultiPhaseSchedulingProblem,
    algorithm_id: str,
    service_model: str,
    raw_schedule: list[dict[str, Any]],
    audit: dict[str, Any],
    makespan: float,
    planning_time_ms: float,
    solver_status: str,
    selection_model: str,
    extra_diagnostics: dict[str, Any],
) -> LogicalSchedulePlan:
    waves_by_id: dict[int, list[dict[str, Any]]] = {}
    for entry in raw_schedule:
        waves_by_id.setdefault(int(entry["wave_id"]), []).append(entry)
    waves: list[LogicalWave] = []
    per_wave: list[WaveDiagnostics] = []
    for wave_id in sorted(waves_by_id):
        entries = sorted(waves_by_id[wave_id], key=lambda item: (int(item["phase"]), int(item["src_gpu"]), int(item["dst_gpu"]), str(item["chunk_id"])))
        flows = tuple(_entry_to_flow(entry, service_model=service_model) for entry in entries)
        duration = max((float(entry["end"]) - float(entry["start"]) for entry in entries), default=0.0)
        waves.append(LogicalWave(wave_id=wave_id, flows=flows, duration=duration))
        per_wave.append(
            WaveDiagnostics(
                wave_id=wave_id,
                selected_flow_ids=tuple(flow.flow_id for flow in flows),
                selected_edges=tuple(
                    {
                        "phase": flow.phase,
                        "src_rank": flow.src_rank,
                        "dst_rank": flow.dst_rank,
                        "byte_count": flow.byte_count,
                        "origin_flow_id": flow.dependency_metadata.get("origin_flow_id", flow.flow_id),
                    }
                    for flow in flows
                ),
                matching_weight=float(sum(flow.byte_count for flow in flows)),
                priority_components={
                    "score_terms": [entry.get("priority", []) for entry in entries],
                    "selection_model": selection_model,
                },
                remaining_bytes_before=float(sum(float(entry.get("residual_before", entry.get("served_volume", 0.0))) for entry in entries)),
                remaining_bytes_after=float(sum(float(entry.get("residual_after", 0.0)) for entry in entries)),
                ready_flow_count_before=len(entries),
                blocked_flow_count_before=len(problem.flow_window.blocked_flows),
                forecast_pressure_summary=_forecast_summary(problem),
                selection_reason=selection_model,
            )
        )
    diagnostics = _base_diagnostics(
        problem=problem,
        algorithm_id=algorithm_id,
        service_model=service_model,
        wave_count=len(waves),
        logical_flow_count=sum(len(wave.flows) for wave in waves),
        planning_time_ms=planning_time_ms,
        makespan=makespan,
        solver_status=solver_status,
        selection_model=selection_model,
        audit=audit,
        extra={**extra_diagnostics, "raw_schedule": raw_schedule},
        per_wave=tuple(per_wave),
    )
    return LogicalSchedulePlan(policy_name=algorithm_id, waves=tuple(waves), diagnostics=diagnostics)


def _entry_to_flow(entry: dict[str, Any], *, service_model: str) -> FlowDemand:
    phase = int(entry["phase"])
    origin_flow_id = str(entry.get("flow_id", f"phase{phase}_src{entry['src_gpu']}_dst{entry['dst_gpu']}"))
    segment_id = str(entry.get("chunk_id", f"{origin_flow_id}_wave{entry['wave_id']}"))
    served = int(round(float(entry.get("served_volume", entry.get("size", 0.0)))))
    return FlowDemand(
        flow_id=segment_id,
        phase=_PHASE_NAMES[phase],
        src_rank=int(entry["src_gpu"]),
        dst_rank=int(entry["dst_gpu"]),
        byte_count=served,
        release_state="ready",
        is_executable=True,
        dependency_metadata={
            "origin_flow_id": origin_flow_id,
            "segment_id": segment_id,
            "service_model": service_model,
            "served_volume": float(entry.get("served_volume", entry.get("size", 0.0))),
            "residual_before": float(entry.get("residual_before", entry.get("served_volume", entry.get("size", 0.0)))),
            "residual_after": float(entry.get("residual_after", 0.0)),
            "start": float(entry.get("start", 0.0)),
            "end": float(entry.get("end", 0.0)),
            "wave_id": int(entry.get("wave_id", 0)),
        },
    )


def _base_diagnostics(
    *,
    problem: MultiPhaseSchedulingProblem,
    algorithm_id: str,
    service_model: str,
    wave_count: int,
    logical_flow_count: int,
    planning_time_ms: float,
    makespan: float,
    solver_status: str,
    selection_model: str,
    audit: dict[str, Any],
    extra: dict[str, Any],
    per_wave: tuple[WaveDiagnostics, ...],
) -> dict[str, Any]:
    future_information_mode = _future_information_mode(problem)
    evaluation_eligible = _evaluation_eligible(problem)
    stable_audit = dict(audit)
    stable_audit["planning_time_ms"] = 0.0
    diag = PolicyDiagnostics(
        policy_name=algorithm_id,
        policy_version="v1",
        information_mode=future_information_mode,
        tie_break_rule="stable wave id, phase, src_rank, dst_rank, chunk_id",
        wave_count=wave_count,
        logical_flow_count=logical_flow_count,
        ready_flow_count=len(problem.flow_window.ready_flows),
        blocked_flow_count=len(problem.flow_window.blocked_flows),
        forecast_flow_count=len(problem.flow_window.forecast_pressure),
        p1_dependency_used=algorithm_id.startswith("U_"),
        p2_forecast_used=problem.options.scheduling_mode == RUNTIME_LOOKAHEAD_MODE and problem.options.prediction_confidence > 0.0,
        p2_source=problem.forecast.source if problem.forecast is not None else "none",
        evaluation_eligible=evaluation_eligible,
        per_wave=per_wave,
        priority_components={
            "selection_model": selection_model,
            "service_model": service_model,
            "future_information_mode": future_information_mode,
        },
    )
    return {
        **diag.to_dict(),
        "algorithm_id": algorithm_id,
        "policy_name": algorithm_id,
        "service_model": service_model,
        "mode": problem.options.scheduling_mode,
        "future_information_mode": future_information_mode,
        "p2_source": problem.forecast.source if problem.forecast is not None else "none",
        "prediction_used": bool(problem.options.scheduling_mode == RUNTIME_LOOKAHEAD_MODE and problem.options.prediction_confidence > 0.0),
        "evaluation_eligible": evaluation_eligible,
        "makespan": float(makespan),
        "logical_service_horizon": float(makespan),
        "planning_time_ms": 0.0,
        "solver_status": solver_status,
        "valid": bool(audit.get("valid", False)),
        "release_barrier_verified": not any("barrier violation" in error for error in audit.get("validation_errors", [])),
        "flow_conservation_verified": not any("volume mismatch" in error or "unexpected flow" in error for error in audit.get("validation_errors", [])),
        "matching_legality_verified": not any("overlap" in error for error in audit.get("validation_errors", [])),
        "audit": stable_audit,
        "online_executor_compatible": False,
        "runtime_latency_comparable": False,
        **extra,
    }


def _audit_raw_schedule(
    problem: MultiPhaseSchedulingProblem,
    raw_schedule: list[dict[str, Any]],
    scheduler_name: str,
    makespan: float,
    planning_time_ms: float,
) -> dict[str, Any]:
    return replay_and_audit_schedule(
        schedule=raw_schedule,
        dispatch_matrix=[list(row) for row in problem.p0_dispatch_matrix],
        combine_matrix=[list(row) for row in problem.p1_return_matrix],
        next_dispatch_matrix=[list(row) for row in problem.p2_next_dispatch_forecast_matrix],
        num_gpus=int(problem.topology.num_gpus),
        expert_compute_delay=float(problem.release_model.expert_compute_delay),
        mode=problem.options.scheduling_mode,
        scheduler_name=scheduler_name,
        planning_time_ms=planning_time_ms,
        reported_makespan=makespan,
        prediction_used=problem.options.scheduling_mode == RUNTIME_LOOKAHEAD_MODE and problem.options.prediction_confidence > 0.0,
    )


def _future_information_mode(problem: MultiPhaseSchedulingProblem) -> str:
    if problem.options.scheduling_mode == EXECUTION_WINDOW_MODE:
        return "oracle_execution_window"
    if problem.options.scheduling_mode == RUNTIME_LOOKAHEAD_MODE and problem.options.prediction_confidence > 0.0:
        return "oracle_predicted_runtime_lookahead"
    return "none"


def _evaluation_eligible(problem: MultiPhaseSchedulingProblem) -> bool:
    if problem.forecast is not None and bool(problem.forecast.oracle):
        return False
    if _future_information_mode(problem) == "oracle_execution_window":
        return False
    return True


def _forecast_summary(problem: MultiPhaseSchedulingProblem) -> dict[str, Any]:
    forecast = problem.forecast
    if forecast is None:
        return {"source": "none", "matrix_total_bytes": 0}
    return {
        "source": forecast.source,
        "digest": forecast.digest,
        "oracle": forecast.oracle,
        "evaluation_eligible": forecast.evaluation_eligible,
        "matrix_total_bytes": forecast.matrix_total_bytes,
    }


def _service_model_for_algorithm(algorithm_id: str) -> str:
    if algorithm_id in {"B_birkhoff", "U_gated_maxweight_matching_atomic", "U_barrier_criticality_global_matching_atomic"}:
        return ATOMIC_SERVICE_MODEL
    if algorithm_id in {"B_birkhoff_wave", "U_gated_maxweight_matching", "U_barrier_criticality_global_matching"}:
        return FLUID_SERVICE_MODEL
    if algorithm_id == "U_lagrangian":
        return LAGRANGIAN_SERVICE_MODEL
    raise ValueError(f"unknown Tier 1 algorithm {algorithm_id!r}")


def _tuple_matrix(matrix: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in matrix)


def _phase_completion_by_dst(schedule: list[dict[str, Any]], phase: int, num_gpus: int) -> list[float]:
    completion = [0.0] * num_gpus
    for entry in schedule:
        if int(entry["phase"]) == phase:
            completion[int(entry["dst_gpu"])] = max(completion[int(entry["dst_gpu"])], float(entry["end"]))
    return completion


def _phase_completion_by_src(schedule: list[dict[str, Any]], phase: int, num_gpus: int) -> list[float]:
    completion = [0.0] * num_gpus
    for entry in schedule:
        if int(entry["phase"]) == phase:
            completion[int(entry["src_gpu"])] = max(completion[int(entry["src_gpu"])], float(entry["end"]))
    return completion


__all__ = [
    "ATOMIC_SERVICE_MODEL",
    "FLUID_SERVICE_MODEL",
    "LAGRANGIAN_SERVICE_MODEL",
    "TIER1_ALGORITHM_IDS",
    "Tier1MultiphasePolicy",
    "is_tier1_algorithm",
    "resolve_tier1_policy",
    "tier1_inventory",
]
