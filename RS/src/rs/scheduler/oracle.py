from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

from .greedy import greedy_schedule_pairwise


@dataclass(frozen=True)
class PairwiseChunk:
    chunk_id: str
    phase: int
    size: int
    src_gpu: int
    dst_gpu: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matrix_to_pairwise_chunks(
    matrix: list[list[int]],
    *,
    phase: int,
    num_gpus: int,
) -> list[PairwiseChunk]:
    chunks: list[PairwiseChunk] = []
    for src in range(num_gpus):
        for dst in range(num_gpus):
            size = int(matrix[src][dst])
            if src == dst or size <= 0:
                continue
            chunks.append(
                PairwiseChunk(
                    chunk_id=f"phase{phase}_src{src}_dst{dst}",
                    phase=phase,
                    size=size,
                    src_gpu=src,
                    dst_gpu=dst,
                )
            )
    return chunks


def _build_phase_chunks(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> list[PairwiseChunk]:
    return [
        *_matrix_to_pairwise_chunks(dispatch_matrix, phase=0, num_gpus=num_gpus),
        *_matrix_to_pairwise_chunks(combine_matrix, phase=1, num_gpus=num_gpus),
        *_matrix_to_pairwise_chunks(next_dispatch_matrix, phase=2, num_gpus=num_gpus),
    ]


def _pairwise_oracle_scipy(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    # Fallback path uses greedy upper bound only; main experiments should hit CP-SAT.
    makespan = greedy_schedule_pairwise(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )
    chunks = _build_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    return {
        "makespan": float(makespan),
        "schedule": [],
        "chunk_count": len(chunks),
        "solver_status": "fallback_greedy",
        "solve_time_ms": 0.0,
        "objective": float(makespan),
        "best_bound": float(makespan),
        "optimality_gap": 0.0,
        "time_limit_ms": 0.0,
    }


def pairwise_oracle(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return _pairwise_oracle_scipy(
            dispatch_matrix,
            combine_matrix,
            next_dispatch_matrix,
            num_gpus,
            model=model,
            expert_compute_delay=expert_compute_delay,
        )

    phase_chunks = _build_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    if not phase_chunks:
        return {
            "makespan": 0.0,
            "schedule": [],
            "chunk_count": 0,
            "solver_status": "empty",
            "solve_time_ms": 0.0,
            "objective": 0.0,
            "best_bound": 0.0,
            "optimality_gap": 0.0,
            "time_limit_ms": 200.0,
        }

    greedy_upper_bound = max(
        1,
        int(
            math.ceil(
                greedy_schedule_pairwise(
                    dispatch_matrix,
                    combine_matrix,
                    next_dispatch_matrix,
                    num_gpus,
                    model=model,
                    expert_compute_delay=expert_compute_delay,
                )
            )
        ),
    )
    horizon = greedy_upper_bound
    sat_model = cp_model.CpModel()

    starts = []
    ends = []
    intervals = []
    for index, chunk in enumerate(phase_chunks):
        start = sat_model.new_int_var(0, horizon, f"start_{index}")
        end = sat_model.new_int_var(0, horizon, f"end_{index}")
        interval = sat_model.new_interval_var(start, int(chunk.size), end, f"interval_{index}")
        starts.append(start)
        ends.append(end)
        intervals.append(interval)

    makespan = sat_model.new_int_var(0, horizon, "makespan")
    for end in ends:
        sat_model.add(makespan >= end)
    sat_model.minimize(makespan)

    for gpu in range(num_gpus):
        if model == "half_duplex":
            constrained = [
                intervals[index]
                for index, chunk in enumerate(phase_chunks)
                if chunk.src_gpu == gpu or chunk.dst_gpu == gpu
            ]
        elif model == "full_duplex":
            sender_intervals = [
                intervals[index]
                for index, chunk in enumerate(phase_chunks)
                if chunk.src_gpu == gpu
            ]
            receiver_intervals = [
                intervals[index]
                for index, chunk in enumerate(phase_chunks)
                if chunk.dst_gpu == gpu
            ]
            if len(sender_intervals) > 1:
                sat_model.add_no_overlap(sender_intervals)
            if len(receiver_intervals) > 1:
                sat_model.add_no_overlap(receiver_intervals)
            constrained = []
        elif model == "incast_only":
            constrained = [
                intervals[index]
                for index, chunk in enumerate(phase_chunks)
                if chunk.dst_gpu == gpu
            ]
        else:
            raise ValueError(f"Unsupported scheduling model: {model}")
        if len(constrained) > 1:
            sat_model.add_no_overlap(constrained)

    for gpu in range(num_gpus):
        for phase_from, phase_to in ((0, 1), (1, 2)):
            if model == "half_duplex":
                from_ends = [
                    ends[index]
                    for index, chunk in enumerate(phase_chunks)
                    if chunk.phase == phase_from and chunk.dst_gpu == gpu
                ]
                to_starts = [
                    starts[index]
                    for index, chunk in enumerate(phase_chunks)
                    if chunk.phase == phase_to and chunk.src_gpu == gpu
                ]
                extra_delay = int(math.ceil(expert_compute_delay)) if phase_to == 1 else 0
            else:
                from_ends = [
                    ends[index]
                    for index, chunk in enumerate(phase_chunks)
                    if chunk.phase == phase_from and chunk.dst_gpu == gpu
                ]
                to_starts = [
                    starts[index]
                    for index, chunk in enumerate(phase_chunks)
                    if chunk.phase == phase_to and chunk.src_gpu == gpu
                ]
                extra_delay = int(math.ceil(expert_compute_delay)) if phase_to == 1 else 0
            if from_ends and to_starts:
                done = sat_model.new_int_var(0, horizon, f"done_{gpu}_{phase_from}")
                for end in from_ends:
                    sat_model.add(done >= end)
                for start in to_starts:
                    sat_model.add(start >= done + extra_delay)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.2
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 0
    solve_start = time.perf_counter()
    status = solver.solve(sat_model)
    solve_time_ms = (time.perf_counter() - solve_start) * 1000.0
    feasible_statuses = {cp_model.OPTIMAL, cp_model.FEASIBLE}
    status_name = solver.status_name(status) if hasattr(solver, "status_name") else str(status)
    best_bound = float(getattr(solver, "best_objective_bound", 0.0))
    objective = float(getattr(solver, "objective_value", 0.0))
    optimality_gap = 0.0
    if status in feasible_statuses and objective > 0.0:
        optimality_gap = max(0.0, objective - best_bound) / objective
    if status not in feasible_statuses:
        return {
            "makespan": None,
            "schedule": [],
            "chunk_count": len(phase_chunks),
            "solver_status": status_name,
            "solve_time_ms": solve_time_ms,
            "objective": objective,
            "best_bound": best_bound,
            "optimality_gap": optimality_gap,
            "time_limit_ms": 200.0,
        }

    schedule = []
    for index, chunk in enumerate(phase_chunks):
        start_value = solver.value(starts[index])
        schedule.append({**chunk.to_dict(), "start": float(start_value), "end": float(start_value + chunk.size)})
    schedule.sort(key=lambda item: item["start"])
    return {
        "makespan": float(solver.value(makespan)),
        "schedule": schedule,
        "chunk_count": len(phase_chunks),
        "solver_status": status_name,
        "solve_time_ms": solve_time_ms,
        "objective": float(solver.value(makespan)),
        "best_bound": best_bound,
        "optimality_gap": optimality_gap,
        "time_limit_ms": 200.0,
    }


def _matrices_from_residual_ready(
    *,
    flows: list[Any],
    residual: dict[str, float],
    num_gpus: int,
    release_time: dict[tuple[int, int], float],
    current_time: float,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    matrices = [[[0] * num_gpus for _ in range(num_gpus)] for _ in range(3)]
    for flow in flows:
        remaining = residual.get(flow.flow_id, 0.0)
        if remaining <= 1e-9:
            continue
        if current_time + 1e-9 < release_time[(flow.phase, flow.src_gpu)]:
            continue
        matrices[flow.phase][flow.src_gpu][flow.dst_gpu] = int(math.ceil(remaining))
    return matrices[0], matrices[1], matrices[2]


def _residual_has_work(residual: dict[str, float]) -> bool:
    return any(value > 1e-9 for value in residual.values())


def _wave_oracle_pairwise(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    atomic: bool,
    planning_budget_ms: float = 500.0,
    no_improve_patience: int = 3,
    min_relative_improvement: float = 0.03,
) -> dict[str, Any]:
    from .multiphase_global import (
        EXECUTION_WINDOW_MODE,
        _collect_real_flows,
        _inbound_remaining,
        _maximum_weight_matching,
        _outbound_loads,
        _ready_flow_candidates,
        replay_and_audit_schedule,
    )

    planning_start = time.perf_counter()
    flows = _collect_real_flows(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        mode=EXECUTION_WINDOW_MODE,
    )
    if not flows:
        return {
            "makespan": 0.0,
            "schedule": [],
            "chunk_count": 0,
            "solver_status": "empty",
            "solve_time_ms": 0.0,
            "objective": 0.0,
            "best_bound": 0.0,
            "optimality_gap": 0.0,
            "time_limit_ms": planning_budget_ms,
            "wave_count": 0,
        }

    residual = {flow.flow_id: float(flow.volume) for flow in flows}
    inbound_remaining = _inbound_remaining(flows, num_gpus)
    release_time = {(phase, gpu): (0.0 if phase == 0 else math.inf) for phase in range(3) for gpu in range(num_gpus)}
    for phase in (1, 2):
        prev_phase = phase - 1
        for gpu in range(num_gpus):
            if inbound_remaining.get((prev_phase, gpu), 0.0) <= 1e-9:
                release_time[(phase, gpu)] = expert_compute_delay if prev_phase == 0 else 0.0
    barrier_done = {(phase, gpu): 0.0 for phase in range(3) for gpu in range(num_gpus)}
    ready_since: dict[str, float] = {}
    downstream_load = _outbound_loads(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        mode=EXECUTION_WINDOW_MODE,
        prediction_confidence=0.0,
    )
    current_time = 0.0
    schedule: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    wave_count = 0
    oracle_active = True
    no_improve_count = 0
    best_projected = float("inf")

    while _residual_has_work(residual):
        ready_candidates = _ready_flow_candidates(
            flows=flows,
            residual=residual,
            ready_since=ready_since,
            current_time=current_time,
            release_time=release_time,
            inbound_remaining=inbound_remaining,
            downstream_load=downstream_load,
            age_scale=max(1.0, current_time + 1.0),
            residual_weight=1.0,
            barrier_weight=1.0,
            age_weight=0.1,
            prediction_weight=0.0,
            mode=EXECUTION_WINDOW_MODE,
            prediction_confidence=0.0,
        )
        if not ready_candidates:
            future = [value for value in release_time.values() if value < math.inf and value > current_time + 1e-9]
            if not future:
                break
            current_time = min(future)
            continue

        selected_entries: list[dict[str, Any]] = []
        if oracle_active:
            ready_dispatch, ready_combine, ready_next = _matrices_from_residual_ready(
                flows=flows,
                residual=residual,
                num_gpus=num_gpus,
                release_time=release_time,
                current_time=current_time,
            )
            subsolve = pairwise_oracle(
                ready_dispatch,
                ready_combine,
                ready_next,
                num_gpus,
                model=model,
                expert_compute_delay=expert_compute_delay,
            )
            status_name = str(subsolve.get("solver_status", "unknown"))
            status_counts[status_name] = status_counts.get(status_name, 0) + 1
            sub_schedule = list(subsolve.get("schedule", []))
            if sub_schedule:
                earliest_start = min(float(entry["start"]) for entry in sub_schedule)
                selected_entries = [
                    entry
                    for entry in sub_schedule
                    if abs(float(entry["start"]) - earliest_start) <= 1e-9
                ]
            planning_elapsed_ms = (time.perf_counter() - planning_start) * 1000.0
            projected_residual = greedy_schedule_pairwise(
                ready_dispatch,
                ready_combine,
                ready_next,
                num_gpus,
                model=model,
                expert_compute_delay=expert_compute_delay,
            )
            projected_total = current_time + float(projected_residual)
            if projected_total < best_projected * (1.0 - min_relative_improvement):
                best_projected = projected_total
                no_improve_count = 0
            else:
                no_improve_count += 1
            if planning_elapsed_ms >= planning_budget_ms or no_improve_count >= no_improve_patience:
                oracle_active = False

        if not selected_entries:
            selected_entries = _maximum_weight_matching(ready_candidates, num_gpus)

        if not selected_entries:
            break

        flow_ids: list[str] = []
        for entry in selected_entries:
            if "flow_id" in entry:
                flow_ids.append(str(entry["flow_id"]))
            else:
                flow_ids.append(f"phase{int(entry['phase'])}_src{int(entry['src_gpu'])}_dst{int(entry['dst_gpu'])}")
        wave_residuals = [residual[flow_id] for flow_id in flow_ids if residual.get(flow_id, 0.0) > 1e-9]
        if not wave_residuals:
            break
        served_common = min(wave_residuals)
        wave_end = current_time
        for entry, flow_id in zip(selected_entries, flow_ids, strict=False):
            if residual.get(flow_id, 0.0) <= 1e-9:
                continue
            phase = int(entry["phase"])
            src = int(entry["src_gpu"])
            dst = int(entry["dst_gpu"])
            served = residual[flow_id] if atomic else served_common
            residual[flow_id] = max(0.0, residual[flow_id] - served)
            inbound_remaining[(phase, dst)] = max(0.0, inbound_remaining[(phase, dst)] - served)
            end = current_time + served
            wave_end = max(wave_end, end if atomic else current_time + served_common)
            barrier_done[(phase, dst)] = max(barrier_done[(phase, dst)], end)
            if phase < 2 and inbound_remaining[(phase, dst)] <= 1e-9:
                release_time[(phase + 1, dst)] = barrier_done[(phase, dst)] + (
                    expert_compute_delay if phase == 0 else 0.0
                )
            schedule.append(
                {
                    "chunk_id": f"{flow_id}_wave{wave_count}",
                    "flow_id": flow_id,
                    "phase": phase,
                    "size": float(served),
                    "served_volume": float(served),
                    "src": src,
                    "dst": dst,
                    "src_gpu": src,
                    "dst_gpu": dst,
                    "start": current_time,
                    "end": end,
                    "wave_id": wave_count,
                    "priority": [],
                }
            )
        current_time = wave_end if atomic else current_time + served_common
        wave_count += 1

    makespan = max((float(entry["end"]) for entry in schedule), default=current_time)
    planning_time_ms = (time.perf_counter() - planning_start) * 1000.0
    audit = replay_and_audit_schedule(
        schedule=schedule,
        dispatch_matrix=dispatch_matrix,
        combine_matrix=combine_matrix,
        next_dispatch_matrix=next_dispatch_matrix,
        num_gpus=num_gpus,
        expert_compute_delay=expert_compute_delay,
        mode=EXECUTION_WINDOW_MODE,
        scheduler_name="wave_oracle_atomic" if atomic else "wave_oracle_fluid",
        planning_time_ms=planning_time_ms,
        reported_makespan=makespan,
        prediction_used=False,
    )
    aggregate_status = "fallback_matching" if not status_counts else ",".join(f"{k}:{v}" for k, v in sorted(status_counts.items()))
    return {
        "makespan": makespan,
        "schedule": schedule,
        "chunk_count": len(_build_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)),
        "solver_status": aggregate_status,
        "solve_time_ms": planning_time_ms,
        "objective": makespan,
        "best_bound": makespan,
        "optimality_gap": 0.0,
        "time_limit_ms": planning_budget_ms,
        "wave_count": wave_count,
        "audit": audit,
        "mode": "execution_window",
        "oracle_active_switched_off": not oracle_active,
    }


def pairwise_wave_oracle(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    planning_budget_ms: float = 500.0,
    no_improve_patience: int = 3,
    min_relative_improvement: float = 0.03,
) -> dict[str, Any]:
    return _wave_oracle_pairwise(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        model=model,
        expert_compute_delay=expert_compute_delay,
        atomic=True,
        planning_budget_ms=planning_budget_ms,
        no_improve_patience=no_improve_patience,
        min_relative_improvement=min_relative_improvement,
    )


def pairwise_fluid_wave_oracle(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    planning_budget_ms: float = 500.0,
    no_improve_patience: int = 3,
    min_relative_improvement: float = 0.03,
) -> dict[str, Any]:
    return _wave_oracle_pairwise(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        model=model,
        expert_compute_delay=expert_compute_delay,
        atomic=False,
        planning_budget_ms=planning_budget_ms,
        no_improve_patience=no_improve_patience,
        min_relative_improvement=min_relative_improvement,
    )
