from __future__ import annotations

import math
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
    model: str = "half_duplex",
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
    }


def pairwise_oracle(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "half_duplex",
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
        return {"makespan": 0.0, "schedule": [], "chunk_count": 0, "solver_status": "empty"}

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
                    if chunk.phase == phase_from and (chunk.src_gpu == gpu or chunk.dst_gpu == gpu)
                ]
                to_starts = [
                    starts[index]
                    for index, chunk in enumerate(phase_chunks)
                    if chunk.phase == phase_to and (chunk.src_gpu == gpu or chunk.dst_gpu == gpu)
                ]
                extra_delay = 0
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
    status = solver.solve(sat_model)
    feasible_statuses = {cp_model.OPTIMAL, cp_model.FEASIBLE}
    if status not in feasible_statuses:
        status_name = solver.status_name(status) if hasattr(solver, "status_name") else str(status)
        return {
            "makespan": None,
            "schedule": [],
            "chunk_count": len(phase_chunks),
            "solver_status": status_name,
        }

    schedule = []
    for index, chunk in enumerate(phase_chunks):
        start_value = solver.value(starts[index])
        schedule.append({**chunk.to_dict(), "start": float(start_value), "end": float(start_value + chunk.size)})
    schedule.sort(key=lambda item: item["start"])
    status_name = solver.status_name(status) if hasattr(solver, "status_name") else str(status)
    return {
        "makespan": float(solver.value(makespan)),
        "schedule": schedule,
        "chunk_count": len(phase_chunks),
        "solver_status": status_name,
    }
