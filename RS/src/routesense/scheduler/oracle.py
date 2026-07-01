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


def _pairwise_oracle_scipy(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> dict[str, Any]:
    try:
        import numpy as np
        import scipy.optimize as spo
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for pairwise oracle scheduling") from exc

    phase_chunks = [
        *_matrix_to_pairwise_chunks(dispatch_matrix, phase=0, num_gpus=num_gpus),
        *_matrix_to_pairwise_chunks(combine_matrix, phase=1, num_gpus=num_gpus),
        *_matrix_to_pairwise_chunks(next_dispatch_matrix, phase=2, num_gpus=num_gpus),
    ]
    if not phase_chunks:
        return {"makespan": 0.0, "schedule": [], "chunk_count": 0, "solver_status": "empty"}

    chunk_count = len(phase_chunks)
    makespan_index = chunk_count
    base_variable_count = chunk_count + 1
    gpu_busy = [0.0] * num_gpus
    for chunk in phase_chunks:
        gpu_busy[chunk.src_gpu] += chunk.size
        gpu_busy[chunk.dst_gpu] += chunk.size
    big_m = max(gpu_busy) + 1.0 if gpu_busy else 1.0

    conflict_pairs: list[tuple[int, int]] = []
    for i in range(chunk_count):
        for j in range(i + 1, chunk_count):
            chunk_i = phase_chunks[i]
            chunk_j = phase_chunks[j]
            if chunk_i.phase != chunk_j.phase:
                continue
            if {chunk_i.src_gpu, chunk_i.dst_gpu}.isdisjoint({chunk_j.src_gpu, chunk_j.dst_gpu}):
                continue
            conflict_pairs.append((i, j))

    binary_offset = base_variable_count
    aux_offset = base_variable_count + len(conflict_pairs)
    aux_variable_count = num_gpus * 2
    total_variable_count = aux_offset + aux_variable_count
    c = np.zeros(total_variable_count, dtype=float)
    c[makespan_index] = 1.0
    lower_bounds = np.zeros(total_variable_count, dtype=float)
    upper_bounds = np.full(total_variable_count, np.inf, dtype=float)
    integrality = np.zeros(total_variable_count, dtype=int)
    for binary_index in range(len(conflict_pairs)):
        integrality[binary_offset + binary_index] = 1
        upper_bounds[binary_offset + binary_index] = 1.0

    rows: list[np.ndarray] = []
    lower_row_bounds: list[float] = []
    upper_row_bounds: list[float] = []

    def add_ub(coeffs: dict[int, float], ub: float) -> None:
        row = np.zeros(total_variable_count, dtype=float)
        for index, value in coeffs.items():
            row[index] = value
        rows.append(row)
        lower_row_bounds.append(-math.inf)
        upper_row_bounds.append(ub)

    for idx, chunk in enumerate(phase_chunks):
        add_ub({idx: 1.0, makespan_index: -1.0}, -float(chunk.size))

    for binary_index, (i, j) in enumerate(conflict_pairs):
        z_index = binary_offset + binary_index
        dur_i = float(phase_chunks[i].size)
        dur_j = float(phase_chunks[j].size)
        add_ub({i: 1.0, j: -1.0, z_index: -big_m}, -dur_i)
        add_ub({j: 1.0, i: -1.0, z_index: big_m}, big_m - dur_j)

    for gpu in range(num_gpus):
        for phase_from, phase_to in ((0, 1), (1, 2)):
            aux_index = aux_offset + gpu * 2 + phase_from
            phase_from_chunks = [
                (idx, chunk)
                for idx, chunk in enumerate(phase_chunks)
                if chunk.phase == phase_from and (chunk.src_gpu == gpu or chunk.dst_gpu == gpu)
            ]
            phase_to_chunks = [
                (idx, chunk)
                for idx, chunk in enumerate(phase_chunks)
                if chunk.phase == phase_to and (chunk.src_gpu == gpu or chunk.dst_gpu == gpu)
            ]
            for from_idx, from_chunk in phase_from_chunks:
                add_ub({from_idx: 1.0, aux_index: -1.0}, -float(from_chunk.size))
            for to_idx, _ in phase_to_chunks:
                add_ub({aux_index: 1.0, to_idx: -1.0}, 0.0)

    constraints = spo.LinearConstraint(
        np.vstack(rows),
        np.asarray(lower_row_bounds, dtype=float),
        np.asarray(upper_row_bounds, dtype=float),
    )
    result = spo.milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=spo.Bounds(lower_bounds, upper_bounds),
        options={"time_limit": 30},
    )
    if not result.success or result.x is None:
        return {
            "makespan": None,
            "schedule": [],
            "chunk_count": chunk_count,
            "solver_status": str(result.message),
        }

    starts = result.x[:chunk_count]
    schedule = []
    for start, chunk in sorted(zip(starts, phase_chunks, strict=False), key=lambda item: float(item[0])):
        schedule.append({**chunk.to_dict(), "start": float(start), "end": float(start) + float(chunk.size)})
    return {
        "makespan": float(result.x[makespan_index]),
        "schedule": schedule,
        "chunk_count": chunk_count,
        "solver_status": str(result.message),
    }


def pairwise_oracle(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> dict[str, Any]:
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return _pairwise_oracle_scipy(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)

    phase_chunks = [
        *_matrix_to_pairwise_chunks(dispatch_matrix, phase=0, num_gpus=num_gpus),
        *_matrix_to_pairwise_chunks(combine_matrix, phase=1, num_gpus=num_gpus),
        *_matrix_to_pairwise_chunks(next_dispatch_matrix, phase=2, num_gpus=num_gpus),
    ]
    if not phase_chunks:
        return {"makespan": 0.0, "schedule": [], "chunk_count": 0, "solver_status": "empty"}

    chunk_count = len(phase_chunks)
    greedy_upper_bound = max(
        1,
        int(math.ceil(greedy_schedule_pairwise(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus))),
    )
    horizon = greedy_upper_bound
    model = cp_model.CpModel()

    starts = []
    ends = []
    intervals = []
    for index, chunk in enumerate(phase_chunks):
        start = model.new_int_var(0, horizon, f"start_{index}")
        end = model.new_int_var(0, horizon, f"end_{index}")
        interval = model.new_interval_var(start, int(chunk.size), end, f"interval_{index}")
        starts.append(start)
        ends.append(end)
        intervals.append(interval)

    makespan = model.new_int_var(0, horizon, "makespan")
    for end in ends:
        model.add(makespan >= end)
    model.minimize(makespan)

    for gpu in range(num_gpus):
        gpu_intervals = [
            intervals[index]
            for index, chunk in enumerate(phase_chunks)
            if chunk.src_gpu == gpu or chunk.dst_gpu == gpu
        ]
        if len(gpu_intervals) > 1:
            model.add_no_overlap(gpu_intervals)

    for gpu in range(num_gpus):
        for phase_from, phase_to in ((0, 1), (1, 2)):
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
            if from_ends and to_starts:
                done = model.new_int_var(0, horizon, f"done_{gpu}_{phase_from}")
                for end in from_ends:
                    model.add(done >= end)
                for start in to_starts:
                    model.add(start >= done)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.2
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    feasible_statuses = {cp_model.OPTIMAL, cp_model.FEASIBLE}
    if status not in feasible_statuses:
        status_name = solver.status_name(status) if hasattr(solver, "status_name") else str(status)
        return {
            "makespan": None,
            "schedule": [],
            "chunk_count": chunk_count,
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
        "chunk_count": chunk_count,
        "solver_status": status_name,
    }
