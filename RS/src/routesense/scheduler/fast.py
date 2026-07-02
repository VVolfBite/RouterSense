from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


_NO_IMPROVE_PATIENCE = 5
_MIN_RELATIVE_IMPROVEMENT = 0.20


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    phase: int
    size: int
    src_gpu: int
    dst_gpu: int


def _matrix_row_sums(matrix: list[list[int]]) -> list[int]:
    return [sum(int(value) for value in row) for row in matrix]


def _matrix_col_sums(matrix: list[list[int]]) -> list[int]:
    return [sum(int(matrix[row][col]) for row in range(len(matrix))) for col in range(len(matrix[0]) if matrix else 0)]


def _phase_matrices(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    return dispatch_matrix, combine_matrix, next_dispatch_matrix


def _extract_phase_orders_from_schedule(schedule: list[dict[str, Any]]) -> dict[int, list[ChunkSpec]]:
    phase_orders: dict[int, list[ChunkSpec]] = {0: [], 1: [], 2: []}
    ordered = sorted(schedule, key=lambda entry: (int(entry["phase"]), float(entry["start"]), float(entry["end"])))
    for entry in ordered:
        phase_orders[int(entry["phase"])].append(
            ChunkSpec(
                chunk_id=str(entry["chunk_id"]),
                phase=int(entry["phase"]),
                size=int(entry["size"]),
                src_gpu=int(entry["src_gpu"]),
                dst_gpu=int(entry["dst_gpu"]),
            )
        )
    return phase_orders


def _phase_completion_times(
    phase_orders: dict[int, list[ChunkSpec]],
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> tuple[dict[int, list[float]], dict[int, list[float]], dict[str, Any]]:
    payload = _schedule_phase_orders(
        phase_orders,
        strategy="tmp_eval",
        solve_time_ms=0.0,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )
    max_gpu = 0
    for chunks in phase_orders.values():
        for chunk in chunks:
            max_gpu = max(max_gpu, chunk.src_gpu, chunk.dst_gpu)
    num_gpus = max_gpu + 1 if phase_orders and max_gpu >= 0 else 0
    send_done: dict[int, list[float]] = {0: [0.0] * num_gpus, 1: [0.0] * num_gpus, 2: [0.0] * num_gpus}
    recv_done: dict[int, list[float]] = {0: [0.0] * num_gpus, 1: [0.0] * num_gpus, 2: [0.0] * num_gpus}
    for entry in payload["schedule"]:
        phase = int(entry["phase"])
        src = int(entry["src_gpu"])
        dst = int(entry["dst_gpu"])
        end = float(entry["end"])
        send_done[phase][src] = max(send_done[phase][src], end)
        recv_done[phase][dst] = max(recv_done[phase][dst], end)
    return send_done, recv_done, payload


def _evaluate_phase_orders(
    phase_orders: dict[int, list[ChunkSpec]],
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> float:
    return float(
        _schedule_phase_orders(
            phase_orders,
            strategy="tmp_eval",
            solve_time_ms=0.0,
            model=model,
            expert_compute_delay=expert_compute_delay,
        )["makespan"]
    )


def _random_swap_orders(
    phase_orders: dict[int, list[ChunkSpec]],
    rng: random.Random,
    *,
    only_phase: int | None = None,
) -> dict[int, list[ChunkSpec]]:
    candidate = _clone_phase_orders(phase_orders)
    phases = [only_phase] if only_phase is not None else [phase for phase, chunks in candidate.items() if len(chunks) >= 2]
    phases = [phase for phase in phases if phase is not None and len(candidate[phase]) >= 2]
    if not phases:
        return candidate
    phase = rng.choice(phases)
    i, j = sorted(rng.sample(range(len(candidate[phase])), 2))
    candidate[phase][i], candidate[phase][j] = candidate[phase][j], candidate[phase][i]
    return candidate


def _gpu_completion(phase_orders: dict[int, list[ChunkSpec]], *, model: str = "full_duplex", expert_compute_delay: float = 0.0) -> dict[int, float]:
    send_done, recv_done, _payload = _phase_completion_times(
        phase_orders,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )
    num_gpus = len(next(iter(send_done.values()), []))
    return {
        gpu: max(
            max(send_done[phase][gpu] for phase in send_done),
            max(recv_done[phase][gpu] for phase in recv_done),
        )
        for gpu in range(num_gpus)
    }


def _collect_phase_chunks(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> dict[int, list[ChunkSpec]]:
    phase_chunks: dict[int, list[ChunkSpec]] = {0: [], 1: [], 2: []}
    for phase, matrix in enumerate((dispatch_matrix, combine_matrix, next_dispatch_matrix)):
        for src in range(num_gpus):
            for dst in range(num_gpus):
                size = int(matrix[src][dst])
                if src == dst or size <= 0:
                    continue
                phase_chunks[phase].append(
                    ChunkSpec(
                        chunk_id=f"phase{phase}_src{src}_dst{dst}",
                        phase=phase,
                        size=size,
                        src_gpu=src,
                        dst_gpu=dst,
                    )
                )
    return phase_chunks


def _schedule_phase_orders(
    phase_orders: dict[int, list[ChunkSpec]],
    *,
    strategy: str,
    solve_time_ms: float,
    priority_lookup: dict[str, tuple[float, ...]] | None = None,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    max_gpu = 0
    for chunks in phase_orders.values():
        for chunk in chunks:
            max_gpu = max(max_gpu, chunk.src_gpu, chunk.dst_gpu)
    num_gpus = max_gpu + 1 if phase_orders and max_gpu >= 0 else 0
    sender_available = [0.0] * num_gpus
    receiver_available = [0.0] * num_gpus
    phase_receiver_done: dict[int, list[float]] = {0: [0.0] * num_gpus, 1: [0.0] * num_gpus}
    schedule: list[dict[str, Any]] = []
    for phase in sorted(phase_orders):
        if phase == 1 and model == "half_duplex" and expert_compute_delay > 0.0:
            sender_available = [value + expert_compute_delay for value in sender_available]
            receiver_available = [value + expert_compute_delay for value in receiver_available]
        for chunk in phase_orders[phase]:
            if model == "half_duplex":
                start = max(sender_available[chunk.src_gpu], receiver_available[chunk.dst_gpu])
            elif model == "full_duplex":
                start = max(sender_available[chunk.src_gpu], receiver_available[chunk.dst_gpu])
                if phase > 0:
                    delay = expert_compute_delay if phase == 1 else 0.0
                    start = max(start, phase_receiver_done[phase - 1][chunk.src_gpu] + delay)
            elif model == "incast_only":
                start = receiver_available[chunk.dst_gpu]
                if phase > 0:
                    delay = expert_compute_delay if phase == 1 else 0.0
                    start = max(start, phase_receiver_done[phase - 1][chunk.src_gpu] + delay)
            else:
                raise ValueError(f"Unsupported scheduling model: {model}")
            end = start + float(chunk.size)
            if model in {"half_duplex", "full_duplex"}:
                sender_available[chunk.src_gpu] = end
            receiver_available[chunk.dst_gpu] = end
            if phase in phase_receiver_done:
                phase_receiver_done[phase][chunk.dst_gpu] = max(phase_receiver_done[phase][chunk.dst_gpu], end)
            schedule.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "phase": chunk.phase,
                    "size": chunk.size,
                    "src": chunk.src_gpu,
                    "dst": chunk.dst_gpu,
                    "src_gpu": chunk.src_gpu,
                    "dst_gpu": chunk.dst_gpu,
                    "start": start,
                    "end": end,
                    "priority": list(priority_lookup.get(chunk.chunk_id, ())) if priority_lookup else [],
                }
            )
    if model == "half_duplex":
        makespan = max(max(sender_available[g], receiver_available[g]) for g in range(num_gpus)) if num_gpus else 0.0
    elif model == "full_duplex":
        makespan = max(sender_available + receiver_available) if num_gpus else 0.0
    else:
        makespan = max(receiver_available) if receiver_available else 0.0
    return {
        "makespan": makespan,
        "schedule": schedule,
        "solve_time_ms": solve_time_ms,
        "strategy": strategy,
    }


def _critical_path_weights(phase_chunks: dict[int, list[ChunkSpec]], num_gpus: int) -> dict[str, float]:
    later_work = [[0.0 for _ in range(4)] for _ in range(num_gpus)]
    for phase in (2, 1, 0):
        for gpu in range(num_gpus):
            later_work[gpu][phase] = later_work[gpu][phase + 1]
        for chunk in phase_chunks.get(phase, []):
            # Note: src and dst later_work are tracked independently per port.
            # In full_duplex, send-port and recv-port are separate resources,
            # so summing both into the priority signal reflects downstream contention on both ports.
            later_work[chunk.src_gpu][phase] += chunk.size
            later_work[chunk.dst_gpu][phase] += chunk.size
    weights: dict[str, float] = {}
    for phase, chunks in phase_chunks.items():
        for chunk in chunks:
            weights[chunk.chunk_id] = float(
                chunk.size + later_work[chunk.src_gpu][phase + 1] + later_work[chunk.dst_gpu][phase + 1]
            )
    return weights


def _phase_orders_cp_lpt(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> tuple[dict[int, list[ChunkSpec]], dict[str, tuple[float, ...]]]:
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    critical = _critical_path_weights(phase_chunks, num_gpus)
    priority_lookup: dict[str, tuple[float, ...]] = {}
    phase_orders: dict[int, list[ChunkSpec]] = {}
    for phase, chunks in phase_chunks.items():
        phase_orders[phase] = sorted(
            chunks,
            key=lambda chunk: (
                critical[chunk.chunk_id],
                chunk.size,
                -chunk.src_gpu,
                -chunk.dst_gpu,
            ),
            reverse=True,
        )
        for chunk in phase_orders[phase]:
            priority_lookup[chunk.chunk_id] = (
                critical[chunk.chunk_id],
                float(chunk.size),
                float(-chunk.src_gpu),
                float(-chunk.dst_gpu),
            )
    return phase_orders, priority_lookup


def _phase_orders_lookahead_lpt(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> tuple[dict[int, list[ChunkSpec]], dict[str, tuple[float, ...]]]:
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    next_inbound = [sum(int(next_dispatch_matrix[src][dst]) for src in range(num_gpus)) for dst in range(num_gpus)]
    next_outbound = [sum(int(next_dispatch_matrix[src][dst]) for dst in range(num_gpus)) for src in range(num_gpus)]
    priority_lookup: dict[str, tuple[float, ...]] = {}
    phase_orders: dict[int, list[ChunkSpec]] = {
        0: sorted(
            phase_chunks[0],
            key=lambda chunk: (next_inbound[chunk.dst_gpu], chunk.size, -chunk.src_gpu, -chunk.dst_gpu),
            reverse=True,
        ),
        1: sorted(
            phase_chunks[1],
            key=lambda chunk: (next_outbound[chunk.dst_gpu], chunk.size, -chunk.src_gpu, -chunk.dst_gpu),
            reverse=True,
        ),
        2: sorted(
            phase_chunks[2],
            key=lambda chunk: (chunk.size, -chunk.src_gpu, -chunk.dst_gpu),
            reverse=True,
        ),
    }
    for phase, chunks in phase_orders.items():
        for chunk in chunks:
            if phase == 0:
                priority_lookup[chunk.chunk_id] = (
                    float(next_inbound[chunk.dst_gpu]),
                    float(chunk.size),
                )
            elif phase == 1:
                priority_lookup[chunk.chunk_id] = (
                    float(next_outbound[chunk.dst_gpu]),
                    float(chunk.size),
                )
            else:
                priority_lookup[chunk.chunk_id] = (float(chunk.size),)
    return phase_orders, priority_lookup


def fast_schedule_lookahead_lpt(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    phase_orders, priority_lookup = _phase_orders_lookahead_lpt(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
    )
    return _schedule_phase_orders(
        phase_orders,
        strategy="lookahead_lpt",
        solve_time_ms=(time.perf_counter() - start) * 1000.0,
        priority_lookup=priority_lookup,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )


def fast_schedule_cp_lpt(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    """Critical-path LPT ordering.

    prediction_aware: True
        Uses next_dispatch_matrix values through _critical_path_weights() to
        compute downstream later_work for phase-0/1 prioritization.
    """
    start = time.perf_counter()
    phase_orders, priority_lookup = _phase_orders_cp_lpt(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    return _schedule_phase_orders(
        phase_orders,
        strategy="cp_lpt",
        solve_time_ms=(time.perf_counter() - start) * 1000.0,
        priority_lookup=priority_lookup,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )


def _birkhoff_decompose(matrix: list[list[int]]) -> list[tuple[int, list[tuple[int, int]]]]:
    residual = np.array(matrix, dtype=int)
    rounds: list[tuple[int, list[tuple[int, int]]]] = []
    while residual.max() > 0:
        edges: list[tuple[int, int]]
        if linear_sum_assignment is not None:
            row_idx, col_idx = linear_sum_assignment(-residual)
            edges = [(int(row), int(col)) for row, col in zip(row_idx, col_idx, strict=False) if residual[row, col] > 0 and row != col]
        else:  # pragma: no cover
            edges = []
            used_rows: set[int] = set()
            used_cols: set[int] = set()
            flat = sorted(
                ((int(residual[row, col]), row, col) for row in range(residual.shape[0]) for col in range(residual.shape[1])),
                reverse=True,
            )
            for value, row, col in flat:
                if value <= 0 or row == col or row in used_rows or col in used_cols:
                    continue
                edges.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
        if not edges:
            break
        weight = min(int(residual[row, col]) for row, col in edges)
        rounds.append((weight, edges))
        for row, col in edges:
            residual[row, col] -= weight
    return rounds


def _rounds_to_phase_order(rounds: list[tuple[int, list[tuple[int, int]]]], *, phase: int) -> list[ChunkSpec]:
    order: list[ChunkSpec] = []
    for round_index, (weight, edges) in enumerate(rounds):
        for src, dst in edges:
            order.append(
                ChunkSpec(
                    chunk_id=f"phase{phase}_round{round_index}_src{src}_dst{dst}",
                    phase=phase,
                    size=int(weight),
                    src_gpu=src,
                    dst_gpu=dst,
                )
            )
    return order


def _birkhoff_round_rank(
    rounds: list[tuple[int, list[tuple[int, int]]]],
    chunks: list[ChunkSpec],
    *,
    phase: int,
) -> dict[tuple[int, int, int], tuple[int, int]]:
    seen: dict[tuple[int, int], int] = {}
    for round_index, (_weight, edges) in enumerate(rounds):
        for src, dst in edges:
            seen.setdefault((src, dst), round_index)
    return {
        (phase, chunk.src_gpu, chunk.dst_gpu): (
            seen.get((chunk.src_gpu, chunk.dst_gpu), len(rounds)),
            -chunk.size,
        )
        for chunk in chunks
    }


def _phase_order_from_round_rank(
    chunks: list[ChunkSpec],
    round_rank: dict[tuple[int, int, int], tuple[int, int]],
) -> tuple[list[ChunkSpec], dict[str, tuple[float, ...]]]:
    ordered = sorted(
        chunks,
        key=lambda chunk: (
            -round_rank[(chunk.phase, chunk.src_gpu, chunk.dst_gpu)][0],
            chunk.size,
            -chunk.src_gpu,
            -chunk.dst_gpu,
        ),
        reverse=True,
    )
    priority_lookup = {
        chunk.chunk_id: (
            float(round_rank[(chunk.phase, chunk.src_gpu, chunk.dst_gpu)][0]),
            float(chunk.size),
        )
        for chunk in ordered
    }
    return ordered, priority_lookup


def _orders_from_round_permutation(
    chunks: list[ChunkSpec],
    rounds: list[tuple[int, list[tuple[int, int]]]],
    round_order: list[int],
    *,
    phase: int,
) -> tuple[list[ChunkSpec], dict[str, tuple[float, ...]]]:
    remapped: dict[tuple[int, int], int] = {}
    for new_index, old_index in enumerate(round_order):
        _weight, edges = rounds[old_index]
        for src, dst in edges:
            remapped.setdefault((src, dst), new_index)
    round_rank = {
        (phase, chunk.src_gpu, chunk.dst_gpu): (
            remapped.get((chunk.src_gpu, chunk.dst_gpu), len(rounds)),
            -chunk.size,
        )
        for chunk in chunks
    }
    return _phase_order_from_round_rank(chunks, round_rank)


def _sample_round_permutations(
    rounds: list[tuple[int, list[tuple[int, int]]]],
    *,
    rng: random.Random,
    max_permutations: int,
) -> list[list[int]]:
    round_count = len(rounds)
    if round_count <= 1:
        return [list(range(round_count))]
    if round_count <= 8:
        all_perms = list(itertools.permutations(range(round_count)))
        if len(all_perms) <= max_permutations:
            return [list(perm) for perm in all_perms]
    sampled: list[list[int]] = [list(range(round_count))]
    seen = {tuple(sampled[0])}
    while len(sampled) < max_permutations:
        perm = list(range(round_count))
        rng.shuffle(perm)
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        sampled.append(perm)
    return sampled


def _phase_workload_by_gpu(phase_orders: dict[int, list[ChunkSpec]], gpu: int, phase: int) -> int:
    return sum(
        chunk.size
        for chunk in phase_orders.get(phase, [])
        if chunk.src_gpu == gpu or chunk.dst_gpu == gpu
    )


def _phase_workload_map(phase_orders: dict[int, list[ChunkSpec]], num_gpus: int) -> dict[tuple[int, int], int]:
    return {
        (phase, gpu): _phase_workload_by_gpu(phase_orders, gpu, phase)
        for phase in range(3)
        for gpu in range(num_gpus)
    }


def _ejection_chain_candidate(
    phase_orders: dict[int, list[ChunkSpec]],
    *,
    g_star: int,
    f_star: int,
    model: str,
    expert_compute_delay: float,
) -> dict[int, list[ChunkSpec]]:
    candidate = _clone_phase_orders(phase_orders)
    target_chunks = list(candidate.get(f_star, []))
    ejected = [chunk for chunk in target_chunks if chunk.src_gpu == g_star or chunk.dst_gpu == g_star]
    if not ejected:
        return candidate
    candidate[f_star] = [chunk for chunk in target_chunks if chunk not in ejected]
    frontier = list(ejected)
    touched = {g_star}
    for _level in range(2):
        affected_gpus: set[int] = set()
        for chunk in frontier:
            for remaining in candidate[f_star]:
                if (
                    remaining.src_gpu in {chunk.src_gpu, chunk.dst_gpu}
                    or remaining.dst_gpu in {chunk.src_gpu, chunk.dst_gpu}
                ):
                    affected_gpus.add(remaining.src_gpu)
                    affected_gpus.add(remaining.dst_gpu)
        affected_gpus.difference_update(touched)
        if not affected_gpus:
            break
        g_aff = max(
            affected_gpus,
            key=lambda gpu: sum(
                item.size
                for item in candidate[f_star]
                if item.src_gpu == gpu or item.dst_gpu == gpu
            ),
        )
        touched.add(g_aff)
        frontier = [chunk for chunk in frontier if chunk.src_gpu == g_aff or chunk.dst_gpu == g_aff] or frontier
        for chunk in sorted(frontier, key=lambda item: item.size, reverse=True):
            candidate = _best_insert_position(candidate, chunk)
    for chunk in sorted(ejected, key=lambda item: item.size, reverse=True):
        candidate = _best_insert_position(candidate, chunk)
    return candidate


def _cp_sat_order_phase_chunks(
    chunks: list[ChunkSpec],
    *,
    num_gpus: int,
    timeout_ms: float,
    model: str,
) -> list[ChunkSpec] | None:
    if len(chunks) <= 1:
        return list(chunks)
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None

    horizon = max(1, sum(chunk.size for chunk in chunks))
    sat_model = cp_model.CpModel()
    starts = []
    ends = []
    intervals = []
    for index, chunk in enumerate(chunks):
        start = sat_model.new_int_var(0, horizon, f"s_{index}")
        end = sat_model.new_int_var(0, horizon, f"e_{index}")
        interval = sat_model.new_interval_var(start, int(chunk.size), end, f"iv_{index}")
        starts.append(start)
        ends.append(end)
        intervals.append(interval)
    makespan = sat_model.new_int_var(0, horizon, "mk")
    for end in ends:
        sat_model.add(makespan >= end)
    sat_model.minimize(makespan)

    for gpu in range(num_gpus):
        if model == "half_duplex":
            constrained = [
                intervals[index]
                for index, chunk in enumerate(chunks)
                if chunk.src_gpu == gpu or chunk.dst_gpu == gpu
            ]
            if len(constrained) > 1:
                sat_model.add_no_overlap(constrained)
            continue
        sender_intervals = [
            intervals[index]
            for index, chunk in enumerate(chunks)
            if chunk.src_gpu == gpu
        ]
        receiver_intervals = [
            intervals[index]
            for index, chunk in enumerate(chunks)
            if chunk.dst_gpu == gpu
        ]
        if model == "full_duplex":
            if len(sender_intervals) > 1:
                sat_model.add_no_overlap(sender_intervals)
        if len(receiver_intervals) > 1:
            sat_model.add_no_overlap(receiver_intervals)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(timeout_ms, 0.1) / 1000.0
    solver.parameters.num_search_workers = 4
    solver.parameters.random_seed = 0
    status = solver.solve(sat_model)
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        return None
    return [
        chunk
        for _start, chunk in sorted(
            ((solver.value(starts[index]), chunk) for index, chunk in enumerate(chunks)),
            key=lambda item: (item[0], -item[1].size, item[1].src_gpu, item[1].dst_gpu),
        )
    ]


def _phase_orders_from_schedule_chunks(
    schedule: list[dict[str, Any]],
    phase_chunks: dict[int, list[ChunkSpec]],
) -> dict[int, list[ChunkSpec]]:
    by_id = {
        chunk.chunk_id: chunk
        for chunks in phase_chunks.values()
        for chunk in chunks
    }
    phase_orders: dict[int, list[ChunkSpec]] = {0: [], 1: [], 2: []}
    for entry in sorted(schedule, key=lambda item: (int(item["phase"]), float(item["start"]), float(item["end"]))):
        chunk = by_id.get(str(entry["chunk_id"]))
        if chunk is not None:
            phase_orders[chunk.phase].append(chunk)
    for phase in phase_orders:
        existing_ids = {chunk.chunk_id for chunk in phase_orders[phase]}
        for chunk in phase_chunks.get(phase, []):
            if chunk.chunk_id not in existing_ids:
                phase_orders[phase].append(chunk)
    return phase_orders


def _merge_subproblem_orders(
    bucket_orders: list[tuple[float, list[ChunkSpec]]],
) -> list[ChunkSpec]:
    """Merge subproblem orders by sender grouping while preserving local order."""

    all_chunks_with_meta: list[tuple[ChunkSpec, int, int]] = []
    for bucket_index, (_score, ordered) in enumerate(bucket_orders):
        for local_rank, chunk in enumerate(ordered):
            all_chunks_with_meta.append((chunk, bucket_index, local_rank))
    all_chunks_with_meta.sort(key=lambda item: (item[0].src_gpu, item[1], item[2]))
    return [chunk for chunk, _bucket, _local in all_chunks_with_meta]


def _subproblem_oracle_phase_order(
    phase_chunks: dict[int, list[ChunkSpec]],
    *,
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    model: str,
    expert_compute_delay: float,
    timeout_ms: float,
) -> dict[int, list[ChunkSpec]] | None:
    try:
        from .oracle import pairwise_oracle
    except Exception:
        return None
    payload = pairwise_oracle(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )
    if not payload.get("schedule"):
        return None
    return _phase_orders_from_schedule_chunks(payload["schedule"], phase_chunks)


def fast_schedule_birkhoff(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    """Per-phase Birkhoff decomposition with barrier-aware simulation.

    prediction_aware: False
        next_dispatch_matrix only defines phase 2 itself; its values do not
        influence phase-0/1 ordering decisions.
    """
    start = time.perf_counter()
    dispatch_rounds = _birkhoff_decompose(dispatch_matrix)
    combine_rounds = _birkhoff_decompose(combine_matrix)
    next_rounds = _birkhoff_decompose(next_dispatch_matrix)
    # Keep original chunk granularity for fair comparison against greedy/oracle.
    # Birkhoff here is used only to rank original chunks by their earliest round.
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    round_rank: dict[tuple[int, int, int], tuple[int, int]] = {}
    for rounds in (dispatch_rounds, combine_rounds, next_rounds):
        pass
    for phase, rounds in ((0, dispatch_rounds), (1, combine_rounds), (2, next_rounds)):
        seen: dict[tuple[int, int], int] = {}
        for round_index, (weight, edges) in enumerate(rounds):
            for src, dst in edges:
                seen.setdefault((src, dst), round_index)
        for chunk in phase_chunks[phase]:
            round_rank[(phase, chunk.src_gpu, chunk.dst_gpu)] = (
                seen.get((chunk.src_gpu, chunk.dst_gpu), len(rounds)),
                -chunk.size,
            )
    phase_orders = {}
    for phase, chunks in phase_chunks.items():
        # Birkhoff round_rank: lower index = higher weight matching = schedule first.
        # -round_rank with reverse=True yields ascending round_rank, so early rounds run first.
        phase_orders[phase] = sorted(
            chunks,
            key=lambda chunk: (
                -round_rank[(phase, chunk.src_gpu, chunk.dst_gpu)][0],
                chunk.size,
                -chunk.src_gpu,
                -chunk.dst_gpu,
            ),
            reverse=True,
        )
    priority_lookup = {
        chunk.chunk_id: (
            float(round_rank[(chunk.phase, chunk.src_gpu, chunk.dst_gpu)][0]),
            float(chunk.size),
        )
        for chunks in phase_orders.values()
        for chunk in chunks
    }
    return _schedule_phase_orders(
        phase_orders,
        strategy="birkhoff",
        solve_time_ms=(time.perf_counter() - start) * 1000.0,
        priority_lookup=priority_lookup,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )


def fast_schedule_phase_aware_greedy(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    downstream = [[0.0] * 3 for _ in range(num_gpus)]
    for phase in (2, 1, 0):
        if phase < 2:
            for gpu in range(num_gpus):
                downstream[gpu][phase] = downstream[gpu][phase + 1]
        for chunk in phase_chunks.get(phase, []):
            downstream[chunk.dst_gpu][phase] += chunk.size
    alpha = 0.5
    priority_lookup: dict[str, tuple[float, ...]] = {}
    for phase in range(3):
        phase_chunks[phase].sort(
            key=lambda chunk: (
                chunk.size + alpha * downstream[chunk.dst_gpu][min(phase + 1, 2)],
                -chunk.src_gpu,
                -chunk.dst_gpu,
            ),
            reverse=True,
        )
        for chunk in phase_chunks[phase]:
            priority_lookup[chunk.chunk_id] = (
                float(chunk.size + alpha * downstream[chunk.dst_gpu][min(phase + 1, 2)]),
                float(chunk.size),
            )
    return _schedule_phase_orders(
        phase_chunks,
        strategy="phase_aware_greedy",
        solve_time_ms=(time.perf_counter() - start) * 1000.0,
        priority_lookup=priority_lookup,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )


def fast_schedule_barrier_aware_birkhoff(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    """Searches among Birkhoff variants while keeping phase-local optimality.

    prediction_aware: False
        It evaluates the three-phase schedule jointly, but phase-2 values do
        not directly guide phase-0/1 priorities beyond candidate evaluation.
    """
    start = time.perf_counter()
    rng = random.Random(0)
    matrices = _phase_matrices(dispatch_matrix, combine_matrix, next_dispatch_matrix)
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    phase_candidates: dict[int, list[tuple[list[ChunkSpec], dict[str, tuple[float, ...]]]]] = {0: [], 1: [], 2: []}
    for phase, matrix in enumerate(matrices):
        for variant in range(3):
            perturbed = np.array(matrix, dtype=float)
            noise = np.array([[rng.uniform(-0.1, 0.1) for _ in row] for row in matrix], dtype=float)
            perturbed = np.maximum(perturbed + noise, 0.0)
            scaled = np.rint(perturbed * 100.0).astype(int).tolist()
            rounds = _birkhoff_decompose(scaled)
            round_rank = _birkhoff_round_rank(rounds, phase_chunks[phase], phase=phase)
            order, lookup = _phase_order_from_round_rank(phase_chunks[phase], round_rank)
            phase_candidates[phase].append((order, lookup))
    best_payload: dict[str, Any] | None = None
    for phase0_orders, lookup0 in phase_candidates[0]:
        for phase1_orders, lookup1 in phase_candidates[1]:
            for phase2_orders, lookup2 in phase_candidates[2]:
                merged = {
                    0: phase0_orders,
                    1: phase1_orders,
                    2: phase2_orders,
                }
                merged_lookup = {}
                merged_lookup.update(lookup0)
                merged_lookup.update(lookup1)
                merged_lookup.update(lookup2)
                payload = _schedule_phase_orders(
                    merged,
                    strategy="barrier_aware_birkhoff",
                    solve_time_ms=0.0,
                    priority_lookup=merged_lookup,
                    model=model,
                    expert_compute_delay=expert_compute_delay,
                )
                if best_payload is None or float(payload["makespan"]) < float(best_payload["makespan"]):
                    best_payload = payload
    assert best_payload is not None
    best_payload["solve_time_ms"] = (time.perf_counter() - start) * 1000.0
    best_payload["strategy"] = "barrier_aware_birkhoff"
    return best_payload


def fast_schedule_randomized_multistart_birkhoff(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    rng = random.Random(1)
    matrices = _phase_matrices(dispatch_matrix, combine_matrix, next_dispatch_matrix)
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    phase_variants: dict[int, list[tuple[list[ChunkSpec], dict[str, tuple[float, ...]]]]] = {0: [], 1: [], 2: []}
    for phase, matrix in enumerate(matrices):
        for _ in range(5):
            perturbed = np.array(matrix, dtype=float)
            noise = np.array([[rng.uniform(-0.25, 0.25) for _ in row] for row in matrix], dtype=float)
            perturbed = np.maximum(perturbed + noise, 0.0)
            scaled = np.rint(perturbed * 50.0).astype(int).tolist()
            rounds = _birkhoff_decompose(scaled)
            round_rank = _birkhoff_round_rank(rounds, phase_chunks[phase], phase=phase)
            order, lookup = _phase_order_from_round_rank(phase_chunks[phase], round_rank)
            phase_variants[phase].append((order, lookup))
    best_payload: dict[str, Any] | None = None
    for orders0, lookup0 in phase_variants[0]:
        for orders1, lookup1 in phase_variants[1]:
            for orders2, lookup2 in phase_variants[2]:
                merged = {0: orders0, 1: orders1, 2: orders2}
                merged_lookup = {}
                merged_lookup.update(lookup0)
                merged_lookup.update(lookup1)
                merged_lookup.update(lookup2)
                payload = _schedule_phase_orders(
                    merged,
                    strategy="randomized_multistart_birkhoff",
                    solve_time_ms=0.0,
                    priority_lookup=merged_lookup,
                    model=model,
                    expert_compute_delay=expert_compute_delay,
                )
                if best_payload is None or float(payload["makespan"]) < float(best_payload["makespan"]):
                    best_payload = payload
    assert best_payload is not None
    best_payload["solve_time_ms"] = (time.perf_counter() - start) * 1000.0
    best_payload["strategy"] = "randomized_multistart_birkhoff"
    return best_payload


def _clone_phase_orders(phase_orders: dict[int, list[ChunkSpec]]) -> dict[int, list[ChunkSpec]]:
    return {phase: list(chunks) for phase, chunks in phase_orders.items()}


def _best_insert_position(
    phase_orders: dict[int, list[ChunkSpec]],
    chunk: ChunkSpec,
) -> dict[int, list[ChunkSpec]]:
    best_orders: dict[int, list[ChunkSpec]] | None = None
    best_makespan = float("inf")
    target_phase = chunk.phase
    for position in range(len(phase_orders[target_phase]) + 1):
        candidate = _clone_phase_orders(phase_orders)
        candidate[target_phase].insert(position, chunk)
        makespan = float(_schedule_phase_orders(candidate, strategy="tmp", solve_time_ms=0.0)["makespan"])
        if makespan < best_makespan:
            best_makespan = makespan
            best_orders = candidate
    return best_orders if best_orders is not None else phase_orders


def fast_schedule_iterated_greedy(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    iterations: int = 10,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    base_orders, priority_lookup = _phase_orders_cp_lpt(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    best_orders = _clone_phase_orders(base_orders)
    best_payload = _schedule_phase_orders(
        best_orders,
        strategy="iterated_greedy",
        solve_time_ms=0.0,
        priority_lookup=priority_lookup,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )
    rng = random.Random(0)
    flattened = [(phase, chunk) for phase, chunks in best_orders.items() for chunk in chunks]
    destroy_count = max(3, len(flattened) // 10) if flattened else 0
    no_improve_count = 0
    prev_best = float(best_payload["makespan"])
    for _ in range(iterations):
        candidate_orders = _clone_phase_orders(best_orders)
        flat_candidate = [(phase, chunk) for phase, chunks in candidate_orders.items() for chunk in chunks]
        if len(flat_candidate) <= destroy_count or destroy_count == 0:
            break
        removed_pairs = rng.sample(flat_candidate, destroy_count)
        removed_chunks = [chunk for _, chunk in removed_pairs]
        removed_ids = {chunk.chunk_id for chunk in removed_chunks}
        for phase in candidate_orders:
            candidate_orders[phase] = [chunk for chunk in candidate_orders[phase] if chunk.chunk_id not in removed_ids]
        removed_chunks.sort(key=lambda chunk: chunk.size, reverse=True)
        for chunk in removed_chunks:
            candidate_orders = _best_insert_position(candidate_orders, chunk)
        candidate_payload = _schedule_phase_orders(
            candidate_orders,
            strategy="iterated_greedy",
            solve_time_ms=0.0,
            priority_lookup=priority_lookup,
            model=model,
            expert_compute_delay=expert_compute_delay,
        )
        if float(candidate_payload["makespan"]) < float(best_payload["makespan"]):
            relative = (prev_best - float(candidate_payload["makespan"])) / max(prev_best, 1e-9)
            no_improve_count = 0 if relative >= _MIN_RELATIVE_IMPROVEMENT else no_improve_count + 1
            prev_best = float(candidate_payload["makespan"])
            best_orders = candidate_orders
            best_payload = candidate_payload
        else:
            no_improve_count += 1
        if no_improve_count >= _NO_IMPROVE_PATIENCE:
            break
    best_payload["solve_time_ms"] = (time.perf_counter() - start) * 1000.0
    return best_payload


def fast_schedule_tabu_search(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    base = fast_schedule_birkhoff(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay)
    current = _extract_phase_orders_from_schedule(base["schedule"])
    best = _clone_phase_orders(current)
    best_makespan = _evaluate_phase_orders(best, model=model, expert_compute_delay=expert_compute_delay)
    tabu: dict[tuple[str, str], int] = {}
    tenure = 7
    iteration = 0
    no_improve_count = 0
    prev_best = best_makespan
    while time.perf_counter() - start <= 0.004:
        iteration += 1
        best_neighbor = None
        best_neighbor_makespan = float("inf")
        best_pair: tuple[str, str] | None = None
        for phase, chunks in current.items():
            for i in range(len(chunks)):
                for j in range(i + 1, len(chunks)):
                    candidate = _clone_phase_orders(current)
                    candidate[phase][i], candidate[phase][j] = candidate[phase][j], candidate[phase][i]
                    pair = tuple(sorted((chunks[i].chunk_id, chunks[j].chunk_id)))
                    makespan = _evaluate_phase_orders(candidate, model=model, expert_compute_delay=expert_compute_delay)
                    is_tabu = tabu.get(pair, 0) > 0
                    if (not is_tabu or makespan < best_makespan) and makespan < best_neighbor_makespan:
                        best_neighbor = candidate
                        best_neighbor_makespan = makespan
                        best_pair = pair
        if best_neighbor is None or best_pair is None:
            break
        for key in list(tabu):
            tabu[key] -= 1
            if tabu[key] <= 0:
                del tabu[key]
        tabu[best_pair] = tenure
        current = best_neighbor
        if best_neighbor_makespan < best_makespan:
            relative = (prev_best - best_neighbor_makespan) / max(prev_best, 1e-9)
            no_improve_count = 0 if relative >= _MIN_RELATIVE_IMPROVEMENT else no_improve_count + 1
            prev_best = best_neighbor_makespan
            best = _clone_phase_orders(best_neighbor)
            best_makespan = best_neighbor_makespan
        else:
            no_improve_count += 1
        if iteration >= 32:
            break
        if no_improve_count >= _NO_IMPROVE_PATIENCE:
            break
    payload = _schedule_phase_orders(best, strategy="tabu_search", solve_time_ms=(time.perf_counter() - start) * 1000.0, model=model, expert_compute_delay=expert_compute_delay)
    return payload


def fast_schedule_lns(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    base = fast_schedule_birkhoff(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay)
    best_orders = _extract_phase_orders_from_schedule(base["schedule"])
    best_makespan = _evaluate_phase_orders(best_orders, model=model, expert_compute_delay=expert_compute_delay)
    no_improve_count = 0
    prev_best = best_makespan
    while time.perf_counter() - start <= 0.003:
        gpu_completion = _gpu_completion(best_orders, model=model, expert_compute_delay=expert_compute_delay)
        bottleneck_gpu = max(gpu_completion, key=gpu_completion.get)
        target_phase = max(
            range(3),
            key=lambda phase: sum(chunk.size for chunk in best_orders[phase] if chunk.src_gpu == bottleneck_gpu or chunk.dst_gpu == bottleneck_gpu),
        )
        candidate_orders = _clone_phase_orders(best_orders)
        removed = [chunk for chunk in candidate_orders[target_phase] if chunk.src_gpu == bottleneck_gpu or chunk.dst_gpu == bottleneck_gpu]
        if not removed:
            break
        removed_ids = {chunk.chunk_id for chunk in removed}
        candidate_orders[target_phase] = [chunk for chunk in candidate_orders[target_phase] if chunk.chunk_id not in removed_ids]
        removed.sort(key=lambda chunk: chunk.size, reverse=True)
        for chunk in removed:
            candidate_orders = _best_insert_position(candidate_orders, chunk)
        makespan = _evaluate_phase_orders(candidate_orders, model=model, expert_compute_delay=expert_compute_delay)
        if makespan < best_makespan:
            relative = (prev_best - makespan) / max(prev_best, 1e-9)
            no_improve_count = 0 if relative >= _MIN_RELATIVE_IMPROVEMENT else no_improve_count + 1
            prev_best = makespan
            best_orders = candidate_orders
            best_makespan = makespan
        else:
            no_improve_count += 1
        if no_improve_count >= _NO_IMPROVE_PATIENCE:
            break
    return _schedule_phase_orders(best_orders, strategy="lns", solve_time_ms=(time.perf_counter() - start) * 1000.0, model=model, expert_compute_delay=expert_compute_delay)


def fast_schedule_simulated_annealing(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    rng = random.Random(7)
    base = fast_schedule_birkhoff(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay)
    current = _extract_phase_orders_from_schedule(base["schedule"])
    current_makespan = _evaluate_phase_orders(current, model=model, expert_compute_delay=expert_compute_delay)
    best = _clone_phase_orders(current)
    best_makespan = current_makespan
    temperature = max(current_makespan * 0.1, 1.0)
    no_improve_count = 0
    prev_best = best_makespan
    while time.perf_counter() - start <= 0.003:
        neighbor = _random_swap_orders(current, rng)
        makespan = _evaluate_phase_orders(neighbor, model=model, expert_compute_delay=expert_compute_delay)
        delta = makespan - current_makespan
        if delta < 0 or rng.random() < np.exp(-delta / max(temperature, 1e-9)):
            current = neighbor
            current_makespan = makespan
            if makespan < best_makespan:
                relative = (prev_best - makespan) / max(prev_best, 1e-9)
                no_improve_count = 0 if relative >= _MIN_RELATIVE_IMPROVEMENT else no_improve_count + 1
                prev_best = makespan
                best = _clone_phase_orders(neighbor)
                best_makespan = makespan
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        temperature *= 0.95
        if temperature < 1e-3 or no_improve_count >= _NO_IMPROVE_PATIENCE:
            break
    return _schedule_phase_orders(best, strategy="simulated_annealing", solve_time_ms=(time.perf_counter() - start) * 1000.0, model=model, expert_compute_delay=expert_compute_delay)


def fast_schedule_lagrangian(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    max_iterations: int = 15,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    """Cross-phase Lagrangian ordering over all provided phase matrices.

    prediction_aware: True
        The optimization uses phase-2 matrix values inside the coupled
        three-phase objective and multiplier updates.
    """
    start = time.perf_counter()
    matrices = _phase_matrices(dispatch_matrix, combine_matrix, next_dispatch_matrix)
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    lambda_g = [0.0] * num_gpus
    best_makespan = float("inf")
    best_orders: dict[int, list[ChunkSpec]] | None = None
    best_lookup: dict[str, tuple[float, ...]] = {}
    no_improve_count = 0
    prev_best = float("inf")

    for iteration in range(max_iterations):
        phase_orders: dict[int, list[ChunkSpec]] = {}
        priority_lookup: dict[str, tuple[float, ...]] = {}
        for phase, matrix in enumerate(matrices):
            row_sums = _matrix_row_sums(matrix)
            col_sums = _matrix_col_sums(matrix)
            rounds = _birkhoff_decompose(matrix)
            round_rank = _birkhoff_round_rank(rounds, phase_chunks[phase], phase=phase)
            chunks = sorted(
                phase_chunks[phase],
                key=lambda chunk: (
                    -(
                        round_rank[(phase, chunk.src_gpu, chunk.dst_gpu)][0]
                        - lambda_g[chunk.src_gpu] * row_sums[chunk.src_gpu] * 0.01
                        - lambda_g[chunk.dst_gpu] * col_sums[chunk.dst_gpu] * 0.01
                    ),
                    chunk.size,
                    -chunk.src_gpu,
                    -chunk.dst_gpu,
                ),
                reverse=True,
            )
            phase_orders[phase] = chunks
            for chunk in chunks:
                priority_lookup[chunk.chunk_id] = (
                    float(round_rank[(phase, chunk.src_gpu, chunk.dst_gpu)][0]),
                    float(lambda_g[chunk.src_gpu]),
                    float(lambda_g[chunk.dst_gpu]),
                    float(chunk.size),
                )

        makespan = _evaluate_phase_orders(phase_orders, model=model, expert_compute_delay=expert_compute_delay)
        if makespan < best_makespan:
            relative = 1.0 if prev_best == float("inf") else (prev_best - makespan) / max(prev_best, 1e-9)
            no_improve_count = 0 if relative >= _MIN_RELATIVE_IMPROVEMENT else no_improve_count + 1
            prev_best = makespan
            best_makespan = makespan
            best_orders = _clone_phase_orders(phase_orders)
            best_lookup = dict(priority_lookup)
        else:
            no_improve_count += 1

        send_done, recv_done, _payload = _phase_completion_times(phase_orders, model=model, expert_compute_delay=expert_compute_delay)
        for phase in recv_done:
            if len(recv_done[phase]) < num_gpus:
                recv_done[phase].extend([0.0] * (num_gpus - len(recv_done[phase])))
        for phase in send_done:
            if len(send_done[phase]) < num_gpus:
                send_done[phase].extend([0.0] * (num_gpus - len(send_done[phase])))
        step_size = learning_rate / float(iteration + 1)
        for gpu in range(num_gpus):
            completion0 = recv_done[0][gpu]
            completion1 = recv_done[1][gpu]
            violation = completion1 - completion0 - expert_compute_delay
            lambda_g[gpu] = max(0.0, lambda_g[gpu] + step_size * violation)
        if no_improve_count >= _NO_IMPROVE_PATIENCE:
            break
    solve_time_ms = (time.perf_counter() - start) * 1000.0
    if best_orders is not None:
        payload = _schedule_phase_orders(
            best_orders,
            strategy="lagrangian",
            solve_time_ms=solve_time_ms,
            priority_lookup=best_lookup,
            model=model,
            expert_compute_delay=expert_compute_delay,
        )
        return payload
    return fast_schedule_birkhoff(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )


def fast_schedule_grasp(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    rng = random.Random(11)
    base_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    best_orders: dict[int, list[ChunkSpec]] | None = None
    best_makespan = float("inf")
    no_improve_count = 0
    prev_best = float("inf")
    for _ in range(5):
        if time.perf_counter() - start > 0.003:
            break
        candidate_orders: dict[int, list[ChunkSpec]] = {0: [], 1: [], 2: []}
        for phase in range(3):
            available = list(base_chunks[phase])
            while available:
                priorities = [float(chunk.size) for chunk in available]
                threshold = max(priorities) * 0.7
                candidates = [chunk for chunk, priority in zip(available, priorities, strict=False) if priority >= threshold]
                chosen = rng.choice(candidates)
                candidate_orders[phase].append(chosen)
                available.remove(chosen)
        local = _clone_phase_orders(candidate_orders)
        no_improve = 0
        current = _evaluate_phase_orders(local, model=model, expert_compute_delay=expert_compute_delay)
        while no_improve < 20 and time.perf_counter() - start <= 0.003:
            neighbor = _random_swap_orders(local, rng)
            score = _evaluate_phase_orders(neighbor, model=model, expert_compute_delay=expert_compute_delay)
            if score < current:
                local = neighbor
                current = score
                no_improve = 0
            else:
                no_improve += 1
        if current < best_makespan:
            relative = 1.0 if prev_best == float("inf") else (prev_best - current) / max(prev_best, 1e-9)
            no_improve_count = 0 if relative >= _MIN_RELATIVE_IMPROVEMENT else no_improve_count + 1
            prev_best = current
            best_makespan = current
            best_orders = local
        else:
            no_improve_count += 1
        if no_improve_count >= _NO_IMPROVE_PATIENCE:
            break
    assert best_orders is not None
    return _schedule_phase_orders(best_orders, strategy="grasp", solve_time_ms=(time.perf_counter() - start) * 1000.0, model=model, expert_compute_delay=expert_compute_delay)


def fast_schedule_completion_balanced(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    priority_lookup: dict[str, tuple[float, ...]] = {}
    for phase, matrix in enumerate(_phase_matrices(dispatch_matrix, combine_matrix, next_dispatch_matrix)):
        row_sums = _matrix_row_sums(matrix)
        col_sums = _matrix_col_sums(matrix)
        lb_f = max(max(row_sums, default=0), max(col_sums, default=0))
        slack = [float(lb_f - max(row_sums[gpu], col_sums[gpu])) for gpu in range(num_gpus)]
        phase_chunks[phase].sort(
            key=lambda chunk: (
                chunk.size - 0.5 * (slack[chunk.src_gpu] + slack[chunk.dst_gpu]) / 2.0,
                -chunk.src_gpu,
                -chunk.dst_gpu,
            ),
            reverse=True,
        )
        for chunk in phase_chunks[phase]:
            priority_lookup[chunk.chunk_id] = (
                float(chunk.size - 0.5 * (slack[chunk.src_gpu] + slack[chunk.dst_gpu]) / 2.0),
                float(chunk.size),
            )
    return _schedule_phase_orders(phase_chunks, strategy="completion_balanced", solve_time_ms=(time.perf_counter() - start) * 1000.0, priority_lookup=priority_lookup, model=model, expert_compute_delay=expert_compute_delay)


def fast_schedule_two_stage(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    """Two-stage Birkhoff-based ordering.

    prediction_aware: False
        Phase-2 traffic is scheduled as its own phase but does not shape
        phase-0/1 priorities.
    """
    start = time.perf_counter()
    matrices = _phase_matrices(dispatch_matrix, combine_matrix, next_dispatch_matrix)
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    phase_orders: dict[int, list[ChunkSpec]] = {}
    priority_lookup: dict[str, tuple[float, ...]] = {}
    for phase, matrix in enumerate(matrices):
        rounds = _birkhoff_decompose(matrix)
        round_rank = _birkhoff_round_rank(rounds, phase_chunks[phase], phase=phase)
        row_sums = _matrix_row_sums(matrix)
        col_sums = _matrix_col_sums(matrix)
        phase_orders[phase] = sorted(
            phase_chunks[phase],
            key=lambda chunk: (
                -round_rank[(phase, chunk.src_gpu, chunk.dst_gpu)][0],
                max(row_sums[chunk.src_gpu], col_sums[chunk.dst_gpu]),
                chunk.size,
                -chunk.src_gpu,
                -chunk.dst_gpu,
            ),
            reverse=True,
        )
        for chunk in phase_orders[phase]:
            priority_lookup[chunk.chunk_id] = (
                float(round_rank[(phase, chunk.src_gpu, chunk.dst_gpu)][0]),
                float(max(row_sums[chunk.src_gpu], col_sums[chunk.dst_gpu])),
                float(chunk.size),
            )
    return _schedule_phase_orders(phase_orders, strategy="two_stage", solve_time_ms=(time.perf_counter() - start) * 1000.0, priority_lookup=priority_lookup, model=model, expert_compute_delay=expert_compute_delay)


def fast_schedule_critical_path_compression(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    base = fast_schedule_birkhoff(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay)
    best_orders = _extract_phase_orders_from_schedule(base["schedule"])
    best_makespan = _evaluate_phase_orders(best_orders, model=model, expert_compute_delay=expert_compute_delay)
    no_improve_count = 0
    prev_best = best_makespan
    for _ in range(4):
        if time.perf_counter() - start > 0.0008:
            break
        payload = _schedule_phase_orders(best_orders, strategy="critical_path_probe", solve_time_ms=0.0, model=model, expert_compute_delay=expert_compute_delay)
        if not payload["schedule"]:
            break
        critical = max(payload["schedule"], key=lambda entry: float(entry["end"]))
        phase = int(critical["phase"])
        chunk_id = str(critical["chunk_id"])
        idx = next((i for i, chunk in enumerate(best_orders[phase]) if chunk.chunk_id == chunk_id), None)
        if idx is None:
            break
        improved = False
        for other_idx in range(max(0, idx - 3), len(best_orders[phase])):
            if other_idx == idx:
                continue
            candidate = _clone_phase_orders(best_orders)
            candidate[phase][idx], candidate[phase][other_idx] = candidate[phase][other_idx], candidate[phase][idx]
            makespan = _evaluate_phase_orders(candidate, model=model, expert_compute_delay=expert_compute_delay)
            if makespan < best_makespan:
                best_orders = candidate
                best_makespan = makespan
                improved = True
                break
        if not improved:
            break
    return _schedule_phase_orders(best_orders, strategy="critical_path_compression", solve_time_ms=(time.perf_counter() - start) * 1000.0, model=model, expert_compute_delay=expert_compute_delay)


def fast_schedule_ibbr(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    """Iterated Birkhoff barrier repair.

    prediction_aware: False
        Local repairs operate on existing phase orders without using phase-2
        traffic values to prioritize phase-0/1 changes.
    """
    start = time.perf_counter()
    base = fast_schedule_birkhoff(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay)
    best_orders = _extract_phase_orders_from_schedule(base["schedule"])
    best_makespan = _evaluate_phase_orders(best_orders, model=model, expert_compute_delay=expert_compute_delay)
    no_improve_count = 0
    prev_best = best_makespan
    for _ in range(4):
        if time.perf_counter() - start > 0.003:
            break
        gpu_completion = _gpu_completion(best_orders, model=model, expert_compute_delay=expert_compute_delay)
        g_star = max(gpu_completion, key=gpu_completion.get)
        improved = False
        for phase in range(3):
            indices = [idx for idx, chunk in enumerate(best_orders[phase]) if chunk.src_gpu == g_star or chunk.dst_gpu == g_star]
            for left, right in zip(indices, indices[1:]):
                candidate = _clone_phase_orders(best_orders)
                candidate[phase][left], candidate[phase][right] = candidate[phase][right], candidate[phase][left]
                makespan = _evaluate_phase_orders(candidate, model=model, expert_compute_delay=expert_compute_delay)
                if makespan < best_makespan:
                    relative = (prev_best - makespan) / max(prev_best, 1e-9)
                    no_improve_count = 0 if relative >= _MIN_RELATIVE_IMPROVEMENT else no_improve_count + 1
                    prev_best = makespan
                    best_orders = candidate
                    best_makespan = makespan
                    improved = True
                    break
            if improved:
                break
        if not improved:
            no_improve_count += 1
        if not improved or no_improve_count >= _NO_IMPROVE_PATIENCE:
            break
    return _schedule_phase_orders(best_orders, strategy="ibbr", solve_time_ms=(time.perf_counter() - start) * 1000.0, model=model, expert_compute_delay=expert_compute_delay)


def fast_schedule_birkhoff_exhaustive(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    max_permutations: int = 1000,
) -> dict[str, Any]:
    """Exhaust or sample Birkhoff round permutations to measure the phase-optimal ceiling.

    Complexity is roughly O(P * E) for generating candidates plus O(K * schedule_eval),
    where K is the number of tested round-order combinations.
    """

    start = time.perf_counter()
    rng = random.Random(0)
    matrices = _phase_matrices(dispatch_matrix, combine_matrix, next_dispatch_matrix)
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    phase_candidates: dict[int, list[tuple[list[ChunkSpec], dict[str, tuple[float, ...]]]]] = {0: [], 1: [], 2: []}

    for phase, matrix in enumerate(matrices):
        rounds = _birkhoff_decompose(matrix)
        permutation_budget = max(1, round(max_permutations ** (1.0 / 3.0)))
        for permutation in _sample_round_permutations(rounds, rng=rng, max_permutations=permutation_budget):
            order, lookup = _orders_from_round_permutation(
                phase_chunks[phase],
                rounds,
                permutation,
                phase=phase,
            )
            phase_candidates[phase].append((order, lookup))
        if not phase_candidates[phase]:
            phase_candidates[phase].append((list(phase_chunks[phase]), {}))

    best_payload: dict[str, Any] | None = None
    checked = 0
    for phase0_orders, lookup0 in phase_candidates[0]:
        for phase1_orders, lookup1 in phase_candidates[1]:
            for phase2_orders, lookup2 in phase_candidates[2]:
                checked += 1
                merged = {0: phase0_orders, 1: phase1_orders, 2: phase2_orders}
                merged_lookup: dict[str, tuple[float, ...]] = {}
                merged_lookup.update(lookup0)
                merged_lookup.update(lookup1)
                merged_lookup.update(lookup2)
                payload = _schedule_phase_orders(
                    merged,
                    strategy="birkhoff_exhaustive",
                    solve_time_ms=0.0,
                    priority_lookup=merged_lookup,
                    model=model,
                    expert_compute_delay=expert_compute_delay,
                )
                if best_payload is None or float(payload["makespan"]) < float(best_payload["makespan"]):
                    best_payload = payload
                if checked >= max_permutations:
                    break
            if checked >= max_permutations:
                break
        if checked >= max_permutations:
            break
    assert best_payload is not None
    best_payload["solve_time_ms"] = (time.perf_counter() - start) * 1000.0
    best_payload["strategy"] = "birkhoff_exhaustive"
    best_payload["checked_combinations"] = checked
    return best_payload


def fast_schedule_ejection_chain_tabu(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    budget_ms: float = 2.5,
    max_iters: int = 20,
    tenure: int = 7,
) -> dict[str, Any]:
    """Cross-phase tabu search with shallow ejection chains.

    Complexity is O(K * N^2 * F) in practice due to bounded local repairs.

    prediction_aware: False
        Search decisions are driven by current schedule bottlenecks rather than
        direct inspection of phase-2 traffic values.
    """

    start = time.perf_counter()
    base = fast_schedule_barrier_aware_birkhoff(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )
    current = _extract_phase_orders_from_schedule(base["schedule"])
    current_makespan = _evaluate_phase_orders(current, model=model, expert_compute_delay=expert_compute_delay)
    best = _clone_phase_orders(current)
    best_makespan = current_makespan
    tabu: dict[tuple[int, int], int] = {}
    no_improve_count = 0
    prev_best = best_makespan

    for _iteration in range(max_iters):
        if (time.perf_counter() - start) * 1000.0 >= budget_ms:
            break
        gpu_completion = _gpu_completion(current, model=model, expert_compute_delay=expert_compute_delay)
        ordered_gpus = sorted(gpu_completion, key=lambda gpu: (-gpu_completion[gpu], gpu))
        workload = _phase_workload_map(current, num_gpus)

        chosen_gpu = None
        chosen_phase = None
        for gpu in ordered_gpus:
            candidate_phase = max(range(3), key=lambda phase: (workload[(phase, gpu)], -phase))
            if tabu.get((gpu, candidate_phase), 0) <= 0:
                chosen_gpu = gpu
                chosen_phase = candidate_phase
                break
        if chosen_gpu is None or chosen_phase is None:
            break

        candidate = _ejection_chain_candidate(
            current,
            g_star=chosen_gpu,
            f_star=chosen_phase,
            model=model,
            expert_compute_delay=expert_compute_delay,
        )
        candidate_makespan = _evaluate_phase_orders(candidate, model=model, expert_compute_delay=expert_compute_delay)

        for key in list(tabu):
            tabu[key] -= 1
            if tabu[key] <= 0:
                del tabu[key]
        tabu[(chosen_gpu, chosen_phase)] = tenure

        if candidate_makespan < best_makespan:
            relative = (prev_best - candidate_makespan) / max(prev_best, 1e-9)
            no_improve_count = 0 if relative >= _MIN_RELATIVE_IMPROVEMENT else no_improve_count + 1
            prev_best = candidate_makespan
            best = _clone_phase_orders(candidate)
            best_makespan = candidate_makespan
            current = candidate
            current_makespan = candidate_makespan
        elif candidate_makespan < current_makespan * 1.005:
            current = candidate
            current_makespan = candidate_makespan
            no_improve_count += 1
        else:
            no_improve_count += 1
        if no_improve_count >= _NO_IMPROVE_PATIENCE:
            break

    return _schedule_phase_orders(
        best,
        strategy="ejection_chain_tabu",
        solve_time_ms=(time.perf_counter() - start) * 1000.0,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )


def fast_schedule_lns_cp_repair(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    budget_ms: float = 4.0,
    max_repair_iters: int = 8,
) -> dict[str, Any]:
    """Large-neighborhood search with CP-SAT phase-local repair around the bottleneck GPU.

    Complexity is O(K * repair_cost) where repair_cost is bounded by a small subproblem.
    """

    start = time.perf_counter()
    base = fast_schedule_barrier_aware_birkhoff(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )
    best_orders = _extract_phase_orders_from_schedule(base["schedule"])
    best_makespan = _evaluate_phase_orders(best_orders, model=model, expert_compute_delay=expert_compute_delay)
    no_improve_count = 0
    prev_best = best_makespan

    for _iteration in range(max_repair_iters):
        if (time.perf_counter() - start) * 1000.0 >= budget_ms:
            break
        gpu_completion = _gpu_completion(best_orders, model=model, expert_compute_delay=expert_compute_delay)
        bottleneck_gpu = max(gpu_completion, key=lambda gpu: (gpu_completion[gpu], -gpu))
        target_phase = max(
            range(3),
            key=lambda phase: _phase_workload_by_gpu(best_orders, bottleneck_gpu, phase),
        )
        candidate_orders = _clone_phase_orders(best_orders)
        phase_chunks = list(candidate_orders[target_phase])
        repair_chunks = [
            chunk
            for chunk in phase_chunks
            if chunk.src_gpu == bottleneck_gpu or chunk.dst_gpu == bottleneck_gpu
        ]
        if not repair_chunks:
            break
        conflict_ids = {
            chunk.chunk_id
            for chunk in phase_chunks
            for anchor in repair_chunks
            if (
                chunk.src_gpu in {anchor.src_gpu, anchor.dst_gpu}
                or chunk.dst_gpu in {anchor.src_gpu, anchor.dst_gpu}
            )
        }
        selected = [chunk for chunk in phase_chunks if chunk.chunk_id in conflict_ids]
        selected = sorted(selected, key=lambda chunk: chunk.size, reverse=True)[:20]
        if len(selected) <= 1:
            break
        ordered = _cp_sat_order_phase_chunks(
            selected,
            num_gpus=num_gpus,
            timeout_ms=0.5,
            model=model,
        )
        if ordered is None:
            ordered = sorted(selected, key=lambda chunk: (chunk.size, -chunk.src_gpu, -chunk.dst_gpu), reverse=True)

        selected_ids = {chunk.chunk_id for chunk in selected}
        skeleton = [chunk for chunk in candidate_orders[target_phase] if chunk.chunk_id not in selected_ids]
        rebuilt = list(skeleton)
        for chunk in ordered:
            temp_orders = _clone_phase_orders(candidate_orders)
            temp_orders[target_phase] = list(rebuilt)
            temp_orders = _best_insert_position(temp_orders, chunk)
            rebuilt = list(temp_orders[target_phase])
        candidate_orders[target_phase] = rebuilt
        candidate_makespan = _evaluate_phase_orders(candidate_orders, model=model, expert_compute_delay=expert_compute_delay)
        if candidate_makespan < best_makespan:
            relative = (prev_best - candidate_makespan) / max(prev_best, 1e-9)
            no_improve_count = 0 if relative >= _MIN_RELATIVE_IMPROVEMENT else no_improve_count + 1
            prev_best = candidate_makespan
            best_orders = candidate_orders
            best_makespan = candidate_makespan
        else:
            no_improve_count += 1
        if no_improve_count >= _NO_IMPROVE_PATIENCE:
            break

    return _schedule_phase_orders(
        best_orders,
        strategy="lns_cp_repair",
        solve_time_ms=(time.perf_counter() - start) * 1000.0,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )


def fast_schedule_decomposed(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    group_size: int = 4,
    sub_timeout_ms: float = 2.0,
) -> dict[str, Any]:
    """Decompose the N-GPU schedule into group-pair subproblems and merge their priorities.

    Complexity is dominated by the number of subgroup repairs, typically O(K * group_size^3).
    """

    start = time.perf_counter()
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    phase_orders: dict[int, list[ChunkSpec]] = {0: [], 1: [], 2: []}
    priority_lookup: dict[str, tuple[float, ...]] = {}

    for phase in range(3):
        buckets: dict[tuple[int, int], list[ChunkSpec]] = {}
        for chunk in phase_chunks[phase]:
            key = (chunk.src_gpu // group_size, chunk.dst_gpu // group_size)
            buckets.setdefault(key, []).append(chunk)
        bucket_orders: list[tuple[float, list[ChunkSpec]]] = []
        for (src_group, dst_group), bucket_chunks in buckets.items():
            ordered = _cp_sat_order_phase_chunks(
                bucket_chunks,
                num_gpus=num_gpus,
                timeout_ms=sub_timeout_ms,
                model=model,
            )
            if ordered is None:
                sub_matrix = [[0] * num_gpus for _ in range(num_gpus)]
                for chunk in bucket_chunks:
                    sub_matrix[chunk.src_gpu][chunk.dst_gpu] = chunk.size
                rounds = _birkhoff_decompose(sub_matrix)
                round_rank = {}
                for round_index, (_weight, edges) in enumerate(rounds):
                    for src, dst in edges:
                        round_rank.setdefault((src, dst), round_index)
                ordered = sorted(
                    bucket_chunks,
                    key=lambda chunk: (round_rank.get((chunk.src_gpu, chunk.dst_gpu), 999), -chunk.size),
                )
            score = float(sum(chunk.size for chunk in ordered))
            if phase < 2:
                score += 0.25 * sum(chunk.size for chunk in phase_chunks[phase + 1] if chunk.src_gpu // group_size == dst_group)
            bucket_orders.append((score, ordered))
        bucket_orders.sort(key=lambda item: item[0], reverse=True)
        phase_orders[phase] = _merge_subproblem_orders(bucket_orders)
        merged_order = phase_orders[phase]
        for order_index, chunk in enumerate(merged_order):
            priority_lookup[chunk.chunk_id] = (float(len(merged_order) - order_index), float(chunk.size))

    return _schedule_phase_orders(
        phase_orders,
        strategy="decomposed",
        solve_time_ms=(time.perf_counter() - start) * 1000.0,
        priority_lookup=priority_lookup,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )


def fast_schedule_quantized_decomposed(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    quantize_alpha: float = 0.1,
    group_size: int = 4,
    sub_timeout_ms: float = 1.0,
) -> dict[str, Any]:
    """Compress small flows, solve the large-flow skeleton, then reinsert deferred chunks.

    Complexity is near decomposed scheduling plus O(deferred * insertion_cost).
    """

    start = time.perf_counter()

    def _quantize(matrix: list[list[int]]) -> tuple[list[list[int]], list[tuple[int, int, int]]]:
        if not matrix:
            return matrix, []
        threshold = max(max(row, default=0) for row in matrix) * quantize_alpha
        compressed = [[0 for _ in row] for row in matrix]
        deferred: list[tuple[int, int, int]] = []
        for src, row in enumerate(matrix):
            scored = [(int(value), dst) for dst, value in enumerate(row) if src != dst and int(value) > 0]
            if not scored:
                continue
            scored.sort(reverse=True)
            keep_count = max(3, num_gpus // 2)
            keep = scored[:keep_count]
            for value, dst in keep:
                compressed[src][dst] += value
            for value, dst in scored[keep_count:]:
                if value < threshold:
                    deferred.append((src, dst, value))
                else:
                    target_dst = keep[0][1]
                    compressed[src][target_dst] += value
        return compressed, deferred

    q_dispatch, deferred_dispatch = _quantize(dispatch_matrix)
    q_combine, deferred_combine = _quantize(combine_matrix)
    q_next, deferred_next = _quantize(next_dispatch_matrix)

    base = fast_schedule_decomposed(
        q_dispatch,
        q_combine,
        q_next,
        num_gpus,
        model=model,
        expert_compute_delay=expert_compute_delay,
        group_size=group_size,
        sub_timeout_ms=sub_timeout_ms,
    )
    phase_orders = _extract_phase_orders_from_schedule(base["schedule"])

    for phase, deferred_items in (
        (0, deferred_dispatch),
        (1, deferred_combine),
        (2, deferred_next),
    ):
        for src, dst, size in sorted(deferred_items, key=lambda item: item[2], reverse=True):
            phase_orders = _best_insert_position(
                phase_orders,
                ChunkSpec(
                    chunk_id=f"phase{phase}_deferred_src{src}_dst{dst}_{size}",
                    phase=phase,
                    size=int(size),
                    src_gpu=src,
                    dst_gpu=dst,
                ),
            )

    return _schedule_phase_orders(
        phase_orders,
        strategy="quantized_decomposed",
        solve_time_ms=(time.perf_counter() - start) * 1000.0,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )


def _critical_gpu(schedule: list[dict[str, Any]]) -> int:
    last_chunk = max(schedule, key=lambda item: (float(item["end"]), int(item["dst_gpu"]), int(item["src_gpu"])))
    return int(last_chunk["dst_gpu"])


def fast_schedule_cp_local_swap(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    rounds: int = 3,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    start = time.perf_counter()
    phase_orders, priority_lookup = _phase_orders_cp_lpt(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    best_orders = _clone_phase_orders(phase_orders)
    best_payload = _schedule_phase_orders(
        best_orders,
        strategy="cp_local_swap",
        solve_time_ms=0.0,
        priority_lookup=priority_lookup,
        model=model,
        expert_compute_delay=expert_compute_delay,
    )
    for _ in range(rounds):
        critical_gpu = _critical_gpu(best_payload["schedule"])
        improved = False
        for phase, chunks in best_orders.items():
            critical_indices = [
                index
                for index, chunk in enumerate(chunks)
                if chunk.src_gpu == critical_gpu or chunk.dst_gpu == critical_gpu
            ]
            for index in critical_indices:
                for candidate_index in range(index):
                    candidate_orders = _clone_phase_orders(best_orders)
                    candidate_phase = candidate_orders[phase]
                    candidate_phase[index], candidate_phase[candidate_index] = candidate_phase[candidate_index], candidate_phase[index]
                    candidate_payload = _schedule_phase_orders(
                        candidate_orders,
                        strategy="cp_local_swap",
                        solve_time_ms=0.0,
                        priority_lookup=priority_lookup,
                        model=model,
                        expert_compute_delay=expert_compute_delay,
                    )
                    if float(candidate_payload["makespan"]) < float(best_payload["makespan"]):
                        best_orders = candidate_orders
                        best_payload = candidate_payload
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    best_payload["solve_time_ms"] = (time.perf_counter() - start) * 1000.0
    return best_payload


def fast_schedule_pairwise(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    candidates = [
        fast_schedule_cp_lpt(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_birkhoff(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_barrier_aware_birkhoff(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_lns(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_simulated_annealing(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_lagrangian(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_grasp(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_ejection_chain_tabu(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_lns_cp_repair(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_quantized_decomposed(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_two_stage(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_ibbr(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
    ]
    eligible = [payload for payload in candidates if float(payload["solve_time_ms"]) <= 5.0]
    best_pool = eligible if eligible else candidates
    best = min(best_pool, key=lambda payload: (float(payload["makespan"]), float(payload["solve_time_ms"])))
    result = dict(best)
    result["strategy"] = f"best_of:{best['strategy']}"
    result["candidates"] = [
        {
            "strategy": payload["strategy"],
            "makespan": payload["makespan"],
            "solve_time_ms": payload["solve_time_ms"],
        }
        for payload in candidates
    ]
    return result
