from __future__ import annotations

import itertools
import random
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
    *,
    sorting_next_dispatch_matrix: list[list[int]] | None = None,
) -> tuple[dict[int, list[ChunkSpec]], dict[str, tuple[float, ...]]]:
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    critical_phase_chunks = _collect_phase_chunks(
        dispatch_matrix,
        combine_matrix,
        sorting_next_dispatch_matrix if sorting_next_dispatch_matrix is not None else next_dispatch_matrix,
        num_gpus,
    )
    critical = _critical_path_weights(critical_phase_chunks, num_gpus)
    priority_lookup: dict[str, tuple[float, ...]] = {}
    phase_orders: dict[int, list[ChunkSpec]] = {}
    for phase, chunks in phase_chunks.items():
        phase_orders[phase] = sorted(
            chunks,
            key=lambda chunk: (
                critical.get(chunk.chunk_id, float(chunk.size)),
                chunk.size,
                -chunk.src_gpu,
                -chunk.dst_gpu,
            ),
            reverse=True,
        )
        for chunk in phase_orders[phase]:
            priority_lookup[chunk.chunk_id] = (
                critical.get(chunk.chunk_id, float(chunk.size)),
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
    sorting_next_dispatch_matrix: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Critical-path LPT ordering.

    prediction_aware: True
        Uses next_dispatch_matrix values through _critical_path_weights() to
        compute downstream later_work for phase-0/1 prioritization.
    """
    start = time.perf_counter()
    phase_orders, priority_lookup = _phase_orders_cp_lpt(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        sorting_next_dispatch_matrix=sorting_next_dispatch_matrix,
    )
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
