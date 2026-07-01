from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    phase: int
    size: int
    src_gpu: int
    dst_gpu: int


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


def fast_schedule_birkhoff(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
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
            best_orders = candidate_orders
            best_payload = candidate_payload
    best_payload["solve_time_ms"] = (time.perf_counter() - start) * 1000.0
    return best_payload


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
    start = time.perf_counter()
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    lambdas: dict[int, list[float]] = {0: [0.0] * num_gpus, 1: [0.0] * num_gpus}
    best_makespan = float("inf")
    best_schedule: dict[str, Any] | None = None

    for _ in range(max_iterations):
        phase_orders: dict[int, list[ChunkSpec]] = {}
        for phase in range(3):
            chunks = list(phase_chunks[phase])

            def penalized_key(chunk: ChunkSpec, _phase: int = phase) -> tuple[float, int, int]:
                base_priority = float(chunk.size)
                penalty = 0.0
                if _phase > 0:
                    penalty += lambdas[_phase - 1][chunk.src_gpu] * 0.5
                if _phase < 2:
                    penalty += lambdas[_phase][chunk.dst_gpu] * 0.5
                return (base_priority + penalty, -chunk.src_gpu, -chunk.dst_gpu)

            chunks.sort(key=penalized_key, reverse=True)
            phase_orders[phase] = chunks

        candidate = _schedule_phase_orders(
            phase_orders,
            strategy="lagrangian",
            solve_time_ms=0.0,
            model=model,
            expert_compute_delay=expert_compute_delay,
        )
        if float(candidate["makespan"]) < best_makespan:
            best_makespan = float(candidate["makespan"])
            best_schedule = dict(candidate)

        schedule = candidate["schedule"]
        recv_done = {0: [0.0] * num_gpus, 1: [0.0] * num_gpus}
        send_start = {1: [float("inf")] * num_gpus, 2: [float("inf")] * num_gpus}
        for entry in schedule:
            phase = int(entry["phase"])
            dst_gpu = int(entry["dst_gpu"])
            src_gpu = int(entry["src_gpu"])
            if phase in recv_done:
                recv_done[phase][dst_gpu] = max(recv_done[phase][dst_gpu], float(entry["end"]))
            if phase in send_start:
                send_start[phase][src_gpu] = min(send_start[phase][src_gpu], float(entry["start"]))

        for gpu in range(num_gpus):
            for phase in (0, 1):
                barrier_time = recv_done[phase][gpu] + (expert_compute_delay if phase == 0 else 0.0)
                earliest_send = send_start[phase + 1][gpu]
                if earliest_send == float("inf"):
                    continue
                violation = barrier_time - earliest_send
                lambdas[phase][gpu] = max(0.0, lambdas[phase][gpu] + learning_rate * violation)
        learning_rate *= 0.95

    solve_time_ms = (time.perf_counter() - start) * 1000.0
    if best_schedule is not None:
        best_schedule["solve_time_ms"] = solve_time_ms
        best_schedule["strategy"] = "lagrangian"
        return best_schedule
    return fast_schedule_birkhoff(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
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
        fast_schedule_birkhoff(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_lagrangian(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_iterated_greedy(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
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
