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
) -> dict[str, Any]:
    max_gpu = 0
    for chunks in phase_orders.values():
        for chunk in chunks:
            max_gpu = max(max_gpu, chunk.src_gpu, chunk.dst_gpu)
    num_gpus = max_gpu + 1 if phase_orders and max_gpu >= 0 else 0
    gpu_available = [0.0] * num_gpus
    schedule: list[dict[str, Any]] = []
    for phase in sorted(phase_orders):
        for chunk in phase_orders[phase]:
            start = max(gpu_available[chunk.src_gpu], gpu_available[chunk.dst_gpu])
            end = start + float(chunk.size)
            gpu_available[chunk.src_gpu] = end
            gpu_available[chunk.dst_gpu] = end
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
    return {
        "makespan": max(gpu_available) if gpu_available else 0.0,
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
    )


def fast_schedule_cp_lpt(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    phase_orders, priority_lookup = _phase_orders_cp_lpt(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    return _schedule_phase_orders(
        phase_orders,
        strategy="cp_lpt",
        solve_time_ms=(time.perf_counter() - start) * 1000.0,
        priority_lookup=priority_lookup,
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
) -> dict[str, Any]:
    start = time.perf_counter()
    base_orders, priority_lookup = _phase_orders_cp_lpt(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    best_orders = _clone_phase_orders(base_orders)
    best_payload = _schedule_phase_orders(best_orders, strategy="iterated_greedy", solve_time_ms=0.0, priority_lookup=priority_lookup)
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
        candidate_payload = _schedule_phase_orders(candidate_orders, strategy="iterated_greedy", solve_time_ms=0.0, priority_lookup=priority_lookup)
        if float(candidate_payload["makespan"]) < float(best_payload["makespan"]):
            best_orders = candidate_orders
            best_payload = candidate_payload
    best_payload["solve_time_ms"] = (time.perf_counter() - start) * 1000.0
    return best_payload


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
) -> dict[str, Any]:
    start = time.perf_counter()
    phase_orders, priority_lookup = _phase_orders_cp_lpt(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    best_orders = _clone_phase_orders(phase_orders)
    best_payload = _schedule_phase_orders(best_orders, strategy="cp_local_swap", solve_time_ms=0.0, priority_lookup=priority_lookup)
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
) -> dict[str, Any]:
    candidates = [
        fast_schedule_lookahead_lpt(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus),
        fast_schedule_cp_lpt(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus),
        fast_schedule_birkhoff(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus),
        fast_schedule_iterated_greedy(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus),
        fast_schedule_cp_local_swap(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus),
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
