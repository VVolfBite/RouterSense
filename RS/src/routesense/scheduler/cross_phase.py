from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from ._common import (
    ChunkSpec,
    _MIN_RELATIVE_IMPROVEMENT,
    _NO_IMPROVE_PATIENCE,
    _birkhoff_decompose,
    _birkhoff_round_rank,
    _collect_phase_chunks,
    _cp_sat_order_phase_chunks,
    _critical_path_weights,
    _evaluate_phase_orders,
    _extract_phase_orders_from_schedule,
    _gpu_completion,
    _matrix_col_sums,
    _matrix_row_sums,
    _merge_subproblem_orders,
    _phase_completion_times,
    _phase_matrices,
    _phase_order_from_round_rank,
    _phase_orders_cp_lpt,
    _phase_orders_lookahead_lpt,
    _phase_workload_by_gpu,
    _schedule_phase_orders,
    _clone_phase_orders,
    _best_insert_position,
)
from .birkhoff import fast_schedule_barrier_aware_birkhoff, fast_schedule_birkhoff
from .global_matching import (
    fast_schedule_u_barrier_criticality_global_matching,
    fast_schedule_u_barrier_criticality_global_matching_atomic,
    fast_schedule_u_barrier_price_adaptive_matching,
    fast_schedule_u_barrier_price_adaptive_matching_atomic,
    fast_schedule_u_gated_greedy_maximal,
    fast_schedule_u_gated_greedy_maximal_atomic,
    fast_schedule_u_gated_maxweight_matching,
    fast_schedule_u_gated_maxweight_matching_atomic,
)

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


def fast_schedule_barrier_aware_birkhoff_wave(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    del model
    start = time.perf_counter()
    base = fast_schedule_barrier_aware_birkhoff(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        expert_compute_delay=expert_compute_delay,
    )
    payload = _phase_local_wave_schedule(
        _extract_phase_orders_from_schedule(base["schedule"]),
        strategy="B_barrier_aware_birkhoff_wave",
        expert_compute_delay=expert_compute_delay,
    )
    payload["solve_time_ms"] = (time.perf_counter() - start) * 1000.0
    return payload

def fast_schedule_lagrangian(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    max_iterations: int = 8,
    learning_rate: float = 0.1,
    budget_ms: float = 5.0,
    sorting_next_dispatch_matrix: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Cross-phase Lagrangian ordering over all provided phase matrices.

    prediction_aware: True
        The optimization uses phase-2 matrix values inside the coupled
        three-phase objective and multiplier updates.
    """
    start = time.perf_counter()
    matrices = _phase_matrices(
        dispatch_matrix,
        combine_matrix,
        sorting_next_dispatch_matrix if sorting_next_dispatch_matrix is not None else next_dispatch_matrix,
    )
    phase_chunks = _collect_phase_chunks(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus)
    lambda_g = [0.0] * num_gpus
    best_makespan = float("inf")
    best_orders: dict[int, list[ChunkSpec]] | None = None
    best_lookup: dict[str, tuple[float, ...]] = {}
    no_improve_count = 0
    prev_best = float("inf")

    for iteration in range(max_iterations):
        if (time.perf_counter() - start) * 1000.0 >= budget_ms:
            break
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
        fast_schedule_lagrangian(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_ibbr(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_u_gated_greedy_maximal(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_u_gated_greedy_maximal_atomic(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_u_gated_maxweight_matching(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_u_gated_maxweight_matching_atomic(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_u_barrier_criticality_global_matching(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_u_barrier_criticality_global_matching_atomic(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_u_barrier_price_adaptive_matching(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
        fast_schedule_u_barrier_price_adaptive_matching_atomic(dispatch_matrix, combine_matrix, next_dispatch_matrix, num_gpus, model=model, expert_compute_delay=expert_compute_delay),
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
