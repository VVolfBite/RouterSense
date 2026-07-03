from __future__ import annotations

import random
import time
from typing import Any

import numpy as np

from ._common import (
    ChunkSpec,
    _birkhoff_decompose,
    _birkhoff_round_rank,
    _collect_phase_chunks,
    _extract_phase_orders_from_schedule,
    _orders_from_round_permutation,
    _phase_matrices,
    _phase_order_from_round_rank,
    _sample_round_permutations,
    _schedule_phase_orders,
)
from .global_matching import EXECUTION_WINDOW_MODE, _run_global_matching_scheduler

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


def _phase_local_wave_schedule(
    phase_orders: dict[int, list[ChunkSpec]],
    *,
    strategy: str,
    expert_compute_delay: float,
) -> dict[str, Any]:
    num_gpus = 0
    for chunks in phase_orders.values():
        for chunk in chunks:
            num_gpus = max(num_gpus, chunk.src_gpu + 1, chunk.dst_gpu + 1)
    matrices = [[[0] * num_gpus for _ in range(num_gpus)] for _ in range(3)]
    base_score_lookup: dict[str, float] = {}
    for phase, chunks in phase_orders.items():
        total = max(len(chunks), 1)
        for index, chunk in enumerate(chunks):
            matrices[phase][chunk.src_gpu][chunk.dst_gpu] = int(chunk.size)
            base_score_lookup[chunk.chunk_id] = float(total - index)
    return _run_global_matching_scheduler(
        matrices[0],
        matrices[1],
        matrices[2],
        num_gpus,
        strategy=strategy,
        mode=EXECUTION_WINDOW_MODE,
        prediction_confidence=0.0,
        expert_compute_delay=expert_compute_delay,
        exact_matching=True,
        wave_quantum=None,
        max_waves=256,
        residual_weight=0.25,
        barrier_weight=0.0,
        age_weight=0.05,
        prediction_weight=0.0,
        adaptive_prices=False,
        price_step=0.0,
        price_decay=0.0,
        price_clip=0.0,
        iteration_budget=1,
        atomic=False,
        base_score_lookup=base_score_lookup,
        base_priority_weight=1.0,
    )


def fast_schedule_birkhoff_wave(
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
    base = fast_schedule_birkhoff(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        expert_compute_delay=expert_compute_delay,
    )
    payload = _phase_local_wave_schedule(
        _extract_phase_orders_from_schedule(base["schedule"]),
        strategy="B_birkhoff_wave",
        expert_compute_delay=expert_compute_delay,
    )
    payload["solve_time_ms"] = (time.perf_counter() - start) * 1000.0
    return payload


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

