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
    _best_insert_position,
    _clone_phase_orders,
    _collect_phase_chunks,
    _cp_sat_order_phase_chunks,
    _ejection_chain_candidate,
    _evaluate_phase_orders,
    _orders_from_round_permutation,
    _extract_phase_orders_from_schedule,
    _gpu_completion,
    _phase_matrices,
    _phase_orders_cp_lpt,
    _phase_workload_by_gpu,
    _phase_workload_map,
    _random_swap_orders,
    _sample_round_permutations,
    _schedule_phase_orders,
)
from .birkhoff import fast_schedule_barrier_aware_birkhoff, fast_schedule_birkhoff

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
    budget_ms: float = 1.5,
    max_iters: int = 8,
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
    budget_ms: float = 1.5,
    max_repair_iters: int = 4,
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
