from __future__ import annotations

from collections import Counter
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch  # type: ignore

from ..scheduler import (
    fast_schedule_barrier_aware_birkhoff,
    fast_schedule_birkhoff,
    fast_schedule_cp_lpt,
    fast_schedule_ibbr,
    fast_schedule_lagrangian,
    fast_schedule_pairwise,
    fast_schedule_u_barrier_criticality_global_matching,
    fast_schedule_u_barrier_price_adaptive_matching,
    fast_schedule_u_gated_greedy_maximal,
    fast_schedule_u_gated_maxweight_matching,
    greedy_schedule_pairwise,
    pairwise_oracle,
)
from ..scheduler.strategy import get_strategy_metadata
from .cross_layer import _decode_predicted_topk_by_sample, _summary_stats, evaluate_gate2, spearman_rank_correlation
from .traffic_matrix import (
    TraceRecord,
    _group_records_by_sample_token_layer,
    _normalize_token_positions,
    build_predicted_traffic,
    build_sample_layer_matrices,
)


SchedulerFn = Callable[[list[list[int]], list[list[int]], list[list[int]], int], dict[str, Any]]


FAST_ALGORITHMS: list[tuple[str, SchedulerFn]] = [
    ("O_cp_lpt", fast_schedule_cp_lpt),
    ("B_birkhoff", fast_schedule_birkhoff),
    ("B_barrier_aware_birkhoff", fast_schedule_barrier_aware_birkhoff),
    ("O_lagrangian", fast_schedule_lagrangian),
    ("O_ibbr", fast_schedule_ibbr),
    ("O_gated_greedy_maximal", fast_schedule_u_gated_greedy_maximal),
    ("O_gated_maxweight_matching", fast_schedule_u_gated_maxweight_matching),
    ("O_barrier_criticality_global_matching", fast_schedule_u_barrier_criticality_global_matching),
    ("O_barrier_price_adaptive_matching", fast_schedule_u_barrier_price_adaptive_matching),
]

PREDICTION_AWARE_ALGORITHMS = {
    "cp_lpt",
    "lagrangian",
    "O_cp_lpt",
    "O_lagrangian",
    "O_gated_greedy_maximal",
    "O_gated_maxweight_matching",
    "O_barrier_criticality_global_matching",
    "O_barrier_price_adaptive_matching",
}


def compute_effective_makespan(
    scheduling_makespan: float,
    phase0_makespan: float,
    expert_compute_delay: float,
    is_prediction_aware: bool,
) -> float:
    if not is_prediction_aware:
        return scheduling_makespan
    hidden_overlap = min(expert_compute_delay, phase0_makespan)
    return scheduling_makespan - hidden_overlap


def _uniform_next_matrix(next_dispatch_matrix: list[list[int]]) -> list[list[int]]:
    size = len(next_dispatch_matrix)
    total = sum(sum(row) for row in next_dispatch_matrix)
    nonzero = size * max(size - 1, 0)
    avg = max(1, total // nonzero) if nonzero > 0 else 1
    return [[0 if row == col else avg for col in range(size)] for row in range(size)]


def run_pairwise_analysis(
    records: list[TraceRecord],
    *,
    hidden_states_by_sample: dict[str, dict[int, torch.Tensor]],
    gate_weights_by_sample: dict[str, dict[int, torch.Tensor]],
    owner_by_expert: dict[int, int],
    num_gpus: int = 4,
    topk: int = 8,
    sample_limit: int | None = None,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    hidden_window_ms: float = 10.0,
    token_to_ms_factor: float = 0.5,
    next_mode: str = "predicted",
    phase2_source: str = "predicted",
    fast_algorithms: list[tuple[str, SchedulerFn]] | None = None,
    skip_oracle: bool = False,
    include_fast_best_of: bool = True,
) -> dict[str, Any]:
    if next_mode not in {"predicted", "zeros"}:
        raise ValueError(f"unsupported next_mode {next_mode!r}")
    if phase2_source not in {"predicted", "actual"}:
        raise ValueError(f"unsupported phase2_source {phase2_source!r}")
    t0 = time.time()
    grouped = _group_records_by_sample_token_layer(records)
    print(f"[perf] _group_records: {time.time() - t0:.2f}s", flush=True)

    token_index: dict[tuple[str, int], list[int]] = {}
    for record in records:
        token_index.setdefault((record.sample_id, record.layer_id), []).append(record.token_position)
    for key, positions in list(token_index.items()):
        deduped_positions = sorted(set(positions))
        token_index[key] = _normalize_token_positions(deduped_positions, num_gpus=num_gpus, max_tokens=256)

    t1 = time.time()
    sample_layer_matrices = build_sample_layer_matrices(records, owner_by_expert=owner_by_expert, num_gpus=num_gpus, grouped=grouped)
    print(f"[perf] build_sample_layer_matrices: {time.time() - t1:.2f}s", flush=True)

    t2 = time.time()
    predicted_topk_by_sample = _decode_predicted_topk_by_sample(
        hidden_states_by_sample=hidden_states_by_sample,
        gate_weights_by_sample=gate_weights_by_sample,
        topk=topk,
    )
    print(f"[perf] _decode_predicted_topk: {time.time() - t2:.2f}s", flush=True)

    sample_ids = sorted({record.sample_id for record in records})
    if sample_limit is not None:
        sample_ids = sample_ids[:sample_limit]
    layer_ids = sorted({record.layer_id for record in records})

    selected_fast_algorithms = fast_algorithms if fast_algorithms is not None else FAST_ALGORITHMS
    algorithm_improvements: dict[str, list[float]] = {name: [] for name, _ in selected_fast_algorithms}
    algorithm_effective_improvements: dict[str, list[float]] = {name: [] for name, _ in selected_fast_algorithms}
    algorithm_latencies: dict[str, list[float]] = {name: [] for name, _ in selected_fast_algorithms}
    algorithm_effective_latencies: dict[str, list[float]] = {name: [] for name, _ in selected_fast_algorithms}
    algorithm_comm_savings_ms: dict[str, list[float]] = {name: [] for name, _ in selected_fast_algorithms}
    algorithm_exposed_latency_ms: dict[str, list[float]] = {name: [] for name, _ in selected_fast_algorithms}
    algorithm_net_benefit_ms: dict[str, list[float]] = {name: [] for name, _ in selected_fast_algorithms}
    algorithm_effective_e2e_time_ms: dict[str, list[float]] = {name: [] for name, _ in selected_fast_algorithms}
    algorithm_e2e_speedup_pct: dict[str, list[float]] = {name: [] for name, _ in selected_fast_algorithms}
    algorithm_failures: dict[str, list[str]] = {name: [] for name, _ in selected_fast_algorithms}
    results: list[dict[str, Any]] = []
    predicted_improvements: list[float] = []
    perfect_improvements: list[float] = []
    fast_improvements: list[float] = []
    oracle_prediction_gaps: list[float] = []
    traffic_correlations: list[float] = []
    greedy_latencies_ms: list[float] = []
    greedy_makespans_ms: list[float] = []
    fast_latencies_ms: list[float] = []
    fast_comm_savings_ms: list[float] = []
    fast_exposed_latency_ms: list[float] = []
    fast_net_benefit_ms: list[float] = []
    fast_effective_e2e_time_ms: list[float] = []
    fast_e2e_speedup_pct: list[float] = []
    oracle_perfect_latencies_ms: list[float] = []
    oracle_predicted_latencies_ms: list[float] = []
    oracle_perfect_comm_savings_ms: list[float] = []
    oracle_perfect_exposed_latency_ms: list[float] = []
    oracle_perfect_net_benefit_ms: list[float] = []
    oracle_perfect_effective_e2e_time_ms: list[float] = []
    oracle_perfect_e2e_speedup_pct: list[float] = []
    oracle_predicted_comm_savings_ms: list[float] = []
    oracle_predicted_exposed_latency_ms: list[float] = []
    oracle_predicted_net_benefit_ms: list[float] = []
    oracle_predicted_effective_e2e_time_ms: list[float] = []
    oracle_predicted_e2e_speedup_pct: list[float] = []
    oracle_perfect_statuses: list[str] = []
    oracle_predicted_statuses: list[str] = []
    pairwise_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    loop_counter = 0

    def compute_e2e_metrics(makespan_tokens: float, latency_ms: float, greedy_tokens: float) -> dict[str, float]:
        greedy_makespan_ms = greedy_tokens * token_to_ms_factor
        algo_makespan_ms = makespan_tokens * token_to_ms_factor
        comm_savings_ms = (greedy_tokens - makespan_tokens) * token_to_ms_factor
        exposed_latency_ms = max(0.0, latency_ms - hidden_window_ms)
        net_benefit_ms = comm_savings_ms - exposed_latency_ms
        effective_e2e_time_ms = algo_makespan_ms + exposed_latency_ms
        e2e_speedup_pct = (net_benefit_ms / greedy_makespan_ms * 100.0) if greedy_makespan_ms > 0.0 else 0.0
        return {
            "greedy_makespan_ms": greedy_makespan_ms,
            "algo_makespan_ms": algo_makespan_ms,
            "comm_savings_ms": comm_savings_ms,
            "exposed_latency_ms": exposed_latency_ms,
            "net_benefit_ms": net_benefit_ms,
            "effective_e2e_time_ms": effective_e2e_time_ms,
            "e2e_speedup_pct": e2e_speedup_pct,
        }

    def matrix_key(matrix: list[list[int]]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in row) for row in matrix)

    def solve_pairwise_cached(phase0: list[list[int]], phase1: list[list[int]], phase2: list[list[int]]) -> tuple[dict[str, Any], bool]:
        key = (matrix_key(phase0), matrix_key(phase1), matrix_key(phase2), num_gpus, model, float(expert_compute_delay))
        cached = pairwise_cache.get(key)
        if cached is None:
            cached = pairwise_oracle(
                phase0,
                phase1,
                phase2,
                num_gpus,
                model=model,
                expert_compute_delay=expert_compute_delay,
            )
            pairwise_cache[key] = cached
            return cached, False
        return cached, True

    for sample_id in sample_ids:
        for idx in range(len(layer_ids) - 1):
            loop_counter += 1
            from_layer = layer_ids[idx]
            to_layer = layer_ids[idx + 1]
            layer_map = sample_layer_matrices.get(sample_id, {})
            if from_layer not in layer_map or to_layer not in layer_map:
                continue

            dispatch_matrix = layer_map[from_layer]
            next_actual_matrix = layer_map[to_layer]
            from .traffic_matrix import combine_matrix_from_dispatch

            combine_matrix = combine_matrix_from_dispatch(dispatch_matrix, 1.0)
            t_build = time.time()
            next_predicted_matrix = build_predicted_traffic(
                sample_id=sample_id,
                from_layer=from_layer,
                to_layer=to_layer,
                grouped_records=grouped,
                predicted_topk_by_sample=predicted_topk_by_sample,
                owner_by_expert=owner_by_expert,
                num_gpus=num_gpus,
                topk=topk,
                token_index=token_index,
            )
            sorting_uniform_matrix = _uniform_next_matrix(next_predicted_matrix)
            if loop_counter <= 5 or loop_counter % 100 == 0:
                print(f"[perf] build_predicted_traffic [{sample_id}..L{from_layer}->{to_layer}]: {time.time() - t_build:.3f}s", flush=True)

            scheduler_next_matrix = next_actual_matrix if phase2_source == "actual" else next_predicted_matrix
            sorting_uniform_matrix = _uniform_next_matrix(scheduler_next_matrix)

            t_greedy = time.time()
            greedy_makespan = greedy_schedule_pairwise(
                dispatch_matrix,
                combine_matrix,
                scheduler_next_matrix,
                num_gpus,
                model=model,
                expert_compute_delay=expert_compute_delay,
            )
            greedy_latency_ms = (time.time() - t_greedy) * 1000.0
            greedy_e2e = compute_e2e_metrics(greedy_makespan, 0.0, greedy_makespan)

            algorithm_results: dict[str, dict[str, Any]] = {}
            for name, scheduler in selected_fast_algorithms:
                try:
                    metadata = get_strategy_metadata(name)
                    extra_kwargs: dict[str, Any] = {}
                    if next_mode == "zeros" and bool(metadata["prediction_aware"]):
                        extra_kwargs["sorting_next_dispatch_matrix"] = sorting_uniform_matrix
                    payload = scheduler(
                        dispatch_matrix,
                        combine_matrix,
                        scheduler_next_matrix,
                        num_gpus,
                        model=model,
                        expert_compute_delay=expert_compute_delay,
                        **extra_kwargs,
                    )
                except Exception as exc:
                    algorithm_failures[name].append(type(exc).__name__)
                    payload = {
                        "makespan": greedy_makespan,
                        "solve_time_ms": 0.0,
                        "schedule": [],
                        "strategy": name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                algorithm_results[name] = payload

            fast_result: dict[str, Any] | None = None
            fast_latency_ms = 0.0
            fast_makespan = greedy_makespan
            if include_fast_best_of:
                fast_result = fast_schedule_pairwise(
                    dispatch_matrix,
                    combine_matrix,
                    scheduler_next_matrix,
                    num_gpus,
                    model=model,
                    expert_compute_delay=expert_compute_delay,
                )
                fast_latency_ms = float(fast_result["solve_time_ms"])
                fast_makespan = float(fast_result["makespan"])

            oracle_perfect: dict[str, Any] = {
                "makespan": None,
                "chunk_count": 0,
                "schedule": [],
                "solver_status": "skipped",
            }
            oracle_predicted: dict[str, Any] = {
                "makespan": None,
                "chunk_count": 0,
                "schedule": [],
                "solver_status": "skipped",
            }
            perfect_solve_time = 0.0
            predicted_solve_time = 0.0
            if not skip_oracle:
                t_solve = time.time()
                oracle_perfect, perfect_cached = solve_pairwise_cached(dispatch_matrix, combine_matrix, next_actual_matrix)
                perfect_solve_time = time.time() - t_solve
                oracle_perfect_statuses.append(str(oracle_perfect.get("solver_status", "unknown")))
                if not perfect_cached or loop_counter <= 5 or loop_counter % 100 == 0:
                    print(f"[perf] pairwise_oracle perfect [{sample_id}..L{from_layer}->{to_layer}]: {perfect_solve_time:.3f}s cached={perfect_cached} (status={oracle_perfect.get('solver_status', '?')})", flush=True)

                t_solve2 = time.time()
                oracle_predicted, predicted_cached = solve_pairwise_cached(dispatch_matrix, combine_matrix, next_predicted_matrix)
                predicted_solve_time = time.time() - t_solve2
                oracle_predicted_statuses.append(str(oracle_predicted.get("solver_status", "unknown")))
                if not predicted_cached or loop_counter <= 5 or loop_counter % 100 == 0:
                    print(f"[perf] pairwise_oracle predicted [{sample_id}..L{from_layer}->{to_layer}]: {predicted_solve_time:.3f}s cached={predicted_cached} (status={oracle_predicted.get('solver_status', '?')})", flush=True)

            oracle_perfect_makespan = float(oracle_perfect["makespan"]) if oracle_perfect["makespan"] is not None else greedy_makespan
            oracle_predicted_makespan = float(oracle_predicted["makespan"]) if oracle_predicted["makespan"] is not None else greedy_makespan

            result_row: dict[str, Any] = {
                "sample_id": sample_id,
                "layer_pair": f"{from_layer}->{to_layer}",
                "dispatch_matrix": dispatch_matrix,
                "combine_matrix": combine_matrix,
                "next_actual_matrix": next_actual_matrix,
                "next_predicted_matrix": next_predicted_matrix,
                "scheduler_next_matrix": scheduler_next_matrix,
                "greedy_makespan": greedy_makespan,
                "greedy_makespan_ms": greedy_e2e["greedy_makespan_ms"],
                "greedy_latency_ms": greedy_latency_ms,
                "greedy_comm_savings_ms": 0.0,
                "greedy_exposed_latency_ms": 0.0,
                "greedy_net_benefit_ms": 0.0,
                "greedy_effective_e2e_time_ms": greedy_e2e["greedy_makespan_ms"],
                "greedy_e2e_speedup_pct": 0.0,
                "baseline_layer_time_ms": hidden_window_ms,
                "fast_makespan": fast_makespan,
                "fast_latency_ms": fast_latency_ms,
                "fast_schedule": [] if fast_result is None else fast_result["schedule"],
                "fast_candidates": [] if fast_result is None else fast_result.get("candidates", []),
                "oracle_perfect_makespan": oracle_perfect_makespan,
                "oracle_predicted_makespan": oracle_predicted_makespan,
                "oracle_perfect_latency_ms": perfect_solve_time * 1000.0,
                "oracle_predicted_latency_ms": predicted_solve_time * 1000.0,
                "oracle_perfect_chunk_count": oracle_perfect["chunk_count"],
                "oracle_predicted_chunk_count": oracle_predicted["chunk_count"],
                "oracle_perfect_schedule": oracle_perfect["schedule"],
                "oracle_predicted_schedule": oracle_predicted["schedule"],
                "oracle_perfect_solver_status": oracle_perfect.get("solver_status"),
                "oracle_predicted_solver_status": oracle_predicted.get("solver_status"),
            }

            for name, payload in algorithm_results.items():
                makespan = float(payload["makespan"])
                latency_ms = float(payload["solve_time_ms"])
                improvement_pct = 0.0 if greedy_makespan == 0.0 else ((greedy_makespan - makespan) / greedy_makespan) * 100.0
                phase0_makespan = max(
                    (
                        float(entry["end"])
                        for entry in payload.get("schedule", [])
                        if int(entry.get("phase", -1)) == 0
                    ),
                    default=0.0,
                )
                prediction_aware = name in PREDICTION_AWARE_ALGORITHMS
                effective_makespan = compute_effective_makespan(
                    makespan,
                    phase0_makespan,
                    expert_compute_delay,
                    prediction_aware,
                )
                effective_improvement_pct = (
                    0.0
                    if greedy_makespan == 0.0
                    else ((greedy_makespan - effective_makespan) / greedy_makespan) * 100.0
                )
                e2e_metrics = compute_e2e_metrics(makespan, latency_ms, greedy_makespan)
                algorithm_improvements[name].append(improvement_pct)
                algorithm_effective_improvements[name].append(effective_improvement_pct)
                algorithm_latencies[name].append(latency_ms)
                algorithm_effective_latencies[name].append(latency_ms)
                algorithm_comm_savings_ms[name].append(e2e_metrics["comm_savings_ms"])
                algorithm_exposed_latency_ms[name].append(e2e_metrics["exposed_latency_ms"])
                algorithm_net_benefit_ms[name].append(e2e_metrics["net_benefit_ms"])
                algorithm_effective_e2e_time_ms[name].append(e2e_metrics["effective_e2e_time_ms"])
                algorithm_e2e_speedup_pct[name].append(e2e_metrics["e2e_speedup_pct"])
                result_row[f"{name}_makespan"] = makespan
                result_row[f"{name}_makespan_ms"] = e2e_metrics["algo_makespan_ms"]
                result_row[f"{name}_effective_makespan"] = effective_makespan
                result_row[f"{name}_latency_ms"] = latency_ms
                result_row[f"{name}_effective_latency_ms"] = latency_ms
                result_row[f"{name}_improvement_pct"] = improvement_pct
                result_row[f"{name}_effective_improvement_pct"] = effective_improvement_pct
                result_row[f"{name}_comm_savings_ms"] = e2e_metrics["comm_savings_ms"]
                result_row[f"{name}_exposed_latency_ms"] = e2e_metrics["exposed_latency_ms"]
                result_row[f"{name}_net_benefit_ms"] = e2e_metrics["net_benefit_ms"]
                result_row[f"{name}_effective_e2e_time_ms"] = e2e_metrics["effective_e2e_time_ms"]
                result_row[f"{name}_e2e_speedup_pct"] = e2e_metrics["e2e_speedup_pct"]
                result_row[f"{name}_prediction_aware"] = prediction_aware
                result_row[f"{name}_schedule"] = payload["schedule"]
                if "error" in payload:
                    result_row[f"{name}_error"] = payload["error"]

            fast_improvement_pct = 0.0 if greedy_makespan == 0.0 else ((greedy_makespan - fast_makespan) / greedy_makespan) * 100.0
            fast_e2e = compute_e2e_metrics(fast_makespan, fast_latency_ms, greedy_makespan) if include_fast_best_of else None
            perfect_improvement_pct = 0.0 if greedy_makespan == 0.0 else ((greedy_makespan - oracle_perfect_makespan) / greedy_makespan) * 100.0
            predicted_improvement_pct = 0.0 if greedy_makespan == 0.0 else ((greedy_makespan - oracle_predicted_makespan) / greedy_makespan) * 100.0
            oracle_perfect_e2e = compute_e2e_metrics(oracle_perfect_makespan, perfect_solve_time * 1000.0, greedy_makespan)
            oracle_predicted_e2e = compute_e2e_metrics(oracle_predicted_makespan, predicted_solve_time * 1000.0, greedy_makespan)
            oracle_prediction_gap_pct = perfect_improvement_pct - predicted_improvement_pct
            flat_actual = [float(value) for row in next_actual_matrix for value in row]
            flat_predicted = [float(value) for row in next_predicted_matrix for value in row]
            traffic_corr = spearman_rank_correlation(flat_actual, flat_predicted)

            result_row["fast_improvement_pct"] = fast_improvement_pct
            result_row["fast_comm_savings_ms"] = 0.0 if fast_e2e is None else fast_e2e["comm_savings_ms"]
            result_row["fast_exposed_latency_ms"] = 0.0 if fast_e2e is None else fast_e2e["exposed_latency_ms"]
            result_row["fast_net_benefit_ms"] = 0.0 if fast_e2e is None else fast_e2e["net_benefit_ms"]
            result_row["fast_effective_e2e_time_ms"] = (
                greedy_e2e["greedy_makespan_ms"] if fast_e2e is None else fast_e2e["effective_e2e_time_ms"]
            )
            result_row["fast_e2e_speedup_pct"] = 0.0 if fast_e2e is None else fast_e2e["e2e_speedup_pct"]
            result_row["perfect_improvement_pct"] = perfect_improvement_pct
            result_row["predicted_improvement_pct"] = predicted_improvement_pct
            result_row["oracle_perfect_comm_savings_ms"] = oracle_perfect_e2e["comm_savings_ms"]
            result_row["oracle_perfect_exposed_latency_ms"] = oracle_perfect_e2e["exposed_latency_ms"]
            result_row["oracle_perfect_net_benefit_ms"] = oracle_perfect_e2e["net_benefit_ms"]
            result_row["oracle_perfect_effective_e2e_time_ms"] = oracle_perfect_e2e["effective_e2e_time_ms"]
            result_row["oracle_perfect_e2e_speedup_pct"] = oracle_perfect_e2e["e2e_speedup_pct"]
            result_row["oracle_predicted_comm_savings_ms"] = oracle_predicted_e2e["comm_savings_ms"]
            result_row["oracle_predicted_exposed_latency_ms"] = oracle_predicted_e2e["exposed_latency_ms"]
            result_row["oracle_predicted_net_benefit_ms"] = oracle_predicted_e2e["net_benefit_ms"]
            result_row["oracle_predicted_effective_e2e_time_ms"] = oracle_predicted_e2e["effective_e2e_time_ms"]
            result_row["oracle_predicted_e2e_speedup_pct"] = oracle_predicted_e2e["e2e_speedup_pct"]
            result_row["oracle_prediction_gap_pct"] = oracle_prediction_gap_pct
            result_row["traffic_correlation"] = traffic_corr

            if not skip_oracle:
                perfect_improvements.append(perfect_improvement_pct)
                predicted_improvements.append(predicted_improvement_pct)
                oracle_prediction_gaps.append(oracle_prediction_gap_pct)
                oracle_perfect_latencies_ms.append(perfect_solve_time * 1000.0)
                oracle_predicted_latencies_ms.append(predicted_solve_time * 1000.0)
            if include_fast_best_of:
                fast_improvements.append(fast_improvement_pct)
                assert fast_e2e is not None
                fast_comm_savings_ms.append(fast_e2e["comm_savings_ms"])
                fast_exposed_latency_ms.append(fast_e2e["exposed_latency_ms"])
                fast_net_benefit_ms.append(fast_e2e["net_benefit_ms"])
                fast_effective_e2e_time_ms.append(fast_e2e["effective_e2e_time_ms"])
                fast_e2e_speedup_pct.append(fast_e2e["e2e_speedup_pct"])
            traffic_correlations.append(traffic_corr)
            greedy_latencies_ms.append(greedy_latency_ms)
            greedy_makespans_ms.append(greedy_e2e["greedy_makespan_ms"])
            if include_fast_best_of:
                fast_latencies_ms.append(fast_latency_ms)
            if not skip_oracle:
                oracle_perfect_comm_savings_ms.append(oracle_perfect_e2e["comm_savings_ms"])
                oracle_perfect_exposed_latency_ms.append(oracle_perfect_e2e["exposed_latency_ms"])
                oracle_perfect_net_benefit_ms.append(oracle_perfect_e2e["net_benefit_ms"])
                oracle_perfect_effective_e2e_time_ms.append(oracle_perfect_e2e["effective_e2e_time_ms"])
                oracle_perfect_e2e_speedup_pct.append(oracle_perfect_e2e["e2e_speedup_pct"])
                oracle_predicted_comm_savings_ms.append(oracle_predicted_e2e["comm_savings_ms"])
                oracle_predicted_exposed_latency_ms.append(oracle_predicted_e2e["exposed_latency_ms"])
                oracle_predicted_net_benefit_ms.append(oracle_predicted_e2e["net_benefit_ms"])
                oracle_predicted_effective_e2e_time_ms.append(oracle_predicted_e2e["effective_e2e_time_ms"])
                oracle_predicted_e2e_speedup_pct.append(oracle_predicted_e2e["e2e_speedup_pct"])
            results.append(result_row)

    predicted_vs_corr = (
        spearman_rank_correlation(predicted_improvements, traffic_correlations)
        if len(predicted_improvements) >= 2
        else 0.0
    )
    summary: dict[str, Any] = {
        "pair_count": len(results),
        "model": model,
        "expert_compute_delay": expert_compute_delay,
        "baseline_layer_time_ms": hidden_window_ms,
        "hidden_window_ms": hidden_window_ms,
        "token_to_ms_factor": token_to_ms_factor,
        "next_mode": next_mode,
        "phase2_source": phase2_source,
        "traffic_correlation": _summary_stats(traffic_correlations),
        "greedy_latency_ms": _summary_stats(greedy_latencies_ms),
        "greedy_makespan_ms": _summary_stats(greedy_makespans_ms),
        "include_fast_best_of": include_fast_best_of,
        "skip_oracle": skip_oracle,
        "predicted_improvement_vs_traffic_correlation": predicted_vs_corr if not skip_oracle else None,
        "gate2_decision": evaluate_gate2(perfect_improvements, predicted_improvements) if not skip_oracle else None,
    }
    if include_fast_best_of:
        summary["fast_improvement_pct"] = _summary_stats(fast_improvements)
        summary["fast_latency_ms"] = _summary_stats(fast_latencies_ms)
        summary["fast_comm_savings_ms"] = _summary_stats(fast_comm_savings_ms)
        summary["fast_exposed_latency_ms"] = _summary_stats(fast_exposed_latency_ms)
        summary["fast_net_benefit_ms"] = _summary_stats(fast_net_benefit_ms)
        summary["fast_effective_e2e_time_ms"] = _summary_stats(fast_effective_e2e_time_ms)
        summary["fast_e2e_speedup_pct"] = _summary_stats(fast_e2e_speedup_pct)
    if not skip_oracle:
        summary["perfect_improvement_pct"] = _summary_stats(perfect_improvements)
        summary["predicted_improvement_pct"] = _summary_stats(predicted_improvements)
        summary["oracle_prediction_gap_pct"] = _summary_stats(oracle_prediction_gaps)
        summary["oracle_perfect_latency_ms"] = _summary_stats(oracle_perfect_latencies_ms)
        summary["oracle_predicted_latency_ms"] = _summary_stats(oracle_predicted_latencies_ms)
        summary["oracle_perfect_comm_savings_ms"] = _summary_stats(oracle_perfect_comm_savings_ms)
        summary["oracle_perfect_exposed_latency_ms"] = _summary_stats(oracle_perfect_exposed_latency_ms)
        summary["oracle_perfect_net_benefit_ms"] = _summary_stats(oracle_perfect_net_benefit_ms)
        summary["oracle_perfect_effective_e2e_time_ms"] = _summary_stats(oracle_perfect_effective_e2e_time_ms)
        summary["oracle_perfect_e2e_speedup_pct"] = _summary_stats(oracle_perfect_e2e_speedup_pct)
        summary["oracle_predicted_comm_savings_ms"] = _summary_stats(oracle_predicted_comm_savings_ms)
        summary["oracle_predicted_exposed_latency_ms"] = _summary_stats(oracle_predicted_exposed_latency_ms)
        summary["oracle_predicted_net_benefit_ms"] = _summary_stats(oracle_predicted_net_benefit_ms)
        summary["oracle_predicted_effective_e2e_time_ms"] = _summary_stats(oracle_predicted_effective_e2e_time_ms)
        summary["oracle_predicted_e2e_speedup_pct"] = _summary_stats(oracle_predicted_e2e_speedup_pct)
        summary["oracle_perfect_solver_statuses"] = dict(Counter(oracle_perfect_statuses))
        summary["oracle_predicted_solver_statuses"] = dict(Counter(oracle_predicted_statuses))
    for name, _scheduler in selected_fast_algorithms:
        metadata = get_strategy_metadata(name)
        summary[f"{name}_improvement_pct"] = _summary_stats(algorithm_improvements[name])
        summary[f"{name}_effective_improvement_pct"] = _summary_stats(algorithm_effective_improvements[name])
        summary[f"{name}_latency_ms"] = _summary_stats(algorithm_latencies[name])
        summary[f"{name}_effective_latency_ms"] = _summary_stats(algorithm_effective_latencies[name])
        summary[f"{name}_comm_savings_ms"] = _summary_stats(algorithm_comm_savings_ms[name])
        summary[f"{name}_exposed_latency_ms"] = _summary_stats(algorithm_exposed_latency_ms[name])
        summary[f"{name}_net_benefit_ms"] = _summary_stats(algorithm_net_benefit_ms[name])
        summary[f"{name}_effective_e2e_time_ms"] = _summary_stats(algorithm_effective_e2e_time_ms[name])
        summary[f"{name}_e2e_speedup_pct"] = _summary_stats(algorithm_e2e_speedup_pct[name])
        summary[f"{name}_prediction_aware"] = bool(metadata["prediction_aware"])
        summary[f"{name}_description"] = str(metadata["description"])
        if algorithm_failures[name]:
            summary[f"{name}_failures"] = dict(Counter(algorithm_failures[name]))

    print(f"[perf] total run_pairwise_analysis: {time.time() - t0:.2f}s", flush=True)
    return {"results": results, "summary": summary}


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
