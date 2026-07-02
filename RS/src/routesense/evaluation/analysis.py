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
    fast_schedule_ejection_chain_tabu,
    fast_schedule_grasp,
    fast_schedule_ibbr,
    fast_schedule_lagrangian,
    fast_schedule_lns,
    fast_schedule_lns_cp_repair,
    fast_schedule_pairwise,
    fast_schedule_quantized_decomposed,
    fast_schedule_simulated_annealing,
    fast_schedule_two_stage,
    greedy_schedule_pairwise,
    pairwise_oracle,
)
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
    ("cp_lpt", fast_schedule_cp_lpt),
    ("birkhoff", fast_schedule_birkhoff),
    ("barrier_aware_birkhoff", fast_schedule_barrier_aware_birkhoff),
    ("lns", fast_schedule_lns),
    ("simulated_annealing", fast_schedule_simulated_annealing),
    ("lagrangian", fast_schedule_lagrangian),
    ("grasp", fast_schedule_grasp),
    ("two_stage", fast_schedule_two_stage),
    ("ibbr", fast_schedule_ibbr),
    ("ejection_chain_tabu", fast_schedule_ejection_chain_tabu),
    ("lns_cp_repair", fast_schedule_lns_cp_repair),
    ("quantized_decomposed", fast_schedule_quantized_decomposed),
]

PREDICTION_AWARE_ALGORITHMS = {"cp_lpt", "lagrangian"}


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
    fast_algorithms: list[tuple[str, SchedulerFn]] | None = None,
    skip_oracle: bool = False,
    include_fast_best_of: bool = True,
) -> dict[str, Any]:
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
    algorithm_failures: dict[str, list[str]] = {name: [] for name, _ in selected_fast_algorithms}
    results: list[dict[str, Any]] = []
    predicted_improvements: list[float] = []
    perfect_improvements: list[float] = []
    fast_improvements: list[float] = []
    oracle_prediction_gaps: list[float] = []
    traffic_correlations: list[float] = []
    greedy_latencies_ms: list[float] = []
    fast_latencies_ms: list[float] = []
    oracle_perfect_latencies_ms: list[float] = []
    oracle_predicted_latencies_ms: list[float] = []
    oracle_perfect_statuses: list[str] = []
    oracle_predicted_statuses: list[str] = []
    pairwise_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    loop_counter = 0

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
            if loop_counter <= 5 or loop_counter % 100 == 0:
                print(f"[perf] build_predicted_traffic [{sample_id}..L{from_layer}->{to_layer}]: {time.time() - t_build:.3f}s", flush=True)

            t_greedy = time.time()
            greedy_makespan = greedy_schedule_pairwise(
                dispatch_matrix,
                combine_matrix,
                next_actual_matrix,
                num_gpus,
                model=model,
                expert_compute_delay=expert_compute_delay,
            )
            greedy_latency_ms = (time.time() - t_greedy) * 1000.0

            algorithm_results: dict[str, dict[str, Any]] = {}
            for name, scheduler in selected_fast_algorithms:
                try:
                    payload = scheduler(
                        dispatch_matrix,
                        combine_matrix,
                        next_predicted_matrix,
                        num_gpus,
                        model=model,
                        expert_compute_delay=expert_compute_delay,
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
                    next_predicted_matrix,
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
                "greedy_makespan": greedy_makespan,
                "greedy_latency_ms": greedy_latency_ms,
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
                algorithm_improvements[name].append(improvement_pct)
                algorithm_effective_improvements[name].append(effective_improvement_pct)
                algorithm_latencies[name].append(latency_ms)
                result_row[f"{name}_makespan"] = makespan
                result_row[f"{name}_effective_makespan"] = effective_makespan
                result_row[f"{name}_latency_ms"] = latency_ms
                result_row[f"{name}_improvement_pct"] = improvement_pct
                result_row[f"{name}_effective_improvement_pct"] = effective_improvement_pct
                result_row[f"{name}_prediction_aware"] = prediction_aware
                result_row[f"{name}_schedule"] = payload["schedule"]
                if "error" in payload:
                    result_row[f"{name}_error"] = payload["error"]

            fast_improvement_pct = 0.0 if greedy_makespan == 0.0 else ((greedy_makespan - fast_makespan) / greedy_makespan) * 100.0
            perfect_improvement_pct = 0.0 if greedy_makespan == 0.0 else ((greedy_makespan - oracle_perfect_makespan) / greedy_makespan) * 100.0
            predicted_improvement_pct = 0.0 if greedy_makespan == 0.0 else ((greedy_makespan - oracle_predicted_makespan) / greedy_makespan) * 100.0
            oracle_prediction_gap_pct = perfect_improvement_pct - predicted_improvement_pct
            flat_actual = [float(value) for row in next_actual_matrix for value in row]
            flat_predicted = [float(value) for row in next_predicted_matrix for value in row]
            traffic_corr = spearman_rank_correlation(flat_actual, flat_predicted)

            result_row["fast_improvement_pct"] = fast_improvement_pct
            result_row["perfect_improvement_pct"] = perfect_improvement_pct
            result_row["predicted_improvement_pct"] = predicted_improvement_pct
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
            traffic_correlations.append(traffic_corr)
            greedy_latencies_ms.append(greedy_latency_ms)
            if include_fast_best_of:
                fast_latencies_ms.append(fast_latency_ms)
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
        "traffic_correlation": _summary_stats(traffic_correlations),
        "greedy_latency_ms": _summary_stats(greedy_latencies_ms),
        "include_fast_best_of": include_fast_best_of,
        "skip_oracle": skip_oracle,
        "predicted_improvement_vs_traffic_correlation": predicted_vs_corr if not skip_oracle else None,
        "gate2_decision": evaluate_gate2(perfect_improvements, predicted_improvements) if not skip_oracle else None,
    }
    if include_fast_best_of:
        summary["fast_improvement_pct"] = _summary_stats(fast_improvements)
        summary["fast_latency_ms"] = _summary_stats(fast_latencies_ms)
    if not skip_oracle:
        summary["perfect_improvement_pct"] = _summary_stats(perfect_improvements)
        summary["predicted_improvement_pct"] = _summary_stats(predicted_improvements)
        summary["oracle_prediction_gap_pct"] = _summary_stats(oracle_prediction_gaps)
        summary["oracle_perfect_latency_ms"] = _summary_stats(oracle_perfect_latencies_ms)
        summary["oracle_predicted_latency_ms"] = _summary_stats(oracle_predicted_latencies_ms)
        summary["oracle_perfect_solver_statuses"] = dict(Counter(oracle_perfect_statuses))
        summary["oracle_predicted_solver_statuses"] = dict(Counter(oracle_predicted_statuses))
    for name, _scheduler in selected_fast_algorithms:
        summary[f"{name}_improvement_pct"] = _summary_stats(algorithm_improvements[name])
        summary[f"{name}_effective_improvement_pct"] = _summary_stats(algorithm_effective_improvements[name])
        summary[f"{name}_latency_ms"] = _summary_stats(algorithm_latencies[name])
        summary[f"{name}_prediction_aware"] = name in PREDICTION_AWARE_ALGORITHMS
        if algorithm_failures[name]:
            summary[f"{name}_failures"] = dict(Counter(algorithm_failures[name]))

    print(f"[perf] total run_pairwise_analysis: {time.time() - t0:.2f}s", flush=True)
    return {"results": results, "summary": summary}


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
