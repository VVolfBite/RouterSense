"""State transitions for multiphase global ready-set scheduling."""

from __future__ import annotations

import math
import time
from typing import Any

from .dependency_model import collect_real_flows, inbound_remaining, outbound_loads
from .flow_model import EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE
from .matching import greedy_maximal_matching, maximum_weight_matching
from .replay import replay_and_audit_schedule
from .scoring import ready_flow_candidates


def run_global_matching_scheduler(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    strategy: str,
    mode: str,
    prediction_confidence: float,
    expert_compute_delay: float,
    exact_matching: bool,
    wave_quantum: float | None,
    max_waves: int,
    residual_weight: float,
    barrier_weight: float,
    age_weight: float,
    prediction_weight: float,
    adaptive_prices: bool,
    price_step: float,
    price_decay: float,
    price_clip: float,
    iteration_budget: int,
    atomic: bool,
    base_score_lookup: dict[str, float] | None = None,
    base_priority_weight: float = 0.0,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    flows = collect_real_flows(dispatch_matrix, combine_matrix, next_dispatch_matrix, mode=mode)
    residual = {flow.flow_id: float(flow.volume) for flow in flows}
    inbound = inbound_remaining(flows, num_gpus)
    release_time = {(phase, gpu): (0.0 if phase == 0 else math.inf) for phase in range(3) for gpu in range(num_gpus)}
    for phase in (1, 2):
        prev_phase = phase - 1
        for gpu in range(num_gpus):
            if inbound.get((prev_phase, gpu), 0.0) <= 1e-9:
                release_time[(phase, gpu)] = expert_compute_delay if prev_phase == 0 else 0.0
    barrier_done = {(phase, gpu): 0.0 for phase in range(3) for gpu in range(num_gpus)}
    ready_since: dict[str, float] = {}
    downstream_load = outbound_loads(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        mode=mode,
        prediction_confidence=prediction_confidence,
    )
    prices = {(phase, gpu): 0.0 for phase in range(2) for gpu in range(num_gpus)}
    current_time = 0.0
    schedule: list[dict[str, Any]] = []
    wave_count = 0
    max_rounds = max(1, max_waves)
    while any(value > 1e-9 for value in residual.values()) and wave_count < max_rounds:
        ready = ready_flow_candidates(
            flows=flows,
            residual=residual,
            ready_since=ready_since,
            current_time=current_time,
            release_time=release_time,
            inbound_remaining=inbound,
            downstream_load=downstream_load,
            age_scale=max(1.0, current_time + 1.0),
            residual_weight=residual_weight,
            barrier_weight=barrier_weight,
            age_weight=age_weight,
            prediction_weight=prediction_weight,
            mode=mode,
            prediction_confidence=prediction_confidence,
            base_score_lookup=base_score_lookup,
            base_priority_weight=base_priority_weight,
        )
        if adaptive_prices:
            for candidate in ready:
                if candidate["phase"] < 2:
                    candidate["score"] += prices[(candidate["phase"], candidate["dst_gpu"])]
        if not ready:
            future = [release for release in release_time.values() if release < math.inf and release > current_time + 1e-9]
            if not future:
                break
            current_time = min(future)
            continue
        chosen = maximum_weight_matching(ready, num_gpus) if exact_matching else greedy_maximal_matching(ready)
        if not chosen:
            break
        duration = min(float(candidate["residual"]) for candidate in chosen)
        if wave_quantum is not None:
            duration = min(duration, float(wave_quantum))
        duration = max(duration, 1e-6)
        wave_end = current_time
        for candidate in chosen:
            flow_id = str(candidate["flow_id"])
            phase = int(candidate["phase"])
            src = int(candidate["src_gpu"])
            dst = int(candidate["dst_gpu"])
            served = max(0.0, residual[flow_id]) if atomic else duration
            residual[flow_id] = max(0.0, residual[flow_id] - served)
            inbound[(phase, dst)] = max(0.0, inbound[(phase, dst)] - served)
            end = current_time + served
            wave_end = max(wave_end, end)
            barrier_done[(phase, dst)] = max(barrier_done[(phase, dst)], end)
            if phase < 2 and inbound[(phase, dst)] <= 1e-9:
                release_time[(phase + 1, dst)] = barrier_done[(phase, dst)] + (expert_compute_delay if phase == 0 else 0.0)
            schedule.append(
                {
                    "chunk_id": f"{flow_id}_wave{wave_count}",
                    "flow_id": flow_id,
                    "phase": phase,
                    "size": float(served),
                    "served_volume": float(served),
                    "src": src,
                    "dst": dst,
                    "src_gpu": src,
                    "dst_gpu": dst,
                    "start": current_time,
                    "end": end,
                    "wave_id": wave_count,
                    "priority": [
                        float(candidate["score"]),
                        float(candidate["barrier_urgency"]),
                        float(candidate["age"]),
                        float(candidate["prediction_bonus"]),
                        float(candidate["base_priority"]),
                    ],
                }
            )
        current_time = wave_end if atomic else current_time + duration
        wave_count += 1
        if adaptive_prices:
            for _ in range(max(1, iteration_budget)):
                for phase in range(2):
                    for gpu in range(num_gpus):
                        pressure = downstream_load.get((phase + 1, gpu), 0.0)
                        remaining = inbound.get((phase, gpu), 0.0)
                        value = 0.0 if pressure <= 0.0 else pressure / max(remaining, 1.0)
                        updated = (1.0 - price_decay) * prices[(phase, gpu)] + price_step * value
                        prices[(phase, gpu)] = max(-price_clip, min(price_clip, updated))
    residual_nonzero = any(value > 1e-9 for value in residual.values())
    solver_status = "max_wave_limit_exceeded" if residual_nonzero and wave_count >= max_rounds else "completed"
    makespan = max((float(entry["end"]) for entry in schedule), default=0.0)
    planning_time_ms = (time.perf_counter() - start_time) * 1000.0
    audit = replay_and_audit_schedule(
        schedule=schedule,
        dispatch_matrix=dispatch_matrix,
        combine_matrix=combine_matrix,
        next_dispatch_matrix=next_dispatch_matrix,
        num_gpus=num_gpus,
        expert_compute_delay=expert_compute_delay,
        mode=mode,
        scheduler_name=strategy,
        planning_time_ms=planning_time_ms,
        reported_makespan=makespan,
        prediction_used=prediction_confidence > 0.0 and mode == RUNTIME_LOOKAHEAD_MODE,
    )
    if residual_nonzero and solver_status == "max_wave_limit_exceeded":
        audit = {
            **audit,
            "valid": False,
            "residual_nonzero": True,
            "validation_errors": [
                *list(audit.get("validation_errors", [])),
                "max_wave_limit_exceeded with residual nonzero",
            ],
        }
    return {
        "makespan": makespan,
        "schedule": schedule,
        "solve_time_ms": planning_time_ms,
        "strategy": strategy,
        "mode": mode,
        "prediction_used": prediction_confidence > 0.0 and mode == RUNTIME_LOOKAHEAD_MODE,
        "wave_count": wave_count,
        "atomic": atomic,
        "audit": audit,
        "solver_status": solver_status,
        "residual_nonzero": residual_nonzero,
        "residual_remaining": {flow_id: value for flow_id, value in residual.items() if value > 1e-9},
    }
