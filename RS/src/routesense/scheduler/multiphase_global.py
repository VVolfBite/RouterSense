from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


EXECUTION_WINDOW_MODE = "execution_window"
RUNTIME_LOOKAHEAD_MODE = "runtime_lookahead"


@dataclass(frozen=True)
class GlobalFlow:
    flow_id: str
    phase: int
    src_gpu: int
    dst_gpu: int
    volume: float


def _collect_real_flows(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    *,
    mode: str,
) -> list[GlobalFlow]:
    matrices = [dispatch_matrix, combine_matrix]
    if mode == EXECUTION_WINDOW_MODE:
        matrices.append(next_dispatch_matrix)
    elif mode != RUNTIME_LOOKAHEAD_MODE:
        raise ValueError(f"unsupported mode {mode!r}")
    flows: list[GlobalFlow] = []
    for phase, matrix in enumerate(matrices):
        for src, row in enumerate(matrix):
            for dst, value in enumerate(row):
                size = float(value)
                if src == dst or size <= 0.0:
                    continue
                flows.append(
                    GlobalFlow(
                        flow_id=f"phase{phase}_src{src}_dst{dst}",
                        phase=phase,
                        src_gpu=src,
                        dst_gpu=dst,
                        volume=size,
                    )
                )
    return flows


def _outbound_loads(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    *,
    mode: str,
    prediction_confidence: float,
) -> dict[tuple[int, int], float]:
    matrices = [dispatch_matrix, combine_matrix, next_dispatch_matrix]
    loads: dict[tuple[int, int], float] = {}
    for phase, matrix in enumerate(matrices):
        scale = 1.0
        if phase == 2 and mode == RUNTIME_LOOKAHEAD_MODE:
            scale = max(0.0, min(1.0, prediction_confidence))
        for gpu, row in enumerate(matrix):
            loads[(phase, gpu)] = scale * float(sum(max(int(value), 0) for value in row))
    return loads


def _inbound_remaining(flows: list[GlobalFlow], num_gpus: int) -> dict[tuple[int, int], float]:
    remaining = {(phase, gpu): 0.0 for phase in range(3) for gpu in range(num_gpus)}
    for flow in flows:
        remaining[(flow.phase, flow.dst_gpu)] += float(flow.volume)
    return remaining


def replay_and_audit_schedule(
    *,
    schedule: list[dict[str, Any]],
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    mode: str = EXECUTION_WINDOW_MODE,
    scheduler_name: str = "unknown",
    planning_time_ms: float = 0.0,
    reported_makespan: float | None = None,
    prediction_used: bool = False,
) -> dict[str, Any]:
    del model  # Current validator is defined only for the full-duplex local-barrier semantics.
    errors: list[str] = []
    target_flows = _collect_real_flows(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        mode=mode,
    )
    target_by_key = {
        (flow.phase, flow.src_gpu, flow.dst_gpu): float(flow.volume)
        for flow in target_flows
    }
    served_by_key = {key: 0.0 for key in target_by_key}
    send_intervals: dict[int, list[tuple[float, float, dict[str, Any]]]] = {gpu: [] for gpu in range(num_gpus)}
    recv_intervals: dict[int, list[tuple[float, float, dict[str, Any]]]] = {gpu: [] for gpu in range(num_gpus)}
    barrier_times = {0: [0.0] * num_gpus, 1: [0.0] * num_gpus, 2: [0.0] * num_gpus}
    recv_remaining = _inbound_remaining(target_flows, num_gpus)

    ordered_schedule = sorted(
        schedule,
        key=lambda item: (float(item["start"]), float(item["end"]), int(item["phase"]), int(item["src_gpu"]), int(item["dst_gpu"])),
    )

    for entry in ordered_schedule:
        phase = int(entry["phase"])
        src = int(entry["src_gpu"])
        dst = int(entry["dst_gpu"])
        start = float(entry["start"])
        end = float(entry["end"])
        served = float(entry.get("served_volume", entry.get("size", 0.0)))
        key = (phase, src, dst)
        if mode == RUNTIME_LOOKAHEAD_MODE and phase == 2:
            errors.append("runtime_lookahead mode contains real phase-2 schedule entry")
        if key not in target_by_key:
            errors.append(f"unexpected flow served: {key}")
            continue
        if served <= 0.0 or end < start:
            errors.append(f"invalid interval for {key}: start={start} end={end} served={served}")
        if phase > 0:
            required = barrier_times[phase - 1][src] + (expert_compute_delay if phase == 1 else 0.0)
            if start + 1e-9 < required:
                errors.append(
                    f"barrier violation for phase={phase}, src={src}: start={start:.6f} < required={required:.6f}"
                )
        send_intervals[src].append((start, end, entry))
        recv_intervals[dst].append((start, end, entry))
        served_by_key[key] += served
        recv_remaining[(phase, dst)] = max(0.0, recv_remaining[(phase, dst)] - served)
        barrier_times[phase][dst] = max(barrier_times[phase][dst], end)

    for gpu, intervals in send_intervals.items():
        intervals.sort(key=lambda item: (item[0], item[1]))
        for (_, prev_end, prev_entry), (start, _end, entry) in zip(intervals, intervals[1:], strict=False):
            if start < prev_end - 1e-9:
                errors.append(
                    f"send_port overlap on gpu {gpu}: {prev_entry['chunk_id']} overlaps {entry['chunk_id']}"
                )
    for gpu, intervals in recv_intervals.items():
        intervals.sort(key=lambda item: (item[0], item[1]))
        for (_, prev_end, prev_entry), (start, _end, entry) in zip(intervals, intervals[1:], strict=False):
            if start < prev_end - 1e-9:
                errors.append(
                    f"recv_port overlap on gpu {gpu}: {prev_entry['chunk_id']} overlaps {entry['chunk_id']}"
                )

    for key, target in target_by_key.items():
        served = served_by_key.get(key, 0.0)
        if abs(served - target) > 1e-6:
            errors.append(f"volume mismatch for {key}: served={served:.6f} target={target:.6f}")
    for key, remaining in recv_remaining.items():
        if remaining > 1e-6:
            errors.append(f"incomplete inbound barrier volume for phase={key[0]} gpu={key[1]} remaining={remaining:.6f}")

    replay_makespan = max((float(entry["end"]) for entry in schedule), default=0.0)
    if reported_makespan is not None and abs(float(reported_makespan) - replay_makespan) > 1e-6:
        errors.append(
            f"reported makespan mismatch: reported={float(reported_makespan):.6f} replay={replay_makespan:.6f}"
        )

    send_busy_time = [
        sum(max(0.0, float(end) - float(start)) for start, end, _entry in send_intervals[gpu])
        for gpu in range(num_gpus)
    ]
    recv_busy_time = [
        sum(max(0.0, float(end) - float(start)) for start, end, _entry in recv_intervals[gpu])
        for gpu in range(num_gpus)
    ]
    if any("wave_id" in entry for entry in schedule):
        wave_count = len({int(entry["wave_id"]) for entry in schedule})
    else:
        wave_count = len(
            {
                (float(entry["start"]), float(entry["end"]))
                for entry in schedule
            }
        )
    return {
        "scheduler_name": scheduler_name,
        "mode": mode,
        "prediction_used": bool(prediction_used),
        "valid": not errors,
        "makespan": float(reported_makespan if reported_makespan is not None else replay_makespan),
        "planning_time_ms": float(planning_time_ms),
        "wave_count": wave_count,
        "replay_makespan": replay_makespan,
        "barrier_times": barrier_times,
        "send_busy_time": send_busy_time,
        "recv_busy_time": recv_busy_time,
        "validation_errors": errors,
    }


def _build_weight_matrix(
    ready_flows: list[dict[str, Any]],
    num_gpus: int,
) -> tuple[np.ndarray, dict[tuple[int, int], dict[str, Any]]]:
    weights = np.zeros((num_gpus, num_gpus), dtype=float)
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in ready_flows:
        src = int(candidate["src_gpu"])
        dst = int(candidate["dst_gpu"])
        score = float(candidate["score"])
        key = (src, dst)
        if score > weights[src, dst]:
            weights[src, dst] = score
            selected[key] = candidate
    return weights, selected


def _maximum_weight_matching(
    ready_flows: list[dict[str, Any]],
    num_gpus: int,
) -> list[dict[str, Any]]:
    if not ready_flows:
        return []
    weights, selected = _build_weight_matrix(ready_flows, num_gpus)
    if linear_sum_assignment is None:  # pragma: no cover
        return _greedy_maximal_matching(ready_flows)
    cost = -weights
    row_idx, col_idx = linear_sum_assignment(cost)
    result: list[dict[str, Any]] = []
    for src, dst in zip(row_idx, col_idx, strict=False):
        key = (int(src), int(dst))
        candidate = selected.get(key)
        if candidate is None or int(src) == int(dst) or weights[src, dst] <= 0.0:
            continue
        result.append(candidate)
    return result


def _greedy_maximal_matching(ready_flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used_src: set[int] = set()
    used_dst: set[int] = set()
    for candidate in sorted(
        ready_flows,
        key=lambda item: (
            float(item["score"]),
            float(item["barrier_urgency"]),
            float(item["age"]),
            float(item["residual"]),
            -int(item["src_gpu"]),
            -int(item["dst_gpu"]),
        ),
        reverse=True,
    ):
        src = int(candidate["src_gpu"])
        dst = int(candidate["dst_gpu"])
        if src in used_src or dst in used_dst:
            continue
        used_src.add(src)
        used_dst.add(dst)
        chosen.append(candidate)
    return chosen


def _ready_flow_candidates(
    *,
    flows: list[GlobalFlow],
    residual: dict[str, float],
    ready_since: dict[str, float],
    current_time: float,
    release_time: dict[tuple[int, int], float],
    inbound_remaining: dict[tuple[int, int], float],
    downstream_load: dict[tuple[int, int], float],
    age_scale: float,
    residual_weight: float,
    barrier_weight: float,
    age_weight: float,
    prediction_weight: float,
    mode: str,
    prediction_confidence: float,
) -> list[dict[str, Any]]:
    max_residual = max((value for value in residual.values() if value > 0.0), default=1.0)
    ready: list[dict[str, Any]] = []
    for flow in flows:
        remaining = residual[flow.flow_id]
        if remaining <= 0.0:
            continue
        release = release_time[(flow.phase, flow.src_gpu)]
        if current_time + 1e-9 < release:
            continue
        ready_since.setdefault(flow.flow_id, current_time)
        barrier_urgency = downstream_load.get((flow.phase + 1, flow.dst_gpu), 0.0) / max(
            inbound_remaining.get((flow.phase, flow.dst_gpu), remaining),
            1.0,
        )
        age = (current_time - ready_since[flow.flow_id]) / max(age_scale, 1.0)
        prediction_bonus = 0.0
        if mode == RUNTIME_LOOKAHEAD_MODE and flow.phase == 1:
            prediction_bonus = prediction_confidence * downstream_load.get((2, flow.dst_gpu), 0.0) / max(max_residual, 1.0)
        score = (
            residual_weight * (remaining / max_residual)
            + barrier_weight * barrier_urgency
            + age_weight * age
            + prediction_weight * prediction_bonus
        )
        ready.append(
            {
                "flow_id": flow.flow_id,
                "phase": flow.phase,
                "src_gpu": flow.src_gpu,
                "dst_gpu": flow.dst_gpu,
                "residual": remaining,
                "release_time": release,
                "barrier_urgency": barrier_urgency,
                "age": age,
                "prediction_bonus": prediction_bonus,
                "score": score,
            }
        )
    return ready


def _run_global_matching_scheduler(
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
) -> dict[str, Any]:
    start_time = time.perf_counter()
    flows = _collect_real_flows(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        mode=mode,
    )
    residual = {flow.flow_id: float(flow.volume) for flow in flows}
    inbound_remaining = _inbound_remaining(flows, num_gpus)
    release_time = {(phase, gpu): (0.0 if phase == 0 else math.inf) for phase in range(3) for gpu in range(num_gpus)}
    for phase in (1, 2):
        prev_phase = phase - 1
        for gpu in range(num_gpus):
            if inbound_remaining.get((prev_phase, gpu), 0.0) <= 1e-9:
                release_time[(phase, gpu)] = expert_compute_delay if prev_phase == 0 else 0.0
    barrier_done = {(phase, gpu): 0.0 for phase in range(3) for gpu in range(num_gpus)}
    ready_since: dict[str, float] = {}
    downstream_load = _outbound_loads(
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
        ready = _ready_flow_candidates(
            flows=flows,
            residual=residual,
            ready_since=ready_since,
            current_time=current_time,
            release_time=release_time,
            inbound_remaining=inbound_remaining,
            downstream_load=downstream_load,
            age_scale=max(1.0, current_time + 1.0),
            residual_weight=residual_weight,
            barrier_weight=barrier_weight,
            age_weight=age_weight,
            prediction_weight=prediction_weight,
            mode=mode,
            prediction_confidence=prediction_confidence,
        )
        if adaptive_prices:
            for candidate in ready:
                if candidate["phase"] < 2:
                    candidate["score"] += prices[(candidate["phase"], candidate["dst_gpu"])]
        if not ready:
            future = [
                release
                for release in release_time.values()
                if release < math.inf and release > current_time + 1e-9
            ]
            if not future:
                break
            current_time = min(future)
            continue

        chosen = _maximum_weight_matching(ready, num_gpus) if exact_matching else _greedy_maximal_matching(ready)
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
            served = duration
            if atomic:
                served = max(0.0, residual[flow_id])
            residual[flow_id] = max(0.0, residual[flow_id] - served)
            inbound_remaining[(phase, dst)] = max(0.0, inbound_remaining[(phase, dst)] - served)
            end = current_time + served
            wave_end = max(wave_end, end)
            barrier_done[(phase, dst)] = max(barrier_done[(phase, dst)], end)
            if phase < 2 and inbound_remaining[(phase, dst)] <= 1e-9:
                release_time[(phase + 1, dst)] = barrier_done[(phase, dst)] + (
                    expert_compute_delay if phase == 0 else 0.0
                )
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
                        remaining = inbound_remaining.get((phase, gpu), 0.0)
                        value = 0.0 if pressure <= 0.0 else pressure / max(remaining, 1.0)
                        updated = (1.0 - price_decay) * prices[(phase, gpu)] + price_step * value
                        prices[(phase, gpu)] = max(-price_clip, min(price_clip, updated))

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
    }


def fast_schedule_u_gated_maxweight_matching(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    mode: str = EXECUTION_WINDOW_MODE,
    prediction_confidence: float = 1.0,
    wave_quantum: float | None = None,
    max_waves: int = 256,
) -> dict[str, Any]:
    """Global ready-set exact max-weight matching across released multi-phase flows."""

    del model
    return _run_global_matching_scheduler(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        strategy="U_gated_maxweight_matching",
        mode=mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
        exact_matching=True,
        wave_quantum=wave_quantum,
        max_waves=max_waves,
        residual_weight=1.0,
        barrier_weight=1.0,
        age_weight=0.1,
        prediction_weight=0.25,
        adaptive_prices=False,
        price_step=0.0,
        price_decay=0.0,
        price_clip=0.0,
        iteration_budget=1,
        atomic=False,
    )


def fast_schedule_u_gated_greedy_maximal(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    mode: str = EXECUTION_WINDOW_MODE,
    prediction_confidence: float = 1.0,
    wave_quantum: float | None = None,
    max_waves: int = 256,
) -> dict[str, Any]:
    """Low-overhead maximal matching baseline over the same global ready set."""

    del model
    return _run_global_matching_scheduler(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        strategy="U_gated_greedy_maximal",
        mode=mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
        exact_matching=False,
        wave_quantum=wave_quantum,
        max_waves=max_waves,
        residual_weight=1.0,
        barrier_weight=1.0,
        age_weight=0.1,
        prediction_weight=0.25,
        adaptive_prices=False,
        price_step=0.0,
        price_decay=0.0,
        price_clip=0.0,
        iteration_budget=1,
        atomic=False,
    )


def fast_schedule_u_barrier_criticality_global_matching(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    mode: str = EXECUTION_WINDOW_MODE,
    prediction_confidence: float = 1.0,
    wave_quantum: float | None = None,
    max_waves: int = 256,
) -> dict[str, Any]:
    """Global matching with stronger barrier-unlock prioritization."""

    del model
    return _run_global_matching_scheduler(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        strategy="U_barrier_criticality_global_matching",
        mode=mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
        exact_matching=True,
        wave_quantum=wave_quantum,
        max_waves=max_waves,
        residual_weight=0.75,
        barrier_weight=1.75,
        age_weight=0.15,
        prediction_weight=0.35,
        adaptive_prices=False,
        price_step=0.0,
        price_decay=0.0,
        price_clip=0.0,
        iteration_budget=1,
        atomic=False,
    )


def fast_schedule_u_barrier_price_adaptive_matching(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    mode: str = EXECUTION_WINDOW_MODE,
    prediction_confidence: float = 1.0,
    wave_quantum: float | None = None,
    max_waves: int = 256,
    iteration_budget: int = 4,
) -> dict[str, Any]:
    """Barrier-priced global matching with bounded adaptive updates."""

    del model
    return _run_global_matching_scheduler(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        strategy="U_barrier_price_adaptive_matching",
        mode=mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
        exact_matching=True,
        wave_quantum=wave_quantum,
        max_waves=max_waves,
        residual_weight=0.65,
        barrier_weight=1.25,
        age_weight=0.10,
        prediction_weight=0.25,
        adaptive_prices=True,
        price_step=0.15,
        price_decay=0.10,
        price_clip=4.0,
        iteration_budget=iteration_budget,
        atomic=False,
    )


def fast_schedule_u_gated_maxweight_matching_atomic(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    mode: str = EXECUTION_WINDOW_MODE,
    prediction_confidence: float = 1.0,
    wave_quantum: float | None = None,
    max_waves: int = 256,
) -> dict[str, Any]:
    """Atomic-wave exact max-weight matching over the global ready set."""

    del model
    return _run_global_matching_scheduler(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        strategy="U_gated_maxweight_matching_atomic",
        mode=mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
        exact_matching=True,
        wave_quantum=wave_quantum,
        max_waves=max_waves,
        residual_weight=1.0,
        barrier_weight=1.0,
        age_weight=0.1,
        prediction_weight=0.25,
        adaptive_prices=False,
        price_step=0.0,
        price_decay=0.0,
        price_clip=0.0,
        iteration_budget=1,
        atomic=True,
    )


def fast_schedule_u_gated_greedy_maximal_atomic(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    mode: str = EXECUTION_WINDOW_MODE,
    prediction_confidence: float = 1.0,
    wave_quantum: float | None = None,
    max_waves: int = 256,
) -> dict[str, Any]:
    """Atomic-wave maximal-matching baseline over the same ready set."""

    del model
    return _run_global_matching_scheduler(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        strategy="U_gated_greedy_maximal_atomic",
        mode=mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
        exact_matching=False,
        wave_quantum=wave_quantum,
        max_waves=max_waves,
        residual_weight=1.0,
        barrier_weight=1.0,
        age_weight=0.1,
        prediction_weight=0.25,
        adaptive_prices=False,
        price_step=0.0,
        price_decay=0.0,
        price_clip=0.0,
        iteration_budget=1,
        atomic=True,
    )


def fast_schedule_u_barrier_criticality_global_matching_atomic(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    mode: str = EXECUTION_WINDOW_MODE,
    prediction_confidence: float = 1.0,
    wave_quantum: float | None = None,
    max_waves: int = 256,
) -> dict[str, Any]:
    """Atomic-wave global matching with stronger barrier-unlock prioritization."""

    del model
    return _run_global_matching_scheduler(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        strategy="U_barrier_criticality_global_matching_atomic",
        mode=mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
        exact_matching=True,
        wave_quantum=wave_quantum,
        max_waves=max_waves,
        residual_weight=0.75,
        barrier_weight=1.75,
        age_weight=0.15,
        prediction_weight=0.35,
        adaptive_prices=False,
        price_step=0.0,
        price_decay=0.0,
        price_clip=0.0,
        iteration_budget=1,
        atomic=True,
    )


def fast_schedule_u_barrier_price_adaptive_matching_atomic(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    mode: str = EXECUTION_WINDOW_MODE,
    prediction_confidence: float = 1.0,
    wave_quantum: float | None = None,
    max_waves: int = 256,
    iteration_budget: int = 4,
) -> dict[str, Any]:
    """Atomic-wave global matching with bounded adaptive barrier prices."""

    del model
    return _run_global_matching_scheduler(
        dispatch_matrix,
        combine_matrix,
        next_dispatch_matrix,
        num_gpus,
        strategy="U_barrier_price_adaptive_matching_atomic",
        mode=mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
        exact_matching=True,
        wave_quantum=wave_quantum,
        max_waves=max_waves,
        residual_weight=0.65,
        barrier_weight=1.25,
        age_weight=0.10,
        prediction_weight=0.25,
        adaptive_prices=True,
        price_step=0.15,
        price_decay=0.10,
        price_clip=4.0,
        iteration_budget=iteration_budget,
        atomic=True,
    )
