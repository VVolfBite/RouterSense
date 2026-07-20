"""Trace-driven barrier vs streaming release scheduling simulator."""

from __future__ import annotations

from typing import Any

from .flow_model import EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE
from .matching import maximum_weight_matching
from .replay import replay_and_audit_schedule
from .scheduler_state import run_global_matching_scheduler


def simulate_barrier_phase_serial(
    *,
    p0_dispatch_matrix: list[list[int]],
    p1_return_matrix: list[list[int]],
    p2_next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    scheduling_mode: str,
    expert_compute_delay: float,
    service_granularity: str,
    chunk_size: float | None,
) -> dict[str, Any]:
    """Simulate Megatron-style phase barrier execution with per-phase matching."""

    schedule: list[dict[str, Any]] = []
    current_time = 0.0
    wave_id = 0
    phase0 = _schedule_matrix_phase(
        matrix=p0_dispatch_matrix,
        phase=0,
        start_time=current_time,
        start_wave_id=wave_id,
        service_granularity=service_granularity,
        chunk_size=chunk_size,
    )
    schedule.extend(phase0["schedule"])
    current_time = float(phase0["makespan"]) + float(expert_compute_delay)
    wave_id = int(phase0["next_wave_id"])

    phase1 = _schedule_matrix_phase(
        matrix=p1_return_matrix,
        phase=1,
        start_time=current_time,
        start_wave_id=wave_id,
        service_granularity=service_granularity,
        chunk_size=chunk_size,
    )
    schedule.extend(phase1["schedule"])
    current_time = float(phase1["makespan"])
    wave_id = int(phase1["next_wave_id"])

    if scheduling_mode == EXECUTION_WINDOW_MODE:
        phase2 = _schedule_matrix_phase(
            matrix=p2_next_dispatch_matrix,
            phase=2,
            start_time=current_time,
            start_wave_id=wave_id,
            service_granularity=service_granularity,
            chunk_size=chunk_size,
        )
        schedule.extend(phase2["schedule"])
        current_time = float(phase2["makespan"])
        wave_id = int(phase2["next_wave_id"])

    audit = replay_and_audit_schedule(
        schedule=schedule,
        dispatch_matrix=p0_dispatch_matrix,
        combine_matrix=p1_return_matrix,
        next_dispatch_matrix=p2_next_dispatch_matrix,
        num_gpus=num_gpus,
        expert_compute_delay=expert_compute_delay,
        mode=scheduling_mode,
        scheduler_name="phase_barrier_streaming_baseline",
        reported_makespan=current_time,
        prediction_used=False,
    )
    release = release_summary(schedule, num_gpus=num_gpus, expert_compute_delay=expert_compute_delay, scheduling_mode=scheduling_mode)
    return {
        "model": "phase_barrier",
        "service_granularity": service_granularity,
        "chunk_size": chunk_size,
        "makespan": current_time,
        "wave_count": len({int(row["wave_id"]) for row in schedule}),
        "schedule": schedule,
        "audit": audit,
        **release,
    }


def simulate_release_aware_streaming(
    *,
    p0_dispatch_matrix: list[list[int]],
    p1_return_matrix: list[list[int]],
    p2_next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    scheduling_mode: str,
    expert_compute_delay: float,
    service_granularity: str,
    chunk_size: float | None,
    prediction_confidence: float = 1.0,
) -> dict[str, Any]:
    """Simulate release-aware P0/P1/P2 ready-set scheduling."""

    result = run_global_matching_scheduler(
        p0_dispatch_matrix,
        p1_return_matrix,
        p2_next_dispatch_matrix,
        num_gpus,
        strategy="release_aware_streaming_simulator",
        mode=scheduling_mode,
        prediction_confidence=prediction_confidence if scheduling_mode == RUNTIME_LOOKAHEAD_MODE else 0.0,
        expert_compute_delay=expert_compute_delay,
        exact_matching=True,
        wave_quantum=_wave_quantum(service_granularity, chunk_size),
        max_waves=4096,
        residual_weight=1.0,
        barrier_weight=1.5,
        age_weight=0.1,
        prediction_weight=0.5,
        adaptive_prices=False,
        price_step=0.0,
        price_decay=0.0,
        price_clip=0.0,
        iteration_budget=1,
        atomic=False,
    )
    release = release_summary(
        list(result["schedule"]),
        num_gpus=num_gpus,
        expert_compute_delay=expert_compute_delay,
        scheduling_mode=scheduling_mode,
    )
    return {
        "model": "release_aware_streaming",
        "service_granularity": service_granularity,
        "chunk_size": chunk_size,
        "makespan": float(result["makespan"]),
        "wave_count": int(result["wave_count"]),
        "schedule": list(result["schedule"]),
        "audit": result["audit"],
        "solver_status": result["solver_status"],
        **release,
    }


def compare_barrier_and_streaming(
    *,
    p0_dispatch_matrix: list[list[int]],
    p1_return_matrix: list[list[int]],
    p2_next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    scheduling_mode: str,
    expert_compute_delay: float,
    service_granularity: str,
    chunk_size: float | None,
) -> dict[str, Any]:
    barrier = simulate_barrier_phase_serial(
        p0_dispatch_matrix=p0_dispatch_matrix,
        p1_return_matrix=p1_return_matrix,
        p2_next_dispatch_matrix=p2_next_dispatch_matrix,
        num_gpus=num_gpus,
        scheduling_mode=scheduling_mode,
        expert_compute_delay=expert_compute_delay,
        service_granularity=service_granularity,
        chunk_size=chunk_size,
    )
    streaming = simulate_release_aware_streaming(
        p0_dispatch_matrix=p0_dispatch_matrix,
        p1_return_matrix=p1_return_matrix,
        p2_next_dispatch_matrix=p2_next_dispatch_matrix,
        num_gpus=num_gpus,
        scheduling_mode=scheduling_mode,
        expert_compute_delay=expert_compute_delay,
        service_granularity=service_granularity,
        chunk_size=chunk_size,
    )
    delta = float(barrier["makespan"]) - float(streaming["makespan"])
    speedup = float(barrier["makespan"]) / max(float(streaming["makespan"]), 1e-9)
    return {
        "scheduling_mode": scheduling_mode,
        "service_granularity": service_granularity,
        "chunk_size": chunk_size,
        "barrier": _summary_without_schedule(barrier),
        "streaming": _summary_without_schedule(streaming),
        "makespan_savings": delta,
        "makespan_reduction_pct": 100.0 * delta / max(float(barrier["makespan"]), 1e-9),
        "speedup": speedup,
        "p1_release_savings_by_rank": [
            float(b) - float(s)
            for b, s in zip(barrier["p1_release_time_by_rank"], streaming["p1_release_time_by_rank"], strict=False)
        ],
        "barrier_schedule": barrier["schedule"],
        "streaming_schedule": streaming["schedule"],
    }


def release_summary(
    schedule: list[dict[str, Any]],
    *,
    num_gpus: int,
    expert_compute_delay: float,
    scheduling_mode: str,
) -> dict[str, Any]:
    p0_inbound = [0.0] * num_gpus
    p1_inbound = [0.0] * num_gpus
    for row in schedule:
        phase = int(row["phase"])
        dst = int(row["dst_gpu"])
        end = float(row["end"])
        if phase == 0:
            p0_inbound[dst] = max(p0_inbound[dst], end)
        elif phase == 1:
            p1_inbound[dst] = max(p1_inbound[dst], end)
    p1_release = [value + float(expert_compute_delay) for value in p0_inbound]
    p2_release = list(p1_inbound) if scheduling_mode == EXECUTION_WINDOW_MODE else [0.0] * num_gpus
    return {
        "p0_inbound_completion_by_rank": p0_inbound,
        "p1_release_time_by_rank": p1_release,
        "p1_inbound_completion_by_rank": p1_inbound,
        "p2_release_time_by_rank": p2_release,
        "rank_idle_proxy": p1_release,
    }


def _schedule_matrix_phase(
    *,
    matrix: list[list[int]],
    phase: int,
    start_time: float,
    start_wave_id: int,
    service_granularity: str,
    chunk_size: float | None,
) -> dict[str, Any]:
    residual = {
        (src, dst): float(value)
        for src, row in enumerate(matrix)
        for dst, value in enumerate(row)
        if src != dst and float(value) > 0.0
    }
    current_time = float(start_time)
    wave_id = int(start_wave_id)
    schedule: list[dict[str, Any]] = []
    while residual:
        ready = [
            {
                "flow_id": f"phase{phase}_src{src}_dst{dst}",
                "phase": phase,
                "src_gpu": src,
                "dst_gpu": dst,
                "residual": volume,
                "score": volume,
            }
            for (src, dst), volume in residual.items()
            if volume > 1e-9
        ]
        chosen = maximum_weight_matching(ready, len(matrix))
        if not chosen:
            raise RuntimeError("phase barrier simulator made no progress")
        duration = min(float(candidate["residual"]) for candidate in chosen)
        quantum = _wave_quantum(service_granularity, chunk_size)
        if quantum is not None:
            duration = min(duration, quantum)
        wave_end = current_time + duration
        for candidate in chosen:
            src = int(candidate["src_gpu"])
            dst = int(candidate["dst_gpu"])
            key = (src, dst)
            served = min(duration, residual[key])
            residual[key] = max(0.0, residual[key] - served)
            schedule.append(
                {
                    "chunk_id": f"phase{phase}_src{src}_dst{dst}_wave{wave_id}",
                    "flow_id": f"phase{phase}_src{src}_dst{dst}",
                    "phase": phase,
                    "size": float(served),
                    "served_volume": float(served),
                    "src": src,
                    "dst": dst,
                    "src_gpu": src,
                    "dst_gpu": dst,
                    "start": current_time,
                    "end": current_time + served,
                    "wave_id": wave_id,
                    "priority": [float(candidate["score"])],
                }
            )
            if residual[key] <= 1e-9:
                del residual[key]
        current_time = wave_end
        wave_id += 1
    return {"schedule": schedule, "makespan": current_time, "next_wave_id": wave_id}


def _wave_quantum(service_granularity: str, chunk_size: float | None) -> float | None:
    if service_granularity == "wave":
        return None
    if service_granularity == "chunk":
        if chunk_size is None or float(chunk_size) <= 0:
            raise ValueError("chunk service granularity requires positive chunk_size")
        return float(chunk_size)
    raise ValueError(f"unsupported service_granularity={service_granularity!r}")


def _summary_without_schedule(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "schedule"}


__all__ = [
    "compare_barrier_and_streaming",
    "release_summary",
    "simulate_barrier_phase_serial",
    "simulate_release_aware_streaming",
]
