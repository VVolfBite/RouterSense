"""Public multiphase ready-set scheduling API."""

from __future__ import annotations

from typing import Any

from .global_ready_set_impl import (
    EXECUTION_WINDOW_MODE,
    RUNTIME_LOOKAHEAD_MODE,
    replay_and_audit_schedule,
    fast_schedule_u_gated_greedy_maximal,
    fast_schedule_u_gated_maxweight_matching,
)


def schedule_global_ready_set(
    dispatch_matrix: list[list[int]],
    p1_return_matrix: list[list[int]],
    p2_next_dispatch_forecast_matrix: list[list[int]],
    num_gpus: int,
    *,
    scheduling_mode: str = RUNTIME_LOOKAHEAD_MODE,
    prediction_confidence: float = 0.0,
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    return fast_schedule_u_gated_maxweight_matching(
        dispatch_matrix,
        p1_return_matrix,
        p2_next_dispatch_forecast_matrix,
        num_gpus,
        mode=scheduling_mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
    )


def schedule_greedy(
    dispatch_matrix: list[list[int]],
    p1_return_matrix: list[list[int]],
    p2_next_dispatch_forecast_matrix: list[list[int]],
    num_gpus: int,
    *,
    scheduling_mode: str = RUNTIME_LOOKAHEAD_MODE,
    prediction_confidence: float = 0.0,
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    return fast_schedule_u_gated_greedy_maximal(
        dispatch_matrix,
        p1_return_matrix,
        p2_next_dispatch_forecast_matrix,
        num_gpus,
        mode=scheduling_mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
    )


__all__ = [
    "EXECUTION_WINDOW_MODE",
    "RUNTIME_LOOKAHEAD_MODE",
    "replay_and_audit_schedule",
    "schedule_global_ready_set",
    "schedule_greedy",
]
