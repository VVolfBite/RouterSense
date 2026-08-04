"""Public multiphase ready-set strategies."""

from __future__ import annotations

from typing import Any

from .flow_model import EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE
from .scheduler_state import run_global_matching_scheduler


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
    return run_global_matching_scheduler(
        dispatch_matrix,
        p1_return_matrix,
        p2_next_dispatch_forecast_matrix,
        num_gpus,
        strategy="U_gated_maxweight_matching",
        mode=scheduling_mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
        exact_matching=True,
        wave_quantum=None,
        max_waves=256,
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
    return run_global_matching_scheduler(
        dispatch_matrix,
        p1_return_matrix,
        p2_next_dispatch_forecast_matrix,
        num_gpus,
        strategy="U_gated_greedy_maximal",
        mode=scheduling_mode,
        prediction_confidence=prediction_confidence,
        expert_compute_delay=expert_compute_delay,
        exact_matching=False,
        wave_quantum=None,
        max_waves=256,
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
