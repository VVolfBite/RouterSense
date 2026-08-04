"""Pure greedy scheduling baselines for offline evaluation."""

from __future__ import annotations

from typing import Any


def greedy_schedule_single_layer(matrix: list[list[int]]) -> float:
    return float(max((sum(row) for row in matrix), default=0))


def greedy_schedule_multi_layer(matrices: list[list[list[int]]]) -> float:
    return float(sum(greedy_schedule_single_layer(matrix) for matrix in matrices))


def greedy_schedule_pairwise(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> float:
    del num_gpus, model
    return float(
        greedy_schedule_single_layer(dispatch_matrix)
        + greedy_schedule_single_layer(combine_matrix)
        + greedy_schedule_single_layer(next_dispatch_matrix)
        + float(expert_compute_delay)
    )


def greedy_baseline_summary(matrix: list[list[int]]) -> dict[str, Any]:
    row_loads = [sum(row) for row in matrix]
    return {
        "makespan": greedy_schedule_single_layer(matrix),
        "max_row_load": max(row_loads, default=0),
        "total_volume": sum(row_loads),
    }


__all__ = [
    "greedy_baseline_summary",
    "greedy_schedule_multi_layer",
    "greedy_schedule_pairwise",
    "greedy_schedule_single_layer",
]
