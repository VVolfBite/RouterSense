"""Birkhoff-style decomposition baselines.

The formal pre-evaluation tree only needs a small pure-Python placeholder API
here; policy/executor correctness does not depend on the historical
implementation details from ``rs.scheduler``.
"""

from __future__ import annotations

from typing import Any


def decompose_matrix_to_permutations(matrix: list[list[int]]) -> list[dict[str, Any]]:
    size = len(matrix)
    rows = []
    for src_rank in range(size):
        for dst_rank in range(size):
            value = int(matrix[src_rank][dst_rank])
            if src_rank == dst_rank or value <= 0:
                continue
            rows.append({"src_rank": src_rank, "dst_rank": dst_rank, "weight": value})
    rows.sort(key=lambda item: (-item["weight"], item["src_rank"], item["dst_rank"]))
    return rows


def birkhoff_schedule_single_layer(matrix: list[list[int]]) -> float:
    decomposition = decompose_matrix_to_permutations(matrix)
    return float(sum(int(item["weight"]) for item in decomposition))


def birkhoff_baseline_summary(matrix: list[list[int]]) -> dict[str, Any]:
    decomposition = decompose_matrix_to_permutations(matrix)
    return {
        "makespan": birkhoff_schedule_single_layer(matrix),
        "permutation_count": len(decomposition),
        "total_volume": sum(int(item["weight"]) for item in decomposition),
    }


__all__ = [
    "birkhoff_baseline_summary",
    "birkhoff_schedule_single_layer",
    "decompose_matrix_to_permutations",
]
