"""Formal Birkhoff baseline contract.

The historical Birkhoff decomposition implementation lives in legacy parking.
Until that implementation is migrated into the formal scheduling tree, the
formal API must fail closed instead of returning placeholder makespans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnsupportedBaselineError(RuntimeError):
    baseline_name: str
    reason: str

    def __str__(self) -> str:
        return f"{self.baseline_name} unsupported in formal mainline: {self.reason}"


def unsupported_birkhoff_summary() -> dict[str, Any]:
    return {
        "baseline_name": "birkhoff",
        "supported": False,
        "solver_status": "unsupported",
        "makespan": None,
        "permutation_count": None,
        "total_volume": None,
        "certified_optimal": False,
        "optimality_gap": None,
        "best_bound": None,
        "reason": "historical implementation remains in legacy/historical_poc and has not been migrated into formal scheduling",
    }


def decompose_matrix_to_permutations(matrix: list[list[int]]) -> list[dict[str, Any]]:
    raise UnsupportedBaselineError(
        baseline_name="birkhoff.decompose_matrix_to_permutations",
        reason="formal tree does not ship a placeholder decomposition",
    )


def birkhoff_schedule_single_layer(matrix: list[list[int]]) -> float:
    raise UnsupportedBaselineError(
        baseline_name="birkhoff_schedule_single_layer",
        reason="formal tree does not expose a fake Birkhoff makespan",
    )


def birkhoff_baseline_summary(matrix: list[list[int]]) -> dict[str, Any]:
    return unsupported_birkhoff_summary()


__all__ = [
    "UnsupportedBaselineError",
    "birkhoff_baseline_summary",
    "birkhoff_schedule_single_layer",
    "decompose_matrix_to_permutations",
    "unsupported_birkhoff_summary",
]
