"""Formal oracle-guided reference contract.

The historical oracle-guided solver remains parked under legacy. The formal
tree must not fabricate optimality metadata or silently fall back to greedy
while reporting zero optimality gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnsupportedReferenceSolver(RuntimeError):
    solver_name: str
    reason: str

    def __str__(self) -> str:
        return f"{self.solver_name} unavailable in formal mainline: {self.reason}"


def unsupported_oracle_guided_result() -> dict[str, Any]:
    return {
        "solver_name": "oracle_guided_reference",
        "supported": False,
        "solver_status": "unsupported",
        "schedule": [],
        "chunk_count": 0,
        "objective": None,
        "best_bound": None,
        "optimality_gap": None,
        "certified_optimal": False,
        "solve_time_ms": None,
        "time_limit_ms": None,
        "reason": "historical oracle-guided solver remains in legacy/historical_poc and has not been migrated into formal scheduling",
    }


def pairwise_oracle(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    *,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    raise UnsupportedReferenceSolver(
        solver_name="pairwise_oracle",
        reason="formal tree does not ship a placeholder oracle fallback",
    )


def schedule_oracle_guided_reference(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise UnsupportedReferenceSolver(
        solver_name="schedule_oracle_guided_reference",
        reason="formal tree does not ship a placeholder oracle fallback",
    )


__all__ = [
    "UnsupportedReferenceSolver",
    "pairwise_oracle",
    "schedule_oracle_guided_reference",
    "unsupported_oracle_guided_result",
]
