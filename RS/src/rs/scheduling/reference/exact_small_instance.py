"""Exact small-instance formal contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnsupportedExactSolve(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return f"exact small-instance solve unsupported in formal mainline: {self.reason}"


def solve_exact_small_instance(*args, **kwargs) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    raise UnsupportedExactSolve(
        reason="no exact CP-SAT or MILP implementation has been migrated into src/rs/scheduling/reference yet",
    )


__all__ = ["UnsupportedExactSolve", "solve_exact_small_instance"]
