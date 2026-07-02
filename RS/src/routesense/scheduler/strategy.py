from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SchedulingContext:
    """Scheduling inputs for pairwise multi-phase communication."""

    dispatch_matrix: list[list[int]]
    combine_matrix: list[list[int]]
    next_dispatch_matrix: list[list[int]]
    num_gpus: int
    model: str = "full_duplex"
    expert_compute_delay: float = 0.0


@dataclass
class SchedulingResult:
    """Normalized strategy output."""

    makespan: float
    schedule: list[dict[str, Any]]
    solve_time_ms: float
    strategy: str


class SchedulingStrategy(ABC):
    """Common interface for all schedulers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable registry name."""

    @abstractmethod
    def solve(self, ctx: SchedulingContext) -> SchedulingResult:
        """Run the scheduling policy."""


_REGISTRY: dict[str, type[SchedulingStrategy]] = {}


def register_strategy(cls: type[SchedulingStrategy]) -> type[SchedulingStrategy]:
    """Class decorator for strategy registration."""

    _REGISTRY[cls().name] = cls
    return cls


def get_strategy(name: str) -> SchedulingStrategy:
    """Construct a registered strategy by name."""

    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown strategy '{name}'. Available: {available}")
    return _REGISTRY[name]()


def list_strategies() -> list[str]:
    """List registered strategy names."""

    return sorted(_REGISTRY)
