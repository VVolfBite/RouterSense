from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....scheduler.strategy import (
    SchedulingContext,
    SchedulingResult,
    SchedulingStrategy,
    get_strategy,
)


@dataclass
class SchedulerDecision:
    strategy: str
    release_order: list[int]
    makespan: float = 0.0
    solve_time_ms: float = 0.0
    schedule: list[dict[str, Any]] | None = None


class Scheduler:
    """Runtime scheduler facade backed by a pluggable strategy."""

    def __init__(self, strategy_name: str = "greedy") -> None:
        self._strategy: SchedulingStrategy = get_strategy(strategy_name)

    @property
    def strategy(self) -> SchedulingStrategy:
        return self._strategy

    def set_strategy(self, strategy_name: str) -> None:
        self._strategy = get_strategy(strategy_name)

    def solve(self, ctx: SchedulingContext) -> SchedulingResult:
        return self._strategy.solve(ctx)

    def plan(self, buckets: list[dict[str, Any]], strategy: str = "fifo") -> SchedulerDecision:
        release_order = list(range(len(buckets)))
        return SchedulerDecision(strategy=strategy, release_order=release_order)
