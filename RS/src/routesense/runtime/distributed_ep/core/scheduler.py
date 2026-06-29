from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SchedulerDecision:
    strategy: str
    release_order: list[int]


class Scheduler:
    def plan(self, buckets: list[dict[str, Any]], strategy: str = "fifo") -> SchedulerDecision:
        release_order = list(range(len(buckets)))
        return SchedulerDecision(strategy=strategy, release_order=release_order)
