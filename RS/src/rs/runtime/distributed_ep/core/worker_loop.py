from __future__ import annotations

from dataclasses import dataclass, field

from .manifest import DispatchPlan


@dataclass
class WorkerLoopState:
    mode: str = "synchronous"
    processed_layers: list[int] = field(default_factory=list)
    processed_route_count: int = 0


class WorkerLoop:
    def __init__(self, mode: str = "synchronous") -> None:
        self.state = WorkerLoopState(mode=mode)

    def record_plan(self, plan: DispatchPlan, rank: int) -> None:
        self.state.processed_layers.append(plan.layer_id)
        self.state.processed_route_count += sum(
            len(shard.route_items)
            for shard in plan.shards
            if shard.destination_rank == rank
        )
