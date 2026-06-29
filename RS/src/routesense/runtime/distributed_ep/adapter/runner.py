from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DistributedRunnerConfig:
    world_size: int
    node_rank: int
    model_id: str
