from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlacementSummary:
    owner_by_expert: dict[int, int]
    expert_ids_by_rank: dict[int, list[int]]


class PlacementStrategy:
    @staticmethod
    def round_robin(num_experts: int, num_gpus: int) -> dict[int, int]:
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
        return {expert_id: expert_id % num_gpus for expert_id in range(num_experts)}

    @staticmethod
    def summarize(owner_by_expert: dict[int, int], world_size: int) -> PlacementSummary:
        expert_ids_by_rank: dict[int, list[int]] = {rank: [] for rank in range(world_size)}
        for expert_id, owner_rank in sorted(owner_by_expert.items()):
            expert_ids_by_rank.setdefault(owner_rank, []).append(expert_id)
        return PlacementSummary(owner_by_expert=owner_by_expert, expert_ids_by_rank=expert_ids_by_rank)
