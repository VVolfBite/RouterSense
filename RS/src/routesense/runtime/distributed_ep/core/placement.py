from __future__ import annotations


class PlacementStrategy:
    @staticmethod
    def round_robin(num_experts: int, num_gpus: int) -> dict[int, int]:
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
        return {expert_id: expert_id % num_gpus for expert_id in range(num_experts)}

