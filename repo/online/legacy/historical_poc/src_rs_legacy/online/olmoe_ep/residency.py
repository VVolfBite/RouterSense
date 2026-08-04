from __future__ import annotations

from typing import Any


def build_full_checkpoint_then_prune_audit(*, placement: dict[int, int], local_rank: int) -> dict[str, Any]:
    local_experts = sorted(expert_id for expert_id, owner in placement.items() if int(owner) == int(local_rank))
    return {
        "expert_residency_mode": "full_checkpoint_then_prune",
        "checkpoint_loading_is_memory_efficient": False,
        "execution_time_nonlocal_expert_parameters": 0,
        "local_rank": int(local_rank),
        "local_expert_ids": local_experts,
    }
