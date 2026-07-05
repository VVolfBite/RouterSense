from __future__ import annotations

import torch.distributed as dist

from rs.runtime.online.megatron_ep.phase import PhaseExecutionPlan, PhaseReadyContext


def _get_process_group_root_safe(group: dist.ProcessGroup | None) -> int:
    if group is None:
        return 0
    if hasattr(dist, "get_process_group_ranks"):
        ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
        return int(ranks[0]) if ranks else 0
    return 0


def run_phase_plan_agreement(
    *,
    local_context: PhaseReadyContext,
    policy,
    group: dist.ProcessGroup | None,
) -> PhaseExecutionPlan:
    world_group = group if group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group=world_group)
    root_rank = int(_get_process_group_root_safe(world_group))
    gathered: list[PhaseReadyContext | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_context, group=world_group)
    global_contexts = tuple(item for item in gathered if item is not None)
    if dist.get_rank(group=world_group) == root_rank:
        root_context = next(ctx for ctx in global_contexts if int(ctx.global_rank) == root_rank)
        payload = policy.build_plan(local_context=root_context, global_contexts=global_contexts)
    else:
        payload = None
    buffer = [payload]
    dist.broadcast_object_list(buffer, src=root_rank, group=world_group)
    assert buffer[0] is not None
    decoded = buffer[0]
    hash_list: list[str | None] = [None for _ in range(world_size)]
    local_hash = decoded.plan_hash
    dist.all_gather_object(hash_list, local_hash, group=world_group)
    if len({item for item in hash_list if item is not None}) != 1:
        raise RuntimeError(f"phase plan hash mismatch: {hash_list}")
    return decoded


__all__ = ["run_phase_plan_agreement"]
