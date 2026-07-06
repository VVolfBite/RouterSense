"""Root-authoritative phase plan agreement exports."""

from __future__ import annotations

import time
from dataclasses import replace

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
    all_gather_start_ns = time.monotonic_ns()
    dist.all_gather_object(gathered, local_context, group=world_group)
    all_gather_end_ns = time.monotonic_ns()
    global_contexts = tuple(item for item in gathered if item is not None)
    if dist.get_rank(group=world_group) == root_rank:
        root_context = next(ctx for ctx in global_contexts if int(ctx.global_rank) == root_rank)
        build_plan_start_ns = time.monotonic_ns()
        payload = policy.build_plan(local_context=root_context, global_contexts=global_contexts)
        build_plan_end_ns = time.monotonic_ns()
    else:
        build_plan_start_ns = time.monotonic_ns()
        build_plan_end_ns = build_plan_start_ns
        payload = None
    buffer = [payload]
    broadcast_start_ns = time.monotonic_ns()
    dist.broadcast_object_list(buffer, src=root_rank, group=world_group)
    broadcast_end_ns = time.monotonic_ns()
    assert buffer[0] is not None
    decoded = buffer[0]
    hash_list: list[str | None] = [None for _ in range(world_size)]
    verify_start_ns = time.monotonic_ns()
    dist.all_gather_object(hash_list, decoded.plan_hash, group=world_group)
    if len({item for item in hash_list if item is not None}) != 1:
        raise RuntimeError(f"phase plan hash mismatch: {hash_list}")
    verify_end_ns = time.monotonic_ns()
    timing_metrics = {
        "all_gather_time_us": (all_gather_end_ns - all_gather_start_ns) / 1000.0,
        "build_plan_time_us": (build_plan_end_ns - build_plan_start_ns) / 1000.0,
        "broadcast_time_us": (broadcast_end_ns - broadcast_start_ns) / 1000.0,
        "verify_time_us": (verify_end_ns - verify_start_ns) / 1000.0,
        "total_agreement_time_us": (verify_end_ns - all_gather_start_ns) / 1000.0,
    }
    return replace(decoded, metrics={**decoded.metrics, **timing_metrics})


__all__ = ["run_phase_plan_agreement"]
