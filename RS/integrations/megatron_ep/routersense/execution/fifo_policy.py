from __future__ import annotations

from dataclasses import replace

import torch.distributed as dist

from integrations.megatron_ep.routersense.phase import IncomingSlot, OutgoingSegment, PhaseExecutionPlan, PhaseReadyContext, TransferLayout


def _outgoing_index(context: PhaseReadyContext) -> dict[tuple[str, int, int], OutgoingSegment]:
    return {
        (context.phase, int(segment.src_rank), int(segment.dst_rank)): segment
        for segment in context.outgoing_segments
    }


def _bundle_index(context: PhaseReadyContext) -> dict[tuple[str, int, int], dict]:
    return {
        (context.phase, int(bundle.outgoing_segment.src_rank), int(bundle.outgoing_segment.dst_rank)): {
            "bundle_id": bundle.bundle_id,
            "atomic_submit": bundle.atomic_submit,
            "payloads": bundle.payloads,
            "payload_slices": bundle.payload_slices,
            "packed_send_layout_id": bundle.outgoing_segment.packed_send_layout_id,
        }
        for bundle in context.transport_bundles
    }


def _incoming_index(context: PhaseReadyContext) -> dict[tuple[str, int, int], IncomingSlot]:
    return {
        (context.phase, int(slot.src_rank), int(slot.dst_rank)): slot
        for slot in context.incoming_slots
    }


def join_transfer_layouts(
    *,
    global_contexts: tuple[PhaseReadyContext, ...],
    phase: str,
) -> tuple[TransferLayout, ...]:
    outgoing_by_rank = {int(ctx.global_rank): _outgoing_index(ctx) for ctx in global_contexts}
    bundle_by_rank = {int(ctx.global_rank): _bundle_index(ctx) for ctx in global_contexts}
    incoming_by_rank = {int(ctx.global_rank): _incoming_index(ctx) for ctx in global_contexts}

    layouts: list[TransferLayout] = []
    for ctx in global_contexts:
        src_rank = int(ctx.global_rank)
        for key, outgoing in outgoing_by_rank[src_rank].items():
            if key[0] != phase:
                continue
            dst_rank = int(outgoing.dst_rank)
            incoming = incoming_by_rank.get(dst_rank, {}).get(key)
            if incoming is None:
                raise ValueError(f"missing incoming slot for flow {key}")
            if int(outgoing.row_count) != int(incoming.row_count):
                raise ValueError(
                    f"row_count mismatch for flow {key}: sender={outgoing.row_count} receiver={incoming.row_count}"
                )
            bundle_info = bundle_by_rank[src_rank][key]
            payload_slices = tuple(
                replace(
                    payload,
                    sender_offset_rows=int(outgoing.send_offset_rows),
                    receiver_offset_rows=int(incoming.receive_offset_rows),
                )
                for payload in bundle_info["payload_slices"]
            )
            layouts.append(
                TransferLayout(
                    transfer_key=f"{phase}:{src_rank}->{dst_rank}",
                    bundle_id=str(bundle_info["bundle_id"]),
                    phase=phase,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    source_peer_index=int(incoming.source_peer_index),
                    destination_peer_index=int(outgoing.destination_peer_index),
                    segment_ordinal=int(outgoing.segment_ordinal),
                    sender_offset_rows=int(outgoing.send_offset_rows),
                    receiver_offset_rows=int(incoming.receive_offset_rows),
                    row_count=int(outgoing.row_count),
                    byte_count=int(outgoing.byte_count),
                    packed_send_layout_id=str(bundle_info["packed_send_layout_id"]),
                    canonical_receive_layout_id=str(incoming.canonical_receive_layout_id),
                    atomic_submit=bool(bundle_info["atomic_submit"]),
                    payloads=tuple(bundle_info["payloads"]),
                    payload_slices=payload_slices,
                )
            )
    return tuple(sorted(layouts, key=lambda item: (int(item.src_rank), int(item.dst_rank), int(item.segment_ordinal))))


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
