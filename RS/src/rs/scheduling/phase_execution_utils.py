"""Pure scheduling helpers for transfer layouts, bucketization, and plan validation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .phase_execution import (
    BucketTask,
    IncomingSlot,
    OutgoingSegment,
    PayloadSlice,
    PhaseExecutionPlan,
    PhaseReadyContext,
    PlanWave,
    TransferLayout,
)


def join_transfer_layouts(
    *,
    global_contexts: tuple[PhaseReadyContext, ...],
    phase: str,
) -> tuple[TransferLayout, ...]:
    def outgoing_index(context: PhaseReadyContext) -> dict[tuple[str, int, int], OutgoingSegment]:
        return {
            (context.phase, int(segment.src_rank), int(segment.dst_rank)): segment
            for segment in context.outgoing_segments
        }

    def bundle_index(context: PhaseReadyContext) -> dict[tuple[str, int, int], dict[str, object]]:
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

    def incoming_index(context: PhaseReadyContext) -> dict[tuple[str, int, int], IncomingSlot]:
        return {
            (context.phase, int(slot.src_rank), int(slot.dst_rank)): slot
            for slot in context.incoming_slots
        }

    outgoing_by_rank = {int(ctx.global_rank): outgoing_index(ctx) for ctx in global_contexts}
    bundle_by_rank = {int(ctx.global_rank): bundle_index(ctx) for ctx in global_contexts}
    incoming_by_rank = {int(ctx.global_rank): incoming_index(ctx) for ctx in global_contexts}

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


def _slice_payload(payload: PayloadSlice, *, row_offset: int, row_count: int) -> PayloadSlice:
    per_row_bytes = int(payload.payload_byte_count / max(int(payload.row_count), 1))
    return PayloadSlice(
        bundle_id=payload.bundle_id,
        tensor_role=payload.tensor_role,
        src_rank=payload.src_rank,
        dst_rank=payload.dst_rank,
        segment_ordinal=payload.segment_ordinal,
        sender_offset_rows=int(payload.sender_offset_rows) + int(row_offset),
        receiver_offset_rows=int(payload.receiver_offset_rows) + int(row_offset),
        row_count=int(row_count),
        dtype=payload.dtype,
        shape_suffix=payload.shape_suffix,
        element_size_bytes=payload.element_size_bytes,
        payload_byte_count=int(per_row_bytes * int(row_count)),
        packed_layout_id=payload.packed_layout_id,
    )


def bucketize_transfer_layouts(
    transfer_layouts: tuple[TransferLayout, ...],
    *,
    bucket_rows: int,
) -> tuple[BucketTask, ...]:
    tasks: list[BucketTask] = []
    for layout in transfer_layouts:
        if int(layout.row_count) <= 0:
            continue
        step = int(layout.row_count) if int(bucket_rows) <= 0 else int(bucket_rows)
        consumed = 0
        bucket_ordinal = 0
        while consumed < int(layout.row_count):
            current_rows = min(step, int(layout.row_count) - consumed)
            payload_slices = tuple(
                _slice_payload(payload, row_offset=consumed, row_count=current_rows)
                for payload in layout.payload_slices
            )
            tasks.append(
                BucketTask(
                    task_id=f"{layout.transfer_key}:bucket:{bucket_ordinal}",
                    bundle_id=layout.bundle_id,
                    phase=layout.phase,
                    src_rank=layout.src_rank,
                    dst_rank=layout.dst_rank,
                    source_peer_index=layout.source_peer_index,
                    destination_peer_index=layout.destination_peer_index,
                    segment_ordinal=layout.segment_ordinal,
                    bucket_ordinal=bucket_ordinal,
                    sender_offset_rows=int(layout.sender_offset_rows) + consumed,
                    receiver_offset_rows=int(layout.receiver_offset_rows) + consumed,
                    row_count=current_rows,
                    byte_count=int(payload_slices[0].payload_byte_count) if payload_slices else 0,
                    packed_send_layout_id=layout.packed_send_layout_id,
                    canonical_receive_layout_id=layout.canonical_receive_layout_id,
                    payload_slices=payload_slices,
                )
            )
            consumed += current_rows
            bucket_ordinal += 1
    return tuple(tasks)


def validate_phase_execution_plan(context: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
    if plan.phase != context.phase:
        raise ValueError(f"phase mismatch: context={context.phase} plan={plan.phase}")
    if plan.plan_key != context.plan_key:
        raise ValueError("plan_key mismatch between context and execution plan")
    local_ids = {segment.segment_id for segment in context.outgoing_segments if segment.is_local}
    coverage: dict[tuple[int, int, int], int] = defaultdict(int)
    for wave in plan.waves:
        outgoing = defaultdict(int)
        incoming = defaultdict(int)
        for task in wave.bucket_tasks:
            if task.phase != context.phase:
                raise ValueError("wave task phase mismatch")
            outgoing[int(task.src_rank)] += 1
            incoming[int(task.dst_rank)] += 1
            if outgoing[int(task.src_rank)] > 1:
                raise ValueError(f"wave {wave.wave_id} has >1 outgoing for rank {task.src_rank}")
            if incoming[int(task.dst_rank)] > 1:
                raise ValueError(f"wave {wave.wave_id} has >1 incoming for rank {task.dst_rank}")
            key = (int(task.src_rank), int(task.dst_rank), int(task.segment_ordinal))
            coverage[key] += int(task.row_count)
            if f"{task.phase}:{task.src_rank}->{task.dst_rank}:{task.segment_ordinal}" in local_ids:
                raise ValueError("local flow leaked into network execution plan")

    expected = {
        (segment.src_rank, segment.dst_rank, segment.segment_ordinal): int(segment.row_count)
        for segment in context.outgoing_segments
        if not segment.is_local and int(segment.row_count) > 0 and int(segment.src_rank) == int(context.global_rank)
    }
    local_coverage = {key: value for key, value in coverage.items() if int(key[0]) == int(context.global_rank)}
    if set(expected) != set(local_coverage):
        missing = sorted(set(expected) - set(local_coverage))
        extra = sorted(set(local_coverage) - set(expected))
        raise ValueError(f"plan coverage mismatch: missing={missing} extra={extra}")
    for key, expected_rows in expected.items():
        if int(local_coverage[key]) != int(expected_rows):
            raise ValueError(f"plan rows mismatch for {key}: expected={expected_rows} actual={local_coverage[key]}")


def row_digest(tasks: tuple[BucketTask, ...]) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple(
        (
            int(task.src_rank),
            int(task.dst_rank),
            int(task.segment_ordinal),
            int(task.sender_offset_rows),
            int(task.row_count),
        )
        for task in tasks
    )


def pack_waves(tasks: list[BucketTask], *, phase: str) -> tuple[PlanWave, ...]:
    waves: list[PlanWave] = []
    pending = list(tasks)
    wave_id = 0
    while pending:
        used_outgoing: set[int] = set()
        used_incoming: set[int] = set()
        selected: list[BucketTask] = []
        remaining: list[BucketTask] = []
        for task in pending:
            src = int(task.src_rank)
            dst = int(task.dst_rank)
            if src in used_outgoing or dst in used_incoming:
                remaining.append(task)
                continue
            selected.append(task)
            used_outgoing.add(src)
            used_incoming.add(dst)
        waves.append(PlanWave(wave_id=wave_id, phase=phase, bucket_tasks=tuple(selected)))
        pending = remaining
        wave_id += 1
    return tuple(waves)
