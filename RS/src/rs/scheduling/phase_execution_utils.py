"""Pure scheduling helpers for transfer layouts, bucketization, and plan validation."""

from __future__ import annotations

from collections import defaultdict
from .phase_execution import (
    AbstractPhaseExecutionPlan,
    BucketTask,
    IncomingSlot,
    OutgoingSegment,
    PackedTensorDescriptor,
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
    def _rows_to_bytes(row_count: int, *, shape_suffix: tuple[int, ...], element_size_bytes: int) -> int:
        multiplier = 1
        for dim in shape_suffix:
            multiplier *= int(dim)
        return int(row_count) * int(multiplier) * int(element_size_bytes)

    def _payload_specs(context: PhaseReadyContext) -> tuple[PackedTensorDescriptor, ...]:
        if context.payload_specs:
            return tuple(context.payload_specs)
        if context.transport_bundles:
            return tuple(context.transport_bundles[0].payloads)
        return ()

    def outgoing_index(context: PhaseReadyContext) -> dict[tuple[str, int, int], OutgoingSegment]:
        return {
            (context.phase, int(segment.src_rank), int(segment.dst_rank)): segment
            for segment in context.outgoing_segments
        }

    def payload_index(context: PhaseReadyContext) -> dict[tuple[str, int, int], dict[str, object]]:
        return {
            (context.phase, int(segment.src_rank), int(segment.dst_rank)): {
                "bundle_id": f"{segment.segment_id}:bundle",
                "atomic_submit": bool(context.atomic_submit),
                "payloads": _payload_specs(context),
                "packed_send_layout_id": segment.packed_send_layout_id,
            }
            for segment in context.outgoing_segments
        }

    def incoming_index(context: PhaseReadyContext) -> dict[tuple[str, int, int], IncomingSlot]:
        return {
            (context.phase, int(slot.src_rank), int(slot.dst_rank)): slot
            for slot in context.incoming_slots
        }

    outgoing_by_rank = {int(ctx.global_rank): outgoing_index(ctx) for ctx in global_contexts}
    payload_by_rank = {int(ctx.global_rank): payload_index(ctx) for ctx in global_contexts}
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
            payload_info = payload_by_rank[src_rank][key]
            payload_slices = tuple(
                PayloadSlice(
                    bundle_id=str(payload_info["bundle_id"]),
                    tensor_role=str(payload.tensor_role),
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    segment_ordinal=int(outgoing.segment_ordinal),
                    sender_offset_rows=int(outgoing.send_offset_rows),
                    receiver_offset_rows=int(incoming.receive_offset_rows),
                    row_count=int(outgoing.row_count),
                    dtype=str(payload.dtype),
                    shape_suffix=tuple(int(v) for v in payload.shape_suffix),
                    element_size_bytes=int(payload.element_size_bytes),
                    payload_byte_count=_rows_to_bytes(
                        int(outgoing.row_count),
                        shape_suffix=tuple(int(v) for v in payload.shape_suffix),
                        element_size_bytes=int(payload.element_size_bytes),
                    ),
                    packed_layout_id=str(outgoing.packed_send_layout_id),
                )
                for payload in tuple(payload_info["payloads"])
            )
            layouts.append(
                TransferLayout(
                    transfer_key=f"{phase}:{src_rank}->{dst_rank}",
                    bundle_id=str(payload_info["bundle_id"]),
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
                    packed_send_layout_id=str(payload_info["packed_send_layout_id"]),
                    canonical_receive_layout_id=str(incoming.canonical_receive_layout_id),
                    atomic_submit=bool(payload_info["atomic_submit"]),
                    payloads=tuple(payload_info["payloads"]),
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


def _rows_to_bytes(row_count: int, *, shape_suffix: tuple[int, ...], element_size_bytes: int) -> int:
    multiplier = 1
    for dim in shape_suffix:
        multiplier *= int(dim)
    return int(row_count) * int(multiplier) * int(element_size_bytes)


def _slice_payload_specs(
    *,
    bundle_id: str,
    payload_specs: tuple[PackedTensorDescriptor, ...],
    src_rank: int,
    dst_rank: int,
    segment_ordinal: int,
    sender_offset_rows: int,
    receiver_offset_rows: int,
    row_count: int,
    packed_layout_id: str,
) -> tuple[PayloadSlice, ...]:
    return tuple(
        PayloadSlice(
            bundle_id=str(bundle_id),
            tensor_role=str(payload.tensor_role),
            src_rank=int(src_rank),
            dst_rank=int(dst_rank),
            segment_ordinal=int(segment_ordinal),
            sender_offset_rows=int(sender_offset_rows),
            receiver_offset_rows=int(receiver_offset_rows),
            row_count=int(row_count),
            dtype=str(payload.dtype),
            shape_suffix=tuple(int(v) for v in payload.shape_suffix),
            element_size_bytes=int(payload.element_size_bytes),
            payload_byte_count=_rows_to_bytes(
                int(row_count),
                shape_suffix=tuple(int(v) for v in payload.shape_suffix),
                element_size_bytes=int(payload.element_size_bytes),
            ),
            packed_layout_id=str(packed_layout_id),
        )
        for payload in payload_specs
    )


def _task_ref_key(*, phase: str, src_rank: int, dst_rank: int, segment_ordinal: int, bucket_ordinal: int) -> tuple[str, int, int, int]:
    # Sender-side segment_ordinal and receiver-side segment_ordinal are not guaranteed
    # to share the same numbering scheme. The stable identity for the current runtime
    # contract is phase + rank pair + bucket ordinal, which matches the emitted task_id.
    return (str(phase), int(src_rank), int(dst_rank), int(bucket_ordinal))


def _local_outgoing_task_catalog(context: PhaseReadyContext, *, bucket_rows: int) -> dict[tuple[str, int, int, int], BucketTask]:
    payload_specs = tuple(context.payload_specs)
    catalog: dict[tuple[str, int, int, int], BucketTask] = {}
    for segment in context.outgoing_segments:
        if segment.is_local or int(segment.row_count) <= 0:
            continue
        step = int(segment.row_count) if int(bucket_rows) <= 0 else int(bucket_rows)
        consumed = 0
        bucket_ordinal = 0
        bundle_id = f"{segment.segment_id}:bundle"
        while consumed < int(segment.row_count):
            current_rows = min(step, int(segment.row_count) - consumed)
            sender_offset = int(segment.send_offset_rows) + consumed
            payload_slices = _slice_payload_specs(
                bundle_id=bundle_id,
                payload_specs=payload_specs,
                src_rank=int(segment.src_rank),
                dst_rank=int(segment.dst_rank),
                segment_ordinal=int(segment.segment_ordinal),
                sender_offset_rows=sender_offset,
                receiver_offset_rows=0,
                row_count=current_rows,
                packed_layout_id=str(segment.packed_send_layout_id),
            )
            task = BucketTask(
                task_id=f"{segment.phase}:{segment.src_rank}->{segment.dst_rank}:bucket:{bucket_ordinal}",
                bundle_id=bundle_id,
                phase=segment.phase,
                src_rank=int(segment.src_rank),
                dst_rank=int(segment.dst_rank),
                source_peer_index=-1,
                destination_peer_index=int(segment.destination_peer_index),
                segment_ordinal=int(segment.segment_ordinal),
                bucket_ordinal=bucket_ordinal,
                sender_offset_rows=sender_offset,
                receiver_offset_rows=0,
                row_count=current_rows,
                byte_count=int(payload_slices[0].payload_byte_count) if payload_slices else 0,
                packed_send_layout_id=str(segment.packed_send_layout_id),
                canonical_receive_layout_id="",
                payload_slices=payload_slices,
            )
            key = _task_ref_key(
                phase=str(task.phase),
                src_rank=int(task.src_rank),
                dst_rank=int(task.dst_rank),
                segment_ordinal=int(task.segment_ordinal),
                bucket_ordinal=int(task.bucket_ordinal),
            )
            if key in catalog:
                raise ValueError(f"duplicate outgoing task key materialization key={key}")
            catalog[key] = task
            consumed += current_rows
            bucket_ordinal += 1
    return catalog


def _local_incoming_task_catalog(context: PhaseReadyContext, *, bucket_rows: int) -> dict[tuple[str, int, int, int], BucketTask]:
    payload_specs = tuple(context.payload_specs)
    catalog: dict[tuple[str, int, int, int], BucketTask] = {}
    for slot in context.incoming_slots:
        if slot.is_local or int(slot.row_count) <= 0:
            continue
        step = int(slot.row_count) if int(bucket_rows) <= 0 else int(bucket_rows)
        consumed = 0
        bucket_ordinal = 0
        bundle_id = f"{slot.phase}:{slot.src_rank}->{slot.dst_rank}:{slot.segment_ordinal}:bundle"
        while consumed < int(slot.row_count):
            current_rows = min(step, int(slot.row_count) - consumed)
            receiver_offset = int(slot.receive_offset_rows) + consumed
            payload_slices = _slice_payload_specs(
                bundle_id=bundle_id,
                payload_specs=payload_specs,
                src_rank=int(slot.src_rank),
                dst_rank=int(slot.dst_rank),
                segment_ordinal=int(slot.segment_ordinal),
                sender_offset_rows=0,
                receiver_offset_rows=receiver_offset,
                row_count=current_rows,
                packed_layout_id="",
            )
            task = BucketTask(
                task_id=f"{slot.phase}:{slot.src_rank}->{slot.dst_rank}:bucket:{bucket_ordinal}",
                bundle_id=bundle_id,
                phase=slot.phase,
                src_rank=int(slot.src_rank),
                dst_rank=int(slot.dst_rank),
                source_peer_index=int(slot.source_peer_index),
                destination_peer_index=-1,
                segment_ordinal=int(slot.segment_ordinal),
                bucket_ordinal=bucket_ordinal,
                sender_offset_rows=0,
                receiver_offset_rows=receiver_offset,
                row_count=current_rows,
                byte_count=int(payload_slices[0].payload_byte_count) if payload_slices else 0,
                packed_send_layout_id="",
                canonical_receive_layout_id=str(slot.canonical_receive_layout_id),
                payload_slices=payload_slices,
            )
            key = _task_ref_key(
                phase=str(task.phase),
                src_rank=int(task.src_rank),
                dst_rank=int(task.dst_rank),
                segment_ordinal=int(task.segment_ordinal),
                bucket_ordinal=int(task.bucket_ordinal),
            )
            if key in catalog:
                raise ValueError(f"duplicate incoming task key materialization key={key}")
            catalog[key] = task
            consumed += current_rows
            bucket_ordinal += 1
    return catalog


def materialize_local_execution_plan(
    *,
    local_context: PhaseReadyContext,
    abstract_plan: AbstractPhaseExecutionPlan,
) -> PhaseExecutionPlan:
    bucket_rows = int(abstract_plan.metrics.get("bucket_rows", 0) or 0)
    outgoing_catalog = _local_outgoing_task_catalog(local_context, bucket_rows=bucket_rows)
    incoming_catalog = _local_incoming_task_catalog(local_context, bucket_rows=bucket_rows)
    waves: list[PlanWave] = []
    for abstract_wave in abstract_plan.waves:
        local_tasks: list[BucketTask] = []
        for task_ref in abstract_wave.task_refs:
            task_key = _task_ref_key(
                phase=str(task_ref.phase),
                src_rank=int(task_ref.src_rank),
                dst_rank=int(task_ref.dst_rank),
                segment_ordinal=int(task_ref.segment_ordinal),
                bucket_ordinal=int(task_ref.bucket_ordinal),
            )
            task = outgoing_catalog.get(task_key)
            if task is None:
                task = incoming_catalog.get(task_key)
            if task is not None:
                local_tasks.append(task)
        waves.append(PlanWave(wave_id=int(abstract_wave.wave_id), phase=str(abstract_wave.phase), bucket_tasks=tuple(local_tasks)))
    plan = PhaseExecutionPlan(
        plan_key=dict(abstract_plan.plan_key),
        phase=str(abstract_plan.phase),
        policy_name=str(abstract_plan.policy_name),
        policy_version=str(abstract_plan.policy_version),
        control_mode=str(abstract_plan.control_mode),
        execution_mode=str(abstract_plan.execution_mode),
        transport_mutation=bool(abstract_plan.transport_mutation),
        is_shadow_only=bool(abstract_plan.is_shadow_only),
        future_hint_mode=str(abstract_plan.future_hint_mode),
        root_rank=int(abstract_plan.root_rank),
        observation_digest=str(abstract_plan.observation_digest),
        plan_hash=str(abstract_plan.plan_hash),
        waves=tuple(waves),
        metrics={**dict(abstract_plan.metrics), "local_materialized": True},
    )
    validate_phase_execution_plan(local_context, plan)
    return plan


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
    wave_tasks: list[list[BucketTask]] = []
    used_outgoing_masks: list[int] = []
    used_incoming_masks: list[int] = []
    for task in tasks:
        src = int(task.src_rank)
        dst = int(task.dst_rank)
        src_mask = 1 << src
        dst_mask = 1 << dst
        placed = False
        for wave_index, (used_outgoing_mask, used_incoming_mask) in enumerate(zip(used_outgoing_masks, used_incoming_masks)):
            if (used_outgoing_mask & src_mask) or (used_incoming_mask & dst_mask):
                continue
            wave_tasks[wave_index].append(task)
            used_outgoing_masks[wave_index] = used_outgoing_mask | src_mask
            used_incoming_masks[wave_index] = used_incoming_mask | dst_mask
            placed = True
            break
        if placed:
            continue
        wave_tasks.append([task])
        used_outgoing_masks.append(src_mask)
        used_incoming_masks.append(dst_mask)
    waves: list[PlanWave] = []
    for wave_id, bucket_tasks in enumerate(wave_tasks):
        waves.append(PlanWave(wave_id=wave_id, phase=phase, bucket_tasks=tuple(bucket_tasks)))
    return tuple(waves)
