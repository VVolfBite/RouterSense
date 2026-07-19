"""Phase context 构建器。

主要函数：
- build_phase_ready_context()：把 dispatcher snapshot、payload contract、
  split 信息、packed tensor 元数据整理成 PhaseReadyContext
这是 phase 级调度与执行之间最重要的桥。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from .contracts import (
    FutureDemandHint,
    IncomingSlot,
    OutgoingSegment,
    PackedTensorDescriptor,
    PayloadSlice,
    PhaseReadyContext,
    TransportBundle,
)


@dataclass(frozen=True)
class RuntimeIdentity:
    run_id: str
    forward_epoch: int
    layer_id: str
    layer_name: str
    global_rank: int
    local_rank: int
    ep_group_ranks: tuple[int, ...]
    ep_group_root_rank: int


@dataclass(frozen=True)
class DispatcherSnapshot:
    dispatcher_class: str
    dispatcher_fingerprint: dict[str, Any]
    expert_placement_hash: str
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]


@dataclass(frozen=True)
class PhasePayloadContract:
    phase: str
    payload_roles: tuple[str, ...]
    atomic_submit: bool


@dataclass(frozen=True)
class PhaseContextBuildRequest:
    plan_key: dict[str, Any]
    runtime_identity: RuntimeIdentity
    topology: dict[str, Any]
    dispatcher_snapshot: DispatcherSnapshot
    payload_contract: PhasePayloadContract
    packed_tensors: tuple[torch.Tensor, ...]
    control_mode: str
    release_state: str
    demand_known_at: str
    payload_exists: bool
    p2_hint: FutureDemandHint | None = None


def _shape_suffix(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape[1:]) if tensor.ndim >= 2 else ()


def _rows_to_bytes(row_count: int, *, shape_suffix: tuple[int, ...], element_size_bytes: int) -> int:
    multiplier = 1
    for dim in shape_suffix:
        multiplier *= int(dim)
    return int(row_count) * int(multiplier) * int(element_size_bytes)


def _phase_send_recv_splits(phase: str, input_splits: tuple[int, ...], output_splits: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if phase == "P0":
        return tuple(int(v) for v in input_splits), tuple(int(v) for v in output_splits)
    return tuple(int(v) for v in output_splits), tuple(int(v) for v in input_splits)


def _bytes_to_rows(byte_count: int, *, shape_suffix: tuple[int, ...], element_size_bytes: int) -> int:
    multiplier = int(element_size_bytes)
    for dim in shape_suffix:
        multiplier *= int(dim)
    if multiplier <= 0:
        raise ValueError("rows_to_bytes inverse requires positive payload row size")
    if int(byte_count) % multiplier != 0:
        raise ValueError(
            f"byte_count={byte_count} is not divisible by payload_row_bytes={multiplier}"
        )
    return int(byte_count) // multiplier


def build_phase_ready_context(request: PhaseContextBuildRequest) -> PhaseReadyContext:
    phase = request.payload_contract.phase
    if len(request.packed_tensors) != len(request.payload_contract.payload_roles):
        raise ValueError(
            "packed_tensors count must match payload_roles count: "
            f"{len(request.packed_tensors)} != {len(request.payload_contract.payload_roles)}"
        )
    runtime_identity = request.runtime_identity
    dispatcher_snapshot = request.dispatcher_snapshot
    packed_tensors = request.packed_tensors
    input_splits = dispatcher_snapshot.input_splits
    output_splits = dispatcher_snapshot.output_splits
    layer_id = runtime_identity.layer_id
    global_rank = int(runtime_identity.global_rank)
    ep_group_ranks = runtime_identity.ep_group_ranks
    release_state = request.release_state
    demand_known_at = request.demand_known_at
    payload_exists = request.payload_exists
    send_splits, recv_splits = _phase_send_recv_splits(phase, input_splits, output_splits)
    per_peer_rows = send_splits
    base_tensor = packed_tensors[0] if packed_tensors else None
    base_suffix = _shape_suffix(base_tensor) if isinstance(base_tensor, torch.Tensor) else ()
    base_elem_size = int(base_tensor.element_size()) if isinstance(base_tensor, torch.Tensor) else 0
    per_peer_bytes = tuple(
        _rows_to_bytes(int(rows), shape_suffix=base_suffix, element_size_bytes=base_elem_size)
        for rows in per_peer_rows
    )
    packed_send_layout_id = hashlib.sha256(
        repr(
            {
                "kind": "packed_send",
                "phase": phase,
                "layer_id": layer_id,
                "splits": list(send_splits),
                "shapes": [tuple(int(dim) for dim in tensor.shape) for tensor in packed_tensors],
                "dtypes": [str(tensor.dtype) for tensor in packed_tensors],
            }
        ).encode("utf-8")
    ).hexdigest()
    canonical_receive_layout_id = hashlib.sha256(
        repr(
            {
                "kind": "canonical_receive",
                "phase": phase,
                "layer_id": layer_id,
                "splits": list(recv_splits),
                "shapes": [tuple(int(dim) for dim in tensor.shape) for tensor in packed_tensors],
                "dtypes": [str(tensor.dtype) for tensor in packed_tensors],
            }
        ).encode("utf-8")
    ).hexdigest()

    payload_descriptors = tuple(
        PackedTensorDescriptor(
            tensor_role=request.payload_contract.payload_roles[index],
            shape=tuple(int(dim) for dim in tensor.shape),
            shape_suffix=_shape_suffix(tensor),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            element_size_bytes=int(tensor.element_size()),
        )
        for index, tensor in enumerate(packed_tensors)
    )

    outgoing_segments: list[OutgoingSegment] = []
    incoming_slots: list[IncomingSlot] = []
    bundles: list[TransportBundle] = []
    send_offset = 0
    running_recv = 0
    for peer_index, rows in enumerate(recv_splits):
        rows = int(rows)
        if peer_index >= len(ep_group_ranks):
            break
        src_rank = int(ep_group_ranks[peer_index])
        dst_rank = int(global_rank)
        slot = IncomingSlot(
            slot_id=f"{phase}:{src_rank}->{dst_rank}:{peer_index}:slot",
            phase=phase,
            src_rank=src_rank,
            dst_rank=dst_rank,
            source_peer_index=peer_index,
            segment_ordinal=peer_index,
            receive_offset_rows=running_recv,
            row_count=rows,
            byte_count=_rows_to_bytes(rows, shape_suffix=base_suffix, element_size_bytes=base_elem_size),
            canonical_receive_layout_id=canonical_receive_layout_id,
            is_local=src_rank == dst_rank,
        )
        incoming_slots.append(slot)
        running_recv += rows

    for peer_index, rows in enumerate(send_splits):
        rows = int(rows)
        if peer_index >= len(ep_group_ranks):
            break
        dst_rank = int(ep_group_ranks[peer_index])
        src_rank = int(runtime_identity.global_rank)
        segment = OutgoingSegment(
            segment_id=f"{phase}:{src_rank}->{dst_rank}:{peer_index}",
            phase=phase,
            src_rank=src_rank,
            dst_rank=dst_rank,
            destination_peer_index=peer_index,
            segment_ordinal=peer_index,
            send_offset_rows=send_offset,
            row_count=rows,
            byte_count=_rows_to_bytes(rows, shape_suffix=base_suffix, element_size_bytes=base_elem_size),
            packed_send_layout_id=packed_send_layout_id,
            is_local=src_rank == dst_rank,
        )
        send_offset += rows
        outgoing_segments.append(segment)

        payload_slices = tuple(
            PayloadSlice(
                bundle_id=f"{segment.segment_id}:bundle",
                tensor_role=descriptor.tensor_role,
                src_rank=src_rank,
                dst_rank=dst_rank,
                segment_ordinal=segment.segment_ordinal,
                sender_offset_rows=segment.send_offset_rows,
                receiver_offset_rows=0,
                row_count=rows,
                dtype=descriptor.dtype,
                shape_suffix=descriptor.shape_suffix,
                element_size_bytes=descriptor.element_size_bytes,
                payload_byte_count=_rows_to_bytes(
                    rows,
                    shape_suffix=descriptor.shape_suffix,
                    element_size_bytes=descriptor.element_size_bytes,
                ),
                packed_layout_id=packed_send_layout_id,
            )
            for descriptor in payload_descriptors
        )
        bundles.append(
            TransportBundle(
                bundle_id=f"{segment.segment_id}:bundle",
                phase=phase,
                atomic_submit=bool(request.payload_contract.atomic_submit),
                outgoing_segment=segment,
                payloads=payload_descriptors,
                payload_slices=payload_slices,
            )
        )

    return PhaseReadyContext(
        plan_key=request.plan_key,
        phase=phase,
        control_mode=request.control_mode,
        forward_epoch=runtime_identity.forward_epoch,
        layer_id=runtime_identity.layer_id,
        layer_name=runtime_identity.layer_name,
        global_rank=runtime_identity.global_rank,
        local_rank=runtime_identity.local_rank,
        ep_group_ranks=runtime_identity.ep_group_ranks,
        ep_group_root_rank=runtime_identity.ep_group_root_rank,
        topology=request.topology,
        dispatcher_class=dispatcher_snapshot.dispatcher_class,
        dispatcher_fingerprint=dispatcher_snapshot.dispatcher_fingerprint,
        expert_placement_hash=dispatcher_snapshot.expert_placement_hash,
        input_splits=tuple(int(v) for v in input_splits),
        output_splits=tuple(int(v) for v in output_splits),
        send_splits=tuple(int(v) for v in send_splits),
        recv_splits=tuple(int(v) for v in recv_splits),
        per_peer_rows=tuple(int(v) for v in per_peer_rows),
        per_peer_bytes=tuple(int(v) for v in per_peer_bytes),
        packed_send_layout_id=packed_send_layout_id,
        canonical_receive_layout_id=canonical_receive_layout_id,
        payload_specs=payload_descriptors,
        atomic_submit=bool(request.payload_contract.atomic_submit),
        outgoing_segments=tuple(outgoing_segments),
        incoming_slots=tuple(incoming_slots),
        transport_bundles=tuple(bundles),
        release_state=release_state,
        demand_known_at=demand_known_at,
        payload_exists=payload_exists,
        p2_hint=request.p2_hint or FutureDemandHint(),
    )


def reconstruct_global_phase_contexts_from_byte_matrix(
    *,
    local_context: PhaseReadyContext,
    matrix: tuple[tuple[int, ...], ...],
    matrix_unit: str = "bytes",
) -> tuple[PhaseReadyContext, ...]:
    world_size = int(len(matrix))
    ep_group_ranks = tuple(int(v) for v in local_context.ep_group_ranks)
    if world_size != len(ep_group_ranks):
        raise ValueError(f"matrix size {world_size} does not match ep group size {len(ep_group_ranks)}")
    if not local_context.payload_specs:
        raise ValueError("payload_specs required to reconstruct global contexts")
    base_spec = local_context.payload_specs[0]
    if str(matrix_unit) == "rows":
        row_matrix = tuple(tuple(int(value) for value in row) for row in matrix)
    else:
        row_matrix = tuple(
            tuple(
                _bytes_to_rows(
                    int(value),
                    shape_suffix=tuple(int(v) for v in base_spec.shape_suffix),
                    element_size_bytes=int(base_spec.element_size_bytes),
                )
                for value in row
            )
            for row in matrix
        )
    contexts: list[PhaseReadyContext] = []
    for rank_index, global_rank in enumerate(ep_group_ranks):
        row = tuple(int(v) for v in row_matrix[rank_index])
        col = tuple(int(row_matrix[src][rank_index]) for src in range(world_size))
        if str(local_context.phase) == "P0":
            input_splits = row
            output_splits = col
        else:
            input_splits = col
            output_splits = row
        send_splits, recv_splits = _phase_send_recv_splits(str(local_context.phase), input_splits, output_splits)
        per_peer_rows = tuple(int(v) for v in send_splits)
        base_suffix = tuple(int(v) for v in base_spec.shape_suffix)
        base_elem_size = int(base_spec.element_size_bytes)
        per_peer_bytes = tuple(
            _rows_to_bytes(int(rows), shape_suffix=base_suffix, element_size_bytes=base_elem_size)
            for rows in per_peer_rows
        )
        outgoing_segments: list[OutgoingSegment] = []
        incoming_slots: list[IncomingSlot] = []
        bundles: list[TransportBundle] = []
        running_recv = 0
        for peer_index, rows in enumerate(recv_splits):
            src_rank = int(ep_group_ranks[peer_index])
            incoming_slots.append(
                IncomingSlot(
                    slot_id=f"{local_context.phase}:{src_rank}->{global_rank}:{peer_index}:slot",
                    phase=str(local_context.phase),
                    src_rank=src_rank,
                    dst_rank=int(global_rank),
                    source_peer_index=int(peer_index),
                    segment_ordinal=int(peer_index),
                    receive_offset_rows=int(running_recv),
                    row_count=int(rows),
                    byte_count=_rows_to_bytes(int(rows), shape_suffix=base_suffix, element_size_bytes=base_elem_size),
                    canonical_receive_layout_id=str(local_context.canonical_receive_layout_id),
                    is_local=bool(src_rank == int(global_rank)),
                )
            )
            running_recv += int(rows)
        send_offset = 0
        for peer_index, rows in enumerate(send_splits):
            dst_rank = int(ep_group_ranks[peer_index])
            receiver_offset_rows = int(sum(int(row_matrix[src_index][peer_index]) for src_index in range(rank_index)))
            segment = OutgoingSegment(
                segment_id=f"{local_context.phase}:{global_rank}->{dst_rank}:{peer_index}",
                phase=str(local_context.phase),
                src_rank=int(global_rank),
                dst_rank=dst_rank,
                destination_peer_index=int(peer_index),
                segment_ordinal=int(peer_index),
                send_offset_rows=int(send_offset),
                row_count=int(rows),
                byte_count=_rows_to_bytes(int(rows), shape_suffix=base_suffix, element_size_bytes=base_elem_size),
                packed_send_layout_id=str(local_context.packed_send_layout_id),
                is_local=bool(int(global_rank) == dst_rank),
            )
            send_offset += int(rows)
            outgoing_segments.append(segment)
            payload_slices = tuple(
                PayloadSlice(
                    bundle_id=f"{segment.segment_id}:bundle",
                    tensor_role=str(spec.tensor_role),
                    src_rank=int(global_rank),
                    dst_rank=dst_rank,
                    segment_ordinal=int(segment.segment_ordinal),
                    sender_offset_rows=int(segment.send_offset_rows),
                    receiver_offset_rows=int(receiver_offset_rows),
                    row_count=int(rows),
                    dtype=str(spec.dtype),
                    shape_suffix=tuple(int(v) for v in spec.shape_suffix),
                    element_size_bytes=int(spec.element_size_bytes),
                    payload_byte_count=_rows_to_bytes(1, shape_suffix=tuple(int(v) for v in spec.shape_suffix), element_size_bytes=int(spec.element_size_bytes)) * int(rows),
                    packed_layout_id=str(local_context.packed_send_layout_id),
                )
                for spec in local_context.payload_specs
            )
            bundles.append(
                TransportBundle(
                    bundle_id=f"{segment.segment_id}:bundle",
                    phase=str(local_context.phase),
                    atomic_submit=bool(local_context.atomic_submit),
                    outgoing_segment=segment,
                    payloads=tuple(local_context.payload_specs),
                    payload_slices=payload_slices,
                )
            )
        contexts.append(
            PhaseReadyContext(
                plan_key=dict(local_context.plan_key),
                phase=str(local_context.phase),
                control_mode=str(local_context.control_mode),
                forward_epoch=int(local_context.forward_epoch),
                layer_id=str(local_context.layer_id),
                layer_name=str(local_context.layer_name),
                global_rank=int(global_rank),
                local_rank=int(rank_index),
                ep_group_ranks=ep_group_ranks,
                ep_group_root_rank=int(local_context.ep_group_root_rank),
                topology={
                    **dict(local_context.topology),
                    "global_rank": int(global_rank),
                    "local_rank": int(rank_index),
                    "ep_group_rank": int(rank_index),
                },
                dispatcher_class=str(local_context.dispatcher_class),
                dispatcher_fingerprint=dict(local_context.dispatcher_fingerprint),
                expert_placement_hash=str(local_context.expert_placement_hash),
                input_splits=tuple(int(v) for v in input_splits),
                output_splits=tuple(int(v) for v in output_splits),
                send_splits=tuple(int(v) for v in send_splits),
                recv_splits=tuple(int(v) for v in recv_splits),
                per_peer_rows=per_peer_rows,
                per_peer_bytes=per_peer_bytes,
                packed_send_layout_id=str(local_context.packed_send_layout_id),
                canonical_receive_layout_id=str(local_context.canonical_receive_layout_id),
                payload_specs=tuple(local_context.payload_specs),
                atomic_submit=bool(local_context.atomic_submit),
                outgoing_segments=tuple(outgoing_segments),
                incoming_slots=tuple(incoming_slots),
                transport_bundles=tuple(bundles),
                release_state=str(local_context.release_state),
                demand_known_at=str(local_context.demand_known_at),
                payload_exists=bool(local_context.payload_exists),
                p2_hint=local_context.p2_hint,
            )
        )
    return tuple(contexts)
