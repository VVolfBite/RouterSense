from __future__ import annotations

import hashlib
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


def build_phase_ready_context(
    *,
    plan_key: dict[str, Any],
    phase: str,
    control_mode: str,
    forward_epoch: int,
    layer_id: str,
    layer_name: str,
    global_rank: int,
    local_rank: int,
    ep_group_ranks: tuple[int, ...],
    ep_group_root_rank: int,
    topology: dict[str, Any],
    dispatcher_class: str,
    dispatcher_fingerprint: dict[str, Any],
    expert_placement_hash: str,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    packed_tensors: tuple[torch.Tensor, ...],
    release_state: str,
    demand_known_at: str,
    payload_exists: bool,
    p2_hint: FutureDemandHint | None = None,
) -> PhaseReadyContext:
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
            tensor_role="hidden_states" if index == 0 else "routing_probs",
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
        src_rank = int(global_rank)
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
                atomic_submit=phase == "P0",
                outgoing_segment=segment,
                payloads=payload_descriptors,
                payload_slices=payload_slices,
            )
        )

    return PhaseReadyContext(
        plan_key=plan_key,
        phase=phase,
        control_mode=control_mode,
        forward_epoch=forward_epoch,
        layer_id=layer_id,
        layer_name=layer_name,
        global_rank=global_rank,
        local_rank=local_rank,
        ep_group_ranks=ep_group_ranks,
        ep_group_root_rank=ep_group_root_rank,
        topology=topology,
        dispatcher_class=dispatcher_class,
        dispatcher_fingerprint=dispatcher_fingerprint,
        expert_placement_hash=expert_placement_hash,
        input_splits=tuple(int(v) for v in input_splits),
        output_splits=tuple(int(v) for v in output_splits),
        send_splits=tuple(int(v) for v in send_splits),
        recv_splits=tuple(int(v) for v in recv_splits),
        per_peer_rows=tuple(int(v) for v in per_peer_rows),
        per_peer_bytes=tuple(int(v) for v in per_peer_bytes),
        packed_send_layout_id=packed_send_layout_id,
        canonical_receive_layout_id=canonical_receive_layout_id,
        outgoing_segments=tuple(outgoing_segments),
        incoming_slots=tuple(incoming_slots),
        transport_bundles=tuple(bundles),
        release_state=release_state,
        demand_known_at=demand_known_at,
        payload_exists=payload_exists,
        p2_hint=p2_hint or FutureDemandHint(),
    )
