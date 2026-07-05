from __future__ import annotations

from integrations.megatron_ep.routersense.phase import BucketTask, PayloadSlice, TransferLayout


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
