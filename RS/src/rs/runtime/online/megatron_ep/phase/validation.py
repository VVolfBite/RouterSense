"""Phase layout 与 bundle 合法性校验。

主要函数：
- validate_layout_offsets_cover_exactly_once()
- validate_p0_atomic_bundle()
用于确保 phase 级 layout 和 P0 原子 bundle 合同没有被破坏。
"""

from __future__ import annotations

from .contracts import PhaseReadyContext


def validate_p0_atomic_bundle(context: PhaseReadyContext) -> bool:
    if context.phase != "P0":
        return True
    return all(bundle.atomic_submit and len(bundle.payload_slices) >= 2 for bundle in context.transport_bundles)


def validate_layout_offsets_cover_exactly_once(context: PhaseReadyContext) -> bool:
    total_send_rows = sum(int(segment.row_count) for segment in context.outgoing_segments)
    if total_send_rows != sum(int(v) for v in context.send_splits):
        return False
    expected_send_offset = 0
    for segment in context.outgoing_segments:
        if int(segment.send_offset_rows) != expected_send_offset:
            return False
        expected_send_offset += int(segment.row_count)
    expected_recv_offset = 0
    for slot in context.incoming_slots:
        if int(slot.receive_offset_rows) != expected_recv_offset:
            return False
        expected_recv_offset += int(slot.row_count)
    return (
        expected_send_offset == total_send_rows
        and expected_recv_offset == sum(int(v) for v in context.recv_splits)
    )
