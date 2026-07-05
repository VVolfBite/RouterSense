from __future__ import annotations

from integrations.megatron_ep.routersense.execution.bucketizer import bucketize_transfer_layouts
from integrations.megatron_ep.routersense.phase import PackedTensorDescriptor, PayloadSlice, TransferLayout


def test_bucket_slice_preserves_both_endpoint_offsets() -> None:
    payload = PayloadSlice(
        bundle_id="bundle-0",
        tensor_role="hidden_states",
        src_rank=0,
        dst_rank=1,
        segment_ordinal=1,
        sender_offset_rows=92,
        receiver_offset_rows=0,
        row_count=100,
        dtype="torch.float16",
        shape_suffix=(8,),
        element_size_bytes=2,
        payload_byte_count=1600,
        packed_layout_id="send-layout",
    )
    layout = TransferLayout(
        transfer_key="P0:0->1",
        bundle_id="bundle-0",
        phase="P0",
        src_rank=0,
        dst_rank=1,
        source_peer_index=0,
        destination_peer_index=1,
        segment_ordinal=1,
        sender_offset_rows=92,
        receiver_offset_rows=0,
        row_count=100,
        byte_count=1600,
        packed_send_layout_id="send-layout",
        canonical_receive_layout_id="recv-layout",
        atomic_submit=True,
        payloads=(
            PackedTensorDescriptor(
                tensor_role="hidden_states",
                shape=(100, 8),
                shape_suffix=(8,),
                dtype="torch.float16",
                device="cuda:0",
                element_size_bytes=2,
            ),
        ),
        payload_slices=(payload,),
    )
    tasks = bucketize_transfer_layouts((layout,), bucket_rows=32)
    assert [task.sender_offset_rows for task in tasks] == [92, 124, 156, 188]
    assert [task.receiver_offset_rows for task in tasks] == [0, 32, 64, 96]
