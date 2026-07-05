from __future__ import annotations

import torch

from integrations.megatron_ep.routersense.phase import build_phase_ready_context, validate_layout_offsets_cover_exactly_once


def test_p1_layout_offsets_cover_exactly_once() -> None:
    hidden = torch.zeros((5, 8), dtype=torch.float16)
    context = build_phase_ready_context(
        plan_key={"layer_id": "0", "phase": "P1"},
        phase="P1",
        control_mode="sync_before_phase",
        forward_epoch=0,
        layer_id="0",
        layer_name="layer0",
        global_rank=0,
        local_rank=0,
        ep_group_ranks=(0, 1),
        ep_group_root_rank=0,
        topology={"single_node": True},
        dispatcher_class="Dispatcher",
        dispatcher_fingerprint={"sha": "x"},
        expert_placement_hash="placement",
        input_splits=(4, 1),
        output_splits=(2, 3),
        packed_tensors=(hidden,),
        release_state="ready",
        demand_known_at="router_ready",
        payload_exists=True,
    )
    assert context.send_splits == (2, 3)
    assert [segment.send_offset_rows for segment in context.outgoing_segments] == [0, 2]
    assert [slot.receive_offset_rows for slot in context.incoming_slots] == [0, 4]
    assert validate_layout_offsets_cover_exactly_once(context) is True
