from __future__ import annotations

import torch

from rs.runtime.online.megatron_ep.phase import build_phase_ready_context


def test_p1_ready_context_semantics() -> None:
    hidden = torch.zeros((6, 8), dtype=torch.float16)
    context = build_phase_ready_context(
        plan_key={"layer_id": "0", "phase": "P1"},
        phase="P1",
        control_mode="sync_before_phase",
        forward_epoch=0,
        layer_id="0",
        layer_name="layer0",
        global_rank=1,
        local_rank=1,
        ep_group_ranks=(0, 1),
        ep_group_root_rank=0,
        topology={"single_node": True},
        dispatcher_class="Dispatcher",
        dispatcher_fingerprint={"sha": "x"},
        expert_placement_hash="placement",
        input_splits=(4, 2),
        output_splits=(3, 3),
        packed_tensors=(hidden,),
        release_state="ready",
        demand_known_at="router_ready",
        payload_exists=True,
    )
    assert context.phase == "P1"
    assert context.release_state == "ready"
    assert context.payload_exists is True
    assert context.send_splits == (3, 3)
