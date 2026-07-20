from __future__ import annotations

import torch

from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    PhaseContextBuildRequest,
    PhasePayloadContract,
    RuntimeIdentity,
    build_phase_ready_context,
    validate_layout_offsets_cover_exactly_once,
)


def test_p0_layout_offsets_cover_exactly_once() -> None:
    hidden = torch.zeros((5, 8), dtype=torch.float16)
    probs = torch.zeros((5,), dtype=torch.float32)
    context = build_phase_ready_context(
        PhaseContextBuildRequest(
            plan_key={"layer_id": "0", "phase": "P0"},
            runtime_identity=RuntimeIdentity("run", 0, "0", "layer0", 0, 0, (0, 1), 0),
            topology={"single_node": True},
            dispatcher_snapshot=DispatcherSnapshot("Dispatcher", {"sha": "x"}, "placement", (2, 3), (4, 1)),
            payload_contract=PhasePayloadContract("P0", ("hidden_states", "routing_probs"), True),
            packed_tensors=(hidden, probs),
            control_mode="sync_before_phase",
            release_state="ready",
            demand_known_at="router_ready",
            payload_exists=True,
        )
    )
    assert context.send_splits == (2, 3)
    assert [segment.send_offset_rows for segment in context.outgoing_segments] == [0, 2]
    assert [slot.receive_offset_rows for slot in context.incoming_slots] == [0, 4]
    assert validate_layout_offsets_cover_exactly_once(context) is True
