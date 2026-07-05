from __future__ import annotations

import torch

from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    PhaseContextBuildRequest,
    PhasePayloadContract,
    RuntimeIdentity,
    build_phase_ready_context,
)


def test_p1_ready_context_semantics() -> None:
    hidden = torch.zeros((6, 8), dtype=torch.float16)
    context = build_phase_ready_context(
        PhaseContextBuildRequest(
            plan_key={"layer_id": "0", "phase": "P1"},
            runtime_identity=RuntimeIdentity("run", 0, "0", "layer0", 1, 1, (0, 1), 0),
            topology={"single_node": True},
            dispatcher_snapshot=DispatcherSnapshot("Dispatcher", {"sha": "x"}, "placement", (4, 2), (3, 3)),
            payload_contract=PhasePayloadContract("P1", ("hidden_states",), False),
            packed_tensors=(hidden,),
            control_mode="sync_before_phase",
            release_state="ready",
            demand_known_at="router_ready",
            payload_exists=True,
        )
    )
    assert context.phase == "P1"
    assert context.release_state == "ready"
    assert context.payload_exists is True
    assert context.send_splits == (3, 3)
