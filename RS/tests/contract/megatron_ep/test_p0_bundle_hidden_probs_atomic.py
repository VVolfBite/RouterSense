from __future__ import annotations

import torch

from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    PhaseContextBuildRequest,
    PhasePayloadContract,
    RuntimeIdentity,
    build_phase_ready_context,
    validate_p0_atomic_bundle,
)


def test_p0_bundle_hidden_probs_atomic() -> None:
    hidden = torch.zeros((4, 16), dtype=torch.float16)
    probs = torch.zeros((4,), dtype=torch.float32)
    context = build_phase_ready_context(
        PhaseContextBuildRequest(
            plan_key={"layer_id": "0", "phase": "P0"},
            runtime_identity=RuntimeIdentity(
                run_id="run",
                forward_epoch=0,
                layer_id="0",
                layer_name="layer0",
                global_rank=0,
                local_rank=0,
                ep_group_ranks=(0, 1),
                ep_group_root_rank=0,
            ),
            topology={"single_node": True},
            dispatcher_snapshot=DispatcherSnapshot(
                dispatcher_class="Dispatcher",
                dispatcher_fingerprint={"sha": "x"},
                expert_placement_hash="placement",
                input_splits=(1, 3),
                output_splits=(2, 2),
            ),
            payload_contract=PhasePayloadContract(
                phase="P0",
                payload_roles=("hidden_states", "routing_probs"),
                atomic_submit=True,
            ),
            packed_tensors=(hidden, probs),
            control_mode="sync_before_phase",
            release_state="ready",
            demand_known_at="router_ready",
            payload_exists=True,
        )
    )
    assert validate_p0_atomic_bundle(context) is True
    assert all(bundle.atomic_submit for bundle in context.transport_bundles)
    assert all([payload.tensor_role for payload in bundle.payloads] == ["hidden_states", "routing_probs"] for bundle in context.transport_bundles)
