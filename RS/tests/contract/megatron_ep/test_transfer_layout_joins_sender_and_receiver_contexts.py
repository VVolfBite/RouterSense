from __future__ import annotations

import torch

from rs.runtime.online.megatron_ep.phase.layout_join import join_transfer_layouts
from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    PhaseContextBuildRequest,
    PhasePayloadContract,
    RuntimeIdentity,
    build_phase_ready_context,
)


def _p0_context(*, rank: int, input_splits: tuple[int, int], output_splits: tuple[int, int]):
    hidden = torch.zeros((sum(input_splits), 8), dtype=torch.float16)
    probs = torch.zeros((sum(input_splits),), dtype=torch.float32)
    return build_phase_ready_context(
        PhaseContextBuildRequest(
            plan_key={"layer_id": "0", "phase": "P0"},
            runtime_identity=RuntimeIdentity(
                run_id="run",
                forward_epoch=0,
                layer_id="0",
                layer_name="layer0",
                global_rank=rank,
                local_rank=rank,
                ep_group_ranks=(0, 1),
                ep_group_root_rank=0,
            ),
            topology={"single_node": True},
            dispatcher_snapshot=DispatcherSnapshot(
                dispatcher_class="Dispatcher",
                dispatcher_fingerprint={"sha": "x"},
                expert_placement_hash="placement",
                input_splits=input_splits,
                output_splits=output_splits,
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


def test_asymmetric_two_rank_receive_offsets() -> None:
    rank0 = _p0_context(rank=0, input_splits=(92, 100), output_splits=(92, 92))
    rank1 = _p0_context(rank=1, input_splits=(92, 100), output_splits=(100, 100))
    layouts = join_transfer_layouts(global_contexts=(rank0, rank1), phase="P0")
    by_pair = {(item.src_rank, item.dst_rank): item for item in layouts}
    flow_01 = by_pair[(0, 1)]
    flow_10 = by_pair[(1, 0)]
    assert flow_01.sender_offset_rows == 92
    assert flow_01.receiver_offset_rows == 0
    assert flow_01.row_count == 100
    assert flow_10.sender_offset_rows == 0
    assert flow_10.receiver_offset_rows == 92
    assert flow_10.row_count == 92
