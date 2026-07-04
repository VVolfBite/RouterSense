from __future__ import annotations

import torch

from integrations.megatron_ep.routersense.policy.bucketed_fifo import BucketedFIFOPolicy
from integrations.megatron_ep.routersense.phase import build_phase_ready_context


def _ctx(rank: int, send_remote: int) -> object:
    hidden = torch.zeros((send_remote + 1, 8), dtype=torch.float16)
    probs = torch.zeros((send_remote + 1,), dtype=torch.float32)
    return build_phase_ready_context(
        plan_key={"layer_id": "0", "phase": "P0"},
        phase="P0",
        control_mode="sync_before_phase",
        forward_epoch=0,
        layer_id="0",
        layer_name="layer0",
        global_rank=rank,
        local_rank=rank,
        ep_group_ranks=(0, 1),
        ep_group_root_rank=0,
        topology={"single_node": True},
        dispatcher_class="Dispatcher",
        dispatcher_fingerprint={"sha": "x"},
        expert_placement_hash="placement",
        input_splits=(1, send_remote) if rank == 0 else (send_remote, 1),
        output_splits=(1, send_remote) if rank == 0 else (send_remote, 1),
        packed_tensors=(hidden, probs),
        release_state="ready",
        demand_known_at="router_ready",
        payload_exists=True,
    )


def test_fifo_wave_constraints() -> None:
    ctx0 = _ctx(0, 3)
    ctx1 = _ctx(1, 3)
    plan = BucketedFIFOPolicy(bucket_rows=0).build_plan(local_context=ctx0, global_contexts=(ctx0, ctx1))
    assert len(plan.waves) == 1
    wave = plan.waves[0]
    assert len(wave.bucket_tasks) == 2
    srcs = {task.src_rank for task in wave.bucket_tasks}
    dsts = {task.dst_rank for task in wave.bucket_tasks}
    assert srcs == {0, 1}
    assert dsts == {0, 1}
