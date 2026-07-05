from __future__ import annotations

import torch

from rs.scheduling.phase_local.fifo import BucketedFIFOPolicy
from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    PhaseContextBuildRequest,
    PhasePayloadContract,
    RuntimeIdentity,
    build_phase_ready_context,
)


def _ctx(rank: int, send_remote: int) -> object:
    hidden = torch.zeros((send_remote + 1, 8), dtype=torch.float16)
    probs = torch.zeros((send_remote + 1,), dtype=torch.float32)
    return build_phase_ready_context(
        PhaseContextBuildRequest(
            plan_key={"layer_id": "0", "phase": "P0"},
            runtime_identity=RuntimeIdentity("run", 0, "0", "layer0", rank, rank, (0, 1), 0),
            topology={"single_node": True},
            dispatcher_snapshot=DispatcherSnapshot(
                "Dispatcher",
                {"sha": "x"},
                "placement",
                (1, send_remote) if rank == 0 else (send_remote, 1),
                (1, send_remote) if rank == 0 else (send_remote, 1),
            ),
            payload_contract=PhasePayloadContract("P0", ("hidden_states", "routing_probs"), True),
            packed_tensors=(hidden, probs),
            control_mode="sync_before_phase",
            release_state="ready",
            demand_known_at="router_ready",
            payload_exists=True,
        )
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
