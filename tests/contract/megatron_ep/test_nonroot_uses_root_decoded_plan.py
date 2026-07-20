from __future__ import annotations

from rs.runtime.online.megatron_ep.control.agreement_wire import decode_plan_tensor, encode_plan_tensor
from rs.runtime.online.megatron_ep.control.shadow_policy.joint_shadow import JointShadowP0P1Policy

from .helpers import make_context, make_observation


def test_nonroot_uses_root_decoded_plan() -> None:
    context = make_context()
    observations = (
        make_observation(rank=0, phase="P0", rows=(0, 5)),
        make_observation(rank=1, phase="P0", rows=(4, 0)),
        make_observation(rank=0, phase="P1", rows=(0, 3)),
        make_observation(rank=1, phase="P1", rows=(2, 0)),
    )
    root_plan = JointShadowP0P1Policy().build_plan(context, observations)
    encoded = encode_plan_tensor(root_plan, 2)
    decoded_nonroot = decode_plan_tensor(encoded)
    assert [(wave.wave_id, wave.release_state, [(f.src_rank, f.dst_rank, f.phase, f.rows, f.bytes) for f in wave.flows]) for wave in decoded_nonroot.ready_waves] == [
        (wave.wave_id, wave.release_state, [(f.src_rank, f.dst_rank, f.phase, f.rows, f.bytes) for f in wave.flows]) for wave in root_plan.ready_waves
    ]
    assert [(wave.wave_id, wave.release_state, [(f.src_rank, f.dst_rank, f.phase, f.rows, f.bytes) for f in wave.flows]) for wave in decoded_nonroot.blocked_future_waves] == [
        (wave.wave_id, wave.release_state, [(f.src_rank, f.dst_rank, f.phase, f.rows, f.bytes) for f in wave.flows]) for wave in root_plan.blocked_future_waves
    ]
