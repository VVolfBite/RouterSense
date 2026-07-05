from __future__ import annotations

from integrations.megatron_ep.routersense.policy.joint_shadow import JointShadowP0P1Policy

from .helpers import make_context, make_observation


def test_joint_shadow_never_executes_p1_before_release() -> None:
    plan = JointShadowP0P1Policy().build_plan(
        make_context(),
        (
            make_observation(rank=0, phase="P0", rows=(0, 5)),
            make_observation(rank=1, phase="P0", rows=(4, 0)),
            make_observation(rank=0, phase="P1", rows=(0, 3)),
            make_observation(rank=1, phase="P1", rows=(2, 0)),
        ),
    )
    assert all(flow.phase == "P0" for wave in plan.ready_waves for flow in wave.flows)
    assert all(flow.phase == "P1" for wave in plan.blocked_future_waves for flow in wave.flows)
