from __future__ import annotations

from integrations.megatron_ep.routersense.policy.joint_shadow import JointShadowP0P1Policy

from integrations.megatron_ep.tests.helpers import make_context, make_observation


def test_joint_shadow_policy_builds_non_empty_full_duplex_wave_plan() -> None:
    context = make_context()
    observations = (
        make_observation(rank=0, phase="P0", rows=(0, 5)),
        make_observation(rank=1, phase="P0", rows=(4, 0)),
        make_observation(rank=0, phase="P1", rows=(0, 3)),
        make_observation(rank=1, phase="P1", rows=(2, 0)),
    )
    plan = JointShadowP0P1Policy().build_plan(context, observations)
    assert plan.policy_name == "joint_shadow_p0p1"
    assert plan.execution_mode == "shadow_only"
    assert plan.transport_mutation is False
    assert len(plan.ready_waves) >= 1
    assert len(plan.blocked_future_waves) >= 1
    ready_covered = {flow.flow_id for wave in plan.ready_waves for flow in wave.flows}
    blocked_covered = {flow.flow_id for wave in plan.blocked_future_waves for flow in wave.flows}
    assert ready_covered == {"0:P0:0->1", "0:P0:1->0"}
    assert blocked_covered == {"0:P1:0->1", "0:P1:1->0"}
    assert plan.metrics["duplex_pair_count"] >= 1
