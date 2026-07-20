from __future__ import annotations

from rs.runtime.online.megatron_ep.control.shadow_policy.native_passthrough_identity import NativePassthroughIdentityPolicy
from .helpers import make_context, make_observation


def test_native_passthrough_identity_policy_emits_shadow_passthrough_plan() -> None:
    policy = NativePassthroughIdentityPolicy()
    context = make_context()
    observations = (
        make_observation(rank=0, phase="P0", rows=(0, 3)),
        make_observation(rank=1, phase="P0", rows=(4, 0)),
    )
    plan = policy.build_plan(context, observations)
    assert plan.policy_name == "native_passthrough_identity"
    assert plan.execution_mode == "native_passthrough"
    assert plan.transport_mutation is False
    assert plan.is_shadow_only is True
    assert len(plan.waves) == 0
