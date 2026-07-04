from __future__ import annotations

from integrations.megatron_ep.routersense.policy.native_order import NativeOrderPolicy

from integrations.megatron_ep.tests.helpers import make_context, make_observation


def test_native_order_policy_emits_no_waves_and_no_mutation() -> None:
    context = make_context()
    plan = NativeOrderPolicy().build_plan(
        context,
        (
            make_observation(rank=0, phase="P0", rows=(0, 5)),
            make_observation(rank=1, phase="P0", rows=(4, 0)),
            make_observation(rank=0, phase="P1", rows=(0, 3)),
            make_observation(rank=1, phase="P1", rows=(2, 0)),
        ),
    )
    assert plan.policy_name == "native_order"
    assert plan.execution_mode == "native_passthrough"
    assert plan.transport_mutation is False
    assert plan.ready_waves == ()
    assert plan.blocked_future_waves == ()
    assert plan.metrics["total_remote_rows"] == 14
