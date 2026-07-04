from __future__ import annotations

from dataclasses import replace

from integrations.megatron_ep.routersense.contracts import PeerFlow, PhaseDemand, RouterSensePlan
from integrations.megatron_ep.routersense.policy.agreement import decode_plan_tensor, encode_plan_tensor
from integrations.megatron_ep.routersense.policy.validation import summarize_plan_metrics
from integrations.megatron_ep.tests.helpers import make_context


def test_identity_plan_wire_roundtrip_preserves_phase_demands() -> None:
    context = make_context()
    flow = PeerFlow(
        flow_id="p0f",
        src_rank=0,
        dst_rank=1,
        phase="P0",
        rows=12,
        bytes=192,
        demand_known_at="router_ready",
        release_state="ready",
        release_dependency="none",
        payload_exists=True,
        is_cross_rank=True,
        is_cross_node=False,
    )
    plan = RouterSensePlan(
        run_id="run",
        step_id="step",
        microbatch_id="mb",
        layer_id="0",
        ep_group_hash=context.ep_group_hash,
        request_table_hash=context.request_table_hash,
        model_revision_hash=context.model_revision_hash,
        expert_placement_hash=context.expert_placement_hash,
        observation_digest="obs",
        plan_hash="placeholder",
        policy_name="native_passthrough_identity",
        policy_version="v1",
        execution_mode="native_passthrough",
        transport_mutation=False,
        future_hint_mode="none",
        control_mode="sync_before_phase",
        is_shadow_only=True,
        phase_demands=(
            PhaseDemand(
                phase="P0",
                demand_known_at="router_ready",
                release_state="ready",
                release_dependency="none",
                payload_exists=True,
                flows=(flow,),
                total_remote_rows=12,
                total_remote_bytes=192,
            ),
        ),
        metrics={},
    )
    plan = replace(plan, metrics=summarize_plan_metrics(plan))
    decoded = decode_plan_tensor(encode_plan_tensor(plan, 2))
    assert len(decoded.phase_demands) == 1
    assert decoded.phase_demands[0].phase == "P0"
    assert decoded.phase_demands[0].flows[0].rows == 12
