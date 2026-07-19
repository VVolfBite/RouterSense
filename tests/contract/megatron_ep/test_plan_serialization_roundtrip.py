from __future__ import annotations

from dataclasses import replace

from rs.runtime.online.megatron_ep.contracts import PeerFlow, PhaseDemand, PlanWave, RouterSensePlan
from rs.runtime.online.megatron_ep.control.agreement_wire import decode_plan_tensor, encode_plan_tensor
from rs.scheduling.validation import summarize_plan_metrics

from .helpers import make_context


def test_root_plan_decode_roundtrip() -> None:
    context = make_context()
    flow = PeerFlow(
        flow_id="f0",
        src_rank=0,
        dst_rank=1,
        phase="P1",
        rows=8,
        bytes=128,
        demand_known_at="router_ready",
        release_state="blocked",
        release_dependency="remote_expert_compute_complete",
        payload_exists=False,
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
        policy_name="joint_shadow_p0p1",
        policy_version="v1",
        execution_mode="shadow_only",
        transport_mutation=False,
        future_hint_mode="none",
        control_mode="sync_before_phase",
        is_shadow_only=True,
        phase_demands=(PhaseDemand(phase="P1", demand_known_at="router_ready", release_state="blocked", release_dependency="remote_expert_compute_complete", payload_exists=False, flows=(flow,), total_remote_rows=8, total_remote_bytes=128),),
        ready_waves=(),
        blocked_future_waves=(PlanWave(wave_id=1, release_state="blocked", flows=(flow,)),),
        metrics={},
    )
    plan = replace(plan, metrics=summarize_plan_metrics(plan))
    encoded = encode_plan_tensor(plan, 2)
    decoded = decode_plan_tensor(encoded)
    assert decoded.policy_name == plan.policy_name
    assert decoded.control_mode == "sync_before_phase"
    assert decoded.is_shadow_only is True
    assert len(decoded.blocked_future_waves) == 1
    assert decoded.blocked_future_waves[0].flows[0].phase == "P1"
    assert decoded.metrics["blocked_future_wave_count"] == 1
