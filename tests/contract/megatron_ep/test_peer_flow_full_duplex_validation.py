from __future__ import annotations

from rs.runtime.online.megatron_ep.contracts import PeerFlow, PhaseDemand, PlanWave, RouterSensePlan
from rs.scheduling.validation import summarize_plan_metrics, validate_shadow_plan

from .helpers import make_context


def _flow(
    flow_id: str,
    src: int,
    dst: int,
    phase: str,
    *,
    rows: int = 4,
    bytes_: int = 64,
    release_state: str = "ready",
    payload_exists: bool = True,
) -> PeerFlow:
    return PeerFlow(
        flow_id=flow_id,
        src_rank=src,
        dst_rank=dst,
        phase=phase,
        rows=rows,
        bytes=bytes_,
        demand_known_at="router_ready",
        release_state=release_state,
        release_dependency="none" if phase == "P0" else "remote_expert_compute_complete",
        payload_exists=payload_exists,
        is_cross_rank=True,
        is_cross_node=False,
    )


def test_full_duplex_shadow_plan_is_valid() -> None:
    context = make_context()
    f01 = _flow("0->1", 0, 1, "P0", release_state="ready", payload_exists=True)
    f10 = _flow("1->0", 1, 0, "P0", release_state="ready", payload_exists=True)
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
        plan_hash="plan",
        policy_name="joint_shadow_p0p1",
        policy_version="v1",
        execution_mode="shadow_only",
        transport_mutation=False,
        future_hint_mode="none",
        control_mode="default_continue",
        is_shadow_only=True,
        phase_demands=(
            PhaseDemand(phase="P0", demand_known_at="router_ready", release_state="ready", release_dependency="none", payload_exists=True, flows=(f01, f10), total_remote_rows=8, total_remote_bytes=128),
        ),
        ready_waves=(PlanWave(wave_id=0, release_state="ready", flows=(f01, f10)),),
        blocked_future_waves=(),
    )
    validate_shadow_plan(context, plan)
    metrics = summarize_plan_metrics(plan)
    assert metrics["duplex_pair_count"] == 1
    assert metrics["wave_count"] == 1
