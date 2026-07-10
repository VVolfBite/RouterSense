from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import AsyncReleaseP2PExecutor, AsyncReleaseP2PExecutorConfig, AsyncReleasePlanBuilder
from rs.scheduling.online_adapters.priority_artifact import PairedUPriorityArtifact, PriorityEntry


def _artifact() -> PairedUPriorityArtifact:
    return PairedUPriorityArtifact(
        source_safe_policy="RS_safe_barrier_criticality",
        raw_u_policy="U_barrier_criticality_global_matching",
        paired_b_policy="B_barrier_criticality_matching",
        selected_policy="U_barrier_criticality_global_matching",
        fallback_to_paired_b=False,
        heuristic_family="barrier_criticality_matching",
        predictor_name="copy_current_dispatch",
        p2_source="copy_current_dispatch",
        priority_entries=(
            PriorityEntry("p0_dispatch", 0, 1, 8, 10.0, 0, 8, "none"),
            PriorityEntry("p1_return", 1, 0, 4, 9.0, 1, 4, "wait_p0_complete"),
        ),
    )


def test_async_release_p2p_executor_orders_recvs_before_sends_deterministically() -> None:
    plan = AsyncReleasePlanBuilder(executor_available=False).build(
        priority_artifact=_artifact(),
        observed_context={"layer_id": "0"},
    )
    executor = AsyncReleaseP2PExecutor(config=AsyncReleaseP2PExecutorConfig(enabled=True))
    report = executor.execute(plan)
    ordered = report["ordered_ops"]
    assert ordered[0]["op_kind"] == "recv"
    assert ordered[1]["op_kind"] == "send"
    assert report["real_collectives_executed"] is False
    assert report["fallback_to_phase_sync"] is True
