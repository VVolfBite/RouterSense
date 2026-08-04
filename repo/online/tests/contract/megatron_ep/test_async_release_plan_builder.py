from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import AsyncReleasePlanBuilder, validate_async_release_execution_plan
from rs.scheduling.online_adapters.priority_artifact import PairedUPriorityArtifact, PriorityEntry


def _artifact() -> PairedUPriorityArtifact:
    return PairedUPriorityArtifact(
        source_safe_policy="RS_safe_barrier_criticality",
        raw_u_policy="U_barrier_criticality_global_matching",
        paired_b_policy="B_barrier_criticality_matching",
        selected_policy="U_barrier_criticality_global_matching",
        fallback_to_paired_b=False,
        heuristic_family="barrier_criticality_matching",
        predictor_name="fate_style_linear",
        p2_source="fate_style_linear",
        priority_entries=(
            PriorityEntry("p0_dispatch", 0, 1, 8, 10.0, 0, 8, "none"),
            PriorityEntry("p1_return", 1, 0, 4, 9.0, 1, 4, "wait_p0_complete"),
        ),
    )


def test_async_release_plan_builder_fails_closed_without_executor() -> None:
    plan = AsyncReleasePlanBuilder(executor_available=False).build(
        priority_artifact=_artifact(),
        observed_context={"layer_id": "0", "phase": "P0"},
    )
    assert plan.fallback_to_phase_sync is True
    assert plan.online_executor_eligible is False
    assert plan.debug_replay_only is True
    result = validate_async_release_execution_plan(plan)
    assert result["valid"] is True
    assert plan.event_table == {}
