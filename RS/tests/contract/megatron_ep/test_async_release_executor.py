from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import (
    AsyncReleaseExecutor,
    AsyncReleaseExecutorConfig,
    AsyncReleasePlanBuilder,
)
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


def test_async_release_executor_defaults_to_phase_sync_fallback() -> None:
    plan = AsyncReleasePlanBuilder(executor_available=False).build(
        priority_artifact=_artifact(),
        observed_context={"layer_id": "0"},
    )
    executor = AsyncReleaseExecutor(config=AsyncReleaseExecutorConfig())
    result = executor.execute_or_fallback(plan, rank=0, world_size=2)
    assert result["async_release_enabled"] is False
    assert result["fallback_to_phase_sync"] is True
    assert result["async_release_real_collectives"] is False


def test_async_release_executor_real_collectives_require_explicit_flags() -> None:
    plan = AsyncReleasePlanBuilder(executor_available=True).build(
        priority_artifact=_artifact(),
        observed_context={"layer_id": "0"},
    )
    executor = AsyncReleaseExecutor(
        config=AsyncReleaseExecutorConfig(enabled=True, dry_run=False, allow_real_collectives=False)
    )
    result = executor.execute_or_fallback(plan, rank=0, world_size=2)
    assert result["allow_real_collectives"] is False
    assert result["fallback_to_phase_sync"] is True
    assert result["fallback_reason"]

