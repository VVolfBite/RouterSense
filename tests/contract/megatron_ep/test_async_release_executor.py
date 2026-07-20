from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import (
    AsyncReleaseExecutor,
    AsyncReleaseExecutorConfig,
    AsyncReleasePlanBuilder,
)
from rs.scheduling.online_adapters.plan_priority import PlanPriorityArtifact, PriorityEntry


def _artifact() -> PlanPriorityArtifact:
    return PlanPriorityArtifact(
        source_policy="safe_pair",
        joint_policy="future:p012:joint:global:rscf",
        local_policy="future:p012:local:global:rscf",
        selected_policy="future:p012:joint:global:rscf",
        fallback_to_local=False,
        heuristic_family="rscf",
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


def test_async_release_executor_flags_cannot_bypass_unimplemented_real_collective_path() -> None:
    plan = AsyncReleasePlanBuilder(executor_available=True).build(
        priority_artifact=_artifact(),
        observed_context={"layer_id": "0"},
    )
    executor = AsyncReleaseExecutor(
        config=AsyncReleaseExecutorConfig(
            enabled=True,
            dry_run=False,
            allow_real_collectives=True,
            real_collective_executor_implemented=False,
        )
    )
    result = executor.execute_or_fallback(plan, rank=0, world_size=2)
    assert result["async_release_real_collectives"] is False
    assert result["fallback_to_phase_sync"] is True
    assert result["fallback_reason"] == "real_collective_path_not_implemented"
