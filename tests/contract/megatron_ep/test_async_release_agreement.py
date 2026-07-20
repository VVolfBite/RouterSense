from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import (
    AsyncReleasePlanBuilder,
    CompiledAsyncReleaseSchedule,
    compile_async_release_schedule,
    gather_and_validate_async_release_schedule,
    validate_async_release_global_agreement,
)
from rs.scheduling.online_adapters.plan_priority import PlanPriorityArtifact, PriorityEntry


def _artifact(byte_count: int = 8) -> PlanPriorityArtifact:
    return PlanPriorityArtifact(
        source_policy="safe_pair",
        joint_policy="future:p012:joint:global:rscf",
        local_policy="future:p012:local:global:rscf",
        selected_policy="future:p012:joint:global:rscf",
        fallback_to_local=False,
        heuristic_family="rscf",
        predictor_name="copy_current_dispatch",
        p2_source="copy_current_dispatch",
        priority_entries=(PriorityEntry("p0_dispatch", 0, 1, byte_count, 10.0, 0, byte_count, "none"),),
    )


def test_async_release_agreement_accepts_identical_compiled_schedules() -> None:
    builder = AsyncReleasePlanBuilder(executor_available=False)
    plan = builder.build(priority_artifact=_artifact(), observed_context={"layer_id": "0"})
    first = compile_async_release_schedule(plan)
    second = compile_async_release_schedule(plan)
    result = validate_async_release_global_agreement((first, second))
    assert result["valid"] is True


def test_async_release_agreement_fails_closed_on_payload_mismatch() -> None:
    builder = AsyncReleasePlanBuilder(executor_available=False)
    first = compile_async_release_schedule(builder.build(priority_artifact=_artifact(8), observed_context={"layer_id": "0"}))
    second = compile_async_release_schedule(builder.build(priority_artifact=_artifact(9), observed_context={"layer_id": "0"}))
    result = validate_async_release_global_agreement((first, second))
    assert result["valid"] is False
    assert any("mismatch" in error for error in result["errors"])


def test_async_release_agreement_fake_backend_path_accepts_identical_schedules() -> None:
    builder = AsyncReleasePlanBuilder(executor_available=False)
    plan = builder.build(priority_artifact=_artifact(), observed_context={"layer_id": "0"})
    first = compile_async_release_schedule(plan)
    second = compile_async_release_schedule(plan)
    result = gather_and_validate_async_release_schedule(first, gathered_schedules=(first, second))
    assert result["valid"] is True


def test_async_release_agreement_detects_variable_length_payload_mismatch() -> None:
    builder = AsyncReleasePlanBuilder(executor_available=False)
    first = compile_async_release_schedule(builder.build(priority_artifact=_artifact(8), observed_context={"layer_id": "0"}))
    second = CompiledAsyncReleaseSchedule(
        task_count=int(first.task_count) + 1,
        tensor_payload=first.tensor_payload[:-1].clone(),
        schema_version=int(first.schema_version),
        digest=str(first.digest),
    )
    result = gather_and_validate_async_release_schedule(first, gathered_schedules=(first, second))
    assert result["valid"] is False
    assert any("mismatch" in error for error in result["errors"])
