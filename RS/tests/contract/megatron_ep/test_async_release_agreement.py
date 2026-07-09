from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import (
    AsyncReleasePlanBuilder,
    compile_async_release_schedule,
    validate_async_release_global_agreement,
)
from rs.scheduling.online_adapters.priority_artifact import PairedUPriorityArtifact, PriorityEntry


def _artifact(byte_count: int = 8) -> PairedUPriorityArtifact:
    return PairedUPriorityArtifact(
        source_safe_policy="RS_safe_barrier_criticality",
        raw_u_policy="U_barrier_criticality_global_matching",
        paired_b_policy="B_barrier_criticality_matching",
        selected_policy="U_barrier_criticality_global_matching",
        fallback_to_paired_b=False,
        heuristic_family="barrier_criticality_matching",
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

