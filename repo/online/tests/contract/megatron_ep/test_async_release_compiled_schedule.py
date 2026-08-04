from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import (
    AsyncReleasePlanBuilder,
    compile_async_release_schedule,
    decode_compiled_async_release_schedule,
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


def test_compile_async_release_schedule_uses_int64_tensor_schema() -> None:
    plan = AsyncReleasePlanBuilder(executor_available=False).build(
        priority_artifact=_artifact(),
        observed_context={"layer_id": "0"},
    )
    schedule = compile_async_release_schedule(plan)
    decoded = decode_compiled_async_release_schedule(schedule)
    assert str(schedule.tensor_payload.dtype) == "torch.int64"
    assert schedule.task_count == 2
    assert decoded["task_count"] == 2
    assert decoded["rows"][0]["global_order_index"] == 0
    assert decoded["rows"][1]["dependency_count"] >= 1
