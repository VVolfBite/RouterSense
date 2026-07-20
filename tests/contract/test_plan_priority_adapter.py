from __future__ import annotations

from rs.scheduling.online_adapters import PlanPriorityAsyncReleaseAdapter, PlanPriorityPhaseSyncAdapter
from rs.scheduling.online_adapters.plan_priority import PlanPriorityArtifact, PriorityEntry


def _artifact() -> PlanPriorityArtifact:
    return PlanPriorityArtifact(
        source_policy="safe_pair",
        joint_policy="future:p012:joint:global:rscf",
        local_policy="future:p012:local:global:rscf",
        selected_policy="future:p012:joint:global:rscf",
        fallback_to_local=False,
        heuristic_family="gated_maxweight_matching",
        predictor_name="fate_style_linear",
        p2_source="predicted_next_dispatch",
        priority_entries=(
            PriorityEntry("p0_dispatch", 0, 1, 8, 10.0, 0, 8, "none"),
            PriorityEntry("p1_return", 1, 0, 4, 9.0, 1, 4, "wait_p0_complete"),
        ),
        heavy_solver_used_offline=False,
    )


def test_phase_sync_adapter_consumes_u_priority_artifact_without_heavy_solver() -> None:
    hint = PlanPriorityPhaseSyncAdapter().build_ordering_hint(_artifact())
    assert hint["source_policy"] == "safe_pair"
    assert hint["joint_policy"] == "future:p012:joint:global:rscf"
    assert hint["local_policy"] == "future:p012:local:global:rscf"
    assert hint["selected_policy"] == "future:p012:joint:global:rscf"
    assert hint["granularity_mode"] == "dynamic_bucket_current"
    assert hint["heavy_solver_used_offline"] is False
    assert len(hint["ordering_hint"]) == 2


def test_async_adapter_emits_release_priority_and_fallback_decision() -> None:
    decision = PlanPriorityAsyncReleaseAdapter().build_release_priority(_artifact(), fallback_required=True)
    assert decision["source_policy"] == "safe_pair"
    assert decision["joint_policy"] == "future:p012:joint:global:rscf"
    assert decision["fallback_decision"] == "fallback_phase_sync"
    assert decision["granularity_mode"] == "dynamic_bucket_current"
    assert decision["heavy_solver_used_offline"] is False
