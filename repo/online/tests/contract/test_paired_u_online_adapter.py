from __future__ import annotations

from rs.scheduling.online_adapters import PairedUAsyncReleaseAdapter, PairedUPhaseSyncAdapter
from rs.scheduling.online_adapters.priority_artifact import PairedUPriorityArtifact, PriorityEntry


def _artifact() -> PairedUPriorityArtifact:
    return PairedUPriorityArtifact(
        source_safe_policy="RS_safe_gated_maxweight",
        raw_u_policy="U_gated_maxweight_matching",
        paired_b_policy="B_gated_maxweight_matching",
        selected_policy="U_gated_maxweight_matching",
        fallback_to_paired_b=False,
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
    hint = PairedUPhaseSyncAdapter().build_ordering_hint(_artifact())
    assert hint["source_safe_policy"] == "RS_safe_gated_maxweight"
    assert hint["source_u_policy"] == "U_gated_maxweight_matching"
    assert hint["paired_b_policy"] == "B_gated_maxweight_matching"
    assert hint["selected_policy"] == "U_gated_maxweight_matching"
    assert hint["granularity_mode"] == "dynamic_bucket_current"
    assert hint["heavy_solver_used_offline"] is False
    assert len(hint["ordering_hint"]) == 2


def test_async_adapter_emits_release_priority_and_fallback_decision() -> None:
    decision = PairedUAsyncReleaseAdapter().build_release_priority(_artifact(), fallback_required=True)
    assert decision["source_safe_policy"] == "RS_safe_gated_maxweight"
    assert decision["source_u_policy"] == "U_gated_maxweight_matching"
    assert decision["fallback_decision"] == "fallback_phase_sync"
    assert decision["granularity_mode"] == "dynamic_bucket_current"
    assert decision["heavy_solver_used_offline"] is False
