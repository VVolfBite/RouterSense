from __future__ import annotations

from rs.scheduling.online_adapters import PairedUAsyncReleaseAdapter, PairedUPhaseSyncAdapter
from rs.scheduling.online_adapters.paired_u import PairedUPriorityArtifact


def _artifact() -> PairedUPriorityArtifact:
    return PairedUPriorityArtifact(
        source_u_policy="U_gated_maxweight_matching",
        paired_b_policy="B_gated_maxweight_matching",
        predictor_name="fate_style_linear",
        p2_source="predicted_next_dispatch",
        priority_table=(
            {"phase": "p0_dispatch", "src_rank": 0, "dst_rank": 1, "priority": 10.0},
            {"phase": "p1_return", "src_rank": 1, "dst_rank": 0, "priority": 9.0},
        ),
        generated_offline_or_shadow=True,
        heavy_solver_used_offline=False,
    )


def test_phase_sync_adapter_consumes_u_priority_artifact_without_heavy_solver() -> None:
    hint = PairedUPhaseSyncAdapter().build_ordering_hint(_artifact())
    assert hint["source_u_policy"] == "U_gated_maxweight_matching"
    assert hint["paired_b_policy"] == "B_gated_maxweight_matching"
    assert hint["granularity_mode"] == "dynamic_bucket_current"
    assert hint["heavy_solver_used_offline"] is False
    assert len(hint["ordering_hint"]) == 2


def test_async_adapter_emits_release_priority_and_fallback_decision() -> None:
    decision = PairedUAsyncReleaseAdapter().build_release_priority(_artifact(), fallback_required=True)
    assert decision["source_u_policy"] == "U_gated_maxweight_matching"
    assert decision["fallback_decision"] == "fallback_phase_sync"
    assert decision["granularity_mode"] == "dynamic_bucket_current"
    assert decision["heavy_solver_used_offline"] is False
