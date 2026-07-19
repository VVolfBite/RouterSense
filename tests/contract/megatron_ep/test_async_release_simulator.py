from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import simulate_async_release
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
        p2_source="predicted_next_dispatch",
        priority_entries=(
            PriorityEntry("p0_dispatch", 1, 0, 6, 12.0, 0, 6, "none"),
            PriorityEntry("p1_return", 0, 1, 8, 11.0, 1, 8, "wait_p0_complete"),
        ),
    )


def test_async_release_simulator_releases_p0_and_blocks_p1_until_ready() -> None:
    report = simulate_async_release(
        p0_dispatch_matrix=((0, 2), (6, 0)),
        p1_return_matrix=((0, 8), (1, 0)),
        predicted_p2_matrix=((0, 4), (3, 0)),
        compute_delay=1.0,
        prediction_lead_time_us=2.0,
        planning_time_us=1.0,
        priority_artifact=_artifact(),
    )
    assert report["dependency_violations"] == 0
    assert report["early_release_task_count"] >= 1
    assert report["blocked_task_count"] >= 1
    assert 0.0 <= report["hidden_planning_fraction"] <= 1.0
    assert report["used_priority_artifact"] is True
    assert report["remote_only_matrix_invariant"] is True


def test_async_release_oracle_p2_is_not_worse_than_zero_hint_on_sensitive_case() -> None:
    zero = simulate_async_release(
        p0_dispatch_matrix=((0, 2), (6, 0)),
        p1_return_matrix=((0, 8), (1, 0)),
        predicted_p2_matrix=((0, 0), (0, 0)),
        compute_delay=0.0,
    )
    oracle = simulate_async_release(
        p0_dispatch_matrix=((0, 2), (6, 0)),
        p1_return_matrix=((0, 8), (1, 0)),
        predicted_p2_matrix=((0, 4), (3, 0)),
        compute_delay=0.0,
    )
    assert oracle["completion_time"] <= zero["completion_time"]


def test_async_release_is_diagonal_invariant_and_tracks_ignored_self_bytes() -> None:
    clean = simulate_async_release(
        p0_dispatch_matrix=((0, 2), (6, 0)),
        p1_return_matrix=((0, 8), (1, 0)),
        predicted_p2_matrix=((0, 4), (3, 0)),
    )
    dirty = simulate_async_release(
        p0_dispatch_matrix=((99, 2), (6, 77)),
        p1_return_matrix=((55, 8), (1, 66)),
        predicted_p2_matrix=((44, 4), (3, 88)),
    )
    assert clean["completion_time"] == dirty["completion_time"]
    assert clean["predicted_p2_total_bytes"] == dirty["predicted_p2_total_bytes"]
    assert dirty["p0_self_bytes_ignored"] > 0
    assert dirty["p1_self_bytes_ignored"] > 0
    assert dirty["predicted_p2_self_bytes_ignored"] > 0
