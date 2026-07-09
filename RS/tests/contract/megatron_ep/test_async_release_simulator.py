from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import simulate_async_release


def test_async_release_simulator_releases_p0_and_blocks_p1_until_ready() -> None:
    report = simulate_async_release(
        p0_dispatch_matrix=((0, 2), (6, 0)),
        p1_return_matrix=((0, 8), (1, 0)),
        predicted_p2_matrix=((0, 4), (3, 0)),
        compute_delay=1.0,
        prediction_lead_time_us=2.0,
        planning_time_us=1.0,
    )
    assert report["dependency_violations"] == 0
    assert report["early_release_task_count"] >= 1
    assert report["blocked_task_count"] >= 1
    assert 0.0 <= report["hidden_planning_fraction"] <= 1.0


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
