from __future__ import annotations

from _current_p12_test_helpers import run_current_p12 as _run
from rs_sim.runtime.metrics.communication import (
    compute_excluded_communication_makespan_ns,
    interval_union_duration_ns,
    nearest_rank_percentile_ns,
    network_active_union_ns,
    summarize_rank_communication_exposure_ns,
)


def test_interval_union_excludes_idle_gaps_without_double_counting_overlap() -> None:
    assert interval_union_duration_ns([(0, 10), (5, 15), (20, 30)]) == 25
    assert compute_excluded_communication_makespan_ns(
        task_ready_complete_intervals=[(0, 10), (20, 30)]
    ) == 20
    assert network_active_union_ns(
        task_start_complete_intervals=[(2, 8), (6, 12), (18, 20)]
    ) == 12


def test_rank_exposure_summary_is_tail_preserving() -> None:
    summary = summarize_rank_communication_exposure_ns((10, 10, 10, 25))
    assert summary.total_ns == 55
    assert summary.mean_ns == 14
    assert summary.max_ns == 25
    assert summary.p95_ns == 25
    assert summary.p99_ns == 25
    assert summary.critical_rank == 3
    assert nearest_rank_percentile_ns((1, 2, 3, 4), 95) == 4


def test_current_p12_reports_compute_excluded_and_rank_local_metrics() -> None:
    runtime = _run(
        run_id="current-p12-communication-only-metrics",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="OVERLAP",
    )
    try:
        records = runtime.current_p12_window_records()
        assert records
        for record in records:
            assert 0 < record.compute_excluded_communication_makespan_ns <= record.window_makespan_ns
            assert 0 < record.network_active_union_ns <= record.compute_excluded_communication_makespan_ns
            assert len(record.rank_communication_exposed_ns_by_rank) == runtime.fixture_input.world_size
            assert record.rank_communication_exposed_ns_max == max(record.rank_communication_exposed_ns_by_rank)
            assert record.rank_communication_exposed_ns_sum == sum(record.rank_communication_exposed_ns_by_rank)
            assert record.rank_communication_exposed_ns_p95 <= record.rank_communication_exposed_ns_max
            assert record.rank_communication_exposed_ns_p99 <= record.rank_communication_exposed_ns_max
    finally:
        runtime.dispose()
