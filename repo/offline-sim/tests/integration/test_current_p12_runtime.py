from __future__ import annotations



from _current_p12_test_helpers import run_current_p12 as _run


def test_current_p12_global_joint_prepares_before_target_truth_and_hides_cost() -> None:
    runtime = _run(
        run_id="current-p12-global-overlap",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="OVERLAP",
    )
    try:
        metrics = runtime.observer_bridge.metrics()
        templates = metrics.current_p12_template_evidence
        assert len(templates) == len(runtime.current_p12_windows)
        assert all(item.template_ready_at_ns < item.target_first_truth_at_ns for item in templates)
        assert all(item.target_bound_at_ns >= item.target_first_truth_at_ns for item in templates)
        lines = {item.line_name: item for item in metrics.line_metrics}
        assert lines["PredictionLine"].hidden_service_ns > 0
        assert lines["ControlLine"].hidden_service_ns > 0
        records = runtime.current_p12_window_records()
        assert len(records) == len(runtime.current_p12_windows)
        assert all(item.terminal and item.window_makespan_ns > 0 for item in records)
        assert all(item.physical_completed_bytes > 0 for item in records)
    finally:
        runtime.dispose()


def test_overlap_and_serialized_have_same_truth_but_different_exposure() -> None:
    overlap = _run(
        run_id="current-p12-overlap",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="OVERLAP",
    )
    serialized = _run(
        run_id="current-p12-serialized",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="SERIALIZED",
    )
    try:
        overlap_lines = {item.line_name: item for item in overlap.observer_bridge.metrics().line_metrics}
        serialized_lines = {item.line_name: item for item in serialized.observer_bridge.metrics().line_metrics}
        assert sum(item.hidden_service_ns for item in overlap_lines.values()) > 0
        assert sum(item.hidden_service_ns for item in serialized_lines.values()) == 0
        assert sum(item.exposed_service_ns for item in overlap_lines.values()) < sum(
            item.exposed_service_ns for item in serialized_lines.values()
        )
        assert [item.truth_digest for item in overlap.current_p12_window_records()] == [
            item.truth_digest for item in serialized.current_p12_window_records()
        ]
    finally:
        overlap.dispose()
        serialized.dispose()


def test_event_recomputes_after_frontier_release_without_repredicting() -> None:
    runtime = _run(
        run_id="current-p12-event",
        algorithm="joint(event(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="OVERLAP",
    )
    try:
        metrics = runtime.observer_bridge.metrics()
        lines = {item.line_name: item for item in metrics.line_metrics}
        assert metrics.frontier_replan_count > 0
        assert lines["PredictionLine"].job_count == len(runtime.current_p12_windows)
        assert lines["ControlLine"].job_count > lines["PredictionLine"].job_count
    finally:
        runtime.dispose()


def test_phase_local_cannot_consume_p2_prediction() -> None:
    runtime = _run(
        run_id="current-p12-local",
        algorithm="local(global_(rscf()))",
        information_mode="PERFECT_P2",
        overlap_mode="OVERLAP",
    )
    try:
        metrics = runtime.observer_bridge.metrics()
        lines = {item.line_name: item for item in metrics.line_metrics}
        assert lines["PredictionLine"].job_count == 0
        assert runtime.run_axes["information_mode"] == "NO_P2_INFORMATION_PHASE_LOCAL"
        assert all(item.prediction_digest == "NO_P2_PREDICTION" for item in runtime.current_p12_window_records())
    finally:
        runtime.dispose()


def test_current_p12_emits_anchor_local_formal_runtime_records():
    runtime = _run(
        run_id="current-p12-formal-records",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="OVERLAP",
    )
    try:
        records = runtime.formal_current_p12_records()
        assert len(records) == len(runtime.current_p12_windows)
        assert all(item.objective_unit == "nanoseconds" for item in records)
        assert all(item.horizon == "P12" for item in records)
        assert all(item.plan_count == 1 for item in records)
        assert all(item.window_makespan_ns < item.run_forward_makespan_ns for item in records)
        assert all(not item.provenance.performance_eligible for item in records)
        assert len({item.paired_key.window_truth_digest for item in records}) == len(records)
    finally:
        runtime.dispose()

