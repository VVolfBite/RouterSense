from __future__ import annotations

from rs_sim.runtime import build_current_p12_integration_runtime
from rs_sim.scheduler import PlanningCostModel
from rs_sim.trace import build_golden_fixture

from _current_p12_test_helpers import run_current_p12
from tests.support.runtime_profiles import synthetic_runtime_profile


def test_serialized_planning_changes_physical_critical_path() -> None:
    cost = PlanningCostModel(
        prediction_base_ns=4_000,
        prediction_per_task_ns=10,
        control_base_ns=4_000,
        control_per_task_ns=10,
        control_per_phase_ns=100,
        binding_base_ns=1_000,
        binding_per_task_ns=5,
        binding_per_phase_ns=50,
    )
    overlap = run_current_p12(
        run_id="current-p12-critical-path",
        paired_instance_id="paired-critical-path",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="OVERLAP",
        runtime_profile=synthetic_runtime_profile(planning_cost_model=cost),
    )
    serialized = run_current_p12(
        run_id="current-p12-critical-path-serialized",
        paired_instance_id="paired-critical-path",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="SERIALIZED",
        runtime_profile=synthetic_runtime_profile(planning_cost_model=cost),
    )
    try:
        # Overlap changes the actual critical path.  It is not required to be
        # monotonic on every synthetic contention instance; the paper-facing
        # overhead claim is based on hidden/exposed line service.
        assert overlap.kernel.now_ns != serialized.kernel.now_ns
        overlap_lines = {item.line_name: item for item in overlap.observer_bridge.metrics().line_metrics}
        serialized_lines = {item.line_name: item for item in serialized.observer_bridge.metrics().line_metrics}
        assert sum(item.hidden_service_ns for item in overlap_lines.values()) > 0
        assert sum(item.hidden_service_ns for item in serialized_lines.values()) == 0
    finally:
        overlap.dispose()
        serialized.dispose()


def test_paired_instance_identity_is_independent_of_treatment_run_id() -> None:
    perfect = build_current_p12_integration_runtime(
        fixture_input=build_golden_fixture(),
        run_id="treatment-perfect",
        paired_instance_id="paired-fixture-0",
        staging_sensitivity="0.25X",
        algorithm="joint(global_(rscf()))",
        information_mode="PERFECT_P2",
        overlap_mode="OVERLAP",
        runtime_profile=synthetic_runtime_profile(local_assembly_latency_ns=5),
    )
    zero = build_current_p12_integration_runtime(
        fixture_input=build_golden_fixture(),
        run_id="treatment-zero",
        paired_instance_id="paired-fixture-0",
        staging_sensitivity="0.25X",
        algorithm="joint(global_(rscf()))",
        information_mode="ZERO_P2",
        overlap_mode="OVERLAP",
        runtime_profile=synthetic_runtime_profile(local_assembly_latency_ns=5),
    )
    perfect.run_to_completion(max_timestamps=20_000)
    zero.run_to_completion(max_timestamps=20_000)
    perfect.assert_terminal()
    zero.assert_terminal()
    try:
        p = perfect.current_p12_window_records()
        z = zero.current_p12_window_records()
        assert tuple(item.task_ids for item in p) == tuple(item.task_ids for item in z)
        assert tuple(item.task_catalogue_digest for item in p) == tuple(
            item.task_catalogue_digest for item in z
        )
        assert tuple(item.task_boundary_digest for item in p) == tuple(
            item.task_boundary_digest for item in z
        )
    finally:
        perfect.dispose()
        zero.dispose()


def test_phase_local_prediction_metrics_are_not_applicable() -> None:
    runtime = run_current_p12(
        run_id="current-p12-local-na",
        algorithm="local(global_(rscf()))",
        information_mode="PERFECT_P2",
        overlap_mode="OVERLAP",
    )
    try:
        records = runtime.current_p12_window_records()
        assert records
        assert all(item.prediction_digest == "NO_P2_PREDICTION" for item in records)
        assert all(item.prediction_quality_digest is None for item in records)
        assert all(item.prediction_absolute_error_bytes is None for item in records)
        assert all(item.prediction_relative_absolute_error_ppm is None for item in records)
    finally:
        runtime.dispose()
