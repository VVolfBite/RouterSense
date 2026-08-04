from __future__ import annotations

from _current_p12_test_helpers import (
    phase_first_start_and_last_complete,
    phase_start_order,
    run_current_p12,
)


def test_joint_runtime_exposes_only_composed_algorithm_identity() -> None:
    runtime = run_current_p12(
        run_id="current-p12-composed-rscf",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="OVERLAP",
    )
    try:
        assert runtime.run_axes["algorithm"] == "joint(global_(rscf()))"
        assert runtime.run_axes["algorithm_core"] == "rscf"
        assert runtime.run_axes["planning_mode"] == "GLOBAL"
        assert runtime.run_axes["planner_scope"] == "WINDOW_JOINT"
        assert runtime.run_axes["safe_scope_selection"] is False
        assert "formal_policy_id" not in runtime.run_axes
        assert "policy_name" not in runtime.run_axes
        assert "algorithm_identity" not in runtime.run_axes

        window_adapters = [
            item
            for item in runtime.observer_bridge.adapters
            if item.current_p12_window is not None
        ]
        boundary_phase_adapters = [
            item
            for item in runtime.observer_bridge.adapters
            if item.current_p12_window is None
        ]
        assert window_adapters
        assert boundary_phase_adapters
        assert all(item.session.spec.core_id == "rscf" for item in window_adapters)
        assert all(item.session.spec.scope.value == "WINDOW_JOINT" for item in window_adapters)
        # Bootstrap/terminal phases use the same registered core; a single-phase
        # problem is naturally wrapped by Local rather than switching kernels.
        assert all(item.session.spec.core_id == "rscf" for item in boundary_phase_adapters)
        assert all(item.session.spec.scope.value == "PHASE_LOCAL" for item in boundary_phase_adapters)
    finally:
        runtime.dispose()


def test_fate_is_consumed_without_copy_fallback_or_hidden_kernel_switch() -> None:
    runtime = run_current_p12(
        run_id="current-p12-fate-only",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        overlap_mode="OVERLAP",
    )
    try:
        records = runtime.current_p12_window_records()
        assert records
        assert all(item.information_mode == "FATE_P2" for item in records)
        assert all(item.prediction_generated for item in records)
        assert all(item.prediction_nonempty for item in records)
        assert all(item.prediction_validated for item in records)
        assert all(item.prediction_consumed for item in records)
        assert all(not item.prediction_fallback for item in records)
        assert all(item.prediction_fallback_reason is None for item in records)
        assert all(item.algorithm_core_run_count == 1 for item in records)
        assert all(item.safe_selector_choice is None for item in records)
    finally:
        runtime.dispose()


def test_fate_and_perfect_match_for_exact_test_artifact_while_zero_differs() -> None:
    runtimes = {}
    try:
        for information_mode in ("FATE_P2", "PERFECT_P2", "ZERO_P2"):
            runtimes[information_mode] = run_current_p12(
                run_id=f"current-p12-information-{information_mode.lower()}",
                paired_instance_id="paired-information-sensitive",
                algorithm="joint(global_(rscf()))",
                information_mode=information_mode,
                overlap_mode="OVERLAP",
            )
        p1_orders = {
            mode: tuple(
                phase_start_order(runtime, window.p1_combine_phase_key)
                for window in runtime.current_p12_windows
            )
            for mode, runtime in runtimes.items()
        }
        p2_orders = {
            mode: tuple(
                phase_start_order(runtime, window.p2_dispatch_phase_key)
                for window in runtime.current_p12_windows
            )
            for mode, runtime in runtimes.items()
        }
        # fixture_with_fate() embeds the exact next-layer routing as a FATE
        # artifact, so FATE and Perfect must generate the same physical plan.
        assert p1_orders["FATE_P2"] == p1_orders["PERFECT_P2"]
        assert p2_orders["FATE_P2"] == p2_orders["PERFECT_P2"]
        # Zero is an explicit ablation and must not be silently treated as FATE.
        assert p2_orders["ZERO_P2"] != p2_orders["FATE_P2"]
        zero_records = runtimes["ZERO_P2"].current_p12_window_records()
        assert all(item.reconciliation_status == "ZERO_HINT_BIND" for item in zero_records)
    finally:
        for runtime in runtimes.values():
            runtime.dispose()


def test_global_runs_one_core_and_p1_barrier_precedes_p2_execution() -> None:
    runtime = run_current_p12(
        run_id="current-p12-global-barrier",
        algorithm="safe(joint(global_(rscf())))",
        information_mode="FATE_P2",
        overlap_mode="OVERLAP",
        max_task_bytes=1 << 20,
    )
    try:
        records = runtime.current_p12_window_records()
        assert records
        assert all(item.algorithm_core_run_count == 1 for item in records)
        assert all(item.incremental_bind_job_count > 0 for item in records)
        assert all(item.template_digest for item in records)
        assert all(
            item.safe_selector_choice in {"PHASE_LOCAL", "WINDOW_JOINT"}
            for item in records
        )
        assert all(item.safe_selector_reason == "SAME_CORE_EXPECTED_COMPLETION" for item in records)

        for window in runtime.current_p12_windows:
            _p1_first, p1_last = phase_first_start_and_last_complete(
                runtime, window.p1_combine_phase_key
            )
            p2_first, _p2_last = phase_first_start_and_last_complete(
                runtime, window.p2_dispatch_phase_key
            )
            # PHASE_BARRIER is the formal default: predicted P2 influences the
            # P1 order, but physical P2 execution cannot overtake unfinished P1.
            assert p2_first >= p1_last
    finally:
        runtime.dispose()
