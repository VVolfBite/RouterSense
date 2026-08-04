from __future__ import annotations

import pytest

from rs_sim import PhaseKey, PhaseKind, SimulationKernel, SubmitOutcome
from rs_sim.transport import (
    ReceiptStateError,
    UnsupportedFormalExecutionModeError,
    build_formal_transport_runtime_driver,
    capture_process_resource_snapshot,
)

from .conftest import (
    AuthorityValidation,
    CompletionLog,
    ControlLog,
    PermitLookup,
    ReleaseLog,
    TaskLookup,
    build_harness,
)


def _run(kernel: SimulationKernel) -> None:
    while kernel.has_events():
        kernel.run_next_timestamp()


def _driver_from_source(source):
    kernel = SimulationKernel()
    return build_formal_transport_runtime_driver(
        kernel=kernel,
        task_lookup=TaskLookup(source.tasks),
        permit_lookup=PermitLookup(source.permits),
        authority_validation=AuthorityValidation(source.phase, source.stamp),
        resource_resolver=source.resolver,
        completion_sink=CompletionLog(),
        resource_release_sink=ReleaseLog(),
        control_delivery_sink=ControlLog(),
        hardware_profile=source.transport.hardware_profile,
    )


def test_filtered_physical_metrics_exactly_reconcile_task_window_and_phase():
    harness = build_harness(task_specs=(("metric-a", 0, 2, 8), ("metric-b", 1, 3, 16)))
    outcome, receipt = harness.transport.prepare_commit(
        harness.batch("metric-a", "metric-b"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None

    prepared = harness.transport.physical_metrics(window_task_ids=("metric-a",))
    assert prepared.selected_task_ids == ("metric-a",)
    assert prepared.outstanding_prepared_receipt_ids == (receipt.receipt_id,)
    assert prepared.outstanding_confirmed_receipt_ids == ()
    assert prepared.task_metrics == ()

    harness.transport.confirm_commit(receipt)
    confirmed = harness.transport.physical_metrics(task_ids=("metric-a",))
    assert confirmed.outstanding_prepared_receipt_ids == ()
    assert confirmed.outstanding_confirmed_receipt_ids == (receipt.receipt_id,)
    assert confirmed.launch_count == 1
    assert confirmed.launch_delay_total_ns == 3
    assert confirmed.physical_completed_bytes == 0

    _run(harness.kernel)
    by_window = harness.transport.physical_metrics(window_task_ids=("metric-a",))
    by_task = harness.transport.physical_metrics(task_ids=("metric-a",))
    by_phase = harness.transport.physical_metrics(phase_keys=(harness.phase,))
    assert by_window.selected_task_ids == by_task.selected_task_ids
    assert by_window.task_metrics == by_task.task_metrics
    assert by_window.launch_metrics == by_task.launch_metrics
    assert by_window.busy_intervals == by_task.busy_intervals
    assert by_window.physical_completed_bytes == 8
    assert by_window.launch_count == 1
    assert by_window.launch_metrics[0].physical_batch_task_ids == (
        "metric-a",
        "metric-b",
    )
    assert by_window.launch_metrics[0].selected_task_ids == ("metric-a",)
    assert {row.resource_kind for row in by_window.busy_intervals} == {
        "LINK_CLASS",
        "LANE",
        "NIC",
    }
    assert by_window.outstanding_confirmed_receipt_ids == ()
    assert by_window.all_resources_free is True
    assert by_window.terminal is True
    assert by_phase.physical_completed_bytes == 24
    assert by_phase.selected_task_ids == ("metric-a", "metric-b")
    assert by_window.metrics_digest == (
        harness.transport.physical_metrics(window_task_ids=("metric-a",)).metrics_digest
    )

    unrelated = PhaseKey("run", "unrelated", 9, PhaseKind.DISPATCH)
    empty = harness.transport.physical_metrics(phase_keys=(unrelated,))
    assert empty.selected_task_ids == ()
    assert empty.physical_completed_bytes == 0
    with pytest.raises(KeyError, match="unknown transport physical metric task IDs"):
        harness.transport.physical_metrics(window_task_ids=("not-a-task",))


def test_live_abort_is_idempotent_and_post_confirm_abort_fails_closed():
    aborted = build_harness(task_specs=(("abort-live", 0, 2, 8),))
    outcome, receipt = aborted.transport.prepare_commit(
        aborted.batch("abort-live"), 0
    )
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    aborted.transport.abort_commit(receipt)
    aborted.transport.abort_commit(receipt)
    assert aborted.transport.prepared_count == 0
    assert aborted.transport.snapshot().busy_src_ranks == ()
    assert aborted.transport.snapshot().busy_lane_ids == ()

    confirmed = build_harness(task_specs=(("abort-confirmed", 0, 2, 8),))
    outcome, receipt = confirmed.transport.prepare_commit(
        confirmed.batch("abort-confirmed"), 0
    )
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    confirmed.transport.confirm_commit(receipt)
    with pytest.raises(ReceiptStateError, match="confirmed receipt cannot be aborted"):
        confirmed.transport.abort_commit(receipt)
    _run(confirmed.kernel)
    confirmed.transport.assert_terminal()


def test_repeated_driver_close_dispose_is_idempotent_and_resets_baseline():
    final_digests = []
    for _ in range(8):
        process_before = capture_process_resource_snapshot()
        source = build_harness(task_specs=(("cycle-task", 0, 2, 8),))
        driver = _driver_from_source(source)
        applied = []
        result = driver.submit_atomic_batch(
            batch=source.batch("cycle-task"),
            commit_time_ns=0,
            apply_receipt=applied.append,
        )
        assert result.outcome is SubmitOutcome.PREPARED
        _run(driver.data_plane.kernel)
        driver.assert_terminal()

        first_close = driver.close()
        second_close = driver.close()
        assert first_close == second_close
        final_digests.append(
            first_close["data_plane"]["final_evidence_digest"]
        )

        first_dispose = driver.dispose()
        second_dispose = driver.dispose()
        assert first_dispose == second_dispose
        assert first_dispose["disposed"] is True
        assert first_dispose["kernel_pending_event_count"] == 0
        assert first_dispose["kernel_callback_registry_disposed"] is True
        for component in ("data_plane", "control_plane"):
            evidence = first_dispose[component]
            assert evidence["owned_child_process_count"] == 0
            assert evidence["owned_thread_count"] == 0
            assert evidence["owned_executor_count"] == 0
            assert evidence["owned_file_handle_count"] == 0
            assert evidence["live_receipt_count"] == 0
            assert evidence["live_transfer_or_request_count"] == 0
        assert driver.data_plane.statistics()["prepare_attempt_count"] == 0
        assert driver.data_plane.statistics()["physical_completed_record_count"] == 0
        assert driver.control_plane.statistics()["published_request_count"] == 0
        assert driver.data_plane.physical_records() == ()
        assert driver.data_plane.terminal_state()["terminal"] is True
        assert driver.control_plane.terminal_state()["terminal"] is True
        assert driver.data_plane.kernel._handlers == {}
        assert driver.data_plane.kernel._evidence_providers == {}
        process_after = capture_process_resource_snapshot()
        assert process_after["thread_count"] == process_before["thread_count"]
        assert process_after["child_process_count"] == process_before["child_process_count"]
        if process_before["open_file_descriptor_count"] is not None:
            assert process_after["open_file_descriptor_count"] <= (
                process_before["open_file_descriptor_count"]
            )

    assert len(set(final_digests)) == 1


def test_formal_full_joint_request_is_explicitly_blocked_by_transport_builder():
    source = build_harness(task_specs=(("order-only", 0, 2, 8),))
    with pytest.raises(
        UnsupportedFormalExecutionModeError,
        match="EXPERIMENTAL_BLOCKED_NOT_LIVE",
    ):
        build_formal_transport_runtime_driver(
            kernel=SimulationKernel(),
            task_lookup=TaskLookup(source.tasks),
            permit_lookup=PermitLookup(source.permits),
            authority_validation=AuthorityValidation(source.phase, source.stamp),
            resource_resolver=source.resolver,
            completion_sink=CompletionLog(),
            resource_release_sink=ReleaseLog(),
            control_delivery_sink=ControlLog(),
            hardware_profile=source.transport.hardware_profile,
            formal_execution_mode="FULL_JOINT",
        )

    order_only = _driver_from_source(source)
    manifest = order_only.manifest_fragment()
    assert manifest["transport_execution_mode"] == "ORDER_ONLY"
    assert manifest["formal_full_joint_status"] == "EXPERIMENTAL_BLOCKED_NOT_LIVE"
