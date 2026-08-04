from __future__ import annotations

from dataclasses import dataclass

import pytest

from rs_sim import (
    LinkClass,
    PhaseKey,
    PhaseKind,
    RowBroadcastRequest,
    SimulationKernel,
    SubmitOutcome,
    make_authority_stamp,
    make_transfer_batch,
    stable_digest,
)
from rs_sim.transport import (
    FormalControlPlaneTransport,
    FormalTransportRuntimeDriver,
    ReceiptStateError,
    build_formal_transport_runtime_driver,
    make_synthetic_profile_sensitivity_set,
)

from .conftest import (
    AuthorityValidation,
    CompletionLog,
    ControlLog,
    PermitLookup,
    ReleaseLog,
    SharedResolver,
    TaskLookup,
    build_harness,
    make_task,
    topology_mixed_nodes,
)


@dataclass
class MultiAuthority:
    stamps: dict[PhaseKey, object]

    def authority_is_current(self, *, phase_key, authority_stamp) -> bool:
        return self.stamps.get(phase_key) == authority_stamp


def _request(phase: PhaseKey, *, src_rank: int, published_at_ns: int) -> RowBroadcastRequest:
    rows = (0, 1, 2, 3)
    payload = tuple(value * 8 for value in rows)
    return RowBroadcastRequest(
        phase_key=phase,
        src_rank=src_rank,
        realized_rows_by_destination=rows,
        payload_bytes_by_destination=payload,
        payload_spec_digest=stable_digest((phase, "payload"), domain="TRANSPORT_TEST_CP"),
        descriptor_digest=stable_digest((phase, src_rank), domain="TRANSPORT_TEST_DESCRIPTOR"),
        published_at_ns=published_at_ns,
        descriptor_payload_bytes=64,
    )


def _scope_counter_map(row):
    return dict(row[3])


def test_public_runtime_driver_aborts_failed_apply_and_confirms_success():
    source = build_harness(task_specs=(("driver-a", 0, 2, 8), ("driver-b", 1, 3, 8)))
    control = ControlLog()
    kernel = SimulationKernel()
    driver = build_formal_transport_runtime_driver(
        kernel=kernel,
        task_lookup=TaskLookup(source.tasks),
        permit_lookup=PermitLookup(source.permits),
        authority_validation=AuthorityValidation(source.phase, source.stamp),
        resource_resolver=source.resolver,
        completion_sink=source.completion,
        resource_release_sink=source.release,
        control_delivery_sink=control,
        hardware_profile=source.transport.hardware_profile,
    )
    assert isinstance(driver, FormalTransportRuntimeDriver)

    def fail_apply(_receipt):
        raise RuntimeError("logical apply failed")

    with pytest.raises(RuntimeError, match="logical apply failed"):
        driver.submit_atomic_batch(
            batch=source.batch("driver-a"),
            commit_time_ns=kernel.now_ns,
            apply_receipt=fail_apply,
        )
    assert driver.data_plane.prepared_count == 0
    assert driver.data_plane.snapshot().busy_src_ranks == ()
    assert driver.data_plane.evidence()["retained_rejected_batches"] == ()

    applied = []
    result = driver.submit_atomic_batch(
        batch=source.batch("driver-b"),
        commit_time_ns=kernel.now_ns,
        apply_receipt=applied.append,
    )
    assert result.outcome is SubmitOutcome.PREPARED
    assert result.applied is True and result.confirmed is True
    assert len(applied) == 1
    while kernel.has_events():
        kernel.run_next_timestamp()
    driver.assert_terminal()
    assert driver.manifest_fragment()["transport_driver_internal_policy_queue"] is False


def test_duplicate_start_and_completion_events_fail_closed():
    harness = build_harness(task_specs=(("dup", 0, 2, 8),))
    outcome, receipt = harness.transport.prepare_commit(harness.batch("dup"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)

    start_event = harness.kernel._queue[0][1]
    harness.kernel.run_next_timestamp()
    with pytest.raises(ReceiptStateError, match="duplicate or unknown transfer start"):
        harness.transport._handle_start(harness.kernel, start_event)

    completion_event = harness.kernel._queue[0][1]
    harness.kernel.run_next_timestamp()
    with pytest.raises(ReceiptStateError, match="duplicate or unknown transfer completion"):
        harness.transport._handle_complete(harness.kernel, completion_event)
    harness.transport.assert_terminal()


def test_synthetic_profile_set_covers_local_intra_inter_and_control_provenance():
    fast = make_synthetic_profile_sensitivity_set(
        profile_set_id="fast",
        profile_provenance="SYNTHETIC_SENSITIVITY_A",
        max_batch_tasks=32,
        local_assembly_latency_ns=7,
        intra_fixed_latency_ns=11,
        inter_fixed_latency_ns=13,
        control_fixed_latency_ns=17,
    )
    slow = make_synthetic_profile_sensitivity_set(
        profile_set_id="slow",
        profile_provenance="SYNTHETIC_SENSITIVITY_A",
        max_batch_tasks=32,
        local_assembly_latency_ns=70,
        intra_fixed_latency_ns=110,
        inter_fixed_latency_ns=130,
        control_fixed_latency_ns=170,
    )
    assert fast.profile_set_digest != slow.profile_set_digest
    manifest = fast.manifest_fragment()
    assert manifest["local_assembly_latency_ns"] == 7
    assert manifest["local_assembly_enters_data_plane"] is False
    assert manifest["synthetic_profile_provenance"] == "SYNTHETIC_SENSITIVITY_A"
    assert manifest["synthetic_profile_performance_eligible"] is False
    assert fast.hardware_profile.performance_eligible is False
    assert fast.control_profile.performance_eligible is False


def test_multiphase_multiwindow_32_chunk_stress_fifo_scoped_stats_and_terminal():
    topology = topology_mixed_nodes()
    phases = (
        PhaseKey("run-r23", "sample", 0, PhaseKind.DISPATCH),
        PhaseKey("run-r23", "sample", 0, PhaseKind.COMBINE),
        PhaseKey("run-r23", "sample", 1, PhaseKind.DISPATCH),
        PhaseKey("run-r23", "sample", 1, PhaseKind.COMBINE),
        PhaseKey("run-r23", "sample", 2, PhaseKind.DISPATCH),
    )
    stamps = {
        phase: make_authority_stamp(
            phase_token=f"token:{index}",
            plan_id=f"plan:{index}",
            phase_plan_epoch=index,
        )
        for index, phase in enumerate(phases)
    }
    tasks = {}
    permits = {}
    batch_by_task = {}
    for index in range(32):
        phase = phases[index % len(phases)]
        dst = 1 if index % 2 == 0 else 2
        task_id = f"chunk-{index:02d}"
        task, permit = make_task(
            phase,
            task_id=task_id,
            src=0,
            dst=dst,
            payload_bytes=8 + index,
            chunk_index=index,
            byte_offset=index * 64,
        )
        tasks[task_id] = task
        permits[task_id] = permit
        link_class = (
            LinkClass.INTRA_NODE
            if topology.rank_to_node[task.src_rank] == topology.rank_to_node[task.dst_rank]
            else LinkClass.INTER_NODE
        )
        batch_by_task[task_id] = make_transfer_batch(
            batch_id=f"batch:{task_id}",
            phase_key=phase,
            task_ids=(task_id,),
            authority_stamp=stamps[phase],
            link_class=link_class,
            topology_digest=topology.topology_digest,
            compiled_at_ns=0,
        )

    kernel = SimulationKernel()
    completion = CompletionLog()
    release = ReleaseLog()
    control_log = ControlLog()
    profile_set = make_synthetic_profile_sensitivity_set(
        profile_set_id="r23-stress",
        max_batch_tasks=4,
        intra_launch_delay_ns=2,
        intra_fixed_latency_ns=5,
        intra_bandwidth_bytes_per_second=2_000_000_000,
        inter_launch_delay_ns=3,
        inter_fixed_latency_ns=7,
        inter_bandwidth_bytes_per_second=1_000_000_000,
        control_fixed_latency_ns=1,
        control_bandwidth_bytes_per_second=64_000_000_000,
    )
    driver = build_formal_transport_runtime_driver(
        kernel=kernel,
        task_lookup=TaskLookup(tasks),
        permit_lookup=PermitLookup(permits),
        authority_validation=MultiAuthority(stamps),
        resource_resolver=SharedResolver(topology),
        completion_sink=completion,
        resource_release_sink=release,
        control_delivery_sink=control_log,
        hardware_profile=profile_set.hardware_profile,
        control_profile=profile_set.control_profile,
    )

    dispatches = [phase for phase in phases if phase.phase_kind is PhaseKind.DISPATCH]
    request_digests = {
        dispatches[2]: driver.publish_dispatch_row(
            _request(dispatches[2], src_rank=2, published_at_ns=6)
        ),
        dispatches[0]: driver.publish_dispatch_row(
            _request(dispatches[0], src_rank=0, published_at_ns=0)
        ),
        dispatches[1]: driver.publish_dispatch_row(
            _request(dispatches[1], src_rank=1, published_at_ns=3)
        ),
    }

    task_ids = tuple(sorted(tasks))
    applied_receipts = []
    retry_count = 0
    for index, task_id in enumerate(task_ids):
        result = driver.submit_atomic_batch(
            batch=batch_by_task[task_id],
            commit_time_ns=kernel.now_ns,
            apply_receipt=applied_receipts.append,
        )
        assert result.outcome is SubmitOutcome.PREPARED
        if index + 1 < len(task_ids):
            next_batch = batch_by_task[task_ids[index + 1]]
            outcome, retained = driver.data_plane.prepare_commit(next_batch, kernel.now_ns)
            assert outcome is SubmitOutcome.RETRYABLE_RESOURCE_BUSY
            assert retained is None
            retry_count += 1
            assert driver.data_plane.evidence()["retained_rejected_batches"] == ()
            assert driver.data_plane.terminal_state()["internal_wait_queue_depth"] == 0
        while kernel.has_events():
            kernel.run_next_timestamp()

    assert retry_count == 31
    assert len(applied_receipts) == 32
    assert len(completion.started) == 32
    assert len(completion.completed) == 32
    assert driver.data_plane.completed_task_ids == task_ids

    fifo_rows = driver.control_plane.fifo_evidence()
    assert tuple(row[1] for row in fifo_rows) == (
        request_digests[dispatches[0]],
        request_digests[dispatches[1]],
        request_digests[dispatches[2]],
    )
    assert tuple(row[4] for row in fifo_rows) == (0, 3, 6)

    data_stats = driver.data_plane.statistics()
    phase_rows = data_stats["phase_mechanism_statistics"]
    window_rows = data_stats["window_mechanism_statistics"]
    assert len(phase_rows) == 5
    assert len(window_rows) == 3
    assert sum(_scope_counter_map(row).get("completed_task_count", 0) for row in phase_rows) == 32
    assert dict(data_stats["resource_wait_retry_counts"])["RANK_ENDPOINT"] == 31
    assert data_stats["physical_completed_bytes"] == sum(task.payload_bytes for task in tasks.values())

    control_stats = driver.control_plane.statistics()
    assert len(control_stats["phase_mechanism_statistics"]) == 3
    assert len(control_stats["window_mechanism_statistics"]) == 3
    assert control_stats["published_request_count"] == 3
    assert control_stats["delivered_request_count"] == 3
    driver.assert_terminal()
    assert kernel.pending_event_count() == 0


def test_same_timestamp_batch_start_and_completion_order_is_exact():
    harness = build_harness(
        task_specs=(("same-a", 0, 2, 8), ("same-b", 1, 3, 8))
    )
    outcome, receipt = harness.transport.prepare_commit(
        harness.batch("same-a", "same-b"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    assert harness.kernel.pending_event_count() == 0
    harness.transport.confirm_commit(receipt)
    assert harness.kernel.pending_event_count() == 2
    while harness.kernel.has_events():
        harness.kernel.run_next_timestamp()

    records = harness.transport.physical_records()
    assert len({record.start_at_ns for record in records}) == 1
    assert len({record.complete_at_ns for record in records}) == 1
    timeline = harness.kernel.timeline()
    starts = [row for row in timeline if row.event_type == harness.transport.START_EVENT]
    completes = [row for row in timeline if row.event_type == harness.transport.COMPLETE_EVENT]
    assert len(starts) == len(completes) == 2
    assert len({(row.time_ns, row.round_index) for row in starts}) == 1
    assert len({(row.time_ns, row.round_index) for row in completes}) == 1
    assert max(row.timeline_index for row in starts) < min(
        row.timeline_index for row in completes
    )


def test_record_event_and_statistics_digests_repeat_ten_times():
    data_summaries = []
    control_summaries = []
    for _ in range(10):
        harness = build_harness(
            task_specs=(("det-a", 0, 2, 8), ("det-b", 1, 3, 8))
        )
        outcome, receipt = harness.transport.prepare_commit(
            harness.batch("det-a", "det-b"), harness.transport.kernel.now_ns)
        assert outcome is SubmitOutcome.PREPARED and receipt is not None
        harness.transport.confirm_commit(receipt)
        while harness.kernel.has_events():
            harness.kernel.run_next_timestamp()
        data_summaries.append(
            (
                harness.transport.physical_record_digest(),
                harness.kernel.event_digest(),
                harness.transport.statistics()["mechanism_statistics_digest"],
            )
        )

        kernel = SimulationKernel()
        sink = ControlLog()
        profile = make_synthetic_profile_sensitivity_set(
            profile_set_id="det-control", max_batch_tasks=2
        ).control_profile
        control = FormalControlPlaneTransport(
            kernel=kernel, profile=profile, delivery_sink=sink
        )
        phases = (
            PhaseKey("det", "cp", 0, PhaseKind.DISPATCH),
            PhaseKey("det", "cp", 1, PhaseKind.DISPATCH),
        )
        control.publish_row(_request(phases[1], src_rank=1, published_at_ns=2))
        control.publish_row(_request(phases[0], src_rank=0, published_at_ns=0))
        while kernel.has_events():
            kernel.run_next_timestamp()
        control_summaries.append(
            (
                control.delivery_digest(),
                kernel.event_digest(),
                control.statistics()["mechanism_statistics_digest"],
                control.evidence()["fifo_evidence_digest"],
            )
        )
    assert len(set(data_summaries)) == 1
    assert len(set(control_summaries)) == 1
