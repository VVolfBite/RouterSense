from __future__ import annotations

from dataclasses import replace

import pytest

from rs_sim import LinkClass, SubmitOutcome, stable_digest
from rs_sim.transport import ReceiptStateError, RejectionCode

from .conftest import build_harness


def test_prepare_holds_resources_but_registers_no_executable_event(harness):
    before_events = harness.kernel.pending_event_count()
    outcome, receipt = harness.transport.prepare_commit(harness.batch("t0"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED
    assert receipt is not None
    assert harness.transport.prepared_count == 1
    assert harness.kernel.pending_event_count() == before_events == 0
    snapshot = harness.transport.snapshot()
    assert snapshot.busy_src_ranks == (0,)
    assert snapshot.busy_dst_ranks == (2,)
    assert snapshot.busy_lane_ids == ("inter-0",)
    assert harness.transport.physical_records() == ()


def test_scheduler_apply_failure_abort_fully_rolls_back_and_is_idempotent(harness):
    baseline = harness.transport.snapshot()
    outcome, receipt = harness.transport.prepare_commit(harness.batch("t0"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.abort_commit(receipt)
    harness.transport.abort_commit(receipt)
    assert harness.transport.prepared_count == 0
    assert harness.kernel.pending_event_count() == 0
    assert harness.transport.snapshot() == baseline
    assert harness.transport.physical_records() == ()


def test_confirm_is_the_only_physical_commit_boundary_and_live_confirm_completes(harness):
    batch = harness.batch("t0")
    outcome, receipt = harness.transport.prepare_commit(batch, harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    assert receipt.authority_stamp is batch.authority_stamp
    assert receipt.authority_stamp == batch.authority_stamp
    harness.transport.confirm_commit(receipt)
    assert harness.transport.prepared_count == 0
    assert harness.kernel.pending_event_count() == 1
    records = harness.transport.physical_records()
    assert len(records) == 1
    assert records[0].committed_at_ns == 0
    while harness.kernel.has_events():
        harness.kernel.run_next_timestamp()
    assert [event.task_id for event in harness.completion.started] == ["t0"]
    assert [event.task_id for event in harness.completion.completed] == ["t0"]
    assert harness.release.phases == [harness.phase]
    assert harness.transport.snapshot().busy_lane_ids == ()


def test_confirmed_receipt_cannot_be_aborted(harness):
    outcome, receipt = harness.transport.prepare_commit(harness.batch("t0"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)
    with pytest.raises(ReceiptStateError):
        harness.transport.abort_commit(receipt)


def test_changed_receipt_is_not_live(harness):
    outcome, receipt = harness.transport.prepare_commit(harness.batch("t0"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    changed = replace(receipt, resource_reservation_digest="changed")
    with pytest.raises(ReceiptStateError):
        harness.transport.confirm_commit(changed)
    assert harness.transport.prepared_count == 1
    assert harness.kernel.pending_event_count() == 0


def test_independent_completion_releases_only_completed_task_resources(harness):
    outcome, receipt = harness.transport.prepare_commit(harness.batch("t0", "t1"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)
    harness.kernel.run_next_timestamp()  # common start
    assert len(harness.completion.started) == 2
    assert len(harness.completion.completed) == 0
    harness.kernel.run_next_timestamp()  # t0 completes first
    assert [event.task_id for event in harness.completion.completed] == ["t0"]
    snapshot = harness.transport.snapshot()
    assert snapshot.busy_src_ranks == (1,)
    assert snapshot.busy_dst_ranks == (3,)
    assert snapshot.busy_lane_ids == ("inter-1",)
    harness.kernel.run_next_timestamp()
    assert [event.task_id for event in harness.completion.completed] == ["t0", "t1"]
    assert harness.transport.snapshot().busy_src_ranks == ()


def test_physical_timing_is_one_batch_launch_and_fixed_per_lane_bandwidth(harness):
    outcome, receipt = harness.transport.prepare_commit(harness.batch("t0", "t1"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)
    by_task = {record.task_id: record for record in harness.transport.physical_records()}
    assert by_task["t0"].start_at_ns == by_task["t1"].start_at_ns == 3
    assert by_task["t0"].complete_at_ns == 18  # 7 fixed + ceil(8 B / 1 GB/s)=8
    assert by_task["t1"].complete_at_ns == 26  # 7 fixed + 16


def test_physical_and_event_digests_are_repeatable():
    outcomes = []
    for _ in range(100):
        h = build_harness()
        result, receipt = h.transport.prepare_commit(h.batch("t0", "t1"), h.transport.kernel.now_ns)
        assert result is SubmitOutcome.PREPARED and receipt is not None
        h.transport.confirm_commit(receipt)
        while h.kernel.has_events():
            h.kernel.run_next_timestamp()
        outcomes.append(
            (
                h.transport.physical_record_digest(),
                h.kernel.event_digest(),
                h.kernel.timeline_digest(),
            )
        )
    assert len(set(outcomes)) == 1
