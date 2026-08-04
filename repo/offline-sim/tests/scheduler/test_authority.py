from __future__ import annotations

from tests.scheduler.conftest import Snapshot
from tests.scheduler.helpers import expectation, make_ready, phase, window


def _setup(stack):
    phase_key = phase()
    stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 128), registered_at_ns=10
    )
    ids = stack.catalogue.task_ids_for_phase(phase_key)
    make_ready(stack, ids)
    return phase_key, ids


def test_stale_authority_is_retryable_and_has_no_side_effect(stack):
    phase_key, ids = _setup(stack)
    plan1 = stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(0), ordered_task_ids=ids, now_ns=40
    )
    attempt = stack.compiler.compile_next(
        phase_key=phase_key, snapshot=Snapshot(max_batch_tasks=1), now_ns=41
    )
    assert attempt.code == "BATCH_READY"
    state_before = stack.runtime.snapshot_digest()
    plan2 = stack.controller.activate_plan(
        phase_key=phase_key,
        window_key=window(0),
        ordered_task_ids=tuple(reversed(ids)),
        now_ns=42,
    )
    result = stack.validator.validate(
        attempt.batch, snapshot=Snapshot(max_batch_tasks=1), now_ns=43
    )
    assert result.code == "RETRYABLE_STALE_AUTHORITY"
    assert stack.runtime.snapshot_digest() == state_before
    assert stack.adapter.plan_view(plan1).status == "ACTIVE"  # immutable caller snapshot
    assert stack.adapter.plan_view(stack.authority.plan(stack.adapter.plan_view(plan1).plan_id)).status == "SUPERSEDED"
    assert stack.adapter.plan_view(plan2).status == "ACTIVE"


def test_supersession_preserves_committed_running_completed_tasks(stack):
    phase_key, ids = _setup(stack)
    first = stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(0), ordered_task_ids=ids, now_ns=40
    )
    first_id = stack.adapter.plan_view(first).plan_id
    stack.authority.commit_batch(phase_key, (ids[0],), at_ns=41)
    stack.authority.mark_running(phase_key, ids[0], at_ns=42)
    stack.authority.mark_completed(phase_key, ids[0], at_ns=43)
    second = stack.controller.activate_plan(
        phase_key=phase_key,
        window_key=window(0),
        ordered_task_ids=(ids[1],),
        now_ns=44,
    )
    second_view = stack.adapter.plan_view(second)
    assert ids[0] in second_view.committed_task_ids
    assert ids[0] not in second_view.remaining_task_ids
    record = stack.authority.record_view(phase_key)
    assert ids[0] in record.completed_task_ids
    assert stack.adapter.plan_view(stack.authority.plan(first_id)).status == "SUPERSEDED"


def test_new_task_can_extend_same_phase_record_without_duplicate_record(stack):
    phase_key = phase()
    stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 64), registered_at_ns=10
    )
    record1 = stack.authority.ensure_phase(phase_key)
    stack.controller.register_expectation(
        expectation(phase_key, 2, 3, 64), registered_at_ns=11
    )
    record2 = stack.authority.ensure_phase(phase_key)
    assert record1.phase_key == record2.phase_key
    assert len(record2.canonical_task_ids) == 2
    assert stack.authority.record_view(phase_key).phase_plan_epoch == 0


def test_rejected_active_plan_invalidates_authority(stack):
    phase_key, ids = _setup(stack)
    plan = stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(0), ordered_task_ids=ids, now_ns=40
    )
    token = stack.authority.authority_token(phase_key)
    rejected = stack.authority.reject_plan(
        stack.adapter.plan_view(plan).plan_id, rejected_at_ns=41
    )
    assert stack.adapter.plan_view(rejected).status == "REJECTED"
    assert stack.authority.active_plan(phase_key) is None
    assert not stack.authority.token_is_current(token)


def test_plan_completion_auto_closes_and_later_event_can_extend_phase(stack):
    phase_key = phase()
    stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 64), registered_at_ns=10
    )
    first_id = stack.catalogue.task_ids_for_phase(phase_key)[0]
    make_ready(stack, (first_id,))
    first = stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(0), ordered_task_ids=(first_id,), now_ns=20
    )
    stack.authority.commit_batch(phase_key, (first_id,), at_ns=21)
    stack.authority.mark_running(phase_key, first_id, at_ns=22)
    stack.authority.mark_completed(phase_key, first_id, at_ns=23)
    assert stack.adapter.plan_view(stack.authority.plan(stack.adapter.plan_view(first).plan_id)).status == "COMPLETED"

    # A later EVENT observation may legally extend the same PhaseKey with a new plan.
    stack.controller.register_expectation(
        expectation(phase_key, 2, 3, 64), registered_at_ns=24
    )
    second_id = stack.catalogue.task_ids_for_phase(phase_key)[1]
    make_ready(stack, (second_id,), permit_at=25, payload_at=26)
    second = stack.controller.activate_plan(
        phase_key=phase_key,
        window_key=window(0),
        ordered_task_ids=(second_id,),
        now_ns=27,
    )
    assert stack.adapter.plan_view(second).status == "ACTIVE"
    stack.authority.commit_batch(phase_key, (second_id,), at_ns=28)
    stack.authority.mark_running(phase_key, second_id, at_ns=29)
    stack.authority.mark_completed(phase_key, second_id, at_ns=30)
    completed = stack.authority.plan(stack.adapter.plan_view(second).plan_id)
    assert stack.adapter.plan_view(completed).status == "COMPLETED"
    assert stack.authority.active_plan(phase_key) is None


def test_explicit_replan_excludes_frozen_tasks_automatically(stack):
    phase_key, ids = _setup(stack)
    stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(0), ordered_task_ids=ids, now_ns=40
    )
    stack.authority.commit_batch(phase_key, (ids[0],), at_ns=41)
    replanned = stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(0), ordered_task_ids=(ids[1],), now_ns=42
    )
    view = stack.adapter.plan_view(replanned)
    assert ids[0] in view.committed_task_ids
    assert view.remaining_task_ids == (ids[1],)


def test_plan_committed_history_survives_later_commits_in_same_plan(stack):
    phase_key, ids = _setup(stack)
    plan = stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(0), ordered_task_ids=ids, now_ns=40
    )
    stack.authority.commit_batch(phase_key, (ids[0],), at_ns=41)
    stack.authority.mark_running(phase_key, ids[0], at_ns=42)
    stack.authority.mark_completed(phase_key, ids[0], at_ns=43)
    stack.authority.commit_batch(phase_key, (ids[1],), at_ns=44)
    plan_view = stack.adapter.plan_view(stack.authority.plan(stack.adapter.plan_view(plan).plan_id))
    assert plan_view.committed_task_ids == ids
    assert plan_view.commit_index == 2
