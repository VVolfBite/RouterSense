from __future__ import annotations

from dataclasses import dataclass
from tests.scheduler.conftest import Snapshot
from rs_sim import SubmitOutcome, make_commit_receipt
from tests.scheduler.helpers import expectation, make_ready, phase, window


@dataclass
class FakeTransport:
    results: list[SubmitOutcome]
    prepared: dict[str, object] = None

    def __post_init__(self):
        self.prepared = {}

    def prepare_commit(self, batch, commit_time_ns: int):
        result = self.results.pop(0) if self.results else SubmitOutcome.PREPARED
        if result is not SubmitOutcome.PREPARED:
            return result, None
        receipt = make_commit_receipt(
            batch=batch,
            commit_time_ns=commit_time_ns,
            resource_reservation_digest="resource",
            transport_snapshot_digest="snapshot",
        )
        self.prepared[receipt.receipt_id] = receipt
        return SubmitOutcome.PREPARED, receipt

    def confirm_commit(self, receipt):
        assert self.prepared.pop(receipt.receipt_id) == receipt

    def abort_commit(self, receipt):
        assert self.prepared.pop(receipt.receipt_id) == receipt


def test_new_high_priority_ready_task_overtakes_uncommitted_but_not_committed(stack):
    phase_key = phase()
    first_tasks = stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 64), registered_at_ns=10
    )
    second_tasks = stack.controller.register_expectation(
        expectation(phase_key, 2, 3, 64), registered_at_ns=11
    )
    ids = stack.catalogue.task_ids_for_phase(phase_key)
    make_ready(stack, ids)
    stack.controller.activate_plan(
        phase_key=phase_key,
        window_key=window(0),
        ordered_task_ids=ids,
        now_ns=20,
    )
    old_attempt = stack.compiler.compile_next(
        phase_key=phase_key, snapshot=Snapshot(max_batch_tasks=1), now_ns=21
    )
    assert old_attempt.batch.task_ids == (stack.adapter.task_view(first_tasks[0]).task_id,)

    new_first = stack.adapter.task_view(second_tasks[0]).task_id
    old_first = stack.adapter.task_view(first_tasks[0]).task_id
    stack.controller.activate_plan(
        phase_key=phase_key,
        window_key=window(0),
        ordered_task_ids=(new_first, old_first),
        now_ns=22,
    )
    new_attempt = stack.compiler.compile_next(
        phase_key=phase_key, snapshot=Snapshot(max_batch_tasks=1), now_ns=23
    )
    assert new_attempt.batch.task_ids == (new_first,)
    stack.authority.commit_batch(phase_key, (new_first,), at_ns=24)

    stack.controller.activate_plan(
        phase_key=phase_key,
        window_key=window(0),
        ordered_task_ids=(old_first,),
        now_ns=25,
    )
    after_commit = stack.compiler.compile_next(
        phase_key=phase_key, snapshot=Snapshot(max_batch_tasks=1), now_ns=26
    )
    assert after_commit.batch.task_ids == (old_first,)
    assert new_first not in after_commit.batch.task_ids


def test_compiler_forms_endpoint_disjoint_batch_and_stops_on_busy_snapshot(stack):
    phase_key = phase()
    stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 64), registered_at_ns=10
    )
    stack.controller.register_expectation(
        expectation(phase_key, 2, 3, 64), registered_at_ns=11
    )
    stack.controller.register_expectation(
        expectation(phase_key, 0, 3, 64), registered_at_ns=12
    )
    ids = stack.catalogue.task_ids_for_phase(phase_key)
    make_ready(stack, ids)
    stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(), ordered_task_ids=stack.catalogue.task_ids_for_phase(phase_key), now_ns=20
    )
    attempt = stack.compiler.compile_next(
        phase_key=phase_key, snapshot=Snapshot(max_batch_tasks=4), now_ns=21
    )
    views = [stack.catalogue.view(task_id) for task_id in attempt.batch.task_ids]
    assert len(views) == 2
    assert len({view.src_rank for view in views}) == 2
    assert len({view.dst_rank for view in views}) == 2

    busy = stack.compiler.compile_next(
        phase_key=phase_key,
        snapshot=Snapshot(max_batch_tasks=4, busy_src_ranks=(0, 2), busy_dst_ranks=(1, 3)),
        now_ns=22,
    )
    assert busy.code == "RETRYABLE_RESOURCE_BUSY"


def test_execution_stabilizer_commits_only_after_transport_acceptance(stack):
    phase_key = phase()
    stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 64), registered_at_ns=10
    )
    task_id = stack.catalogue.task_ids_for_phase(phase_key)[0]
    make_ready(stack, (task_id,))
    stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(), ordered_task_ids=stack.catalogue.task_ids_for_phase(phase_key), now_ns=20
    )
    transport = FakeTransport(
        results=[SubmitOutcome.RETRYABLE_RESOURCE_BUSY, SubmitOutcome.PREPARED]
    )
    first = stack.controller.stabilizer.stabilize(
        phase_key=phase_key,
        snapshot_provider=lambda: Snapshot(max_batch_tasks=1),
        transport=transport,
        now_ns=21,
    )
    assert first.terminal_code == "RETRYABLE_RESOURCE_BUSY"
    assert stack.runtime.facts(task_id).state == "READY_UNCOMMITTED"

    second = stack.controller.stabilizer.stabilize(
        phase_key=phase_key,
        snapshot_provider=lambda: Snapshot(max_batch_tasks=1),
        transport=transport,
        now_ns=22,
    )
    assert second.accepted_task_ids == (task_id,)
    assert stack.runtime.facts(task_id).state == "COMMITTED"


def test_prepare_commit_rolls_back_when_scheduler_apply_fails(stack, monkeypatch):
    import pytest

    phase_key = phase()
    stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 64), registered_at_ns=10
    )
    task_id = stack.catalogue.task_ids_for_phase(phase_key)[0]
    make_ready(stack, (task_id,))
    stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(), ordered_task_ids=stack.catalogue.task_ids_for_phase(phase_key), now_ns=20
    )
    transport = FakeTransport(results=[SubmitOutcome.PREPARED])

    def fail_apply(_receipt):
        raise RuntimeError("injected scheduler commit failure")

    monkeypatch.setattr(stack.authority, "apply_commit_receipt", fail_apply)
    with pytest.raises(RuntimeError, match="injected"):
        stack.controller.stabilizer.stabilize(
            phase_key=phase_key,
            snapshot_provider=lambda: Snapshot(max_batch_tasks=1),
            transport=transport,
            now_ns=21,
        )
    assert transport.prepared == {}
    assert stack.runtime.facts(task_id).state == "READY_UNCOMMITTED"


def test_wrong_receipt_authority_stamp_aborts_and_fails_closed(stack):
    import pytest
    from dataclasses import replace
    from rs_sim import make_authority_stamp
    from rs_sim.scheduler.execution.compiler import CompilationError

    phase_key = phase()
    stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 64), registered_at_ns=10
    )
    task_id = stack.catalogue.task_ids_for_phase(phase_key)[0]
    make_ready(stack, (task_id,))
    stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(), ordered_task_ids=stack.catalogue.task_ids_for_phase(phase_key), now_ns=20
    )

    class WrongAuthorityTransport(FakeTransport):
        aborted = False

        def prepare_commit(self, batch, commit_time_ns: int):
            outcome, receipt = super().prepare_commit(batch, commit_time_ns)
            wrong = make_authority_stamp(
                phase_token=batch.authority_stamp.phase_token,
                plan_id="wrong-plan",
                phase_plan_epoch=batch.authority_stamp.phase_plan_epoch,
            )
            bad = replace(receipt, authority_stamp=wrong)
            self.prepared.pop(receipt.receipt_id)
            self.prepared[bad.receipt_id] = bad
            return outcome, bad

        def abort_commit(self, receipt):
            self.aborted = True
            return super().abort_commit(receipt)

    transport = WrongAuthorityTransport(results=[SubmitOutcome.PREPARED])
    with pytest.raises(CompilationError, match="does not echo"):
        stack.controller.stabilizer.stabilize(
            phase_key=phase_key,
            snapshot_provider=lambda: Snapshot(max_batch_tasks=1),
            transport=transport,
            now_ns=21,
        )
    assert transport.aborted
    assert stack.runtime.facts(task_id).state == "READY_UNCOMMITTED"


def test_wrong_receipt_batch_topology_task_set_or_digest_aborts(stack):
    import pytest
    from dataclasses import replace
    from rs_sim.scheduler.execution.compiler import CompilationError

    mutations = (
        lambda receipt: replace(receipt, batch_id="wrong-batch"),
        lambda receipt: replace(receipt, topology_digest="wrong-topology"),
        lambda receipt: replace(receipt, task_ids=("wrong-task",)),
        lambda receipt: replace(receipt, batch_digest="wrong-digest"),
    )
    for mutate in mutations:
        local = __import__("tests.scheduler.conftest", fromlist=["build_stack"]).build_stack()
        phase_key = phase()
        local.controller.register_expectation(
            expectation(phase_key, 0, 1, 64), registered_at_ns=10
        )
        task_id = local.catalogue.task_ids_for_phase(phase_key)[0]
        make_ready(local, (task_id,))
        local.controller.activate_plan(
            phase_key=phase_key, window_key=window(), ordered_task_ids=local.catalogue.task_ids_for_phase(phase_key), now_ns=20
        )

        class MutatingTransport(FakeTransport):
            aborted = False

            def prepare_commit(self, batch, commit_time_ns: int):
                outcome, receipt = super().prepare_commit(batch, commit_time_ns)
                bad = mutate(receipt)
                self.prepared[receipt.receipt_id] = bad
                return outcome, bad

            def abort_commit(self, receipt):
                self.aborted = True
                return super().abort_commit(receipt)

        transport = MutatingTransport(results=[SubmitOutcome.PREPARED])
        with pytest.raises(CompilationError, match="does not echo"):
            local.controller.stabilizer.stabilize(
                phase_key=phase_key,
                snapshot_provider=lambda: Snapshot(max_batch_tasks=1),
                transport=transport,
                now_ns=21,
            )
        assert transport.aborted
        assert local.runtime.facts(task_id).state == "READY_UNCOMMITTED"


def test_nonprepared_result_with_live_receipt_is_aborted(stack):
    import pytest
    from rs_sim.scheduler.execution.compiler import CompilationError

    phase_key = phase()
    stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 64), registered_at_ns=10
    )
    task_id = stack.catalogue.task_ids_for_phase(phase_key)[0]
    make_ready(stack, (task_id,))
    stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(), ordered_task_ids=stack.catalogue.task_ids_for_phase(phase_key), now_ns=20
    )

    class InvalidOutcomeTransport(FakeTransport):
        aborted = False

        def prepare_commit(self, batch, commit_time_ns: int):
            _, receipt = super().prepare_commit(batch, commit_time_ns)
            return SubmitOutcome.RETRYABLE_RESOURCE_BUSY, receipt

        def abort_commit(self, receipt):
            self.aborted = True
            return super().abort_commit(receipt)

    transport = InvalidOutcomeTransport(results=[SubmitOutcome.PREPARED])
    with pytest.raises(CompilationError, match="non-PREPARED"):
        stack.controller.stabilizer.stabilize(
            phase_key=phase_key,
            snapshot_provider=lambda: Snapshot(max_batch_tasks=1),
            transport=transport,
            now_ns=21,
        )
    assert transport.aborted
    assert stack.runtime.facts(task_id).state == "READY_UNCOMMITTED"


def test_multi_task_logical_commit_is_atomic_when_one_task_is_not_ready(stack):
    import pytest
    from rs_sim.scheduler.errors import AuthorityError

    phase_key = phase()
    stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 64), registered_at_ns=10
    )
    stack.controller.register_expectation(
        expectation(phase_key, 2, 3, 64), registered_at_ns=11
    )
    first, second = stack.catalogue.task_ids_for_phase(phase_key)
    make_ready(stack, (first,))
    stack.controller.activate_plan(
        phase_key=phase_key, window_key=window(), ordered_task_ids=stack.catalogue.task_ids_for_phase(phase_key), now_ns=20
    )
    with pytest.raises(AuthorityError, match="READY_UNCOMMITTED"):
        stack.authority.commit_batch(phase_key, (first, second), at_ns=21)
    assert stack.runtime.facts(first).state == "READY_UNCOMMITTED"
    assert stack.runtime.facts(second).state == "PENDING_DEPENDENCY"
    assert stack.authority.record_view(phase_key).committed_task_ids == ()
