from __future__ import annotations

from tests.scheduler.conftest import build_stack
from tests.scheduler.helpers import expectation, phase


def test_ranges_are_complete_nonoverlapping_and_zero_edges_create_no_tasks(stack):
    phase_key = phase()
    tasks = stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 150), registered_at_ns=11
    )
    views = [stack.adapter.task_view(task) for task in tasks]
    assert [(view.byte_offset, view.payload_bytes) for view in views] == [
        (0, 64),
        (64, 64),
        (128, 22),
    ]
    assert sum(view.payload_bytes for view in views) == 150
    assert stack.controller.register_expectation(
        expectation(phase_key, 0, 2, 0), registered_at_ns=12
    ) == ()


def test_p2_p0_alias_registration_is_idempotent(stack):
    shared_phase = phase(layer=2, kind="DISPATCH_LAYER_2")
    exp = expectation(shared_phase, 1, 3, 128)
    first = stack.controller.register_expectation(exp, registered_at_ns=100)
    digest_before = stack.catalogue.digest()
    second = stack.controller.register_expectation(exp, registered_at_ns=999)
    assert [stack.adapter.task_view(item).task_id for item in first] == [
        stack.adapter.task_view(item).task_id for item in second
    ]
    assert stack.catalogue.digest() == digest_before
    assert len(stack.catalogue.tasks_for_phase(shared_phase)) == 2


def test_task_and_catalogue_digest_is_identical_across_100_runs():
    digests = set()
    task_ids = set()
    for _ in range(100):
        stack = build_stack(chunk_bytes=64, alignment_bytes=16)
        phase_key = phase()
        stack.controller.register_expectation(
            expectation(phase_key, 0, 1, 150), registered_at_ns=11
        )
        stack.controller.register_expectation(
            expectation(phase_key, 2, 3, 96), registered_at_ns=12
        )
        digests.add(stack.catalogue.digest())
        task_ids.add(stack.catalogue.task_ids_for_phase(phase_key))
    assert len(digests) == 1
    assert len(task_ids) == 1


def test_task_becomes_ready_only_after_permit_and_source_payload(stack):
    phase_key = phase()
    tasks = stack.controller.register_expectation(
        expectation(phase_key, 0, 1, 64), registered_at_ns=11
    )
    task_id = stack.adapter.task_view(tasks[0]).task_id
    stack.runtime.note_permit(task_id, at_ns=20)
    assert stack.runtime.facts(task_id).state == "PENDING_DEPENDENCY"
    stack.runtime.note_source_payload_ready(task_id, at_ns=25)
    facts = stack.runtime.facts(task_id)
    assert facts.state == "READY_UNCOMMITTED"
    assert facts.ready_at_ns == 25


def test_adapter_rejects_inconsistent_edge_identity(stack):
    from types import SimpleNamespace
    from rs_sim.scheduler.errors import SharedSchemaError

    phase_key = phase()
    valid = expectation(phase_key, 0, 1, 64)
    invalid = SimpleNamespace(
        edge_key=valid.edge_key,
        phase_key=valid.phase_key,
        src_rank=2,
        dst_rank=valid.dst_rank,
        total_expected_payload_bytes=valid.total_expected_payload_bytes,
        expectation_digest=valid.expectation_digest,
        origin=valid.origin,
        created_at_ns=valid.created_at_ns,
        zero_edge=valid.zero_edge,
        descriptor_digest_or_none=valid.descriptor_digest_or_none,
    )
    try:
        stack.controller.register_expectation(invalid, registered_at_ns=11)
    except SharedSchemaError:
        pass
    else:
        raise AssertionError("inconsistent edge identity was accepted")


def test_nonzero_local_diagonal_expectation_creates_no_network_task(stack):
    phase_key = phase()
    tasks = stack.controller.register_expectation(
        expectation(phase_key, 0, 0, 64), registered_at_ns=10
    )
    assert tasks == ()
    assert stack.catalogue.task_ids_for_phase(phase_key) == ()


def test_canonical_identity_excludes_expectation_and_registration_time():
    first = build_stack(chunk_bytes=64, alignment_bytes=16)
    second = build_stack(chunk_bytes=64, alignment_bytes=16)
    phase_key = phase()
    first.controller.register_expectation(
        expectation(phase_key, 0, 1, 128, created_at_ns=10),
        registered_at_ns=20,
    )
    second.controller.register_expectation(
        expectation(phase_key, 0, 1, 128, created_at_ns=10_000),
        registered_at_ns=20_000,
    )
    assert first.catalogue.task_ids_for_phase(phase_key) == second.catalogue.task_ids_for_phase(phase_key)
    assert first.catalogue.phase_digest(phase_key) == second.catalogue.phase_digest(phase_key)
