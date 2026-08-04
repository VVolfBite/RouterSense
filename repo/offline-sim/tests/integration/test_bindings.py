from __future__ import annotations

from rs_sim import (
    CanonicalTransferTask,
    EdgeKey,
    ExpectationOrigin,
    PhaseExecutionRecord,
    PhaseKey,
    PhaseKind,
    PlanStatus,
    PlanVersion,
    ReceiveExpectation,
    WindowKey,
)
from rs_sim.runtime import SchemaExpectationFactory, build_scheduling_stack
from rs_sim.scheduler import TaskizationSpec


def test_scheduler_uses_shared_schema_objects() -> None:
    phase = PhaseKey("run", "sample", 2, PhaseKind.DISPATCH)
    edge = EdgeKey(phase, 0, 1)
    expectation = ReceiveExpectation(
        edge_key=edge,
        phase_key=phase,
        src_rank=0,
        dst_rank=1,
        total_expected_payload_bytes=10,
        expectation_digest="exp-digest",
        origin=ExpectationOrigin.DISPATCH_DESCRIPTOR,
        created_at_ns=7,
        zero_edge=False,
        descriptor_digest_or_none="descriptor-digest",
    )
    stack = build_scheduling_stack(
        taskization_spec=TaskizationSpec(chunk_bytes=4, alignment_bytes=1)
    )
    tasks = stack.controller.register_expectation(expectation, registered_at_ns=7)
    assert [task.payload_bytes for task in tasks] == [4, 4, 2]
    assert all(isinstance(task, CanonicalTransferTask) for task in tasks)

    plan = stack.controller.activate_plan(
        phase_key=phase,
        window_key=WindowKey("run", "sample", 2),
        ordered_task_ids=tuple(task.task_id for task in tasks),
        now_ns=8,
    )
    record = stack.authority.record(phase)
    assert isinstance(plan, PlanVersion)
    assert plan.status is PlanStatus.ACTIVE
    assert isinstance(record, PhaseExecutionRecord)
    assert record.active_plan_id == plan.plan_id


def test_expectation_factory_maps_backend_origins() -> None:
    phase = PhaseKey("run", "sample", 1, PhaseKind.COMBINE)
    edge = EdgeKey(phase, 1, 0)
    expectation = SchemaExpectationFactory().create_receive_expectation(
        edge_key=edge,
        phase_key=phase,
        src_rank=1,
        dst_rank=0,
        total_expected_payload_bytes=8,
        expectation_digest="e",
        origin="REALIZED_DISPATCH_TRANSPOSE",
        created_at_ns=3,
        zero_edge=False,
        descriptor_digest_or_none=None,
    )
    assert expectation.origin is ExpectationOrigin.COMBINE_REALIZED


def test_shared_binding_digest_is_stable() -> None:
    from rs_sim.runtime import shared_binding_digest

    first = shared_binding_digest()
    second = shared_binding_digest()
    assert first == second
    assert len(first) == 64
