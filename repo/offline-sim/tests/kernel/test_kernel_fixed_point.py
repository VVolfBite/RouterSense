from __future__ import annotations

import pytest

from rs_sim import (
    KernelFault,
    KernelFaultCode,
    KernelPhase,
    PastEventError,
    ProgressSignal,
    RecursiveExecutionError,
    SimulationKernel,
)


def test_same_time_later_phase_current_round_earlier_phase_next_round() -> None:
    kernel = SimulationKernel()

    def root(k: SimulationKernel, _event):
        k.schedule(
            time_ns=k.now_ns,
            phase_priority=KernelPhase.THREE_LINE_JOB_TRANSITIONS,
            producer="test",
            event_type="later",
            ordinal=0,
        )
        k.schedule(
            time_ns=k.now_ns,
            phase_priority=KernelPhase.AUTHORITATIVE_STATE_UPDATES,
            producer="test",
            event_type="earlier",
            ordinal=0,
        )
        return ProgressSignal(authoritative_state_updates=1)

    kernel.register_event_handler("root", root)
    kernel.register_event_handler("later", lambda _k, _e: None)
    kernel.register_event_handler("earlier", lambda _k, _e: None)
    kernel.schedule(
        time_ns=10,
        phase_priority=KernelPhase.DESCRIPTOR_OBSERVATION_DELIVERY,
        producer="test",
        event_type="root",
        ordinal=0,
    )

    processed = kernel.run_next_timestamp()
    assert [(e.event_type, e.round_index, int(e.phase_priority)) for e in processed] == [
        ("root", 0, 3),
        ("later", 0, 5),
        ("earlier", 1, 2),
    ]


def test_zero_duration_completion_returns_to_phase_one_next_round() -> None:
    kernel = SimulationKernel()

    def launch(k: SimulationKernel, _event):
        k.schedule(
            time_ns=k.now_ns,
            phase_priority=KernelPhase.COMPLETION_COLLECTION,
            producer="transport",
            event_type="complete",
            ordinal=0,
            subject_id="task-0",
        )
        return ProgressSignal(successful_commits=1)

    kernel.register_event_handler("launch", launch)
    kernel.register_event_handler(
        "complete",
        lambda _k, _e: ProgressSignal(authoritative_state_updates=1),
    )
    kernel.schedule(
        time_ns=100,
        phase_priority=KernelPhase.EXECUTION_STABILIZATION_SUBMIT,
        producer="scheduler",
        event_type="launch",
        ordinal=0,
    )

    processed = kernel.run_next_timestamp()
    assert [(e.event_type, e.round_index, int(e.phase_priority)) for e in processed] == [
        ("launch", 0, 7),
        ("complete", 1, 1),
    ]


def test_past_event_and_recursive_execution_are_rejected() -> None:
    kernel = SimulationKernel()

    def illegal_recursive(k: SimulationKernel, _event):
        with pytest.raises(RecursiveExecutionError):
            k.run_next_timestamp()
        return None

    kernel.register_event_handler("event", illegal_recursive)
    kernel.schedule(
        time_ns=5,
        phase_priority=KernelPhase.COMPLETION_COLLECTION,
        producer="test",
        event_type="event",
        ordinal=0,
    )
    kernel.run_next_timestamp()

    with pytest.raises(PastEventError):
        kernel.schedule(
            time_ns=5,
            phase_priority=KernelPhase.COMPLETION_COLLECTION,
            producer="test",
            event_type="event",
            ordinal=1,
        )


def test_causal_cycle_has_fail_closed_evidence() -> None:
    kernel = SimulationKernel(max_stabilization_rounds=4)

    def loop(k: SimulationKernel, event):
        k.schedule(
            time_ns=k.now_ns,
            phase_priority=KernelPhase.COMPLETION_COLLECTION,
            producer="cycle",
            event_type="loop",
            ordinal=event.ordinal + 1,
        )
        return ProgressSignal(authoritative_state_updates=1, notes=("loop",))

    kernel.register_event_handler("loop", loop)
    kernel.register_evidence_provider("backend", lambda: {"unreleased": ("rank-0",)})
    kernel.schedule(
        time_ns=0,
        phase_priority=KernelPhase.COMPLETION_COLLECTION,
        producer="cycle",
        event_type="loop",
        ordinal=0,
    )

    with pytest.raises(KernelFault) as captured:
        kernel.run_next_timestamp()
    evidence = captured.value.evidence
    assert evidence.fault_code is KernelFaultCode.CAUSAL_CYCLE
    assert evidence.pending_event_count == 1
    assert len(evidence.round_summaries) == 4
    assert evidence.external_evidence[0][0] == "backend"


def test_deadlock_no_progress_has_evidence() -> None:
    kernel = SimulationKernel()
    kernel.register_evidence_provider(
        "scheduler",
        lambda: {
            "ready_but_unscheduled_tasks": ("task-0",),
            "blocked_reason": "no receive permit",
        },
    )

    with pytest.raises(KernelFault) as captured:
        kernel.run_until_complete(lambda: False)
    evidence = captured.value.evidence
    assert evidence.fault_code is KernelFaultCode.DEADLOCK_NO_PROGRESS
    assert evidence.pending_event_count == 0
    assert evidence.next_event_time_ns is None
    assert evidence.external_evidence[0][0] == "scheduler"


def test_registered_work_predicate_detects_deadlock_at_timestamp_boundary() -> None:
    kernel = SimulationKernel(work_remaining_predicate=lambda: True)
    kernel.register_event_handler("seed", lambda _k, _e: None)
    kernel.schedule(
        time_ns=7,
        phase_priority=KernelPhase.COMPLETION_COLLECTION,
        producer="test",
        event_type="seed",
        ordinal=0,
    )

    with pytest.raises(KernelFault) as captured:
        kernel.run_next_timestamp()
    assert captured.value.evidence.fault_code is KernelFaultCode.DEADLOCK_NO_PROGRESS
