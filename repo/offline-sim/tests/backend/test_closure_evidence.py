from __future__ import annotations

import pytest

from rs_sim.backend import (
    BackendSealReadinessPort,
    DuplicateRegistrationError,
    IllegalTransitionError,
    PhaseClosureSummary,
)

from .conftest import Expectation, Phase, make_system


def _close_dispatch(system, phase: Phase) -> PhaseClosureSummary:
    system.backend.on_dispatch_descriptor_delivered(
        phase_key=phase,
        src_rank=0,
        payload_bytes_by_destination=(4, 8),
        descriptor_digest="dispatch-row-0",
        delivered_at_ns=10,
    )
    assert system.backend.phase_closure_summary(phase_key=phase) is None
    assert not system.observer.payloads("PHASE_CLOSURE_SUMMARY_READY")

    system.backend.on_dispatch_descriptor_delivered(
        phase_key=phase,
        src_rank=1,
        payload_bytes_by_destination=(0, 6),
        descriptor_digest="dispatch-row-1",
        delivered_at_ns=14,
    )
    return system.backend.require_phase_closure_summary(phase_key=phase)


def test_dispatch_closure_summary_is_complete_stable_and_fact_only() -> None:
    system = make_system(world_size=2, capacity=32)
    phase = Phase(0, 0, "DISPATCH")

    assert isinstance(system.backend, BackendSealReadinessPort)
    summary = _close_dispatch(system, phase)

    assert summary.seal_ready is True
    assert summary.closure_generation == 1
    assert summary.finalized_at_ns == 14
    assert summary.expected_descriptor_count == 2
    assert summary.delivered_descriptor_count == 2
    assert summary.expected_expectation_count == 4
    assert summary.expectation_count == 4
    assert summary.zero_expectation_count == 1
    assert summary.local_nonzero_expectation_count == 2
    assert summary.remote_task_expectation_count == 1
    assert summary.remote_task_expected_payload_bytes == 8
    assert summary.remote_task_expectation_inputs[0].src_rank == 0
    assert summary.remote_task_expectation_inputs[0].dst_rank == 1
    assert summary.remote_task_expectation_inputs[0].expected_payload_bytes == 8
    assert summary.closure_digest
    assert summary.all_expectations_digest
    assert summary.remote_task_inputs_digest

    rows = system.observer.payloads("PHASE_CLOSURE_SUMMARY_READY")
    assert len(rows) == 1
    assert rows[0]["summary"] == summary
    assert rows[0]["closure_digest"] == summary.closure_digest

    # Exact replay is idempotent and does not create a second generation.
    system.backend.on_dispatch_descriptor_delivered(
        phase_key=phase,
        src_rank=1,
        payload_bytes_by_destination=(0, 6),
        descriptor_digest="dispatch-row-1",
        delivered_at_ns=14,
    )
    assert system.backend.require_phase_closure_summary(phase_key=phase) == summary
    assert len(system.observer.payloads("PHASE_CLOSURE_SUMMARY_READY")) == 1


def test_late_descriptor_or_expectation_after_closure_fails_closed() -> None:
    system = make_system(world_size=2, capacity=32)
    phase = Phase(0, 0, "DISPATCH")
    summary = _close_dispatch(system, phase)

    with pytest.raises(DuplicateRegistrationError):
        system.backend.on_dispatch_descriptor_delivered(
            phase_key=phase,
            src_rank=1,
            payload_bytes_by_destination=(1, 5),
            descriptor_digest="conflicting-row-1",
            delivered_at_ns=15,
        )

    late = Expectation(
        edge_key=("late-edge", phase, 0, 1),
        phase_key=phase,
        src_rank=0,
        dst_rank=1,
        total_expected_payload_bytes=8,
        expectation_digest="late-expectation",
        origin="LATE_TEST",
        created_at_ns=16,
        zero_edge=False,
        descriptor_digest_or_none="dispatch-row-0",
    )
    with pytest.raises(IllegalTransitionError, match="late expectation"):
        system.receiver.register_expectation(
            late, descriptor_digest_or_none="dispatch-row-0"
        )

    with pytest.raises(IllegalTransitionError, match="conflicting closure"):
        system.receiver.finalize_expectation_closure(
            phase_key=phase, closure_digest="different-digest"
        )
    assert system.backend.require_phase_closure_summary(phase_key=phase) == summary


def test_combine_closure_has_no_descriptor_count_and_exact_remote_inputs() -> None:
    system = make_system(world_size=2, capacity=32)
    phase = Phase(0, 0, "COMBINE")

    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=phase,
        original_rank=0,
        realized_dispatch_payload_bytes_by_expert=(3, 0),
        created_at_ns=7,
    )
    assert system.backend.phase_closure_summary(phase_key=phase) is None

    system.backend.register_combine_expectations_from_realized_dispatch(
        combine_phase_key=phase,
        original_rank=1,
        realized_dispatch_payload_bytes_by_expert=(5, 2),
        created_at_ns=9,
    )
    summary = system.backend.require_phase_closure_summary(phase_key=phase)
    assert summary.phase_kind == "COMBINE"
    assert summary.expected_descriptor_count == 0
    assert summary.delivered_descriptor_count == 0
    assert summary.expectation_count == 4
    assert summary.zero_expectation_count == 1
    assert summary.local_nonzero_expectation_count == 2
    assert summary.remote_task_expectation_count == 1
    assert summary.remote_task_expected_payload_bytes == 5
    assert summary.finalized_at_ns == 9


def test_closure_digest_is_deterministic_across_replays() -> None:
    digests = set()
    for _ in range(50):
        system = make_system(world_size=2, capacity=32)
        summary = _close_dispatch(system, Phase(0, 0, "DISPATCH"))
        digests.add(summary.closure_digest)
    assert len(digests) == 1
