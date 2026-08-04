from __future__ import annotations

import pytest

from rs_sim.scheduler.execution.window_arbiter import (
    PhaseFrontier,
    WindowArbitrationContext,
    make_window_decision,
)
from tests.scheduler.helpers import expectation, make_ready, phase, window


def test_window_arbiter_contract_selects_prefix_of_one_independent_phase(stack):
    phase_a = phase(layer=1)
    phase_b = phase(layer=2)
    stack.controller.register_expectation(
        expectation(phase_a, 0, 1, 128), registered_at_ns=10
    )
    stack.controller.register_expectation(
        expectation(phase_b, 2, 3, 64), registered_at_ns=11
    )
    ids_a = stack.catalogue.task_ids_for_phase(phase_a)
    ids_b = stack.catalogue.task_ids_for_phase(phase_b)
    make_ready(stack, (*ids_a, *ids_b))
    stack.controller.activate_plan(
        phase_key=phase_a, window_key=window(), ordered_task_ids=ids_a, now_ns=20
    )
    stack.controller.activate_plan(
        phase_key=phase_b, window_key=window(), ordered_task_ids=ids_b, now_ns=21
    )
    frontier_a = PhaseFrontier.build(
        phase_key=phase_a,
        authority_stamp=stack.authority.authority_stamp(phase_a),
        ready_task_ids=ids_a,
    )
    frontier_b = PhaseFrontier.build(
        phase_key=phase_b,
        authority_stamp=stack.authority.authority_stamp(phase_b),
        ready_task_ids=ids_b,
    )
    context = WindowArbitrationContext.build(
        window_key=window(),
        frontiers=(frontier_a, frontier_b),
        transport_snapshot_digest="snapshot",
        observed_at_ns=22,
    )
    decision = make_window_decision(
        context,
        selected_phase_token=frontier_a.authority_stamp.phase_token,
        selected_task_ids=(ids_a[0],),
        reason="contract-test-selection",
    )
    assert decision.selected_task_ids == (ids_a[0],)


def test_window_arbiter_contract_rejects_cross_phase_or_nonprefix_selection(stack):
    phase_a = phase(layer=1)
    phase_b = phase(layer=2)
    stack.controller.register_expectation(
        expectation(phase_a, 0, 1, 128), registered_at_ns=10
    )
    stack.controller.register_expectation(
        expectation(phase_b, 2, 3, 64), registered_at_ns=11
    )
    ids_a = stack.catalogue.task_ids_for_phase(phase_a)
    ids_b = stack.catalogue.task_ids_for_phase(phase_b)
    make_ready(stack, (*ids_a, *ids_b))
    stack.controller.activate_plan(
        phase_key=phase_a, window_key=window(), ordered_task_ids=ids_a, now_ns=20
    )
    stack.controller.activate_plan(
        phase_key=phase_b, window_key=window(), ordered_task_ids=ids_b, now_ns=21
    )
    frontier_a = PhaseFrontier.build(
        phase_key=phase_a,
        authority_stamp=stack.authority.authority_stamp(phase_a),
        ready_task_ids=ids_a,
    )
    frontier_b = PhaseFrontier.build(
        phase_key=phase_b,
        authority_stamp=stack.authority.authority_stamp(phase_b),
        ready_task_ids=ids_b,
    )
    context = WindowArbitrationContext.build(
        window_key=window(),
        frontiers=(frontier_a, frontier_b),
        transport_snapshot_digest="snapshot",
        observed_at_ns=22,
    )
    with pytest.raises(ValueError, match="prefix"):
        make_window_decision(
            context,
            selected_phase_token=frontier_a.authority_stamp.phase_token,
            selected_task_ids=(ids_a[-1],),
            reason="invalid-nonprefix",
        )
    with pytest.raises(ValueError, match="prefix"):
        make_window_decision(
            context,
            selected_phase_token=frontier_a.authority_stamp.phase_token,
            selected_task_ids=(ids_a[0], ids_b[0]),
            reason="invalid-cross-phase",
        )


def test_wave_arbiter_keeps_selection_inside_earliest_task_boundary_wave(stack):
    from rs_sim.scheduler.execution.window_arbiter import ReleaseFrontierWaveArbiter

    phase_a = phase(layer=4)
    phase_b = phase(layer=5)
    stack.controller.register_expectation(expectation(phase_a, 0, 1, 256), registered_at_ns=10)
    stack.controller.register_expectation(expectation(phase_b, 2, 3, 256), registered_at_ns=10)
    ids_a = stack.catalogue.task_ids_for_phase(phase_a)
    ids_b = stack.catalogue.task_ids_for_phase(phase_b)
    make_ready(stack, (*ids_a, *ids_b))
    stack.controller.activate_plan(phase_key=phase_a, window_key=window(), ordered_task_ids=ids_a, now_ns=20)
    stack.controller.activate_plan(phase_key=phase_b, window_key=window(), ordered_task_ids=ids_b, now_ns=20)
    frontier_a = PhaseFrontier.build(
        phase_key=phase_a,
        authority_stamp=stack.authority.authority_stamp(phase_a),
        ready_task_ids=ids_a,
    )
    frontier_b = PhaseFrontier.build(
        phase_key=phase_b,
        authority_stamp=stack.authority.authority_stamp(phase_b),
        ready_task_ids=ids_b,
    )
    context = WindowArbitrationContext.build(
        window_key=window(),
        frontiers=(frontier_a, frontier_b),
        transport_snapshot_digest="snapshot-wave",
        observed_at_ns=22,
    )
    arbiter = ReleaseFrontierWaveArbiter(
        preferred_waves=((ids_b[0], ids_a[0]),),
        preferred_task_ids=(ids_b[0], ids_a[0]),
        max_prefix_tasks=8,
    )
    decision = arbiter.select(context)
    assert decision.selected_phase_token == frontier_b.authority_stamp.phase_token
    assert decision.selected_task_ids == (ids_b[0],)
    assert "TASK_BOUNDARY_WAVE" in decision.reason
