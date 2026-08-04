from __future__ import annotations

import pytest

from rs_sim.scheduler.execution.lines import ThreeLineServices
from rs_sim.scheduler.decorators.planning_gate import PlanningGate, PlanningMode, PlanningTrigger


def test_event_trigger_set_is_explicitly_configurable_and_empty_is_invalid():
    with pytest.raises(ValueError, match="non-empty"):
        PlanningGate(PlanningMode.EVENT, event_triggers=())

    delegated = PlanningGate(PlanningMode.EVENT)
    decision = delegated.on_observation(
        "SOURCE_PAYLOAD_READY:src:0", changed=True, closure_satisfied=False
    )
    assert decision.action == "CREATE_PLAN_VERSION"
    assert decision.trigger is PlanningTrigger.TASK_READY


def test_event_plans_only_on_configured_changed_observation_not_resource_release():
    gate = PlanningGate(
        PlanningMode.EVENT,
        event_triggers=(PlanningTrigger.DESCRIPTOR_DELIVERY, PlanningTrigger.TASK_READY),
    )
    disabled = gate.on_observation(
        "permit-0",
        trigger=PlanningTrigger.PERMIT_GRANTED,
        changed=True,
        closure_satisfied=False,
    )
    assert disabled.action == "NO_ACTION"
    assert disabled.plan_count == 0
    first = gate.on_observation(
        "row-0",
        trigger=PlanningTrigger.DESCRIPTOR_DELIVERY,
        changed=True,
        closure_satisfied=False,
    )
    assert first.action == "CREATE_PLAN_VERSION"
    assert first.plan_count == 1
    duplicate = gate.on_observation(
        "row-0",
        trigger=PlanningTrigger.DESCRIPTOR_DELIVERY,
        changed=True,
        closure_satisfied=False,
    )
    assert duplicate.action == "NO_ACTION"
    release = gate.on_resource_release()
    assert release.action == "STABILIZE_EXECUTION"
    assert release.plan_count == 1


def test_global_creates_zero_before_closure_and_exactly_one_after_closure():
    gate = PlanningGate(PlanningMode.GLOBAL)
    assert gate.plan_count == 0
    assert gate.on_observation(
        "row-0",
        trigger=PlanningTrigger.DESCRIPTOR_DELIVERY,
        changed=True,
        closure_satisfied=False,
    ).action == "NO_ACTION"
    assert gate.plan_count == 0
    closure = gate.on_observation(
        "row-0",
        trigger=PlanningTrigger.DESCRIPTOR_DELIVERY,
        changed=False,
        closure_satisfied=True,
    )
    assert closure.action == "CREATE_PLAN_VERSION"
    assert closure.plan_count == 1
    assert gate.on_observation(
        "row-2",
        trigger=PlanningTrigger.TASK_READY,
        changed=True,
        closure_satisfied=True,
    ).action == "NO_ACTION"
    assert gate.on_resource_release().action == "STABILIZE_EXECUTION"
    assert gate.plan_count == 1


def test_three_lines_are_independent_fifo_single_server_nonpreemptive():
    lines = ThreeLineServices()
    p0 = lines.prediction.submit(job_id="p0", arrival_at_ns=10, duration_ns=10, payload={"x": 0})
    p1 = lines.prediction.submit(job_id="p1", arrival_at_ns=10, duration_ns=5, payload={"x": 1})
    c0 = lines.control.submit(job_id="c0", arrival_at_ns=12, duration_ns=3, payload={"x": 2})
    e0 = lines.execution_binding.submit(job_id="e0", arrival_at_ns=11, duration_ns=4, payload={"x": 3})
    assert p0.start_at_ns == 10 and p0.finish_at_ns == 20
    assert p1.start_at_ns == 20 and p1.finish_at_ns == 25
    assert c0.start_at_ns == 12 and c0.finish_at_ns == 15
    assert e0.start_at_ns == 11 and e0.finish_at_ns == 15


def test_exposed_delay_includes_queueing_after_hide_deadline():
    lines = ThreeLineServices()
    lines.control.submit(
        job_id="first",
        arrival_at_ns=0,
        duration_ns=100,
        hide_until_ns=100,
        payload={"job": 1},
    )
    second = lines.control.submit(
        job_id="second",
        arrival_at_ns=0,
        duration_ns=10,
        hide_until_ns=50,
        payload={"job": 2},
    )
    assert second.queue_wait_ns == 100
    assert second.exposed_service_ns == 10
    assert second.exposed_delay_ns == 60
    metrics = lines.control.metrics()
    assert metrics.exposed_delay_ns == 60
