from __future__ import annotations

from rs_sim.scheduler.execution.lines import ThreeLineServices
from tests.scheduler.conftest import build_stack
from tests.scheduler.helpers import expectation, make_ready, phase, window


def test_plan_authority_and_line_digests_repeat_100_times():
    digests = set()
    for _ in range(100):
        stack = build_stack()
        phase_key = phase()
        stack.controller.register_expectation(
            expectation(phase_key, 0, 1, 128), registered_at_ns=10
        )
        stack.controller.register_expectation(
            expectation(phase_key, 2, 3, 64), registered_at_ns=11
        )
        ids = stack.catalogue.task_ids_for_phase(phase_key)
        make_ready(stack, ids)
        stack.controller.activate_plan(
            phase_key=phase_key,
            window_key=window(),
            ordered_task_ids=tuple(sorted(ids, key=lambda task_id: (-stack.catalogue.get(task_id).payload_bytes, task_id))),
            now_ns=40,
        )
        lines = ThreeLineServices()
        lines.prediction.submit(job_id="p", arrival_at_ns=1, duration_ns=2, payload={"a": 1})
        lines.control.submit(job_id="c", arrival_at_ns=2, duration_ns=3, payload={"b": 2})
        lines.execution_binding.submit(job_id="e", arrival_at_ns=3, duration_ns=4, payload={"c": 3})
        digests.add((stack.catalogue.digest(), stack.authority.digest(), lines.digest()))
    assert len(digests) == 1
