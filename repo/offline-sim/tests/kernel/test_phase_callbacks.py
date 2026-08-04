from __future__ import annotations

from rs_sim import KernelPhase, ProgressSignal, SimulationKernel


def test_phase_callbacks_are_name_sorted_and_reach_fixed_point() -> None:
    kernel = SimulationKernel()
    calls: list[str] = []
    state = {"first": True}

    def callback_b(_kernel: SimulationKernel):
        calls.append("b")
        return None

    def callback_a(_kernel: SimulationKernel):
        calls.append("a")
        if state["first"]:
            state["first"] = False
            return ProgressSignal(authoritative_state_updates=1)
        return None

    kernel.register_phase_callback(
        KernelPhase.AUTHORITATIVE_STATE_UPDATES, "b", callback_b
    )
    kernel.register_phase_callback(
        KernelPhase.AUTHORITATIVE_STATE_UPDATES, "a", callback_a
    )
    kernel.register_event_handler("seed", lambda _k, _e: None)
    kernel.schedule(
        time_ns=0,
        phase_priority=KernelPhase.COMPLETION_COLLECTION,
        producer="test",
        event_type="seed",
        ordinal=0,
    )

    kernel.run_next_timestamp()
    assert calls == ["a", "b", "a", "b"]
