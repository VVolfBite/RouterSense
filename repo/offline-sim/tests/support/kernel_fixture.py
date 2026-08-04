from __future__ import annotations

from dataclasses import dataclass

from rs_sim.backend.core.simulation import ProgressSignal, SimulationKernel
from rs_sim.contracts.schema import KernelPhase


@dataclass(frozen=True, slots=True)
class KernelDeterminismFixtureResult:
    event_digest: str
    timeline_digest: str
    completion_order: tuple[str, ...]
    timeline_shape: tuple[tuple[str, int, int], ...]


def run_kernel_determinism_fixture() -> KernelDeterminismFixtureResult:
    """Run the shared-schema cross-phase zero-delay determinism fixture."""

    kernel = SimulationKernel()
    completion_order: list[str] = []

    def descriptor(k: SimulationKernel, _event):
        for dst in range(4):
            k.schedule(
                time_ns=k.now_ns,
                phase_priority=KernelPhase.BACKEND_RECEIVER_CLOSURE_RELEASE,
                producer="receiver",
                event_type="permit",
                ordinal=dst,
                subject_id=f"edge-{dst}",
                attributes=(("dst", str(dst)), ("src", "0")),
            )
        return ProgressSignal(authoritative_state_updates=1)

    def permit(k: SimulationKernel, event):
        k.schedule(
            time_ns=k.now_ns,
            phase_priority=KernelPhase.EXECUTION_STABILIZATION_SUBMIT,
            producer="transport",
            event_type="submit",
            ordinal=event.ordinal,
            subject_id=event.subject_id,
        )
        return ProgressSignal(authoritative_state_updates=1)

    def submit(k: SimulationKernel, event):
        k.schedule(
            time_ns=k.now_ns,
            phase_priority=KernelPhase.COMPLETION_COLLECTION,
            producer="transport",
            event_type="complete",
            ordinal=event.ordinal,
            subject_id=event.subject_id,
        )
        return ProgressSignal(successful_commits=1)

    def complete(_k: SimulationKernel, event):
        completion_order.append(event.subject_id)
        return ProgressSignal(authoritative_state_updates=1)

    kernel.register_event_handler("descriptor", descriptor)
    kernel.register_event_handler("permit", permit)
    kernel.register_event_handler("submit", submit)
    kernel.register_event_handler("complete", complete)
    kernel.schedule(
        time_ns=20,
        phase_priority=KernelPhase.DESCRIPTOR_OBSERVATION_DELIVERY,
        producer="control",
        event_type="descriptor",
        ordinal=0,
        subject_id="row-0",
    )
    kernel.run_until_complete(lambda: len(completion_order) == 4)

    return KernelDeterminismFixtureResult(
        event_digest=kernel.event_digest(),
        timeline_digest=kernel.timeline_digest(),
        completion_order=tuple(completion_order),
        timeline_shape=tuple(
            (row.event_type, row.round_index, int(row.phase_priority))
            for row in kernel.timeline()
        ),
    )
