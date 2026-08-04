from __future__ import annotations

from rs_sim import SimulationKernel, SubmitOutcome
from rs_sim.transport import FormalControlPlaneTransport

from .conftest import ControlLog, build_harness
from .test_control_plane import profile, request


def _run_once() -> dict[str, str]:
    harness = build_harness()
    outcome, receipt = harness.transport.prepare_commit(
        harness.batch("t0", "t1"), harness.transport.kernel.now_ns)
    assert outcome is SubmitOutcome.PREPARED and receipt is not None
    harness.transport.confirm_commit(receipt)
    while harness.kernel.has_events():
        harness.kernel.run_next_timestamp()

    kernel = SimulationKernel()
    sink = ControlLog()
    control = FormalControlPlaneTransport(
        kernel=kernel, profile=profile(), delivery_sink=sink
    )
    control.publish_row(request(src_rank=0))
    control.publish_row(request(src_rank=1))
    while kernel.has_events():
        kernel.run_next_timestamp()

    return {
        "physical_record_digest": harness.transport.physical_record_digest(),
        "data_event_digest": harness.kernel.event_digest(),
        "data_timeline_digest": harness.kernel.timeline_digest(),
        "control_delivery_digest": control.delivery_digest(),
        "control_event_digest": kernel.event_digest(),
        "control_timeline_digest": kernel.timeline_digest(),
    }


def test_transport_determinism_repeats_without_pinned_release_fixture() -> None:
    assert _run_once() == _run_once()
