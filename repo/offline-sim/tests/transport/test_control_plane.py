from __future__ import annotations

from rs_sim import (
    PhaseKey,
    PhaseKind,
    SimulationKernel,
    make_control_plane_profile,
    make_exact_row_descriptor,
    make_row_broadcast_request,
    stable_digest,
)
from rs_sim.transport import (
    FormalControlPlaneTransport,
    build_formal_transports,
    make_default_synthetic_control_profile,
    make_default_synthetic_hardware_profile,
)

from .conftest import ControlLog, build_harness


def profile():
    return make_control_plane_profile(
        profile_id="control-test",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        fixed_latency_ns=5,
        bandwidth_bytes_per_second=1_000_000_000,
    )


def request(*, src_rank, published_at_ns=10, payload_bytes=10):
    phase = PhaseKey("run", "control", 0, PhaseKind.DISPATCH)
    descriptor = make_exact_row_descriptor(
        phase_key=phase,
        src_rank=src_rank,
        realized_rows_by_destination=(0, 1, 0, 0),
        payload_bytes_by_destination=(0, 4, 0, 0),
        payload_spec_digest="payload-spec",
        published_at_ns=published_at_ns,
        descriptor_payload_bytes=payload_bytes,
    )
    return make_row_broadcast_request(descriptor)


def test_same_arrival_requests_complete_in_publish_fifo_order():
    kernel = SimulationKernel()
    sink = ControlLog()
    control = FormalControlPlaneTransport(kernel=kernel, profile=profile(), delivery_sink=sink)
    first = request(src_rank=0)
    second = request(src_rank=1)
    first_digest = control.publish_row(first)
    second_digest = control.publish_row(second)
    while kernel.has_events():
        kernel.run_next_timestamp()
    assert [item.request_digest for item in sink.deliveries] == [first_digest, second_digest]
    assert sink.deliveries[0].delivery_start_ns == 10
    assert sink.deliveries[0].delivered_at_ns == 25
    assert sink.deliveries[1].delivery_start_ns == 20
    assert sink.deliveries[1].delivered_at_ns == 35


def test_delivery_callback_occurs_in_descriptor_delivery_phase():
    kernel = SimulationKernel()
    sink = ControlLog()
    control = FormalControlPlaneTransport(kernel=kernel, profile=profile(), delivery_sink=sink)
    control.publish_row(request(src_rank=0))
    while kernel.has_events():
        kernel.run_next_timestamp()
    rows = [row for row in kernel.timeline() if row.event_type == control.DELIVERY_EVENT]
    assert len(rows) == 1
    assert rows[0].phase_priority.name == "DESCRIPTOR_OBSERVATION_DELIVERY"


def test_control_plane_is_independent_of_data_plane_resources():
    h = build_harness()
    sink = ControlLog()
    control = FormalControlPlaneTransport(kernel=h.kernel, profile=profile(), delivery_sink=sink)
    outcome, receipt = h.transport.prepare_commit(h.batch("t0", "t1"), h.transport.kernel.now_ns)
    assert receipt is not None
    before = h.transport.snapshot()
    control.publish_row(request(src_rank=0, published_at_ns=0))
    assert h.transport.snapshot() == before
    manifest = control.manifest_fragment()
    assert manifest["control_plane_shares_data_nic"] is False


def test_control_delivery_digest_and_event_digest_repeat():
    values = []
    for _ in range(100):
        kernel = SimulationKernel()
        sink = ControlLog()
        control = FormalControlPlaneTransport(kernel=kernel, profile=profile(), delivery_sink=sink)
        control.publish_row(request(src_rank=0))
        control.publish_row(request(src_rank=1))
        while kernel.has_events():
            kernel.run_next_timestamp()
        values.append((control.delivery_digest(), kernel.event_digest()))
    assert len(set(values)) == 1


def test_public_request_digest_is_canonical():
    kernel = SimulationKernel()
    sink = ControlLog()
    control = FormalControlPlaneTransport(kernel=kernel, profile=profile(), delivery_sink=sink)
    item = request(src_rank=0)
    assert control.publish_row(item) == stable_digest(item, domain="ROW_BROADCAST_REQUEST")


def test_default_builder_profiles_are_explicitly_synthetic_and_not_performance_eligible():
    h = build_harness()
    sink = ControlLog()
    bundle = build_formal_transports(
        kernel=SimulationKernel(),
        task_lookup=h.transport.task_lookup,
        permit_lookup=h.transport.permit_lookup,
        authority_validation=h.transport.authority_validation,
        resource_resolver=h.resolver,
        completion_sink=h.completion,
        resource_release_sink=h.release,
        control_delivery_sink=sink,
    )
    manifest = bundle.manifest_fragment()
    assert manifest["transport_transport"] is True
    assert manifest["profile_provenance"] == "SYNTHETIC_TEST_ONLY"
    assert manifest["performance_eligible"] is False
    assert manifest["control_plane_profile_provenance"] == "SYNTHETIC_TEST_ONLY"
    assert manifest["control_plane_performance_eligible"] is False
    assert manifest["control_plane_shares_data_nic"] is False


def test_future_requests_are_fifo_by_actual_arrival_not_call_order():
    kernel = SimulationKernel()
    sink = ControlLog()
    control = FormalControlPlaneTransport(kernel=kernel, profile=profile(), delivery_sink=sink)
    late = request(src_rank=0, published_at_ns=20)
    early = request(src_rank=1, published_at_ns=10)
    late_digest = control.publish_row(late)
    early_digest = control.publish_row(early)
    while kernel.has_events():
        kernel.run_next_timestamp()
    assert [item.request_digest for item in sink.deliveries] == [early_digest, late_digest]
    assert sink.deliveries[0].delivery_start_ns == 10
    # Propagation is pipelined; the next row starts after serialization,
    # before the preceding row reaches its destination.
    assert sink.deliveries[1].delivery_start_ns == 20


def test_descriptor_published_in_the_past_fails_closed():
    kernel = SimulationKernel()
    from rs_sim import KernelPhase

    kernel.register_event_handler("advance", lambda _kernel, _event: None)
    kernel.schedule(
        time_ns=10,
        phase_priority=KernelPhase.COMPLETION_COLLECTION,
        event_type="advance",
        producer="test",
        ordinal=0,
    )
    kernel.run_next_timestamp()
    control = FormalControlPlaneTransport(
        kernel=kernel, profile=profile(), delivery_sink=ControlLog()
    )
    import pytest

    with pytest.raises(ValueError, match="precede"):
        control.publish_row(request(src_rank=0, published_at_ns=9))
