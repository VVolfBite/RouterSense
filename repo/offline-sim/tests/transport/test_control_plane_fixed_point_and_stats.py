from __future__ import annotations

from rs_sim import SimulationKernel, make_control_plane_profile
from rs_sim.transport import FormalControlPlaneTransport

from .conftest import ControlLog
from .test_control_plane import request


def _run(kernel):
    while kernel.has_events():
        kernel.run_next_timestamp()


def test_many_same_timestamp_arrivals_use_publish_sequence_not_event_hash_order():
    kernel = SimulationKernel()
    sink = ControlLog()
    profile = make_control_plane_profile(
        profile_id="same-time-fifo",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        fixed_latency_ns=1,
        bandwidth_bytes_per_second=1_000_000_000,
    )
    control = FormalControlPlaneTransport(
        kernel=kernel, profile=profile, delivery_sink=sink
    )
    expected = [
        control.publish_row(request(src_rank=rank, published_at_ns=10, payload_bytes=0))
        for rank in range(20)
    ]
    _run(kernel)
    assert [delivery.request_digest for delivery in sink.deliveries] == expected
    assert control.terminal_state()["terminal"] is True


def test_zero_duration_control_delivery_uses_fixed_point_rounds_without_time_padding():
    kernel = SimulationKernel()
    sink = ControlLog()
    profile = make_control_plane_profile(
        profile_id="zero-duration-control",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        fixed_latency_ns=0,
        bandwidth_bytes_per_second=1_000_000_000,
    )
    control = FormalControlPlaneTransport(
        kernel=kernel, profile=profile, delivery_sink=sink
    )
    first = control.publish_row(request(src_rank=0, published_at_ns=10, payload_bytes=0))
    second = control.publish_row(request(src_rank=1, published_at_ns=10, payload_bytes=0))
    kernel.run_next_timestamp()
    assert kernel.now_ns == 10
    assert [delivery.request_digest for delivery in sink.deliveries] == [first, second]
    assert [delivery.delivered_at_ns for delivery in sink.deliveries] == [10, 10]
    assert control.statistics()["total_service_time_ns"] == 0
    assert control.terminal_state()["terminal"] is True


def test_control_statistics_report_bytes_queue_wait_and_latency():
    kernel = SimulationKernel()
    sink = ControlLog()
    profile = make_control_plane_profile(
        profile_id="control-stats",
        profile_provenance="SYNTHETIC_TEST_ONLY",
        performance_eligible=False,
        fixed_latency_ns=5,
        bandwidth_bytes_per_second=1_000_000_000,
    )
    control = FormalControlPlaneTransport(
        kernel=kernel, profile=profile, delivery_sink=sink
    )
    control.publish_row(request(src_rank=0, published_at_ns=10, payload_bytes=10))
    control.publish_row(request(src_rank=1, published_at_ns=10, payload_bytes=10))
    _run(kernel)
    stats = control.statistics()
    assert stats["published_request_count"] == 2
    assert stats["published_payload_bytes"] == 20
    assert stats["delivered_request_count"] == 2
    assert stats["delivered_payload_bytes"] == 20
    assert stats["total_service_time_ns"] == 20
    assert stats["total_queue_wait_ns"] == 10
    assert stats["max_end_to_end_latency_ns"] == 25
    assert stats["channel_utilization_rational"] == (20, 25)
