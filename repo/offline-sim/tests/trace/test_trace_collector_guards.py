from __future__ import annotations

import pytest

from rs_sim.trace.build.collector import TraceCollector
from rs_sim.trace.schema.fixtures import build_builtin_fixtures
from rs_sim.trace.schema.model import TraceValidationError


def test_trace_collector_rejects_empty_identity_and_negative_time() -> None:
    fixture = build_builtin_fixtures()[0]
    with pytest.raises(TraceValidationError):
        TraceCollector("", fixture.provenance)
    with pytest.raises(TraceValidationError):
        TraceCollector("fixture", fixture.provenance, initial_time_ns=-1)


def test_trace_collector_rejects_duplicate_window_identity_and_world_size_drift() -> None:
    fixture = build_builtin_fixtures()[0]
    source = fixture.windows[0]
    collector = TraceCollector("fixture", fixture.provenance)
    kwargs = dict(
        window_id=source.window_id,
        layer_id=source.layer_id,
        request_id=source.request_id,
        decode_step=source.decode_step,
        is_bootstrap_p0=True,
        mapping=source.mapping,
        routing=source.routing,
        local_compute=source.local_compute,
        dispatch_payload_spec=source.dispatch_payload_spec,
        combine_payload_spec=source.combine_payload_spec,
        descriptor_metadata_spec=source.descriptor_metadata_spec,
        metadata=source.metadata,
    )
    collector.record_window(**kwargs)
    with pytest.raises(TraceValidationError):
        collector.record_window(**{**kwargs, "is_bootstrap_p0": False})
