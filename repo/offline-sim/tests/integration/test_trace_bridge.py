from __future__ import annotations

from pathlib import Path

from rs_sim import PhaseKind
from rs_sim.runtime import keys_for_trace_window, payload_bytes_for_phase
from rs_sim.trace.io.serialization import load_fixture


def test_trace_fixture_maps_to_canonical_keys_and_payloads() -> None:
    fixture = load_fixture(Path("fixtures/trace/ep4_train_balanced.json"))
    window = fixture.windows[0]
    keys = keys_for_trace_window(run_id="trace-run", trace_window=window)
    assert keys.dispatch_phase_key.phase_kind is PhaseKind.DISPATCH
    assert keys.combine_phase_key.phase_kind is PhaseKind.COMBINE
    dispatch = payload_bytes_for_phase(window, keys.dispatch_phase_key)
    combine = payload_bytes_for_phase(window, keys.combine_phase_key)
    assert len(dispatch) == fixture.world_size
    assert len(combine) == fixture.world_size
    assert all(len(row) == fixture.world_size for row in dispatch)
