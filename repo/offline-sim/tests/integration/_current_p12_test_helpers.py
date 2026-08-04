from __future__ import annotations

from dataclasses import replace

from rs_sim.runtime import build_current_p12_integration_runtime
from rs_sim.scheduler.prediction.fate_p2 import FATE_METADATA_KEY, canonical_fate_metadata
from rs_sim.trace import build_golden_fixture
from tests.support.runtime_profiles import synthetic_runtime_profile


def fixture_with_fate():
    fixture = build_golden_fixture()
    windows = list(fixture.windows)
    for index in range(len(windows) - 1):
        current, nxt = windows[index], windows[index + 1]
        metadata = dict(current.metadata)
        metadata[FATE_METADATA_KEY] = canonical_fate_metadata(
            predictor_id="fate-integration-test",
            source_layer_id=current.layer_id,
            target_layer_id=nxt.layer_id,
            confidence_ppm=900_000,
            routing_rows=tuple(tuple(int(value) for value in row) for row in nxt.dispatch_rows),
            estimator_kind="CROSS_LAYER_GATE_TEST",
        )
        windows[index] = replace(current, metadata=metadata)
    return replace(fixture, windows=tuple(windows))


def run_current_p12(**kwargs):
    kwargs.setdefault("max_task_bytes", 1 << 20)
    kwargs.setdefault("release_mode", "PHASE_BARRIER")
    runtime = build_current_p12_integration_runtime(
        fixture_input=fixture_with_fate(),
        run_id=kwargs.pop("run_id", "current-p12-test"),
        staging_sensitivity="0.25X",
        runtime_profile=kwargs.pop(
            "runtime_profile", synthetic_runtime_profile(local_assembly_latency_ns=5)
        ),
        **kwargs,
    )
    runtime.run_to_completion(max_timestamps=20_000)
    runtime.assert_terminal()
    return runtime


def phase_start_order(runtime, phase_key):
    timeline = runtime.data_plane.formal_runtime_metrics()["statistics"]["transfer_timeline"]
    rows = sorted(
        (item for item in timeline if item[1] == phase_key),
        key=lambda item: (int(item[5]), str(item[0])),
    )
    return tuple(str(item[0]) for item in rows)


def phase_first_start_and_last_complete(runtime, phase_key):
    timeline = runtime.data_plane.formal_runtime_metrics()["statistics"]["transfer_timeline"]
    rows = tuple(item for item in timeline if item[1] == phase_key)
    return min(int(item[5]) for item in rows), max(int(item[7]) for item in rows)
