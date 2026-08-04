from dataclasses import replace

from rs_sim.runtime import build_current_p12_integration_runtime
from rs_sim.scheduler.prediction.fate_p2 import FATE_METADATA_KEY, canonical_fate_metadata
from rs_sim.trace import build_golden_fixture


def _fixture_with_fate():
    fixture = build_golden_fixture()
    windows = list(fixture.windows)
    for index in range(len(windows) - 1):
        current, nxt = windows[index], windows[index + 1]
        metadata = dict(current.metadata)
        metadata[FATE_METADATA_KEY] = canonical_fate_metadata(
            predictor_id="fate-runtime-test",
            source_layer_id=current.layer_id,
            target_layer_id=nxt.layer_id,
            confidence_ppm=900_000,
            routing_rows=tuple(tuple(int(value) for value in row) for row in nxt.dispatch_rows),
            estimator_kind="CROSS_LAYER_GATE_TEST",
        )
        windows[index] = replace(current, metadata=metadata)
    return replace(fixture, windows=tuple(windows))


def test_fate_joint_runs_with_p1_completion_barrier() -> None:
    runtime = build_current_p12_integration_runtime(
        fixture_input=_fixture_with_fate(),
        run_id="fate-barrier-runtime-test",
        release_mode="PHASE_BARRIER",
        algorithm="joint(global_(rscf()))",
        information_mode="FATE_P2",
        max_task_bytes=262_144,
        max_window_prefix_tasks=64,
    )
    runtime.run_to_completion(max_timestamps=200_000)
    runtime.assert_terminal()
    records = runtime.current_p12_window_records()
    assert records
    assert runtime.run_axes["release_mode"] == "PHASE_BARRIER"
    assert runtime.run_axes["information_mode"] == "FATE_P2"
    assert all(item.prediction_generated for item in records)
    assert all(item.prediction_validated for item in records)
    assert all(item.prediction_consumed for item in records)
