from __future__ import annotations

from pathlib import Path


def test_refactor_structure_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src/rs/contracts/result.py",
        root / "src/rs/offline/router_prediction/metadata.py",
        root / "src/rs/online/olmoe_ep/runtime.py",
        root / "src/rs/legacy/trace_replay/metadata.py",
        root / "src/rs/scheduler/oracle.py",
        root / "src/rs/scheduler/greedy.py",
        root / "src/rs/scheduler/fast.py",
        root / "src/rs/evaluation/cross_layer.py",
        root / "src/rs/evaluation/traffic_matrix.py",
        root / "src/rs/evaluation/analysis.py",
        root / "scripts",
        root / "channel/ins/.gitkeep",
        root / "channel/reply/.gitkeep",
        root / "experiments/poc_line1/exp_trace.py",
        root / "experiments/poc_line1/exp_cross_layer.py",
        root / "experiments/poc_line1/exp_oracle.py",
        root / "experiments/poc_line1/exp_pairwise.py",
        root / "experiments/offline/exp_router_prediction.py",
        root / "experiments/offline/_bootstrap.py",
        root / "experiments/offline/fit_ep_cost_model.py",
        root / "experiments/offline/exp_calibrated_schedule.py",
        root / "experiments/online/collect_native_ep_trace.py",
        root / "experiments/online/_bootstrap.py",
        root / "experiments/online/bench_native_ep.py",
        root / "experiments/online/bench_scheduled_ep.py",
        root / "experiments/legacy/_bootstrap.py",
        root / "experiments/legacy/exp_trace_replay.py",
    ]
    for path in required:
        assert path.exists(), path


def test_old_local_test_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src/rs/runtime/local_test").exists()
