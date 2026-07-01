from __future__ import annotations

from pathlib import Path


def test_refactor_structure_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src/routesense/scheduler/oracle.py",
        root / "src/routesense/scheduler/greedy.py",
        root / "src/routesense/scheduler/fast.py",
        root / "src/routesense/evaluation/cross_layer.py",
        root / "src/routesense/evaluation/traffic_matrix.py",
        root / "src/routesense/evaluation/analysis.py",
        root / "scripts",
        root / "channel/ins/.gitkeep",
        root / "channel/reply/.gitkeep",
        root / "experiments/poc_line1/exp_trace.py",
        root / "experiments/poc_line1/exp_cross_layer.py",
        root / "experiments/poc_line1/exp_oracle.py",
        root / "experiments/poc_line1/exp_pairwise.py",
    ]
    for path in required:
        assert path.exists(), path


def test_old_local_test_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src/routesense/runtime/local_test").exists()
