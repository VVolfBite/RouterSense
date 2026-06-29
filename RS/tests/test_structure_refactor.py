from __future__ import annotations

from pathlib import Path


def test_runtime_local_test_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "routesense" / "runtime" / "local_test" / "inference.py").exists()


def test_prerun_entrypoints_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "experiments" / "prerun" / "single_gpu_smoke.py").exists()
    assert (root / "experiments" / "prerun" / "router_trace_smoke.py").exists()
    assert (root / "experiments" / "prerun" / "probe_architecture.py").exists()
    assert (root / "experiments" / "prerun" / "text_inference.py").exists()


def test_distributed_ep_split_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "routesense" / "runtime" / "distributed_ep" / "core" / "collective.py").exists()
    assert (root / "src" / "routesense" / "runtime" / "distributed_ep" / "adapter" / "olmoe_adapter.py").exists()


def test_top_level_scheduler_oracle_are_not_runtime_modules():
    root = Path(__file__).resolve().parents[1]
    scheduler_dir = root / "src" / "routesense" / "scheduler"
    oracle_dir = root / "src" / "routesense" / "oracle"
    assert not any(path.name != ".gitkeep" for path in scheduler_dir.glob("*"))
    assert not any(path.name != ".gitkeep" for path in oracle_dir.glob("*"))
