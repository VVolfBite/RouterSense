from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_dry(config_path: Path, output_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.online.run_strategy_comparison",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )


def _base(*, line: str, output_mode: str, strategies: list[str]) -> dict:
    return {
        "model": {"path": "/tmp/model", "model_id": "test-model", "trust_remote_code": False},
        "topology": {"ep_size": 4, "launcher": {"kind": "torchrun"}},
        "runtime": {"line": line, "output_mode": output_mode, "precision": "bf16", "dispatcher": "alltoall"},
        "workload": {"prompts": "configs/workload/smoke_prompts.json"},
        "strategies": [{"name": name} for name in strategies],
        "execution": {"repetitions": 1},
        "comparison": {"baseline_strategy": strategies[0]},
    }


def test_phase_sync_public_surface_maps_baseline(tmp_path: Path) -> None:
    config = tmp_path / "phase-sync.yaml"
    config.write_text(yaml.safe_dump(_base(line="phase_sync", output_mode="paper", strategies=["native", "birkhoff_phase_local_sync"]), sort_keys=False), encoding="utf-8")
    out = tmp_path / "out"
    _run_dry(config, out)
    native = yaml.safe_load((out / "generated_configs/native_rep0.yaml").read_text(encoding="utf-8"))
    baseline = yaml.safe_load((out / "generated_configs/birkhoff_phase_local_sync_rep0.yaml").read_text(encoding="utf-8"))
    assert native["execution"]["mode"] == "native_passthrough"
    assert baseline["online_policy"]["name"] == "birkhoff_bucket_phase_local"
    assert baseline["execution"]["mode"] == "phase_sync_wave"


def test_future_axis_strategy_preserves_planner_and_prediction(tmp_path: Path) -> None:
    payload = _base(line="async_release", output_mode="paper", strategies=["routersense_future_p012_joint_global_rscf_async"])
    payload["prediction"] = {"name": "fate_cross_layer_gate", "config": {"second_hop_predictor_id": "bridge_copy_current"}}
    config = tmp_path / "future.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    out = tmp_path / "out"
    _run_dry(config, out)
    generated = yaml.safe_load((out / "generated_configs/routersense_future_p012_joint_global_rscf_async_rep0.yaml").read_text(encoding="utf-8"))
    params = generated["online_policy"]["parameters"]
    assert generated["online_policy"]["name"] == "prepared_priority"
    assert params["planner_id"] == "future:p012:joint:global:rscf"
    assert params["planning_timing"] == "previous_layer"
    assert params["online_p2_predictor"] == "fate_cross_layer_gate"
    assert generated["execution"]["mode"] == "joint_window_async_p2p"


def test_debug_replay_enables_trace_without_tensor_capture(tmp_path: Path) -> None:
    config = tmp_path / "debug.yaml"
    config.write_text(yaml.safe_dump(_base(line="async_release", output_mode="debug_replay", strategies=["routersense_current_p012_joint_event_rscf_async"]), sort_keys=False), encoding="utf-8")
    out = tmp_path / "out"
    _run_dry(config, out)
    generated = yaml.safe_load((out / "generated_configs/routersense_current_p012_joint_event_rscf_async_rep0.yaml").read_text(encoding="utf-8"))
    assert generated["observation"]["profile"] == "debug"
    assert generated["observation"]["replay_trace_enabled"] is True
    assert generated["observation"]["capture_enabled"] is False


def test_recommended_config_exposes_only_public_surface() -> None:
    payload = yaml.safe_load((REPO_ROOT / "configs/comparison/formal_runtime_smoke_4gpu.yaml").read_text(encoding="utf-8"))
    assert payload["runtime"]["line"] == "async_release"
    assert "control_mode" not in payload["runtime"]
    assert "bucket_rows" not in payload["execution"]
    for strategy in payload["strategies"]:
        assert set(strategy) == {"name"}


def test_child_env_normalizes_invalid_omp_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    from rs.experiments_support.strategy_comparison import child_env
    for value, expected in [("", "1"), ("abc", "1"), ("4", "4")]:
        monkeypatch.setenv("OMP_NUM_THREADS", value)
        assert child_env()["OMP_NUM_THREADS"] == expected


def test_strategy_comparison_read_summary_requires_result_bundle(tmp_path: Path) -> None:
    from rs.experiments_support.strategy_comparison_runner import read_summary
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"details": {"obsolete": True}}), encoding="utf-8")
    assert read_summary(run_dir) == {}
