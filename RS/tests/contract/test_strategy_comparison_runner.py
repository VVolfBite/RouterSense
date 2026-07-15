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


def test_strategy_comparison_dry_run_legacy_config_still_works(tmp_path: Path) -> None:
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"path": "/tmp/model", "model_id": "test-model", "trust_remote_code": False},
                "topology": {"ep_size": 2, "launcher": {"kind": "torchrun"}},
                "runtime": {"precision": "fp16", "dispatcher": "alltoall"},
                "workload": {"prompts": "configs/workload/smoke_prompts.json"},
                "observation": {
                    "profile": "debug",
                    "capture_enabled": True,
                    "capture_layer_selector": "1",
                    "capture_phase_selector": "P0",
                    "replay_trace_enabled": True,
                },
                "validation": {"save_logits": True, "stop_after_selected_layer": True},
                "strategies": [
                    {
                        "name": "disabled",
                        "description": "baseline",
                        "family": "native",
                        "policy": "",
                        "execution_mode": "native_passthrough",
                        "control_mode": "default_continue",
                        "p2_hint_mode": "none",
                        "calibrated_p2": False,
                    },
                    {
                        "name": "routersense_p0p1p2_hint",
                        "description": "candidate",
                        "family": "semantic",
                        "policy": "routersense_p0p1p2_hint",
                        "execution_mode": "phase_sync_wave",
                        "control_mode": "sync_before_phase",
                        "p2_hint_mode": "calibrated_artifact",
                        "calibrated_p2": True,
                    },
                ],
                "execution": {
                    "repetitions": 2,
                    "bucket_rows": 16,
                    "p0_weight": 1.0,
                    "p1_reservation_weight": 1.0,
                    "p2_hint_weight": 1.0,
                    "schedule_layer_selector": "1",
                    "schedule_phase_selector": "both",
                },
                "comparison": {"baseline_strategy": "disabled"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    _run_dry(config_path, output_dir)

    report = json.loads((output_dir / "comparison_report.json").read_text(encoding="utf-8"))
    assert report["baseline"] == "disabled"
    assert len(report["strategies"]) == 2
    disabled_cmd = (output_dir / "per_strategy" / "disabled" / "rep0" / "command.txt").read_text(encoding="utf-8")
    hint_cmd = (output_dir / "per_strategy" / "routersense_p0p1p2_hint" / "rep1" / "command.txt").read_text(encoding="utf-8")
    assert "experiments.online.collect_native_ep_trace" in disabled_cmd
    assert "routersense_p0p1p2_hint_rep1.yaml" in hint_cmd
    generated = yaml.safe_load((output_dir / "generated_configs" / "routersense_p0p1p2_hint_rep0.yaml").read_text(encoding="utf-8"))
    assert generated["online_policy"]["name"] == "routersense_p0p1p2_hint"
    assert generated["online_policy"]["p2"]["mode"] == "calibrated_artifact"
    assert generated["execution"]["bucket_rows"] == 16
    assert generated["observation"]["profile"] == "debug"
    assert generated["observation"]["capture_enabled"] is True
    assert generated["observation"]["replay_trace_enabled"] is True


def test_strategy_comparison_dry_run_public_config_maps_phase_sync(tmp_path: Path) -> None:
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"path": "/tmp/model", "model_id": "test-model", "trust_remote_code": False},
                "topology": {"ep_size": 4, "launcher": {"kind": "torchrun"}},
                "runtime": {
                    "line": "phase_sync",
                    "output_mode": "paper",
                    "precision": "fp16",
                    "dispatcher": "alltoall",
                },
                "workload": {"prompts": "configs/workload/comparison_256x128_prompts.json"},
                "validation": {"save_logits": False, "stop_after_selected_layer": False},
                "strategies": [
                    {"name": "disabled"},
                    {"name": "birkhoff_phase_local"},
                    {"name": "routersense_p0p1p2_hint"},
                ],
                "execution": {
                    "repetitions": 1,
                    "p0_weight": 1.0,
                    "p1_reservation_weight": 1.0,
                    "p2_hint_weight": 1.0,
                    "schedule_layer_selector": "all",
                    "schedule_phase_selector": "both",
                },
                "comparison": {"baseline_strategy": "disabled"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    _run_dry(config_path, output_dir)

    disabled_cmd = (output_dir / "per_strategy" / "disabled" / "rep0" / "command.txt").read_text(encoding="utf-8")
    birkhoff_cmd = (output_dir / "per_strategy" / "birkhoff_phase_local" / "rep0" / "command.txt").read_text(encoding="utf-8")
    hint_cmd = (output_dir / "per_strategy" / "routersense_p0p1p2_hint" / "rep0" / "command.txt").read_text(encoding="utf-8")
    assert "experiments.online.collect_native_ep_trace" in disabled_cmd
    assert "experiments.online.run_policy_correctness" in birkhoff_cmd
    assert "experiments.online.run_policy_correctness" in hint_cmd

    disabled_cfg = yaml.safe_load((output_dir / "generated_configs" / "disabled_rep0.yaml").read_text(encoding="utf-8"))
    birkhoff_cfg = yaml.safe_load((output_dir / "generated_configs" / "birkhoff_phase_local_rep0.yaml").read_text(encoding="utf-8"))
    hint_cfg = yaml.safe_load((output_dir / "generated_configs" / "routersense_p0p1p2_hint_rep0.yaml").read_text(encoding="utf-8"))

    assert disabled_cfg["runtime"]["control_mode"] == "none"
    assert disabled_cfg["execution"]["mode"] == "native_passthrough"
    assert disabled_cfg["observation"]["profile"] == "perf"
    assert disabled_cfg["observation"]["replay_trace_enabled"] is False
    assert disabled_cfg["execution"]["bucket_rows"] == 0

    assert birkhoff_cfg["online_policy"]["name"] == "birkhoff_phase_local"
    assert birkhoff_cfg["runtime"]["control_mode"] == "sync_before_phase"
    assert birkhoff_cfg["execution"]["mode"] == "phase_sync_wave"
    assert birkhoff_cfg["online_policy"]["p2"]["mode"] == "none"

    assert hint_cfg["online_policy"]["name"] == "routersense_p0p1p2_hint"
    assert hint_cfg["runtime"]["control_mode"] == "sync_before_phase"
    assert hint_cfg["execution"]["mode"] == "multiphase_pending_window"
    assert hint_cfg["online_policy"]["p2"]["mode"] == "calibrated_artifact"


def test_strategy_comparison_dry_run_public_debug_replay_maps_trace_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"path": "/tmp/model"},
                "topology": {"ep_size": 2, "launcher": {"kind": "torchrun"}},
                "runtime": {
                    "line": "phase_sync",
                    "output_mode": "debug_replay",
                    "precision": "fp16",
                    "dispatcher": "alltoall",
                },
                "workload": {"prompts": "configs/workload/smoke_prompts.json"},
                "strategies": [{"name": "routersense_p0p1p2_hint"}],
                "execution": {
                    "repetitions": 1,
                    "p0_weight": 1.0,
                    "p1_reservation_weight": 1.0,
                    "p2_hint_weight": 1.0,
                    "schedule_layer_selector": "all",
                    "schedule_phase_selector": "both",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    _run_dry(config_path, output_dir)
    generated = yaml.safe_load((output_dir / "generated_configs" / "routersense_p0p1p2_hint_rep0.yaml").read_text(encoding="utf-8"))
    assert generated["observation"]["profile"] == "debug"
    assert generated["observation"]["replay_trace_enabled"] is True
    assert generated["observation"]["capture_enabled"] is False


def test_async_release_public_runtime_line_dry_run_passes(tmp_path: Path) -> None:
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"path": "/tmp/model"},
                "topology": {"ep_size": 2, "launcher": {"kind": "torchrun"}},
                "runtime": {
                    "line": "async_release",
                    "output_mode": "paper",
                    "precision": "fp16",
                    "dispatcher": "alltoall",
                },
                "workload": {"prompts": "configs/workload/smoke_prompts.json"},
                "strategies": [{"name": "birkhoff_phase_local_async_p2p"}],
                "execution": {"repetitions": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    proc = subprocess.run(
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
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    generated = yaml.safe_load((output_dir / "generated_configs" / "birkhoff_phase_local_async_p2p_rep0.yaml").read_text(encoding="utf-8"))
    assert generated["execution"]["mode"] == "joint_window_async_p2p"
    assert generated["runtime"]["control_mode"] == "sync_before_phase"


def test_recommended_config_does_not_expose_legacy_low_level_fields() -> None:
    payload = yaml.safe_load(
        Path(REPO_ROOT, "configs/comparison/natural_256x128_4gpu.yaml").read_text(encoding="utf-8")
    )
    runtime = payload.get("runtime", {}) or {}
    execution = payload.get("execution", {}) or {}
    observation = payload.get("observation", {}) or {}
    assert runtime["line"] == "phase_sync"
    assert runtime["output_mode"] == "paper"
    assert "control_mode" not in runtime
    assert "bucket_rows" not in execution
    assert "profile" not in observation
    for strategy in payload.get("strategies", []):
        assert set(strategy.keys()) == {"name"}


def test_child_env_normalizes_invalid_omp_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.online.run_strategy_comparison import _child_env

    monkeypatch.setenv("OMP_NUM_THREADS", "")
    env = _child_env()
    assert env["OMP_NUM_THREADS"] == "1"

    monkeypatch.setenv("OMP_NUM_THREADS", "abc")
    env = _child_env()
    assert env["OMP_NUM_THREADS"] == "1"

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    env = _child_env()
    assert env["OMP_NUM_THREADS"] == "4"
