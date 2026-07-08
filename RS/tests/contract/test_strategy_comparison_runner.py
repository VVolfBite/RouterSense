from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_strategy_comparison_dry_run_generates_commands_and_report(tmp_path: Path) -> None:
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"path": "/tmp/model", "model_id": "test-model", "trust_remote_code": False},
                "topology": {"ep_size": 2, "launcher": {"kind": "torchrun"}},
                "runtime": {"precision": "fp16", "dispatcher": "alltoall"},
                "workload": {"prompts": "configs/workload/smoke_prompts.json"},
                "observation": {"profile": "debug", "capture_enabled": True, "capture_layer_selector": "1", "capture_phase_selector": "P0"},
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
        cwd="/root/autodl-tmp/RouterSense/RS",
        env={"PYTHONPATH": "src"},
    )

    report = json.loads((output_dir / "comparison_report.json").read_text(encoding="utf-8"))
    assert report["baseline"] == "disabled"
    assert len(report["strategies"]) == 2
    assert report["strategies"][0]["family"] == "native"
    assert report["strategies"][1]["family"] == "semantic"
    disabled_cmd = (output_dir / "per_strategy" / "disabled" / "rep0" / "command.txt").read_text(encoding="utf-8")
    hint_cmd = (output_dir / "per_strategy" / "routersense_p0p1p2_hint" / "rep1" / "command.txt").read_text(encoding="utf-8")
    assert "torchrun" in disabled_cmd
    assert "experiments.online.collect_native_ep_trace" in disabled_cmd
    assert "routersense_p0p1p2_hint_rep1.yaml" in hint_cmd
    generated = yaml.safe_load((output_dir / "generated_configs" / "routersense_p0p1p2_hint_rep0.yaml").read_text(encoding="utf-8"))
    assert generated["online_policy"]["name"] == "routersense_p0p1p2_hint"
    assert generated["online_policy"]["p2"]["mode"] == "calibrated_artifact"
    assert generated["execution"]["schedule"]["layer_selector"] == "1"
    assert generated["observation"]["profile"] == "debug"
    assert generated["observation"]["capture_enabled"] is True
    assert generated["validation"]["save_logits"] is True
    assert generated["validation"]["stop_after_selected_layer"] is True
    disabled_cfg = yaml.safe_load((output_dir / "generated_configs" / "disabled_rep0.yaml").read_text(encoding="utf-8"))
    assert disabled_cfg["run"]["kind"] == "online_observe"
    assert disabled_cfg["online_policy"]["name"] == "disabled"
    assert disabled_cfg["online_policy"]["parameters"]["p2_hint_weight"] == 0.0


def test_child_env_normalizes_invalid_omp_threads(monkeypatch) -> None:
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
