from __future__ import annotations

from pathlib import Path

import pytest

from rs.core.experiment_config import build_launch_command, load_run_config


ROOT = Path(__file__).resolve().parents[1]


def test_load_offline_trace_config() -> None:
    config = load_run_config(config_path=ROOT / "configs/experiment/offline_trace_olmoe.yaml")
    assert config.run.kind == "offline_trace"
    assert config.model.model_id
    assert config.topology.launcher.kind == "python"


def test_load_online_policy_correctness_config() -> None:
    config = load_run_config(config_path=ROOT / "configs/experiment/online_policy_correctness_local_2gpu.yaml")
    assert config.run.kind == "online_policy_correctness"
    assert config.execution.mode == "phase_sync_wave"
    assert config.runtime.control_mode == "sync_before_phase"
    assert config.online_policy.name == "phase_barrier_fifo"


def test_online_policy_correctness_accepts_multiphase_pending_window_for_hint_policy() -> None:
    config = load_run_config(
        config_path=ROOT / "configs/experiment/online_policy_correctness_local_2gpu.yaml",
        overrides=[
            "online_policy.name=routersense_p0p1p2_hint",
            "execution.mode=multiphase_pending_window",
            "online_policy.p2.mode=calibrated_artifact",
            "online_policy.parameters.p2_hint_weight=1.0",
        ],
    )
    assert config.execution.mode == "multiphase_pending_window"
    assert config.online_policy.name == "routersense_p0p1p2_hint"


def test_build_launch_command_uses_torchrun_for_online() -> None:
    config = load_run_config(config_path=ROOT / "configs/experiment/online_observe_local_2gpu.yaml")
    command = build_launch_command(
        config=config,
        config_path=str(ROOT / "configs/experiment/online_observe_local_2gpu.yaml"),
    )
    assert command[0] == "torchrun"
    assert "experiments.online.collect_native_ep_trace" in command


def test_build_launch_command_uses_python_for_offline() -> None:
    config = load_run_config(config_path=ROOT / "configs/experiment/offline_trace_olmoe.yaml")
    command = build_launch_command(
        config=config,
        config_path=str(ROOT / "configs/experiment/offline_trace_olmoe.yaml"),
    )
    assert command[:2] == ["python", "-m"]
    assert "experiments.offline.collect_router_trace" in command


def test_online_observe_rejects_enabled_policy() -> None:
    with pytest.raises(ValueError):
        load_run_config(
            config_path=ROOT / "configs/experiment/online_observe_local_2gpu.yaml",
            overrides=["online_policy.name=bucketed_fifo"],
        )


def test_multiphase_pending_window_rejects_non_hint_policy() -> None:
    with pytest.raises(ValueError):
        load_run_config(
            config_path=ROOT / "configs/experiment/online_policy_correctness_local_2gpu.yaml",
            overrides=[
                "online_policy.name=greedy_ready_set",
                "execution.mode=multiphase_pending_window",
            ],
        )


def test_unknown_override_fails() -> None:
    with pytest.raises(ValueError):
        load_run_config(
            config_path=ROOT / "configs/experiment/offline_trace_olmoe.yaml",
            overrides=["runtime.not_a_field=1"],
        )


def test_unknown_yaml_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
run:
  kind: offline_trace
  name: bad
model:
  config: configs/model/olmoe_1b_7b_instruct.yaml
topology:
  launcher:
    kind: python
runtime:
  precision: bf16
  bad_field: 1
online_policy:
  name: disabled
offline_study:
  policies: []
execution:
  mode: native_passthrough
observation:
  profile: minimal
validation:
  save_logits: false
artifact:
  artifact_root: artifacts/test
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_run_config(config_path=path)
