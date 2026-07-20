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
    assert config.online_policy.name == "fifo_bucket"


def test_online_policy_correctness_accepts_prepared_priority_with_explicit_planner() -> None:
    config = load_run_config(
        config_path=ROOT / "configs/experiment/online_policy_correctness_local_2gpu.yaml",
        overrides=[
            "online_policy.name=prepared_priority",
            "online_policy.parameters.planner_id=future:p012:joint:global:rscf",
            "execution.mode=joint_window_async_p2p",
            "online_policy.p2.mode=calibrated_artifact",
            "online_policy.parameters.p2_hint_weight=1.0",
        ],
    )
    assert config.execution.mode == "joint_window_async_p2p"
    assert config.online_policy.name == "prepared_priority"
    assert config.online_policy.parameters.planner_id == "future:p012:joint:global:rscf"


def test_prepared_priority_requires_explicit_planner_id() -> None:
    with pytest.raises(ValueError, match="requires online_policy.parameters.planner_id"):
        load_run_config(
            config_path=ROOT / "configs/experiment/online_policy_correctness_local_2gpu.yaml",
            overrides=[
                "online_policy.name=prepared_priority",
                "execution.mode=joint_window_async_p2p",
                "online_policy.p2.mode=calibrated_artifact",
                "online_policy.parameters.p2_hint_weight=1.0",
            ],
        )


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
            overrides=["online_policy.name=fifo_bucket"],
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


def test_string_bool_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad-bool.yaml"
    path.write_text(
        """
run:
  kind: online_observe
  name: bad-bool
model:
  config: configs/model/olmoe_1b_7b_instruct.yaml
topology:
  launcher:
    kind: torchrun
    nproc_per_node: 2
  ep_size: 2
runtime:
  precision: bf16
online_policy:
  name: disabled
execution:
  mode: native_passthrough
observation:
  profile: debug
  capture_expert_trace: "false"
validation:
  save_logits: false
artifact:
  artifact_root: artifacts/test
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="capture_expert_trace must be a boolean"):
        load_run_config(config_path=path)


def test_string_int_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad-int.yaml"
    path.write_text(
        """
run:
  kind: online_policy_correctness
  name: bad-int
model:
  model_id: fixture/model
topology:
  launcher:
    kind: torchrun
    nproc_per_node: "2"
  ep_size: 2
runtime:
  precision: bf16
  control_mode: sync_before_phase
online_policy:
  name: fifo_bucket
execution:
  mode: phase_sync_wave
observation:
  profile: execution
validation:
  save_logits: false
artifact:
  artifact_root: artifacts/test
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nproc_per_node must be an integer"):
        load_run_config(config_path=path)


def test_execution_schedule_selected_layer_ids_are_loaded(tmp_path: Path) -> None:
    path = tmp_path / "selected-layer-config.yaml"
    path.write_text(
        """
run:
  kind: online_policy_correctness
  name: selected-layer-config
model:
  model_id: fixture/model
topology:
  launcher:
    kind: torchrun
    nproc_per_node: 2
  ep_size: 2
runtime:
  precision: bf16
  control_mode: sync_before_phase
online_policy:
  name: fifo_bucket
execution:
  mode: phase_sync_wave
  schedule:
    layer_selector: selected
    phase_selector: both
    selected_layer_ids: ["0", "1"]
observation:
  profile: execution
validation:
  save_logits: false
artifact:
  artifact_root: artifacts/test
""".strip(),
        encoding="utf-8",
    )
    config = load_run_config(config_path=path)
    assert config.execution.schedule.layer_selector == "selected"
    assert config.execution.schedule.selected_layer_ids == ("0", "1")


def test_capture_expert_trace_is_preserved_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "capture-expert-trace.yaml"
    path.write_text(
        """
run:
  kind: online_observe
  name: capture-expert-trace
model:
  config: configs/model/olmoe_1b_7b_instruct.yaml
topology:
  launcher:
    kind: torchrun
    nproc_per_node: 2
  ep_size: 2
runtime:
  precision: bf16
online_policy:
  name: disabled
execution:
  mode: native_passthrough
observation:
  profile: debug
  capture_expert_trace: true
validation:
  save_logits: false
artifact:
  artifact_root: artifacts/test
""".strip(),
        encoding="utf-8",
    )
    config = load_run_config(config_path=path)
    assert config.observation.profile == "debug"
    assert config.observation.capture_expert_trace is True
