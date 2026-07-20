from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from rs.core.experiment_config import load_run_config
from rs.experiments_support.deployed_strategy import prepare_deployed_strategy
from rs.topology import (
    infer_model_row_contract,
    resolve_runtime_link_cost_profile,
    write_link_cost_profile,
)


ROOT = Path(__file__).resolve().parents[2]


def test_policy_correctness_entrypoint_imports_runtime_link_cost_resolver() -> None:
    source = (ROOT / "experiments/online/run_policy_correctness.py").read_text(encoding="utf-8")
    assert "from rs.topology import resolve_runtime_link_cost_profile" in source


def test_prepare_deployed_strategy_expands_public_async_config(tmp_path: Path) -> None:
    payload = prepare_deployed_strategy(
        comparison_config=ROOT / "configs/official/online_p012_deploy_smoke.yaml",
        strategy_name="routersense_future_p012_joint_global_rscf_async",
        output_dir=tmp_path / "run",
        model_path="/models/olmoe",
    )
    generated = Path(str(payload["generated_config"]))
    assert generated.is_file()
    assert payload["run_kind"] == "online_policy_correctness"
    text = generated.read_text(encoding="utf-8")
    assert "planning_timing: previous_layer" in text
    assert "planner_id: future:p012:joint:global:rscf" in text
    model_text = Path(str(payload["generated_model_config"])).read_text(encoding="utf-8")
    assert "/models/olmoe" in model_text
    assert "allenai/OLMoE-1B-7B-0924-Instruct" in model_text
    assert "max_new_tokens: 32" in model_text
    topology_text = Path(str(payload["generated_topology_config"])).read_text(encoding="utf-8")
    assert "nproc_per_node: 4" in topology_text
    assert "size: 4" in topology_text


def test_deployed_strategy_cli_dry_run_is_rank_local(tmp_path: Path) -> None:
    env = dict(os.environ)
    env.update({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "4", "PYTHONPATH": str(ROOT / "src")})
    completed = subprocess.run(
        [
            "python",
            "-m",
            "experiments.online.run_deployed_strategy",
            "--comparison-config",
            "configs/official/online_phase_sync.yaml",
            "--strategy",
            "birkhoff_phase_local_sync",
            "--output-dir",
            str(tmp_path / "run"),
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["world_size"] == 4
    assert payload["run_kind"] == "online_policy_correctness"
    assert payload["dry_run"] is True


def test_prepare_deployed_strategy_uses_actual_multinode_layout_and_cost_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("RS_LINK_COST_PROFILE", "/remote/profile.json")
    payload = prepare_deployed_strategy(
        comparison_config=ROOT / "configs/official/online_p012_deploy_smoke.yaml",
        strategy_name="routersense_future_p012_joint_global_rscf_async",
        output_dir=tmp_path / "run",
        model_path="/models/olmoe",
    )
    topology_text = Path(str(payload["generated_topology_config"])).read_text(encoding="utf-8")
    assert "nnodes: 2" in topology_text
    assert "nproc_per_node: 2" in topology_text
    assert "size: 4" in topology_text
    assert "scope: multi_node" in topology_text
    assert "cost_profile: /remote/profile.json" in topology_text
    assert "require_cost_profile: true" in topology_text
    parsed = load_run_config(config_path=str(payload["generated_config"]))
    assert parsed.topology.launcher.nnodes == 2
    assert parsed.topology.launcher.nproc_per_node == 2
    assert parsed.topology.ep_size == 4
    assert parsed.topology.cost_profile == "/remote/profile.json"
    assert parsed.topology.require_cost_profile is True


def _write_test_link_profile(path: Path, *, model_path: Path, row_bytes: int | None = None) -> Path:
    model_contract = infer_model_row_contract(model_path, precision="fp16")
    effective_row_bytes = int(model_contract["row_bytes"] if row_bytes is None else row_bytes)
    write_link_cost_profile(
        path,
        {
            "world_size": 4,
            "ranks_per_node": 2,
            "rank_to_node": [0, 0, 1, 1],
            "row_bytes": effective_row_bytes,
            "edge_slope_us_per_row": [[1.0, 2.0, 8.0, 8.0] for _ in range(4)],
            "edge_intercept_us": [[0.0, 0.5, 4.0, 4.0] for _ in range(4)],
            "wave_launch_us": 0.0,
            "source": "contract-fixture",
            "metadata": {"model_contract": model_contract},
        },
    )
    return path


def test_runtime_accepts_matching_link_profile_and_rejects_model_row_mismatch(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"hidden_size": 128}', encoding="utf-8")
    profile_path = _write_test_link_profile(tmp_path / "profile.json", model_path=model_path)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("RS_LINK_COST_PROFILE", str(profile_path))
    payload = prepare_deployed_strategy(
        comparison_config=ROOT / "configs/official/online_p012_deploy_smoke.yaml",
        strategy_name="routersense_future_p012_joint_global_rscf_async",
        output_dir=tmp_path / "run",
        model_path=str(model_path),
    )
    config = load_run_config(config_path=str(payload["generated_config"]))
    planner_config, metadata = resolve_runtime_link_cost_profile(
        configured_path=config.topology.cost_profile,
        source_config_path=config.source_config_path,
        repository_root=ROOT,
        model_path=model_path,
        precision=config.runtime.precision,
        world_size=4,
        local_world_size=2,
        require_profile=config.topology.require_cost_profile,
    )
    assert planner_config["cost_profile_id"]
    assert planner_config["ranks_per_node"] == 2
    assert metadata["mode"] == "measured_pairwise"
    assert metadata["row_bytes"] == 256

    bad_profile = _write_test_link_profile(
        tmp_path / "bad-profile.json",
        model_path=model_path,
        row_bytes=512,
    )
    monkeypatch.setenv("RS_LINK_COST_PROFILE", str(bad_profile))
    bad_payload = prepare_deployed_strategy(
        comparison_config=ROOT / "configs/official/online_p012_deploy_smoke.yaml",
        strategy_name="routersense_future_p012_joint_global_rscf_async",
        output_dir=tmp_path / "bad-run",
        model_path=str(model_path),
    )
    bad_config = load_run_config(config_path=str(bad_payload["generated_config"]))
    with pytest.raises(RuntimeError, match="row_bytes"):
        resolve_runtime_link_cost_profile(
            configured_path=bad_config.topology.cost_profile,
            source_config_path=bad_config.source_config_path,
            repository_root=ROOT,
            model_path=model_path,
            precision=bad_config.runtime.precision,
            world_size=4,
            local_world_size=2,
            require_profile=bad_config.topology.require_cost_profile,
        )
