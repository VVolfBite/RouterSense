from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from rs.experiments_support.deployed_strategy import prepare_deployed_strategy


ROOT = Path(__file__).resolve().parents[2]


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
