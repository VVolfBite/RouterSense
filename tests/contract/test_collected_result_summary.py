from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _inventory(tmp_path: Path) -> Path:
    path = tmp_path / "inventory.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "cluster_name": "summary-test",
                "nodes": [
                    {
                        "name": "node0",
                        "host": "127.0.0.1",
                        "port": 22,
                        "ssh_user": "user",
                        "node_rank": 0,
                        "current_gpu_count": 2,
                        "target_gpu_count": 2,
                        "paths": {
                            "remote_rs_root": "/tmp/rs",
                            "model_cache": "/tmp/models",
                            "artifact_root": "/tmp/artifacts",
                        },
                    },
                    {
                        "name": "node1",
                        "host": "127.0.0.1",
                        "port": 22,
                        "ssh_user": "user",
                        "node_rank": 1,
                        "current_gpu_count": 2,
                        "target_gpu_count": 2,
                        "paths": {
                            "remote_rs_root": "/tmp/rs",
                            "model_cache": "/tmp/models",
                            "artifact_root": "/tmp/artifacts",
                        },
                    },
                ],
                "rendezvous": {"master_node": "node0", "master_port": 29500, "backend": "c10d"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _run(inventory: Path, collected: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [
            "python",
            "scripts/deploy/summarize_collected_run.py",
            str(inventory),
            "--run-id",
            "run",
            "--input-dir",
            str(collected),
            "--apply",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _successful_tree(root: Path) -> None:
    for node in ("node0", "node1"):
        node_root = root / node / "logs"
        node_root.mkdir(parents=True)
        (node_root / f"{node}.exit").write_text("0\n", encoding="utf-8")
    result = root / "node0" / "per_strategy" / "rscf" / "run" / "summary.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"status": "ready"}), encoding="utf-8")


def test_collected_result_summary_accepts_complete_success(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    collected = tmp_path / "collected"
    _successful_tree(collected)
    completed = _run(inventory, collected)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["result_ready"] is True


def test_collected_result_summary_rejects_runtime_failure_status(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    collected = tmp_path / "collected"
    _successful_tree(collected)
    result = collected / "node0" / "per_strategy" / "rscf" / "run" / "summary.json"
    result.write_text(json.dumps({"status": "runtime_failure"}), encoding="utf-8")
    completed = _run(inventory, collected)
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "FAIL"
    assert any("explicit failure" in item for item in payload["failures"])


def test_collected_result_summary_rejects_missing_master_result(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    collected = tmp_path / "collected"
    for node in ("node0", "node1"):
        node_root = collected / node / "logs"
        node_root.mkdir(parents=True)
        (node_root / f"{node}.exit").write_text("0\n", encoding="utf-8")
    completed = _run(inventory, collected)
    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert "rendezvous master result_bundle/summary/comparison_report missing" in payload["failures"]
