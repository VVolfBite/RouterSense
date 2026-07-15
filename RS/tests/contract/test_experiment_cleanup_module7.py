from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

from rs.core.config_normalization import normalize_run_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_official_entrypoints_help() -> None:
    for script in (
        "experiments/run_offline_replay.py",
        "experiments/run_online_phase_sync.py",
        "experiments/run_online_async_release.py",
        "experiments/dev/run_validation.py",
        "experiments/reporting/build_report.py",
    ):
        proc = _run(script, "--help")
        assert proc.returncode == 0, proc.stderr + proc.stdout


def test_official_configs_normalize_and_components_exist() -> None:
    for path in (
        "configs/official/offline_replay.yaml",
        "configs/official/online_phase_sync.yaml",
        "configs/official/online_async_release.yaml",
        "configs/official/gpu_c2_correctness.yaml",
        "configs/official/gpu_a2_performance.yaml",
    ):
        config_path = REPO_ROOT / path
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        normalized = normalize_run_config(payload, source_path=config_path)
        assert normalized.schema_version == 1
        if path != "configs/official/offline_replay.yaml":
            assert normalized.model.get("local_path")
            assert normalized.topology.get("launcher")
    assert (REPO_ROOT / "configs/components/models/olmoe_1b_7b_instruct.yaml").exists()
    assert (REPO_ROOT / "configs/components/topologies/local_4gpu.yaml").exists()
    assert (REPO_ROOT / "configs/components/workloads/comparison_64_prompts.json").exists()


def test_validation_entry_dry_run_suites() -> None:
    for suite in ("b2", "c2", "a2"):
        proc = _run("experiments/dev/run_validation.py", "--suite", suite)
        assert proc.returncode == 0, proc.stderr + proc.stdout


def test_validation_entry_execute_removes_dry_run(monkeypatch) -> None:
    from experiments.dev import run_validation

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd=None, env=None, check=False):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["env"] = dict(env or {})
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_validation.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["run_validation.py", "--suite", "b2", "--execute"])
    try:
        run_validation.main()
    except SystemExit as exc:
        assert int(exc.code or 0) == 0
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--dry-run" not in cmd


def test_validation_entry_execute_after_suite_is_not_swallowed(monkeypatch) -> None:
    from experiments.dev import run_validation

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd=None, env=None, check=False):
        captured["cmd"] = list(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_validation.subprocess, "run", _fake_run)
    monkeypatch.setattr(sys, "argv", ["run_validation.py", "--suite", "c2", "--execute"])
    try:
        run_validation.main()
    except SystemExit as exc:
        assert int(exc.code or 0) == 0
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--dry-run" not in cmd


def test_unified_report_builder_reads_structured_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "offline_run"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "reports").mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "offline_run",
                "run_type": "offline",
                "status": "completed",
                "artifact_schema_version": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics" / "summary.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"canonical_policy_name": "fifo_bucket", "hint_type": "zero_hint"},
                    {"canonical_policy_name": "birkhoff_bucket_phase_local", "hint_type": "copy_current_dispatch"},
                ],
                "invariants": [{"fixture_id": "fixture"}],
            }
        ),
        encoding="utf-8",
    )
    proc = _run("experiments/reporting/build_report.py", "--input", str(run_dir), "--report-type", "offline")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    report = json.loads((run_dir / "reports" / "offline.json").read_text(encoding="utf-8"))
    assert report["row_count"] == 2
    assert report["invariant_count"] == 1
