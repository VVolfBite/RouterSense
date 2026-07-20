from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = "deploy/inventory/hosts.example.yaml"


def _run_with_args(script: str, *extra: str) -> dict[str, object]:
    env = dict(os.environ)
    env.pop("RSSH_PASSWORD", None)
    env.pop("SSHPASS", None)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        ["bash", script, INVENTORY, *extra],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    return json.loads(completed.stdout)


def _run(script: str) -> dict[str, object]:
    return _run_with_args(script)


def test_deploy_dry_runs_do_not_require_credentials_or_contact_hosts() -> None:
    sync = _run("scripts/deploy/sync_repo.sh")
    prepare = _run("scripts/deploy/prepare_cluster_environment.sh")
    parity = _run("scripts/deploy/verify_repo_parity.sh")

    assert sync["apply_mode"] is False
    assert all(row["status"] == "DRY_RUN" for row in sync["targets"])
    assert sync["schema_version"] == "routersense.deploy.repo_sync.v2"
    assert all(str(row["remote_root"]).endswith("/RouterSense") for row in sync["targets"])
    assert all("deploy/inventory/hosts.example.yaml" in str(row["inventory_target"]) for row in sync["targets"])
    assert prepare["apply_mode"] is False
    assert all(row["status"] == "DRY_RUN" for row in prepare["targets"])
    assert parity["verification_status"] == "DRY_RUN"
    assert parity["REPO_PARITY_PASS"] is False
    assert all(row["status"] == "DRY_RUN" for row in parity["nodes"])


def test_launch_remote_renders_expected_two_node_four_rank_plan() -> None:
    payload = _run("scripts/deploy/launch_remote.sh")
    assert payload["dry_run"] is True
    assert payload["nnodes"] == 2
    assert payload["nproc_per_node"] == 2
    assert payload["world_size"] == 4
    assert payload["gpu_capacity_sufficient"] is True
    assert payload["status"] == "DRY_RUN"
    commands = "\n".join(row["command"] for row in payload["commands"])
    assert "experiments.online.run_deployed_strategy" in commands
    assert "formal dry-run entrypoint" not in commands
    assert all("set +e" in row["detached_command"] for row in payload["commands"])
    assert all(".exit" in row["detached_command"] for row in payload["commands"])


def test_model_prefetch_and_sync_are_safe_dry_runs() -> None:
    prefetch = _run("scripts/deploy/prefetch_model.sh")
    sync = _run("scripts/deploy/sync_model_cache.sh")
    mounted = _run("scripts/deploy/verify_mounted_model.sh")
    runtime_environment = _run("scripts/deploy/verify_runtime_environment.sh")
    calibration = _run("scripts/deploy/calibrate_cluster_links.sh")
    assert prefetch["apply_mode"] is False
    assert prefetch["status"] in {"READY", "DRY_RUN_MISSING"}
    assert prefetch["before"]["required_files_present"] is False
    assert sync["apply_mode"] is False
    assert sync["status"] == "DRY_RUN"
    assert all(row["status"] == "DRY_RUN" for row in sync["nodes"])
    assert mounted["status"] == "DRY_RUN"
    assert runtime_environment["status"] == "DRY_RUN"
    assert calibration["status"] == "DRY_RUN"
    assert calibration["world_size"] == 4


def test_collect_and_stop_entrypoints_are_safe_dry_runs() -> None:
    collect = _run_with_args("scripts/deploy/collect_remote_logs.sh", "--run-id", "dryrun")
    summary = _run_with_args("scripts/deploy/summarize_collected_run.sh", "--run-id", "dryrun")
    stop = _run_with_args("scripts/deploy/stop_remote_jobs.sh", "--run-id", "dryrun")
    assert collect["status"] == "DRY_RUN"
    assert summary["status"] == "DRY_RUN"
    assert stop["status"] == "DRY_RUN"
    assert all(row["status"] == "DRY_RUN" for row in collect["nodes"])
    assert all(row["status"] == "DRY_RUN" for row in stop["nodes"])


def test_access_probe_and_full_pipeline_are_safe_dry_runs(tmp_path: Path) -> None:
    access = _run("scripts/deploy/verify_cluster_access.sh")
    run_id = f"contract-{tmp_path.name}"
    pipeline = _run_with_args("scripts/deploy/run_deployment_pipeline.sh", "--run-id", run_id)

    assert access["apply_mode"] is False
    assert access["status"] == "DRY_RUN"
    assert all(row["status"] == "DRY_RUN" for row in access["nodes"])

    assert pipeline["apply_mode"] is False
    assert pipeline["status"] == "DRY_RUN_PASS"
    stage_names = [row["name"] for row in pipeline["stages"]]
    assert stage_names == [
        "access",
        "repo_sync",
        "repo_parity",
        "environment",
        "model_sync",
        "model_preflight",
        "runtime_environment_preflight",
        "link_calibration",
        "launch",
        "collect",
        "result_validation",
    ]
    assert all(row["status"] == "PASS" for row in pipeline["stages"])


def test_pipeline_rejects_ambiguous_link_profile_modes() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    missing_profile = subprocess.run(
        [
            "python",
            "scripts/deploy/run_deployment_pipeline.py",
            INVENTORY,
            "--skip-link-calibration",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert missing_profile.returncode != 0
    assert "requires --link-cost-profile" in (missing_profile.stdout + missing_profile.stderr)

    conflicting_profile = subprocess.run(
        [
            "python",
            "scripts/deploy/run_deployment_pipeline.py",
            INVENTORY,
            "--link-cost-profile",
            "outputs/existing-profile.json",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert conflicting_profile.returncode != 0
    assert "only valid with --skip-link-calibration" in (
        conflicting_profile.stdout + conflicting_profile.stderr
    )
