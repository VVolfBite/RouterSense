#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "validation_source"

IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", ".pytest_cache", "*.pyc", "*.pyo", "*.nbc", "*.nbi",
    "*.egg-info", "outputs", "models", "checkpoints", "traces", "wandb", "archive",
)


def _run(argv: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    completed = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {argv}; see {log}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble and run the RouterSense two-node one-command suite")
    parser.add_argument("--formal-repo", type=Path, required=True, help="RouterSense repository containing RS/src/rs")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=ROOT / "server_runs")
    parser.add_argument("--execute", action="store_true", help="Actually SSH to the cluster; default is a full dry-run")
    parser.add_argument("--skip-local-tests", action="store_true")
    args = parser.parse_args()

    formal = args.formal_repo.expanduser().resolve()
    if not (formal / "RS" / "src" / "rs").is_dir():
        raise SystemExit(f"formal repository is missing RS/src/rs: {formal}")
    if not args.experiment.is_file() or not args.deployment.is_file():
        raise SystemExit("experiment/deployment config is missing")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workspace = args.work_root.expanduser().resolve() / f"workspace-{stamp}"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(DRIVER, workspace, ignore=IGNORE)
    shutil.copytree(formal / "RS", workspace / "RS", ignore=IGNORE)

    logs = workspace / "controller_logs"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workspace / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["ROUTERSENSE_REPO_ROOT"] = str(workspace)

    _run([sys.executable, str(ROOT / "tools" / "apply_overlay.py"), str(workspace)], cwd=ROOT, env=env, log=logs / "apply_overlay.log")
    _run([sys.executable, str(ROOT / "tools" / "verify_applied_tree.py"), str(workspace),
          "--output", str(workspace / "integration_verify.json")], cwd=ROOT, env=env, log=logs / "verify_overlay.log")

    if not args.skip_local_tests:
        _run([sys.executable, "-m", "pytest", "-q", "tests/test_deploy_dryrun.py", "tests/test_pipeline_config.py"],
             cwd=workspace, env=env, log=logs / "controller_driver_tests.log")
        _run([sys.executable, "-m", "routersense_sched.cli", "doctor", "--repo-root", str(workspace),
              "--experiment", str(args.experiment.resolve()), "--deployment", str(args.deployment.resolve())],
             cwd=workspace, env=env, log=logs / "doctor.log")
        _run([sys.executable, "-m", "routersense_sched.cli", "validate", "--repo-root", str(workspace)],
             cwd=workspace, env=env, log=logs / "planner_validate.log")

    command = [
        sys.executable, "-m", "routersense_sched.cli", "run",
        "--repo-root", str(workspace),
        "--experiment", str(args.experiment.resolve()),
        "--deployment", str(args.deployment.resolve()),
        "--mode", "deploy-all",
    ]
    if args.execute:
        command.append("--execute")
    completed = subprocess.run(command, cwd=workspace, env=env, text=True, capture_output=True)
    (logs / "deploy_all.log").write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
    payload = {
        "schema_version": "routersense.server.one_click.v1",
        "workspace": str(workspace),
        "formal_repo": str(formal),
        "experiment": str(args.experiment.resolve()),
        "deployment": str(args.deployment.resolve()),
        "execute": bool(args.execute),
        "returncode": completed.returncode,
        "deploy_log": str(logs / "deploy_all.log"),
    }
    (workspace / "ONE_CLICK_RESULT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
