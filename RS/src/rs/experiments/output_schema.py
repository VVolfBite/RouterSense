from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    raw_dir: Path
    metrics_dir: Path
    reports_dir: Path
    logs_dir: Path
    failures_dir: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_output(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def detect_git_state(repo_root: Path) -> tuple[str, bool]:
    sha = _git_output(repo_root, "rev-parse", "HEAD")
    dirty = bool(_git_output(repo_root, "status", "--short"))
    return sha, dirty


def build_output_layout(output_dir: Path) -> OutputLayout:
    root = output_dir.resolve()
    raw_dir = root / "raw"
    metrics_dir = root / "metrics"
    reports_dir = root / "reports"
    logs_dir = root / "logs"
    failures_dir = root / "failures"
    for path in (root, raw_dir, metrics_dir, reports_dir, logs_dir, failures_dir):
        path.mkdir(parents=True, exist_ok=True)
    return OutputLayout(
        root=root,
        raw_dir=raw_dir,
        metrics_dir=metrics_dir,
        reports_dir=reports_dir,
        logs_dir=logs_dir,
        failures_dir=failures_dir,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def capture_environment() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "hostname": socket.gethostname(),
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "captured_at": _utc_now(),
    }


def initialize_run_artifacts(
    *,
    repo_root: Path,
    output_dir: Path,
    run_type: str,
    official_entrypoint: str,
    config_snapshot: dict[str, Any],
    manifest_overrides: dict[str, Any] | None = None,
) -> OutputLayout:
    layout = build_output_layout(output_dir)
    commit_sha, git_dirty = detect_git_state(repo_root)
    manifest: dict[str, Any] = {
        "run_id": layout.root.name,
        "run_type": str(run_type),
        "commit_sha": commit_sha,
        "git_dirty": bool(git_dirty),
        "config_schema_version": int(config_snapshot.get("schema_version", 0) or 0),
        "official_entrypoint": str(official_entrypoint),
        "runtime_line": str((config_snapshot.get("runtime", {}) or {}).get("line", "")),
        "policy_id": str((config_snapshot.get("policy", {}) or {}).get("name", "")),
        "predictor_id": str((config_snapshot.get("prediction", {}) or {}).get("name", "")),
        "bucket_rows": (config_snapshot.get("traffic", {}) or {}).get("bucket_rows", []),
        "model": config_snapshot.get("model", {}),
        "topology": config_snapshot.get("topology", {}),
        "workload": config_snapshot.get("workload", {}),
        "world_size": int((config_snapshot.get("topology", {}) or {}).get("world_size", (config_snapshot.get("topology", {}) or {}).get("ep_size", 1)) or 1),
        "start_time": _utc_now(),
        "end_time": "",
        "status": "running",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    write_json(layout.root / "manifest.json", manifest)
    write_json(layout.root / "environment.json", capture_environment())
    write_yaml(layout.root / "config_snapshot.yaml", config_snapshot)
    write_json(layout.root / "status.json", {"status": "running", "updated_at": _utc_now()})
    return layout


def update_status(layout: OutputLayout, *, status: str, extra: dict[str, Any] | None = None) -> None:
    manifest_path = layout.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = str(status)
    manifest["end_time"] = _utc_now()
    if extra:
        manifest.update(extra)
    write_json(manifest_path, manifest)
    status_payload: dict[str, Any] = {"status": str(status), "updated_at": _utc_now()}
    if extra:
        status_payload.update(extra)
    write_json(layout.root / "status.json", status_payload)

