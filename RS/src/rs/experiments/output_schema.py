from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from rs.runtime.guards import InvariantContext, RouterSenseInvariantError, invariant_mode_allows_dirty_git, normalize_invariant_mode, require_invariant
from rs.scheduling.catalog import resolve_algorithm_id
from rs.scheduling.bucketizer import BUCKET_MODE_DYNAMIC_CURRENT, BUCKET_MODE_FIXED_ROWS


ARTIFACT_SCHEMA_VERSION = 1
_CANONICAL_ONLINE_PREDICTORS = {"none", "zero_hint", "copy_current_dispatch", "history_ema"}
_CANONICAL_OFFLINE_PREDICTORS = _CANONICAL_ONLINE_PREDICTORS | {"ridge_linear_trace_predictor", "perfect_trace_hint", "shuffled_control"}


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


def _manifest_sha(repo_root: Path) -> tuple[str, str]:
    manifest_path = repo_root / "handoff" / "manifest.json"
    if not manifest_path.is_file():
        return "", ""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return "", ""
    final_sha = str(payload.get("final_sha", "") or "").strip()
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return final_sha, digest


def resolve_commit_identity(*, repo_root: Path, run_plan_commit_sha: str = "") -> tuple[str, bool, str, str]:
    direct = str(run_plan_commit_sha or "").strip()
    if direct:
        return direct, False, "run_plan", ""
    env_sha = str(os.environ.get("ROUTERSENSE_COMMIT_SHA", "") or "").strip()
    if env_sha:
        return env_sha, False, "env", ""
    manifest_sha, manifest_digest = _manifest_sha(repo_root)
    if manifest_sha:
        return manifest_sha, False, "handoff_manifest", manifest_digest
    sha, dirty = detect_git_state(repo_root)
    if sha:
        return sha, bool(dirty), "git", ""
    return "", False, "", manifest_digest


def _config_digest(config_snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(config_snapshot, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_official_entrypoint_config(
    *,
    config_snapshot: dict[str, Any],
    expected_runtime_line: str | None,
    official_entrypoint: str,
) -> str:
    schema_version = int(config_snapshot.get("schema_version", 0) or 0)
    runtime = dict(config_snapshot.get("runtime", {}) or {})
    traffic = dict(config_snapshot.get("traffic", {}) or {})
    policy = dict(config_snapshot.get("policy", {}) or {})
    prediction = dict(config_snapshot.get("prediction", {}) or {})
    online_policy = dict(config_snapshot.get("online_policy", {}) or {})
    invariant_mode = normalize_invariant_mode(runtime.get("invariant_mode", "diagnostic"))
    require_invariant(
        schema_version == 1,
        context=InvariantContext(stage="startup", error_code="RS-STARTUP-001"),
        message="official entrypoints require schema_version=1",
        expected=1,
        actual=schema_version,
    )
    if expected_runtime_line is not None:
        require_invariant(
            str(runtime.get("line", "")) == str(expected_runtime_line),
            context=InvariantContext(stage="startup", error_code="RS-CONFIG-001"),
            message="runtime.line does not match official entrypoint",
            expected=expected_runtime_line,
            actual=runtime.get("line", ""),
        )
    raw_bucket_rows = traffic.get("bucket_rows", 0)
    bucket_rows_are_dynamic = False
    if isinstance(raw_bucket_rows, list):
        bucket_rows_are_dynamic = all(int(item) == 0 for item in raw_bucket_rows)
    else:
        bucket_rows_are_dynamic = int(raw_bucket_rows or 0) == 0
    bucket_mode_default = BUCKET_MODE_DYNAMIC_CURRENT if bucket_rows_are_dynamic else BUCKET_MODE_FIXED_ROWS
    bucket_mode = str(traffic.get("bucket_mode", bucket_mode_default))
    require_invariant(
        bucket_mode in {BUCKET_MODE_DYNAMIC_CURRENT, BUCKET_MODE_FIXED_ROWS},
        context=InvariantContext(stage="startup", error_code="RS-CONFIG-002"),
        message="bucket_mode must be canonical",
        expected=[BUCKET_MODE_DYNAMIC_CURRENT, BUCKET_MODE_FIXED_ROWS],
        actual=bucket_mode,
    )
    bucket_rows = raw_bucket_rows
    bucket_values = bucket_rows if isinstance(bucket_rows, list) else [bucket_rows]
    for value in bucket_values:
        ivalue = int(value)
        if bucket_mode == BUCKET_MODE_DYNAMIC_CURRENT:
            require_invariant(
                ivalue == 0,
                context=InvariantContext(stage="startup", error_code="RS-CONFIG-003"),
                message="dynamic_current requires bucket_rows=0",
                expected=0,
                actual=ivalue,
            )
        else:
            require_invariant(
                ivalue > 0,
                context=InvariantContext(stage="startup", error_code="RS-CONFIG-004"),
                message="fixed_rows bucket_rows must be positive",
                expected="> 0",
                actual=ivalue,
            )
            require_invariant(
                (ivalue & (ivalue - 1)) == 0,
                context=InvariantContext(stage="startup", error_code="RS-CONFIG-005"),
                message="fixed_rows bucket_rows must be a power of two",
                expected="power_of_two",
                actual=ivalue,
            )
    policy_name = str(policy.get("name", "") or online_policy.get("name", "")).strip()
    if policy_name and policy_name != "disabled":
        resolved = resolve_algorithm_id(policy_name)
        require_invariant(
            resolved.requested_name == resolved.canonical_name,
            context=InvariantContext(stage="startup", error_code="RS-CONFIG-006"),
            message="policy name must be canonical in official configs",
            expected=resolved.canonical_name,
            actual=policy_name,
        )
        if expected_runtime_line in {"phase_sync", "async_release"}:
            require_invariant(
                bool(resolved.spec.online_eligible) and not bool(resolved.spec.reference_only),
                context=InvariantContext(stage="startup", error_code="RS-CONFIG-007"),
                message="online official config requested a non-deployable policy",
                expected="online_eligible canonical policy",
                actual=policy_name,
            )
    predictor_name = str(prediction.get("name", "") or online_policy.get("parameters", {}).get("online_p2_predictor", "")).strip()
    if predictor_name:
        allowed_predictors = _CANONICAL_OFFLINE_PREDICTORS if expected_runtime_line == "offline_replay" else _CANONICAL_ONLINE_PREDICTORS
        require_invariant(
            predictor_name in allowed_predictors,
            context=InvariantContext(stage="startup", error_code="RS-CONFIG-008"),
            message="predictor name must be canonical and eligible for this entrypoint",
            expected=sorted(allowed_predictors),
            actual=predictor_name,
        )
    return invariant_mode


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
    run_plan_commit_sha: str = "",
    manifest_overrides: dict[str, Any] | None = None,
) -> OutputLayout:
    layout = build_output_layout(output_dir)
    commit_sha, git_dirty, commit_sha_source, source_archive_digest = resolve_commit_identity(
        repo_root=repo_root,
        run_plan_commit_sha=str(run_plan_commit_sha),
    )
    runtime = dict(config_snapshot.get("runtime", {}) or {})
    invariant_mode = normalize_invariant_mode(runtime.get("invariant_mode", "diagnostic"))
    require_invariant(
        invariant_mode_allows_dirty_git(invariant_mode) or not git_dirty,
        context=InvariantContext(stage="startup", error_code="RS-STARTUP-002"),
        message="evaluation_strict/runtime_safe runs require a clean git tree",
        expected=False,
        actual=git_dirty,
    )
    require_invariant(
        bool(commit_sha),
        context=InvariantContext(stage="startup", error_code="RS-STARTUP-003"),
        message="commit SHA must be available",
        expected="non-empty",
        actual=commit_sha,
    )
    manifest: dict[str, Any] = {
        "run_id": layout.root.name,
        "run_type": str(run_type),
        "commit_sha": commit_sha,
        "commit_sha_source": str(commit_sha_source),
        "source_commit_sha": commit_sha,
        "runtime_commit_sha": commit_sha,
        "result_commit_sha": commit_sha,
        "git_dirty": bool(git_dirty),
        "config_schema_version": int(config_snapshot.get("schema_version", 0) or 0),
        "config_digest": _config_digest(config_snapshot),
        "source_tree_digest": commit_sha,
        "source_archive_digest": str(source_archive_digest),
        "official_entrypoint": str(official_entrypoint),
        "runtime_line": str((config_snapshot.get("runtime", {}) or {}).get("line", "")),
        "policy_id": str((config_snapshot.get("policy", {}) or {}).get("name", "")),
        "predictor_id": str((config_snapshot.get("prediction", {}) or {}).get("name", "")),
        "bucket_mode": str((config_snapshot.get("traffic", {}) or {}).get("bucket_mode", "")),
        "bucket_rows": (config_snapshot.get("traffic", {}) or {}).get("bucket_rows", []),
        "model": config_snapshot.get("model", {}),
        "topology": config_snapshot.get("topology", {}),
        "workload": config_snapshot.get("workload", {}),
        "world_size": int((config_snapshot.get("topology", {}) or {}).get("world_size", (config_snapshot.get("topology", {}) or {}).get("ep_size", 1)) or 1),
        "start_time": _utc_now(),
        "end_time": "",
        "status": "running",
        "invariant_mode": invariant_mode,
        "valid_for_evaluation": bool(invariant_mode != "diagnostic"),
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
