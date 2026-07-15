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

from rs.core.contracts.result import ResultBundle
from rs.core.contracts.provenance import compute_source_tree_digest, resolve_commit_identity
from rs.runtime.guards import InvariantContext, RouterSenseInvariantError, invariant_mode_allows_dirty_git, normalize_invariant_mode, require_invariant
from rs.scheduling.catalog import resolve_algorithm_id
from rs.scheduling.bucketizer import BUCKET_MODE_DYNAMIC_CURRENT, BUCKET_MODE_FIXED_ROWS


ARTIFACT_SCHEMA_VERSION = 1
_CANONICAL_ONLINE_PREDICTORS = {"none", "zero_hint", "copy_current_dispatch", "history_ema"}
_CANONICAL_OFFLINE_PREDICTORS = _CANONICAL_ONLINE_PREDICTORS | {"ridge_linear_trace_predictor", "perfect_trace_hint", "shuffled_control"}


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    config_dir: Path
    raw_dir: Path
    metrics_dir: Path
    reports_dir: Path
    logs_dir: Path
    failures_dir: Path


RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"


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
    sha, dirty, _source = resolve_commit_identity(repo_root)
    return sha, dirty


def _config_digest(config_snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(config_snapshot, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_int(value: Any, *, field_name: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise RouterSenseInvariantError(
                stage="startup",
                error_code="RS-CONFIG-INT",
                message=f"{field_name} must be an integer",
            )
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise RouterSenseInvariantError(
            stage="startup",
            error_code="RS-CONFIG-INT",
            message=f"{field_name} must be an integer",
            actual=type(value).__name__,
        )
    return int(value)


def _strict_bool(value: Any, *, field_name: str, default: bool | None = None) -> bool:
    if value is None:
        if default is None:
            raise RouterSenseInvariantError(
                stage="startup",
                error_code="RS-CONFIG-BOOL",
                message=f"{field_name} must be a boolean",
            )
        value = default
    if not isinstance(value, bool):
        raise RouterSenseInvariantError(
            stage="startup",
            error_code="RS-CONFIG-BOOL",
            message=f"{field_name} must be a boolean",
            actual=type(value).__name__,
        )
    return value


def validate_official_entrypoint_config(
    *,
    config_snapshot: dict[str, Any],
    expected_runtime_line: str | None,
    official_entrypoint: str,
) -> str:
    schema_version = _strict_int(config_snapshot.get("schema_version", 0), field_name="schema_version", default=0)
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
        bucket_rows_are_dynamic = all(_strict_int(item, field_name=f"traffic.bucket_rows[{index}]") == 0 for index, item in enumerate(raw_bucket_rows))
    else:
        bucket_rows_are_dynamic = _strict_int(raw_bucket_rows if raw_bucket_rows is not None else 0, field_name="traffic.bucket_rows", default=0) == 0
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
    for index, value in enumerate(bucket_values):
        ivalue = _strict_int(value, field_name=f"traffic.bucket_rows[{index}]")
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
    config_dir = root / "config"
    raw_dir = root / "raw"
    metrics_dir = root / "metrics"
    reports_dir = root / "reports"
    logs_dir = root / "logs"
    failures_dir = root / "failures"
    for path in (root, config_dir, raw_dir, metrics_dir, reports_dir, logs_dir, failures_dir):
        path.mkdir(parents=True, exist_ok=True)
    return OutputLayout(
        root=root,
        config_dir=config_dir,
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


def write_result_bundle(path: Path, bundle: ResultBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def write_manifest(layout: OutputLayout, payload: dict[str, Any]) -> None:
    write_json(layout.root / "manifest.json", payload)


def read_manifest(layout: OutputLayout) -> dict[str, Any]:
    return json.loads((layout.root / "manifest.json").read_text(encoding="utf-8"))


def write_run_status(layout: OutputLayout, payload: dict[str, Any]) -> None:
    write_json(layout.root / "status.json", payload)


def write_layout_result_bundle(layout: OutputLayout, bundle: ResultBundle) -> None:
    write_result_bundle(layout.root / "result_bundle.json", bundle)


def write_resolved_configs(
    layout: OutputLayout,
    *,
    normalized_config: dict[str, Any],
    consumed_config: dict[str, Any] | None = None,
    legacy_bridge_config: dict[str, Any] | None = None,
) -> None:
    write_yaml(layout.config_dir / "normalized.yaml", normalized_config)
    write_json(layout.config_dir / "normalized.json", normalized_config)
    if consumed_config is not None:
        write_yaml(layout.config_dir / "consumed.yaml", consumed_config)
        write_json(layout.config_dir / "consumed.json", consumed_config)
    if legacy_bridge_config is not None:
        write_yaml(layout.config_dir / "legacy_bridge.yaml", legacy_bridge_config)


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
        "source_commit_sha": commit_sha,
        "runtime_commit_sha": commit_sha,
        "result_commit_sha": commit_sha,
        "git_dirty": _strict_bool(git_dirty, field_name="git_dirty", default=False),
        "config_schema_version": _strict_int(config_snapshot.get("schema_version", 0), field_name="schema_version", default=0),
        "config_digest": _config_digest(config_snapshot),
        "source_tree_digest": compute_source_tree_digest(repo_root),
        "official_entrypoint": str(official_entrypoint),
        "runtime_line": str((config_snapshot.get("runtime", {}) or {}).get("line", "")),
        "policy_id": str((config_snapshot.get("policy", {}) or {}).get("name", "")),
        "predictor_id": str((config_snapshot.get("prediction", {}) or {}).get("name", "")),
        "bucket_mode": str((config_snapshot.get("traffic", {}) or {}).get("bucket_mode", "")),
        "bucket_rows": (config_snapshot.get("traffic", {}) or {}).get("bucket_rows", []),
        "model": config_snapshot.get("model", {}),
        "topology": config_snapshot.get("topology", {}),
        "workload": config_snapshot.get("workload", {}),
        "world_size": _strict_int(
            (config_snapshot.get("topology", {}) or {}).get("world_size", (config_snapshot.get("topology", {}) or {}).get("ep_size", 1)),
            field_name="topology.world_size",
            default=1,
        ),
        "start_time": _utc_now(),
        "end_time": "",
        "status": "running",
        "invariant_mode": invariant_mode,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    write_manifest(layout, manifest)
    write_json(layout.root / "environment.json", capture_environment())
    write_yaml(layout.root / "config_snapshot.yaml", config_snapshot)
    write_run_status(layout, {"status": RUN_STATUS_RUNNING, "updated_at": _utc_now()})
    return layout


def update_status(layout: OutputLayout, *, status: str, extra: dict[str, Any] | None = None) -> None:
    manifest = read_manifest(layout)
    manifest["status"] = str(status)
    manifest["end_time"] = _utc_now()
    if extra:
        manifest.update(extra)
    write_manifest(layout, manifest)
    status_payload: dict[str, Any] = {"status": str(status), "updated_at": _utc_now()}
    if extra:
        status_payload.update(extra)
    write_run_status(layout, status_payload)
