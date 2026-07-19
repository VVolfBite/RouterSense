from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from rs.core.contracts.provenance import compute_source_tree_digest, resolve_commit_identity


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class OfficialOutputLayout:
    root: Path
    config_dir: Path
    measurements_dir: Path
    raw_dir: Path
    summary_dir: Path
    failures_dir: Path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def initialize_official_output(
    *,
    repo_root: Path,
    output_dir: Path,
    run_type: str,
    official_entrypoint: str,
    config_snapshot: dict[str, Any],
) -> OfficialOutputLayout:
    root = output_dir.resolve()
    layout = OfficialOutputLayout(
        root=root,
        config_dir=root / "config",
        measurements_dir=root / "measurements",
        raw_dir=root / "raw",
        summary_dir=root / "summary",
        failures_dir=root / "failures",
    )
    for path in (layout.root, layout.config_dir, layout.measurements_dir, layout.raw_dir, layout.summary_dir, layout.failures_dir):
        path.mkdir(parents=True, exist_ok=True)
    commit_sha, git_dirty, provenance_source = resolve_commit_identity(repo_root)
    manifest = {
        "run_id": root.name,
        "run_type": str(run_type),
        "official_entrypoint": str(official_entrypoint),
        "commit_sha": str(commit_sha),
        "git_dirty": bool(git_dirty),
        "provenance_source": str(provenance_source),
        "source_tree_digest": compute_source_tree_digest(repo_root),
        "status": "running",
        "started_at": _utc_now(),
        "config_schema_version": int(config_snapshot.get("schema_version", 0) or 0),
        "runtime_line": str((config_snapshot.get("runtime", {}) or {}).get("line", "")),
        "policy_id": str((config_snapshot.get("policy", {}) or {}).get("name", "")),
    }
    _write_json(layout.root / "manifest.json", manifest)
    _write_json(layout.root / "status.json", {"status": "running", "updated_at": _utc_now()})
    return layout


def write_official_configs(
    layout: OfficialOutputLayout,
    *,
    normalized_config: dict[str, Any],
    consumed_config: dict[str, Any],
) -> None:
    _write_yaml(layout.config_dir / "normalized.yaml", normalized_config)
    _write_json(layout.config_dir / "normalized.json", normalized_config)
    _write_yaml(layout.config_dir / "consumed.yaml", consumed_config)
    _write_json(layout.config_dir / "consumed.json", consumed_config)


def update_official_status(layout: OfficialOutputLayout, *, status: str, extra: dict[str, Any] | None = None) -> None:
    manifest_path = layout.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = str(status)
    manifest["ended_at"] = _utc_now()
    if extra:
        manifest.update(extra)
    _write_json(manifest_path, manifest)
    payload = {"status": str(status), "updated_at": _utc_now()}
    if extra:
        payload.update(extra)
    _write_json(layout.root / "status.json", payload)


__all__ = [
    "OfficialOutputLayout",
    "initialize_official_output",
    "update_official_status",
    "write_official_configs",
]
