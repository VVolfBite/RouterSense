"""Source/result provenance contracts and helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceProvenance:
    git_commit: str | None = None
    git_dirty: bool | None = None
    source_archive_sha256: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def resolve_source_manifest(repo_root: Path) -> dict[str, Any] | None:
    candidates = (
        repo_root / "source_manifest.json",
        repo_root.parent / "source_manifest.json",
    )
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if payload.get("authoritative") is True:
                return payload
    return None


def compute_source_tree_digest(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in repo_root.rglob("*") if item.is_file()):
        if path.name == "source_manifest.json" and path.parent == repo_root.parent:
            continue
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_verified_source_manifest(repo_root: Path) -> dict[str, Any] | None:
    manifest = resolve_source_manifest(repo_root)
    if manifest is None:
        return None
    expected_digest = str(manifest.get("source_tree_digest", "") or "")
    if not expected_digest:
        return None
    try:
        actual_digest = compute_source_tree_digest(repo_root)
    except FileNotFoundError:
        return None
    if actual_digest != expected_digest:
        return None
    return manifest


def resolve_commit_identity(repo_root: Path) -> tuple[str, bool, str]:
    git_sha = _git_output(repo_root, "rev-parse", "HEAD")
    if git_sha:
        git_dirty = bool(_git_output(repo_root, "status", "--short"))
        return git_sha, git_dirty, "git"
    env_sha = str(os.environ.get("ROUTERSENSE_COMMIT_SHA", "")).strip()
    if env_sha:
        env_dirty = str(os.environ.get("ROUTERSENSE_GIT_DIRTY", "")).strip().lower()
        return env_sha, env_dirty in {"1", "true", "yes"}, "env"
    manifest = resolve_verified_source_manifest(repo_root)
    if manifest is not None:
        return (
            str(manifest.get("commit_sha", "") or "unknown"),
            bool(manifest.get("git_dirty", False)),
            "source_manifest",
        )
    return "unknown", False, "unknown"
