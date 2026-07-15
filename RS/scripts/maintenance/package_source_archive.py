#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INCLUDED = (
    "src",
    "experiments",
    "configs",
    "tests",
    "scripts",
    "docs",
    "README.md",
    "pyproject.toml",
)
DEFAULT_EXCLUDED = (
    ".git",
    ".pytest_cache",
    "__pycache__",
    "*.pyc",
    "outputs",
    "artifacts",
    "logs",
    "*.zip",
    "*.tar.gz",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.safetensors",
    "*.ckpt",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_output(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _detect_git_state() -> tuple[str, str, bool]:
    commit_sha = _git_output("rev-parse", "HEAD")
    branch = _git_output("branch", "--show-current")
    dirty = bool(_git_output("status", "--short"))
    return commit_sha, branch, dirty


def _matches_excluded(relative_posix: str, *, scope: str) -> bool:
    patterns = list(DEFAULT_EXCLUDED)
    if scope == "mainline":
        patterns.extend(["legacy", "legacy/*"])
    for pattern in patterns:
        if fnmatch.fnmatch(relative_posix, pattern) or fnmatch.fnmatch(Path(relative_posix).name, pattern):
            return True
        if relative_posix.startswith(f"{pattern}/"):
            return True
    return False


def _iter_included_paths(*, scope: str) -> list[Path]:
    selected: list[Path] = []
    for item in DEFAULT_INCLUDED:
        path = REPO_ROOT / item
        if path.exists():
            selected.append(path)
    if scope == "full" and (REPO_ROOT / "legacy").exists():
        selected.append(REPO_ROOT / "legacy")
    return selected


def _copy_tree(staging_root: Path, *, scope: str) -> tuple[list[str], list[str]]:
    included_paths: list[str] = []
    excluded_patterns = list(DEFAULT_EXCLUDED)
    if scope == "mainline":
        excluded_patterns.extend(["legacy", "legacy/*"])
    target_root = staging_root / "RS"
    target_root.mkdir(parents=True, exist_ok=True)
    for source in _iter_included_paths(scope=scope):
        if source.is_file():
            relative = source.relative_to(REPO_ROOT)
            relative_posix = relative.as_posix()
            if _matches_excluded(relative_posix, scope=scope):
                continue
            destination = target_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            included_paths.append(relative_posix)
            continue
        for item in source.rglob("*"):
            if item.is_dir():
                continue
            relative = item.relative_to(REPO_ROOT)
            relative_posix = relative.as_posix()
            if _matches_excluded(relative_posix, scope=scope):
                continue
            destination = target_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            included_paths.append(relative_posix)
    return sorted(included_paths), sorted(dict.fromkeys(excluded_patterns))


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_source_manifest(staging_root: Path, *, scope: str, included_paths: list[str], excluded_patterns: list[str]) -> dict[str, object]:
    commit_sha, branch, dirty = _detect_git_state()
    manifest = {
        "schema_version": 1,
        "authoritative": True,
        "commit_sha": commit_sha or os.environ.get("ROUTERSENSE_COMMIT_SHA", "unknown"),
        "branch": branch or "unknown",
        "git_dirty": dirty if commit_sha else bool(os.environ.get("ROUTERSENSE_GIT_DIRTY", "")),
        "created_at": _utc_now(),
        "archive_format": "tar.gz",
        "scope": scope,
        "included_paths": included_paths,
        "excluded_patterns": excluded_patterns,
        "source_tree_digest": _tree_digest(staging_root / "RS"),
        "self_check_status": "pending",
        "self_check_commands": [],
    }
    (staging_root / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _make_archive(staging_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tf:
        for path in sorted(staging_root.rglob("*")):
            if not path.is_file():
                continue
            arcname = PurePosixPath(path.relative_to(staging_root).as_posix())
            tf.add(path, arcname=str(arcname))


def _self_check(archive_path: Path, *, scope: str) -> dict[str, object]:
    commands: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="rs_source_check_") as tmp:
        unpack_root = Path(tmp)
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.startswith("/") or ".." in PurePosixPath(member.name).parts:
                    raise ValueError(f"unsafe archive entry: {member.name}")
            tf.extractall(unpack_root)
        rs_root = unpack_root / "RS"
        source_manifest = json.loads((unpack_root / "source_manifest.json").read_text(encoding="utf-8"))
        if source_manifest["source_tree_digest"] != _tree_digest(rs_root):
            raise ValueError("source_tree_digest mismatch after unpack")
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src{os.pathsep}{existing_pythonpath}"
        env["ROUTERSENSE_COMMIT_SHA"] = str(source_manifest.get("commit_sha", "unknown"))
        env["ROUTERSENSE_GIT_DIRTY"] = "1" if bool(source_manifest.get("git_dirty", False)) else "0"
        checks = [
            ["python", "-c", "import rs; print('IMPORT_OK')"],
            ["python", "-m", "pytest", "--collect-only", "-q"],
            [
                "python",
                "experiments/run_offline_replay.py",
                "--config",
                "configs/official/offline_replay.yaml",
                "--output-dir",
                str(rs_root / "outputs" / "archive_offline_smoke"),
            ],
        ]
        for command in checks:
            proc = subprocess.run(command, cwd=str(rs_root), env=env, text=True, capture_output=True, check=False)
            commands.append(
                {
                    "command": command,
                    "exit_code": proc.returncode,
                    "stdout_tail": proc.stdout[-800:],
                    "stderr_tail": proc.stderr[-800:],
                }
            )
            if proc.returncode != 0:
                return {"status": "failed", "commands": commands}
    return {"status": "passed", "commands": commands}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("mainline", "full"), default="mainline")
    parser.add_argument("archive_path")
    args = parser.parse_args()

    archive_path = Path(args.archive_path).resolve()
    with tempfile.TemporaryDirectory(prefix="rs_source_stage_") as tmp:
        staging_root = Path(tmp)
        included_paths, excluded_patterns = _copy_tree(staging_root, scope=str(args.scope))
        manifest = _write_source_manifest(
            staging_root,
            scope=str(args.scope),
            included_paths=included_paths,
            excluded_patterns=excluded_patterns,
        )
        _make_archive(staging_root, archive_path)
        self_check = _self_check(archive_path, scope=str(args.scope))
        manifest["self_check_status"] = str(self_check["status"])
        manifest["self_check_commands"] = list(self_check["commands"])
        (staging_root / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        _make_archive(staging_root, archive_path)
        if str(self_check["status"]) != "passed":
            print(json.dumps({"status": "failed", "archive_path": str(archive_path), "self_check": self_check}, ensure_ascii=False, indent=2))
            return 1
    print(json.dumps({"status": "passed", "archive_path": str(archive_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
