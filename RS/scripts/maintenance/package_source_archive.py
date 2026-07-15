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
import zipfile
import yaml
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "src"))

from rs.core.contracts.provenance import resolve_commit_identity, resolve_verified_source_manifest

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


def _detect_provenance_state() -> dict[str, object]:
    commit_sha = _git_output("rev-parse", "HEAD")
    branch = _git_output("branch", "--show-current")
    if commit_sha:
        return {
            "commit_sha": commit_sha,
            "branch": branch,
            "git_dirty": bool(_git_output("status", "--short")),
            "provenance_source": "git",
            "parent_commit_sha": None,
            "parent_source_tree_digest": None,
        }
    resolved_sha, resolved_dirty, source = resolve_commit_identity(REPO_ROOT)
    verified_manifest = resolve_verified_source_manifest(REPO_ROOT)
    parent_commit_sha = None
    parent_source_tree_digest = None
    if verified_manifest is not None:
        parent_commit_sha = str(verified_manifest.get("commit_sha", "") or "")
        parent_source_tree_digest = str(verified_manifest.get("source_tree_digest", "") or "")
    return {
        "commit_sha": resolved_sha,
        "branch": branch or str((verified_manifest or {}).get("branch", "") or "unknown"),
        "git_dirty": bool(resolved_dirty),
        "provenance_source": str(source),
        "parent_commit_sha": parent_commit_sha or None,
        "parent_source_tree_digest": parent_source_tree_digest or None,
    }


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
    provenance = _detect_provenance_state()
    manifest = {
        "schema_version": 1,
        "authoritative": True,
        "commit_sha": str(provenance["commit_sha"] or os.environ.get("ROUTERSENSE_COMMIT_SHA", "unknown")),
        "branch": str(provenance["branch"] or "unknown"),
        "git_dirty": bool(provenance["git_dirty"]),
        "provenance_source": str(provenance["provenance_source"]),
        "parent_commit_sha": provenance["parent_commit_sha"],
        "parent_source_tree_digest": provenance["parent_source_tree_digest"],
        "created_at": _utc_now(),
        "archive_format": "unknown",
        "scope": scope,
        "included_paths": included_paths,
        "excluded_patterns": excluded_patterns,
        "source_tree_digest": _tree_digest(staging_root / "RS"),
        "self_check_status": "pending",
        "self_check_commands": [],
    }
    (staging_root / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _archive_format(archive_path: Path) -> str:
    lower_name = archive_path.name.lower()
    if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz"):
        return "tar.gz"
    if lower_name.endswith(".zip"):
        return "zip"
    raise ValueError(f"unsupported archive format for {archive_path.name!r}; expected .tar.gz, .tgz, or .zip")


def _make_archive(staging_root: Path, archive_path: Path, *, archive_format: str) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_format == "tar.gz":
        with tarfile.open(archive_path, "w:gz") as tf:
            for path in sorted(staging_root.rglob("*")):
                if not path.is_file():
                    continue
                arcname = PurePosixPath(path.relative_to(staging_root).as_posix())
                tf.add(path, arcname=str(arcname))
        return
    if archive_format == "zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(staging_root.rglob("*")):
                if not path.is_file():
                    continue
                arcname = PurePosixPath(path.relative_to(staging_root).as_posix())
                zf.write(path, arcname=str(arcname))
        return
    raise ValueError(f"unsupported archive format {archive_format!r}")


def _self_check(archive_path: Path, *, scope: str, archive_format: str) -> dict[str, object]:
    commands: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="rs_source_check_") as tmp:
        unpack_root = Path(tmp)
        if archive_format == "tar.gz":
            with tarfile.open(archive_path, "r:gz") as tf:
                for member in tf.getmembers():
                    if member.name.startswith("/") or ".." in PurePosixPath(member.name).parts:
                        raise ValueError(f"unsafe archive entry: {member.name}")
                try:
                    tf.extractall(unpack_root, filter="data")
                except TypeError:
                    tf.extractall(unpack_root)
        elif archive_format == "zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.infolist():
                    if member.filename.startswith("/") or ".." in PurePosixPath(member.filename).parts:
                        raise ValueError(f"unsafe archive entry: {member.filename}")
                zf.extractall(unpack_root)
        else:
            raise ValueError(f"unsupported archive format {archive_format!r}")
        rs_root = unpack_root / "RS"
        source_manifest = json.loads((unpack_root / "source_manifest.json").read_text(encoding="utf-8"))
        if source_manifest["source_tree_digest"] != _tree_digest(rs_root):
            raise ValueError("source_tree_digest mismatch after unpack")
        offline_config = yaml.safe_load((rs_root / "configs" / "official" / "offline_replay.yaml").read_text(encoding="utf-8"))
        if not isinstance(offline_config, dict):
            raise ValueError("official offline replay config must be a mapping")
        runtime_section = dict(offline_config.get("runtime", {}) or {})
        runtime_section["invariant_mode"] = "diagnostic"
        offline_config["runtime"] = runtime_section
        selfcheck_config = unpack_root / "offline_replay_selfcheck.yaml"
        selfcheck_config.write_text(yaml.safe_dump(offline_config, sort_keys=False), encoding="utf-8")
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
                str(selfcheck_config),
                "--output-dir",
                "outputs/selfcheck_offline",
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
    archive_format = _archive_format(archive_path)
    with tempfile.TemporaryDirectory(prefix="rs_source_stage_") as tmp:
        staging_root = Path(tmp)
        included_paths, excluded_patterns = _copy_tree(staging_root, scope=str(args.scope))
        manifest = _write_source_manifest(
            staging_root,
            scope=str(args.scope),
            included_paths=included_paths,
            excluded_patterns=excluded_patterns,
        )
        manifest["archive_format"] = archive_format
        (staging_root / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        _make_archive(staging_root, archive_path, archive_format=archive_format)
        self_check = _self_check(archive_path, scope=str(args.scope), archive_format=archive_format)
        manifest["self_check_status"] = str(self_check["status"])
        manifest["self_check_commands"] = list(self_check["commands"])
        (staging_root / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        _make_archive(staging_root, archive_path, archive_format=archive_format)
        if str(self_check["status"]) != "passed":
            print(json.dumps({"status": "failed", "archive_path": str(archive_path), "self_check": self_check}, ensure_ascii=False, indent=2))
            return 1
    print(json.dumps({"status": "passed", "archive_path": str(archive_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
