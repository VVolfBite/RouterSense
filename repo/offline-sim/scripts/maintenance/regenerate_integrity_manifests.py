from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_JSON = ROOT / "SOURCE_MANIFEST.json"
CHECKSUMS_TXT = ROOT / "FILE_SHA256SUMS.txt"
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
}
SOURCE_DIRS = ("src", "tests", "scripts", "configs", "fixtures")
ROOT_FILES = ("pyproject.toml", "SNAPSHOT_STATUS_CN.txt")


def _excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return (
        (rel.parts and rel.parts[0] in EXCLUDED_PARTS)
        or "__pycache__" in rel.parts
        or any(part.endswith(".egg-info") for part in rel.parts)
        or path.suffix == ".pyc"
        or path in {MANIFEST_JSON, CHECKSUMS_TXT}
    )


def _source_files() -> tuple[Path, ...]:
    files: set[Path] = set()
    for name in ROOT_FILES:
        path = ROOT / name
        if path.is_file():
            files.add(path)
    for directory in SOURCE_DIRS:
        root = ROOT / directory
        if not root.exists():
            continue
        files.update(
            path
            for path in root.rglob("*")
            if path.is_file() and not _excluded(path)
        )
    return tuple(sorted(files, key=lambda path: path.relative_to(ROOT).as_posix()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    files = _source_files()
    rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    payload = {
        "schema_version": "RS_SIM_SOURCE_MANIFEST",
        "package_version": str(version),
        "file_count": len(rows),
        "files": rows,
    }
    MANIFEST_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_files = (*files, MANIFEST_JSON)
    CHECKSUMS_TXT.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
            for path in sorted(
                checksum_files, key=lambda item: item.relative_to(ROOT).as_posix()
            )
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
