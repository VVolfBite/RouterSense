from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


def _bash_available() -> bool:
    if shutil.which("bash") is None:
        return False
    probe = subprocess.run(
        ["bash", "-lc", "exit 0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0


def _script_command(script: Path, *args: str) -> list[str]:
    if os.name == "nt":
        if not _bash_available():
            pytest.skip("bash is required on Windows to execute archive shell scripts")
        return ["bash", str(script), *args]
    return [str(script), *args]


def test_package_source_only_mainline_excludes_legacy_and_runtime(tmp_path):
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "mainline.tar.gz"
    subprocess.run(
        _script_command(root / "RS" / "tools" / "archive" / "package_source_only.sh", "--scope", "mainline", str(archive)),
        check=True,
    )
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True).splitlines()
    assert "RS/src/rs/__init__.py" in listing
    assert not any(line.startswith("legacy/poc1/") for line in listing)
    assert not any(line.startswith("legacy/poc2/") for line in listing)
    assert not any(line.startswith("outputs/") for line in listing if line != "RS/outputs/.gitkeep")
    assert not any(line.startswith("artifacts/") for line in listing if line != "RS/artifacts/.gitkeep")
    assert "RS/README.md" in listing


def test_package_source_only_full_includes_legacy(tmp_path):
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "full.tar.gz"
    subprocess.run(
        _script_command(root / "RS" / "tools" / "archive" / "package_source_only.sh", "--scope", "full", str(archive)),
        check=True,
    )
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True).splitlines()
    assert "legacy/poc1/src/routesense_poc1/__init__.py" in listing
    assert "legacy/poc2/src/routesense_poc2/__init__.py" in listing


def test_verify_source_archive_matches_head_mainline(tmp_path):
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "mainline.tar.gz"
    subprocess.run(
        _script_command(root / "RS" / "tools" / "archive" / "package_source_only.sh", "--scope", "mainline", str(archive)),
        check=True,
    )
    subprocess.run(
        _script_command(root / "RS" / "tools" / "archive" / "verify_source_archive_matches_head.sh", "--scope", "mainline", str(archive)),
        check=True,
    )


def test_archive_unpack_allows_pytest_from_rs_dir(tmp_path):
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "mainline.tar.gz"
    subprocess.run(
        _script_command(root / "RS" / "tools" / "archive" / "package_source_only.sh", "--scope", "mainline", str(archive)),
        check=True,
    )
    unpack_dir = tmp_path / "unpack"
    unpack_dir.mkdir()
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(unpack_dir)
    rs_dir = unpack_dir / "RS"
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src:{existing_pythonpath}"
    result = subprocess.run(["python", "-c", "from rs.topology.paths import resolve_rs_root; print(resolve_rs_root().name)"], cwd=rs_dir, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
