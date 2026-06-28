from __future__ import annotations

import subprocess
from pathlib import Path


def test_package_source_only_mainline_excludes_legacy_and_runtime(tmp_path):
    root = Path("/root/autodl-tmp/RouterSense")
    archive = tmp_path / "mainline.tar.gz"
    subprocess.run([str(root / "scripts" / "package_source_only.sh"), "--scope", "mainline", str(archive)], check=True)
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True).splitlines()
    assert "RS/src/routesense/__init__.py" in listing
    assert not any(line.startswith("legacy/poc1/") for line in listing)
    assert not any(line.startswith("legacy/poc2/") for line in listing)
    assert not any(line.startswith("outputs/") for line in listing if line != "RS/outputs/.gitkeep")
    assert not any(line.startswith("artifacts/") for line in listing if line != "RS/artifacts/.gitkeep")
    assert "RS/README.md" in listing


def test_package_source_only_full_includes_legacy(tmp_path):
    root = Path("/root/autodl-tmp/RouterSense")
    archive = tmp_path / "full.tar.gz"
    subprocess.run([str(root / "scripts" / "package_source_only.sh"), "--scope", "full", str(archive)], check=True)
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True).splitlines()
    assert "legacy/poc1/src/routesense_poc1/__init__.py" in listing
    assert "legacy/poc2/src/routesense_poc2/__init__.py" in listing


def test_verify_source_archive_matches_head_mainline(tmp_path):
    root = Path("/root/autodl-tmp/RouterSense")
    archive = tmp_path / "mainline.tar.gz"
    subprocess.run([str(root / "scripts" / "package_source_only.sh"), "--scope", "mainline", str(archive)], check=True)
    subprocess.run([str(root / "scripts" / "verify_source_archive_matches_head.sh"), "--scope", "mainline", str(archive)], check=True)
