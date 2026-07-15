from __future__ import annotations

import os
import subprocess
import tarfile
import json
from pathlib import Path

import pytest
from rs.core.contracts.provenance import resolve_source_manifest

def _package_command(script: Path, *, scope: str, archive: Path) -> list[str]:
    return ["python", str(script), "--scope", scope, str(archive)]


def test_package_source_only_mainline_excludes_legacy_and_runtime(tmp_path):
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "mainline.tar.gz"
    subprocess.run(
        _package_command(root / "RS" / "scripts" / "maintenance" / "package_source_archive.py", scope="mainline", archive=archive),
        check=True,
    )
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True).splitlines()
    assert "RS/src/rs/__init__.py" in listing
    assert not any(line.startswith("RS/legacy/") for line in listing)
    assert not any(line.startswith("RS/outputs/") or line == "RS/outputs" for line in listing)
    assert not any(line.startswith("RS/artifacts/") or line == "RS/artifacts" for line in listing)
    assert not any(line.startswith("RS/deploy/logs/") or line == "RS/deploy/logs" for line in listing)
    assert not any(line.startswith("RS/archives/") or line == "RS/archives" for line in listing)
    assert not any(line.startswith("RS/integrations/") or line == "RS/integrations" for line in listing)
    assert "RS/README.md" in listing


def test_package_source_only_full_includes_legacy(tmp_path):
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "full.tar.gz"
    subprocess.run(
        _package_command(root / "RS" / "scripts" / "maintenance" / "package_source_archive.py", scope="full", archive=archive),
        check=True,
    )
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True).splitlines()
    if not (root / "RS" / "legacy").exists():
        assert not any(line.startswith("RS/legacy/") for line in listing)
        return
    assert "RS/legacy/historical_poc/README.md" in listing
    assert any(line.startswith("RS/legacy/historical_poc/") for line in listing)


def test_verify_source_archive_matches_head_mainline(tmp_path):
    root = Path(__file__).resolve().parents[2]
    if not (root / "RS" / ".git").exists():
        pytest.skip("repository provenance check requires a Git checkout")
    archive = tmp_path / "mainline.tar.gz"
    subprocess.run(
        _package_command(root / "RS" / "scripts" / "maintenance" / "package_source_archive.py", scope="mainline", archive=archive),
        check=True,
    )
    subprocess.run(
        ["bash", str(root / "RS" / "scripts" / "maintenance" / "archive" / "verify_source_archive_matches_head.sh"), "--scope", "mainline", str(archive)]
        if os.name != "nt"
        else ["python", "-c", "import json, subprocess, sys, tarfile; from pathlib import Path; archive=Path(sys.argv[1]); repo=Path(sys.argv[2]); manifest=json.loads(tarfile.open(archive, 'r:gz').extractfile('source_manifest.json').read().decode('utf-8')); head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip(); assert manifest['commit_sha']==head; print('VERIFY_OK')", str(archive), str(root / "RS")],
        check=True,
    )


def test_archive_unpack_allows_pytest_from_rs_dir(tmp_path):
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "mainline.tar.gz"
    subprocess.run(
        _package_command(root / "RS" / "scripts" / "maintenance" / "package_source_archive.py", scope="mainline", archive=archive),
        check=True,
    )
    unpack_dir = tmp_path / "unpack"
    unpack_dir.mkdir()
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(unpack_dir)
    rs_dir = unpack_dir / "RS"
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src{os.pathsep}{existing_pythonpath}"
    result = subprocess.run(["python", "-c", "from rs.topology.paths import resolve_rs_root; print(resolve_rs_root().name)"], cwd=rs_dir, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_source_manifest_is_explicitly_authoritative(tmp_path):
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "mainline.tar.gz"
    subprocess.run(
        _package_command(root / "RS" / "scripts" / "maintenance" / "package_source_archive.py", scope="mainline", archive=archive),
        check=True,
    )
    with tarfile.open(archive, "r:gz") as tf:
        manifest = json.loads(tf.extractfile("source_manifest.json").read().decode("utf-8"))
    assert manifest["authoritative"] is True


def test_resolve_source_manifest_requires_explicit_authoritative(tmp_path):
    repo_root = tmp_path / "RS"
    repo_root.mkdir()
    (repo_root / "source_manifest.json").write_text(json.dumps({"commit_sha": "abc"}), encoding="utf-8")
    assert resolve_source_manifest(repo_root) is None
    (repo_root / "source_manifest.json").write_text(
        json.dumps({"authoritative": True, "commit_sha": "abc"}, indent=2),
        encoding="utf-8",
    )
    resolved = resolve_source_manifest(repo_root)
    assert resolved is not None
    assert resolved["commit_sha"] == "abc"
