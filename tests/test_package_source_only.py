from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest
from rs.core.contracts.provenance import _iter_canonical_digest_entries, compute_source_tree_digest, resolve_source_manifest, resolve_verified_source_manifest



def _repo_root() -> Path:
    """Resolve the checkout root without assuming its directory is named RS."""

    return Path(__file__).resolve().parents[1]

def _package_command(script: Path, *, scope: str, archive: Path) -> list[str]:
    return ["python", str(script), "--scope", scope, "--skip-self-check", "--skip-repack-check", str(archive)]


@pytest.fixture(scope="module")
def packaged_source_archives(tmp_path_factory):
    """Build each package shape once for the module.

    These tests inspect the same immutable checkout. Rebuilding the full source
    archive in every test adds substantial I/O and can make an otherwise
    healthy packaging suite look hung on constrained CI hosts.
    """

    repo_root = _repo_root()
    root = tmp_path_factory.mktemp("source_archives")
    script = repo_root / "scripts" / "maintenance" / "package_source_archive.py"
    archives = {
        "mainline_tar": root / "mainline.tar.gz",
        "full_tar": root / "full.tar.gz",
        "mainline_zip": root / "mainline.zip",
    }
    subprocess.run(_package_command(script, scope="mainline", archive=archives["mainline_tar"]), check=True)
    subprocess.run(_package_command(script, scope="full", archive=archives["full_tar"]), check=True)
    subprocess.run(_package_command(script, scope="mainline", archive=archives["mainline_zip"]), check=True)
    return {"repo_root": repo_root, **archives}


def test_package_source_only_mainline_excludes_legacy_and_runtime(packaged_source_archives):
    repo_root = packaged_source_archives["repo_root"]
    archive = packaged_source_archives["mainline_tar"]
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True).splitlines()
    assert "RS/src/rs/__init__.py" in listing
    assert "RS/deploy/README.md" in listing
    assert "RS/DEPLOYMENT_HANDOFF.md" in listing
    assert "RS/task-test-deploy.md" in listing
    assert "RS/.gitignore" in listing
    assert "RS/deploy/inventory/hosts.example.yaml" in listing
    assert "RS/scripts/deploy/launch_remote.sh" in listing
    assert not any(line.startswith("RS/legacy/") for line in listing)
    assert not any(line.startswith("RS/tests/legacy/") for line in listing)
    assert not any(line.startswith("RS/outputs/") or line == "RS/outputs" for line in listing)
    assert not any(line.startswith("RS/artifacts/") or line == "RS/artifacts" for line in listing)
    assert not any(line.startswith("RS/deploy/logs/") or line == "RS/deploy/logs" for line in listing)
    assert not any(line.startswith("RS/archives/") or line == "RS/archives" for line in listing)
    assert not any(line.startswith("RS/integrations/") or line == "RS/integrations" for line in listing)
    assert not any(line.startswith("RS/archive/") or line == "RS/archive" for line in listing)
    assert not any(line.startswith("RS/configs/archive/") or line == "RS/configs/archive" for line in listing)
    assert not any(line.startswith("RS/experiments/archive/") or line == "RS/experiments/archive" for line in listing)
    assert "RS/README.md" in listing


def test_package_source_only_full_includes_legacy(packaged_source_archives):
    repo_root = packaged_source_archives["repo_root"]
    archive = packaged_source_archives["full_tar"]
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True).splitlines()
    if not (repo_root / "legacy").exists():
        assert not any(line.startswith("RS/legacy/") for line in listing)
        return
    assert "RS/legacy/historical_poc/README.md" in listing
    assert any(line.startswith("RS/legacy/historical_poc/") for line in listing)


def test_verify_source_archive_matches_head_mainline(packaged_source_archives):
    repo_root = packaged_source_archives["repo_root"]
    if not (repo_root / ".git").exists():
        pytest.skip("repository provenance check requires a Git checkout")
    archive = packaged_source_archives["mainline_tar"]
    subprocess.run(
        ["bash", str(repo_root / "scripts" / "maintenance" / "archive" / "verify_source_archive_matches_head.sh"), "--scope", "mainline", str(archive)]
        if os.name != "nt"
        else ["python", "-c", "import json, subprocess, sys, tarfile; from pathlib import Path; archive=Path(sys.argv[1]); repo=Path(sys.argv[2]); manifest=json.loads(tarfile.open(archive, 'r:gz').extractfile('source_manifest.json').read().decode('utf-8')); head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip(); assert manifest['commit_sha']==head; print('VERIFY_OK')", str(archive), str(repo_root)],
        check=True,
    )


def test_archive_unpack_allows_pytest_from_rs_dir(tmp_path, packaged_source_archives):
    archive = packaged_source_archives["mainline_tar"]
    unpack_dir = tmp_path / "unpack"
    unpack_dir.mkdir()
    with tarfile.open(archive, "r:gz") as tf:
        try:
            tf.extractall(unpack_dir, filter="data")
        except TypeError:
            tf.extractall(unpack_dir)
    rs_dir = unpack_dir / "RS"
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src{os.pathsep}{existing_pythonpath}"
    result = subprocess.run(["python", "-c", "from rs.topology.paths import resolve_rs_root; print(resolve_rs_root().name)"], cwd=rs_dir, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_source_manifest_is_explicitly_authoritative(packaged_source_archives):
    archive = packaged_source_archives["mainline_tar"]
    with tarfile.open(archive, "r:gz") as tf:
        manifest = json.loads(tf.extractfile("source_manifest.json").read().decode("utf-8"))
    assert manifest["authoritative"] is True


def test_package_source_only_zip_is_real_zip_with_posix_paths(packaged_source_archives):
    archive = packaged_source_archives["mainline_zip"]
    with zipfile.ZipFile(archive, "r") as zf:
        names = zf.namelist()
        assert "source_manifest.json" in names
        assert "RS/README.md" in names
        assert all("\\" not in name for name in names)
        manifest = json.loads(zf.read("source_manifest.json").decode("utf-8"))
    assert manifest["archive_format"] == "zip"
    assert manifest["digest_algorithm"] == "sha256_path_content"
    assert manifest["digest_order"] == "posix_casefold_then_original_v1"


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


def test_verified_archive_repackage_preserves_commit_identity(tmp_path):
    repo_root = _repo_root()
    first_archive = tmp_path / "first.tar.gz"
    subprocess.run(
        _package_command(repo_root / "scripts" / "maintenance" / "package_source_archive.py", scope="mainline", archive=first_archive),
        check=True,
    )
    unpack_dir = tmp_path / "unpack"
    unpack_dir.mkdir()
    with tarfile.open(first_archive, "r:gz") as tf:
        try:
            tf.extractall(unpack_dir, filter="data")
        except TypeError:
            tf.extractall(unpack_dir)
    rs_dir = unpack_dir / "RS"
    manifest_path = unpack_dir / "source_manifest.json"
    verified = resolve_verified_source_manifest(rs_dir)
    assert verified is not None
    second_archive = tmp_path / "second.tar.gz"
    env = dict(os.environ)
    env.pop("ROUTERSENSE_COMMIT_SHA", None)
    env.pop("ROUTERSENSE_GIT_DIRTY", None)
    subprocess.run(
        ["python", str(rs_dir / "scripts" / "maintenance" / "package_source_archive.py"), "--scope", "mainline", "--skip-self-check", "--skip-repack-check", str(second_archive)],
        cwd=rs_dir,
        env=env,
        check=True,
    )
    with tarfile.open(first_archive, "r:gz") as tf:
        first_manifest = json.loads(tf.extractfile("source_manifest.json").read().decode("utf-8"))
    with tarfile.open(second_archive, "r:gz") as tf:
        second_manifest = json.loads(tf.extractfile("source_manifest.json").read().decode("utf-8"))
    assert manifest_path.exists()
    assert second_manifest["commit_sha"] == first_manifest["commit_sha"]
    assert second_manifest["provenance_source"] == "source_manifest"
    assert second_manifest["parent_commit_sha"] == first_manifest["commit_sha"]
    assert second_manifest["parent_source_tree_digest"] == first_manifest["source_tree_digest"]


def test_source_tree_digest_is_stable_across_enumeration_order(tmp_path):
    repo_root = tmp_path / "RS"
    (repo_root / "alpha").mkdir(parents=True)
    (repo_root / "Zeta").mkdir(parents=True)
    (repo_root / "alpha" / "Beta.txt").write_text("beta\n", encoding="utf-8")
    (repo_root / "alpha" / "beta.txt").write_text("beta-lower\n", encoding="utf-8")
    (repo_root / "Zeta" / "gamma.txt").write_text("gamma\n", encoding="utf-8")
    (repo_root / "README.md").write_text("root\n", encoding="utf-8")

    canonical_digest = compute_source_tree_digest(repo_root)
    files = [path for path in repo_root.rglob("*") if path.is_file()]
    reversed_entries = _iter_canonical_digest_entries(repo_root, reversed(files))
    forward_entries = _iter_canonical_digest_entries(repo_root, files)

    assert [relative for relative, _ in forward_entries] == [relative for relative, _ in reversed_entries]

    import hashlib

    digest = hashlib.sha256()
    for relative_posix, path in reversed_entries:
        digest.update(relative_posix.encode("utf-8"))
        digest.update(path.read_bytes())
    assert digest.hexdigest() == canonical_digest
