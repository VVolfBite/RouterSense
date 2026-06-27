from __future__ import annotations

import subprocess
from pathlib import Path


def _init_clean_repo_from_worktree(root: Path, target: Path) -> None:
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude=.git",
            "--exclude=artifacts",
            "--exclude=outputs",
            "--exclude=logs",
            "--exclude=.pytest_cache",
            "--exclude=__pycache__",
            f"{root}/",
            str(target),
        ],
        check=True,
    )
    subprocess.run(["git", "init"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=target, check=True)
    subprocess.run(["git", "add", "."], cwd=target, check=True)
    subprocess.run(["git", "commit", "-m", "test snapshot"], cwd=target, check=True)


def test_package_source_only_excludes_runtime_dirs(tmp_path):
    root = Path("/root/autodl-tmp/RouterSense")
    script = root / "scripts" / "package_source_only.sh"
    archive = tmp_path / "src.tar.gz"
    subprocess.run([str(script), str(archive)], cwd=root, check=True)
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True)
    assert "outputs/" not in listing
    assert "artifacts/" not in listing
    assert ".pytest_cache/" not in listing
    assert "__pycache__/" not in listing
    assert "src/routesense_poc2/stress.py" in listing
    assert "experiment/poc2/stress_suite.py" in listing
    assert "experiment/poc2/analyze_dependency_predictiveness.py" in listing
    assert "experiment/poc2/analyze_stress_results.py" in listing
    assert "docs/poc2_stress_suite_contract.md" in listing
    assert "PACKAGE_MANIFEST.sha256" in listing
    assert "SOURCE_COMMIT.txt" in listing
    assert "SOURCE_TREE_SHA256.txt" in listing

    manifest = subprocess.check_output(["tar", "-xOzf", str(archive), "PACKAGE_MANIFEST.sha256"], text=True)
    assert "src/routesense_poc2/stress.py" in manifest
    assert "experiment/poc2/stress_suite.py" in manifest


def test_verify_source_archive_matches_head(tmp_path):
    root = Path("/root/autodl-tmp/RouterSense")
    clone = tmp_path / "repo"
    _init_clean_repo_from_worktree(root, clone)
    archive = tmp_path / "src.tar.gz"
    subprocess.run([str(clone / "scripts" / "package_source_only.sh"), str(archive)], cwd=clone, check=True)
    subprocess.run([str(clone / "scripts" / "verify_source_archive_matches_head.sh"), str(archive)], cwd=clone, check=True)


def test_archive_contains_source_truth_files(tmp_path):
    root = Path("/root/autodl-tmp/RouterSense")
    archive = tmp_path / "src.tar.gz"
    subprocess.run([str(root / "scripts" / "package_source_only.sh"), str(archive)], cwd=root, check=True)
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True).splitlines()
    assert "SOURCE_COMMIT.txt" in listing
    assert "SOURCE_TREE_SHA256.txt" in listing
