from __future__ import annotations

from pathlib import Path

from routesense.evaluation import write_artifact_bundle


def test_artifact_bundle_writes_expected_files(tmp_path: Path):
    write_artifact_bundle(
        tmp_path,
        summary={"ok": True},
        goal="# goal\n",
        config={"x": 1},
        environment={"y": 2},
    )
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "environment.json").exists()
    assert (tmp_path / "goal.md").exists()
