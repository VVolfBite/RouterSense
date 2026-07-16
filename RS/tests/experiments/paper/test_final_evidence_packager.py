from __future__ import annotations

import json
from pathlib import Path

from scripts.maintenance.package_final_scheduling_evidence import build_portable_zip, build_relative_artifact_index, fresh_unpack_verify, write_relative_checksums


def test_checksums_are_relative_and_no_bom(tmp_path) -> None:
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "CAPABILITY_AUDIT.md").write_text("x\n", encoding="utf-8", newline="\n")
    path = write_relative_checksums(tmp_path)
    data = path.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf")
    line = path.read_text(encoding="utf-8").strip()
    assert "  audit/CAPABILITY_AUDIT.md" in line
    assert "\\" not in line


def test_artifact_index_is_relative(tmp_path) -> None:
    (tmp_path / "runtime" / "B" / "phase_sync").mkdir(parents=True)
    (tmp_path / "runtime" / "B" / "phase_sync" / "parity.json").write_text("{}\n", encoding="utf-8", newline="\n")
    index = build_relative_artifact_index(tmp_path)
    assert index["runtime/B/phase_sync/parity.json"] == "runtime/B/phase_sync/parity.json"
    assert all(":\\" not in value for value in index.values())


def test_portable_zip_uses_posix_paths(tmp_path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "x.json").write_text("{}\n", encoding="utf-8", newline="\n")
    write_relative_checksums(tmp_path)
    zip_path = tmp_path / "x.zip"
    build_portable_zip(tmp_path, zip_path)
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert all("\\" not in name for name in zf.namelist())


def test_fresh_unpack_verify_detects_complete_minimal_package(tmp_path) -> None:
    for relative, payload in {
        "source/source_manifest.json": {"source_tree_digest": "bad"},
        "results/result_bundle.json": {"artifact_index": {"audit/CAPABILITY_AUDIT.md": "audit/CAPABILITY_AUDIT.md"}},
        "audit/CAPABILITY_AUDIT.md": "# x\n",
        "oracle/oracle_control_summary.json": {"status": "READY"},
        "runtime/B/phase_sync/formal_runner_summary.json": {},
        "runtime/B/async_release/formal_runner_summary.json": {},
        "runtime/U/phase_sync/formal_runner_summary.json": {},
        "runtime/U/async_release/formal_runner_summary.json": {},
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, dict):
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
        else:
            path.write_text(str(payload), encoding="utf-8", newline="\n")
    import zipfile
    src_zip = tmp_path / "source" / "canonical_source.zip"
    with zipfile.ZipFile(src_zip, "w") as zf:
        zf.writestr("RS/README.md", "x\n")
    write_relative_checksums(tmp_path)
    zip_path = tmp_path / "pkg.zip"
    build_portable_zip(tmp_path, zip_path)
    verification = fresh_unpack_verify(zip_path, "deadbeef")
    assert verification.portable_zip_paths is True
