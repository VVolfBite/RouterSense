from __future__ import annotations

from experiments.paper.result_bundle import sha256_file


def test_result_bundle_sha256(tmp_path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("abc", encoding="utf-8")
    assert len(sha256_file(path)) == 64
