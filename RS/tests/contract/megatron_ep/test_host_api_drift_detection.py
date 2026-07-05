from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.online.run_injection_smoke import _assert_expected_fingerprint


def test_host_api_drift_detection_raises_on_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "fingerprint.json"
    path.write_text(json.dumps({"dispatchers": {"layer": {"module_sha256": "abc"}}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        _assert_expected_fingerprint(str(path), {"dispatchers": {"layer": {"module_sha256": "def"}}})
