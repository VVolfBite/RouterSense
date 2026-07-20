"""Fingerprint validation helpers for runtime injection smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def assert_expected_fingerprint(expected_path: str | None, actual: dict[str, Any]) -> None:
    if not expected_path:
        return
    path = Path(expected_path)
    expected = json.loads(path.read_text(encoding="utf-8"))
    if expected != actual:
        raise RuntimeError(
            "Host API drift detected: actual dispatcher fingerprint does not match expected fingerprint"
        )


__all__ = ["assert_expected_fingerprint"]

