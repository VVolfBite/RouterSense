from __future__ import annotations

"""Reporting-stage evaluation helpers."""

from pathlib import Path
from typing import Any

from .analysis import write_json
from .artifacts import collect_environment_snapshot, write_artifact_bundle


def write_csv(_path: str | Path, _rows: list[dict[str, Any]]) -> None:
    raise NotImplementedError("CSV export is not implemented in the current evaluation pipeline.")


__all__ = [
    "collect_environment_snapshot",
    "write_artifact_bundle",
    "write_csv",
    "write_json",
]
