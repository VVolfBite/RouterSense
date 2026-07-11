from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReportBundle:
    report_type: str
    title: str
    summary: dict[str, Any]
    markdown: str


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "manifest.json")

