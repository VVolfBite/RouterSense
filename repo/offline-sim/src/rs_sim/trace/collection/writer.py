"""Small rank-local artifact writer used inside instrumented model processes."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class RankArtifactWriter:
    def __init__(self, output_dir: Path, *, global_rank: int, source_rank: int) -> None:
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        stem = f"rank{source_rank:04d}-global{global_rank:04d}"
        self.routing_path = self.raw_dir / f"{stem}_source_expert_counts.jsonl"
        self.compute_path = self.raw_dir / f"{stem}_local_compute.jsonl"
        self.warning_path = self.raw_dir / f"{stem}_capture_warnings.jsonl"
        self.fate_path = self.raw_dir / f"{stem}_fate_p2_rows.jsonl"
        self.manifest_path = self.raw_dir / f"{stem}_capture_manifest.json"
        self._lock = threading.Lock()

    def append_routing(self, payload: dict[str, Any]) -> None:
        self._append(self.routing_path, payload)

    def append_compute(self, payload: dict[str, Any]) -> None:
        self._append(self.compute_path, payload)

    def append_warning(self, payload: dict[str, Any]) -> None:
        self._append(self.warning_path, payload)

    def append_fate(self, payload: dict[str, Any]) -> None:
        self._append(self.fate_path, payload)

    def write_manifest(self, payload: dict[str, Any]) -> None:
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(fd, (line + "\n").encode("utf-8"))
            finally:
                os.close(fd)
