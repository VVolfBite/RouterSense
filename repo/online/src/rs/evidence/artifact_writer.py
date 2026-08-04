from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rs.evidence.serialization import EvidenceSerializer


class FilesystemArtifactWriter:
    def __init__(self, *, root_dir: str | Path, serializer: EvidenceSerializer | None = None) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.serializer = serializer or EvidenceSerializer()

    def _resolve_target(self, relative_path: str) -> Path:
        candidate = (self.root_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.root_dir)
        except ValueError as exc:
            raise ValueError("artifact path escapes root_dir") from exc
        return candidate

    def write_text(self, *, relative_path: str, payload: str) -> str:
        target = self._resolve_target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_name(f"{target.name}.tmp")
        temp_target.write_text(payload, encoding="utf-8")
        temp_target.replace(target)
        manifest_path = target.with_name(f"{target.name}.manifest.json")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        manifest_path.write_text(
            json.dumps(
                {
                    "artifact_path": str(target.relative_to(self.root_dir)).replace("\\", "/"),
                    "schema": "artifact_text.v1",
                    "sha256": digest,
                    "size_bytes": len(payload.encode("utf-8")),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return str(target)
