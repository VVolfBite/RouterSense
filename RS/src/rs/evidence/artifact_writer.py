from __future__ import annotations

from pathlib import Path

from rs.evidence.serialization import EvidenceSerializer


class FilesystemArtifactWriter:
    def __init__(self, *, root_dir: str | Path, serializer: EvidenceSerializer | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.serializer = serializer or EvidenceSerializer()

    def write_text(self, *, relative_path: str, payload: str) -> str:
        target = self.root_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        return str(target)
