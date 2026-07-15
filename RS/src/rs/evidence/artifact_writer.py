from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from rs.core.contracts.artifact import ArtifactRecord
from rs.core.contracts.result import ResultBundle
from rs.evidence.serialization import EvidenceSerializer


class FilesystemArtifactWriter:
    def __init__(self, *, root_dir: str | Path, serializer: EvidenceSerializer | None = None) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.serializer = serializer or EvidenceSerializer()
        self._manifest_path = self.root_dir / "manifest.json"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self._manifest_path.exists():
            self._manifest_path.write_text(
                json.dumps({"artifacts": []}, ensure_ascii=True, sort_keys=True, indent=2),
                encoding="utf-8",
            )

    def _resolve_target(self, relative_path: str) -> Path:
        normalized = str(PurePosixPath(relative_path))
        candidate = (self.root_dir / normalized).resolve()
        try:
            candidate.relative_to(self.root_dir)
        except ValueError as exc:
            raise ValueError("artifact path escapes root_dir") from exc
        return candidate

    def _write_bytes(self, *, relative_path: str, payload: bytes) -> ArtifactRecord:
        target = self._resolve_target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_name(f"{target.name}.tmp")
        with temp_target.open("wb") as handle:
            handle.write(payload)
            handle.flush()
        temp_target.replace(target)
        return ArtifactRecord(
            relative_path=str(target.relative_to(self.root_dir)).replace("\\", "/"),
            schema="",
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            producer="",
            claim_role="",
        )

    def _append_manifest(self, record: ArtifactRecord) -> None:
        payload = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        payload.setdefault("artifacts", []).append(record.to_dict())
        temp = self._manifest_path.with_name("manifest.json.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")
        temp.replace(self._manifest_path)

    def write(
        self,
        *,
        category: str,
        name: str,
        payload: Mapping[str, object] | str,
        format: str,
        schema: str,
        producer: str,
        claim_role: str,
    ) -> ArtifactRecord:
        category_name = str(PurePosixPath(category))
        file_name = str(PurePosixPath(name))
        if format == "text":
            encoded = str(payload).encode("utf-8")
        elif format == "json":
            encoded = json.dumps(dict(payload) if isinstance(payload, Mapping) else payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        elif format == "result_bundle":
            if not isinstance(payload, ResultBundle):
                raise TypeError("result_bundle format requires a ResultBundle payload")
            encoded = self.serializer.serialize_result(payload).encode("utf-8")
        else:
            raise ValueError(f"unsupported artifact format: {format}")
        base_record = self._write_bytes(relative_path=f"{category_name}/{file_name}", payload=encoded)
        record = ArtifactRecord(
            relative_path=base_record.relative_path,
            schema=str(schema),
            sha256=base_record.sha256,
            size_bytes=base_record.size_bytes,
            producer=str(producer),
            claim_role=str(claim_role),
        )
        self._append_manifest(record)
        return record

    def write_text(self, *, relative_path: str, payload: str) -> str:
        record = self.write(
            category=str(PurePosixPath(relative_path).parent),
            name=PurePosixPath(relative_path).name,
            payload=str(payload),
            format="text",
            schema="artifact_text.v2",
            producer="legacy_write_text",
            claim_role="diagnostic",
        )
        return str(self._resolve_target(record.relative_path))
