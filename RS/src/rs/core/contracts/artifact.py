from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class ArtifactRecord:
    relative_path: str
    schema: str
    sha256: str
    size_bytes: int
    producer: str
    claim_role: str

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": str(self.relative_path),
            "schema": str(self.schema),
            "sha256": str(self.sha256),
            "size_bytes": int(self.size_bytes),
            "producer": str(self.producer),
            "claim_role": str(self.claim_role),
        }


class ArtifactWriter(Protocol):
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
        ...
