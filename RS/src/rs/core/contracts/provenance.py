"""Source/result provenance contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceProvenance:
    git_commit: str | None = None
    git_dirty: bool | None = None
    source_archive_sha256: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
