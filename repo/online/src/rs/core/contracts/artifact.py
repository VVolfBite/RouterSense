from __future__ import annotations

from typing import Mapping, Protocol


class ArtifactWriter(Protocol):
    def write(self, *, category: str, name: str, payload: Mapping[str, object]) -> str:
        ...
