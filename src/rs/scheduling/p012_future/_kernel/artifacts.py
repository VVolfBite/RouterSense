from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any
import numpy as np

from .contracts import _digest, _freeze_metadata, semantic_metadata
from .plan import CompactWindowPlan, WindowPlan


@dataclass(frozen=True)
class ForecastArtifact:
    request_digest: str
    planner_id: str
    planner_family: str
    branch: str
    raw_template: tuple
    plan: CompactWindowPlan | WindowPlan
    hint_rows: object
    metadata: Mapping[str, Any]
    _digest_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(str(self.request_digest)) != 64:
            raise ValueError("request_digest must be a SHA-256 digest")
        frozen_template = []
        for value in self.raw_template:
            if isinstance(value, np.ndarray):
                arr = np.ascontiguousarray(value).copy(); arr.setflags(write=False); frozen_template.append(arr)
            else:
                frozen_template.append(value)
        hint = np.ascontiguousarray(np.asarray(self.hint_rows, dtype=np.int32)).copy()
        if hint.ndim != 2 or hint.shape[0] != hint.shape[1]:
            raise ValueError("hint_rows must be a square matrix")
        hint.setflags(write=False)
        object.__setattr__(self, "raw_template", tuple(frozen_template))
        object.__setattr__(self, "hint_rows", hint)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def semantic_payload(self) -> dict:
        return {
            "forecast_artifact_semantic_version": "forecast_artifact_v4",
            "request_digest": self.request_digest,
            "planner_id": self.planner_id,
            "planner_family": self.planner_family,
            "branch": self.branch,
            "plan_digest": self.plan.semantic_digest(),
            "hint_rows": self.hint_rows.tolist(),
            "metadata": semantic_metadata(self.metadata),
        }

    def semantic_digest(self) -> str:
        cached = self._digest_cache
        if cached is None:
            cached = _digest(self.semantic_payload()); object.__setattr__(self, "_digest_cache", cached)
        return cached


__all__ = ["ForecastArtifact"]
