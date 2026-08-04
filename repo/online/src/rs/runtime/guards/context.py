from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InvariantContext:
    stage: str
    error_code: str
    rank: int | None = None
    layer_name: str | None = None
    layer_id: int | None = None
    forward_epoch: int | None = None
    phase: str | None = None
    tensor_role: str | None = None
    logical_plan_digest: str = ""
    compiled_plan_digest: str = ""
    execution_plan_digest: str = ""
    metadata: dict[str, Any] | None = None

