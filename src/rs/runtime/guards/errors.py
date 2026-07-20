from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InvariantFailure:
    error_code: str
    stage: str
    message: str
    rank: int | None = None
    layer_name: str | None = None
    layer_id: int | None = None
    forward_epoch: int | None = None
    phase: str | None = None
    tensor_role: str | None = None
    expected: Any = None
    actual: Any = None
    logical_plan_digest: str = ""
    compiled_plan_digest: str = ""
    execution_plan_digest: str = ""


class RouterSenseInvariantError(RuntimeError):
    def __init__(self, failure: InvariantFailure) -> None:
        self.failure = failure
        super().__init__(f"[{failure.error_code}] {failure.stage}: {failure.message}")


class RuntimeStateFieldError(RouterSenseInvariantError):
    pass

