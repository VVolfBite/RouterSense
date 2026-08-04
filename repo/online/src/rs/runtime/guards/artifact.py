from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import RouterSenseInvariantError


def write_failure_artifact(path: Path, *, error: RouterSenseInvariantError, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "error_code": error.failure.error_code,
        "stage": error.failure.stage,
        "message": error.failure.message,
        "rank": error.failure.rank,
        "layer_name": error.failure.layer_name,
        "layer_id": error.failure.layer_id,
        "forward_epoch": error.failure.forward_epoch,
        "phase": error.failure.phase,
        "tensor_role": error.failure.tensor_role,
        "expected": error.failure.expected,
        "actual": error.failure.actual,
        "logical_plan_digest": error.failure.logical_plan_digest,
        "compiled_plan_digest": error.failure.compiled_plan_digest,
        "execution_plan_digest": error.failure.execution_plan_digest,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
