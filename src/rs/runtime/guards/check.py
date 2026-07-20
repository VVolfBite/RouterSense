from __future__ import annotations

from typing import Any

from .context import InvariantContext
from .errors import InvariantFailure, RouterSenseInvariantError


def require_invariant(
    condition: bool,
    *,
    context: InvariantContext,
    message: str,
    expected: Any = None,
    actual: Any = None,
) -> None:
    if condition:
        return
    raise RouterSenseInvariantError(
        InvariantFailure(
            error_code=str(context.error_code),
            stage=str(context.stage),
            message=str(message),
            rank=context.rank,
            layer_name=context.layer_name,
            layer_id=context.layer_id,
            forward_epoch=context.forward_epoch,
            phase=context.phase,
            tensor_role=context.tensor_role,
            expected=expected,
            actual=actual,
            logical_plan_digest=str(context.logical_plan_digest),
            compiled_plan_digest=str(context.compiled_plan_digest),
            execution_plan_digest=str(context.execution_plan_digest),
        )
    )

