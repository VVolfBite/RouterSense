from __future__ import annotations

"""Pure, family-agnostic Safe pair selection kernel.

This module owns only the decision rule.  It does not build plans, validate
RouterSense contracts, import runtime code, or know historical U/B names.
Offline algorithm adapters and formal online Planner wrappers both call this
same function.
"""

from dataclasses import dataclass
import math
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SafeCandidate(Generic[T]):
    role: str
    payload: T | None
    valid: bool
    objective: float
    score: float
    error: str | None = None

    def validate(self) -> None:
        if self.role not in {"joint", "local"}:
            raise ValueError("Safe candidate role must be joint/local")
        if self.valid:
            if self.payload is None:
                raise ValueError("valid Safe candidate requires payload")
            if not math.isfinite(float(self.objective)) or float(self.objective) < 0.0:
                raise ValueError("valid Safe candidate objective must be finite and non-negative")
            if not math.isfinite(float(self.score)) or float(self.score) < 0.0:
                raise ValueError("valid Safe candidate score must be finite and non-negative")
        elif self.payload is not None:
            raise ValueError("invalid Safe candidate must not carry payload")


@dataclass(frozen=True)
class SafeSelection(Generic[T]):
    selected: SafeCandidate[T]
    joint: SafeCandidate[T]
    local: SafeCandidate[T]
    reason: str


class SafePairSelectionError(RuntimeError):
    pass


def select_safe_pair(
    *,
    joint: SafeCandidate[T],
    local: SafeCandidate[T],
    tie_break: str = "local",
    minimum_joint_gain: float = 0.0,
) -> SafeSelection[T]:
    joint.validate(); local.validate()
    if joint.role != "joint" or local.role != "local":
        raise ValueError("Safe pair candidates must be passed as Joint then Local")
    if tie_break not in {"local", "joint"}:
        raise ValueError("tie_break must be local or joint")
    if not math.isfinite(float(minimum_joint_gain)) or float(minimum_joint_gain) < 0.0:
        raise ValueError("minimum_joint_gain must be finite and non-negative")
    if not joint.valid and not local.valid:
        raise SafePairSelectionError(
            f"both Safe candidates invalid: joint={joint.error!r}, local={local.error!r}"
        )
    if joint.valid and not local.valid:
        return SafeSelection(joint, joint, local, "local_invalid")
    if local.valid and not joint.valid:
        return SafeSelection(local, joint, local, "joint_invalid")
    margin = float(minimum_joint_gain)
    if joint.score + margin < local.score:
        return SafeSelection(joint, joint, local, "joint_lower_cost")
    if local.score + 1e-12 < joint.score + margin:
        return SafeSelection(local, joint, local, "local_lower_or_joint_margin_not_met")
    selected = local if tie_break == "local" else joint
    return SafeSelection(selected, joint, local, f"tie_prefer_{tie_break}")


__all__ = [
    "SafeCandidate",
    "SafePairSelectionError",
    "SafeSelection",
    "select_safe_pair",
]
