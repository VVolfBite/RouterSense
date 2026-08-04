"""Stable identifiers and primitive validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from rs_sim.backend.core.errors import BackendContractError


def require_time_ns(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BackendContractError(f"{field} must be a non-negative int nanosecond value")
    return value


def require_nonnegative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BackendContractError(f"{field} must be a non-negative int")
    return value


def require_positive_int(value: Any, *, field: str) -> int:
    value = require_nonnegative_int(value, field=field)
    if value == 0:
        raise BackendContractError(f"{field} must be positive")
    return value


def stable_digest(parts: Iterable[Any], *, prefix: str) -> str:
    payload = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def stable_semantic_event_id(
    *, event_kind: str, time_ns: int, semantic_parts: Iterable[Any]
) -> str:
    return stable_digest(
        ["BACKEND", time_ns, 4, event_kind, *semantic_parts], prefix="backend-event"
    )
