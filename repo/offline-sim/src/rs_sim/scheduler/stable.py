from __future__ import annotations

"""Deterministic serialization helpers.

The implementation intentionally rejects opaque objects instead of falling back
onto ``repr`` or process-randomized ``hash`` output.
"""

import dataclasses
import enum
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_data(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise TypeError(
            "float is not permitted in authoritative RS-SIM scheduling serialization"
        )
    if isinstance(value, enum.Enum):
        return {"enum": value.__class__.__qualname__, "name": value.name}
    stable_payload = getattr(value, "stable_payload", None)
    if callable(stable_payload):
        return canonical_data(stable_payload())
    if dataclasses.is_dataclass(value):
        return {
            field.name: canonical_data(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        items = [(str(key), canonical_data(item)) for key, item in value.items()]
        items.sort(key=lambda pair: pair[0])
        return {key: item for key, item in items}
    if isinstance(value, (tuple, list)):
        return [canonical_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [canonical_data(item) for item in value]
        normalized.sort(key=stable_json)
        return normalized
    asdict = getattr(value, "_asdict", None)
    if callable(asdict):
        return canonical_data(asdict())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return canonical_data(to_dict())
    raise TypeError(
        f"opaque value {type(value).__module__}.{type(value).__qualname__} has no stable payload"
    )


def stable_json(value: Any) -> str:
    return json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_digest(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: Any, *, length: int = 24) -> str:
    if not prefix or ":" in prefix:
        raise ValueError("stable id prefix must be non-empty and colon-free")
    digest = stable_digest(value)
    return f"{prefix}:{digest[: int(length)]}"
