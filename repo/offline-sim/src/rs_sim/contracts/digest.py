from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from collections.abc import Mapping
from typing import Any


class StableSerializationError(TypeError):
    """Raised when a value cannot be represented by the stable codec."""


def _canonicalize(value: Any) -> Any:
    """Convert supported immutable values to a canonical JSON tree.

    Authoritative RS-SIM objects intentionally reject floats and unordered
    containers. Time and all cost fields are integer nanoseconds.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise StableSerializationError(
            "float is not permitted in authoritative RS-SIM serialization"
        )
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, enum.Enum):
        return {
            "__enum__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "value": _canonicalize(value.value),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "fields": {
                field.name: _canonicalize(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, list):
        raise StableSerializationError("list is mutable; use tuple")
    if isinstance(value, (set, frozenset)):
        raise StableSerializationError("unordered containers are not permitted")
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise StableSerializationError("mapping keys must be strings")
        return {
            key: _canonicalize(value[key])
            for key in sorted(value)
        }
    raise StableSerializationError(
        f"unsupported stable serialization type: {type(value).__qualname__}"
    )


def stable_json_dumps(value: Any) -> str:
    """Return a deterministic UTF-8 JSON representation."""

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_digest(value: Any, *, domain: str = "RS_SIM_CANONICAL_TASKIZATION") -> str:
    """Return a domain-separated SHA-256 digest for a supported value."""

    payload = stable_json_dumps(value).encode("utf-8")
    prefix = domain.encode("utf-8") + b"\x00"
    return hashlib.sha256(prefix + payload).hexdigest()
