"""Merge-friendly default adapters for shared-schema immutable objects.

They read attributes/mapping keys only; they do not define shared schemas.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any, Callable

from rs_sim.backend.core.errors import BackendContractError


class AttributeSharedObjectAdapter:
    """Read frozen objects supplied by shared-schema using attributes or mapping keys."""

    def get(self, obj: Any, field: str) -> Any:
        if isinstance(obj, Mapping) and field in obj:
            return obj[field]
        if hasattr(obj, field):
            return getattr(obj, field)
        raise BackendContractError(
            f"shared object {type(obj).__name__} is missing required field {field!r}"
        )

    def stable_key(self, value: Any) -> str:
        if value is None or isinstance(value, (str, int, bool)):
            return json.dumps(value, separators=(",", ":"), sort_keys=True)
        if isinstance(value, tuple):
            return json.dumps(
                [self.stable_key(item) for item in value], separators=(",", ":")
            )
        if isinstance(value, Mapping):
            normalized = {
                str(key): self.stable_key(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
            return json.dumps(normalized, separators=(",", ":"), sort_keys=True)
        if dataclasses.is_dataclass(value):
            return json.dumps(
                dataclasses.asdict(value),
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
        stable_digest = getattr(value, "stable_digest", None)
        if callable(stable_digest):
            return str(stable_digest())
        digest = getattr(value, "digest", None)
        if isinstance(digest, str):
            return digest
        raise BackendContractError(
            f"no deterministic stable key adapter for {type(value).__name__}"
        )


class CallablePhaseSemantics:
    """Phase semantics backed by integration-provided callables."""

    def __init__(
        self,
        *,
        phase_kind: Callable[[Any], str],
        phase_sort_key: Callable[[Any], str],
    ) -> None:
        self._phase_kind = phase_kind
        self._phase_sort_key = phase_sort_key
        self._kind_cache: dict[int, tuple[Any, str]] = {}
        self._sort_key_cache: dict[int, tuple[Any, str]] = {}

    def phase_kind(self, phase_key: Any) -> str:
        cache_key = id(phase_key)
        cached = self._kind_cache.get(cache_key)
        if cached is not None and cached[0] is phase_key:
            return cached[1]
        kind = self._phase_kind(phase_key).upper()
        if kind not in {"DISPATCH", "COMBINE"}:
            raise BackendContractError(
                f"phase kind must be DISPATCH or COMBINE, got {kind!r}"
            )
        self._kind_cache[cache_key] = (phase_key, kind)
        return kind

    def phase_sort_key(self, phase_key: Any) -> str:
        cache_key = id(phase_key)
        cached = self._sort_key_cache.get(cache_key)
        if cached is not None and cached[0] is phase_key:
            return cached[1]
        value = str(self._phase_sort_key(phase_key))
        self._sort_key_cache[cache_key] = (phase_key, value)
        return value
