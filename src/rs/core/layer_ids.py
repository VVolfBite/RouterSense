from __future__ import annotations

import re
from typing import Any


_SPLIT_RE = re.compile(r"(\d+)")


def stable_layer_sort_key(value: Any) -> tuple[Any, ...]:
    text = str(value)
    if text.isdigit():
        return (0, int(text), text)
    parts: list[Any] = []
    for part in _SPLIT_RE.split(text):
        if not part:
            continue
        parts.append((0, int(part)) if part.isdigit() else (1, part))
    return (1, *parts, text)


def stable_layer_ids(values: Any) -> list[str]:
    return sorted((str(item) for item in values), key=stable_layer_sort_key)


def stable_layer_count_map(payload: dict[Any, Any]) -> dict[str, int]:
    return {key: int(payload[key]) for key in stable_layer_ids(payload.keys())}
