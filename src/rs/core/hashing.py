"""Deterministic hashing helpers for contracts, plans, and artifacts."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any


def stable_hash_json(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def stable_hash_dict(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return stable_hash_json(canonical)
