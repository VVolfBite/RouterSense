from __future__ import annotations

import json
from hashlib import sha256
from typing import Any


def build_plan_protocol_stub(*, plan_id: str, plan_hash: str, layer_id: int, microbatch_id: str, information_mode: str) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "layer_id": layer_id,
        "microbatch_id": microbatch_id,
        "information_mode": information_mode,
    }


def compute_plan_hash(protocol_payload: dict[str, Any]) -> str:
    canonical = json.dumps(protocol_payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def assert_plan_hash_agreement(plan_hashes: list[str]) -> None:
    unique = sorted(set(str(item) for item in plan_hashes))
    if len(unique) != 1:
        raise RuntimeError(f"plan hash mismatch across ranks: {unique}")
