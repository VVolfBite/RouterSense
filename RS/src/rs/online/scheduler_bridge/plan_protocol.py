from __future__ import annotations

from typing import Any


def build_plan_protocol_stub(*, plan_id: str, plan_hash: str, layer_id: int, microbatch_id: str, information_mode: str) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "layer_id": layer_id,
        "microbatch_id": microbatch_id,
        "information_mode": information_mode,
    }
