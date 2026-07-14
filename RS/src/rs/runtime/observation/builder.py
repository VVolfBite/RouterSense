from __future__ import annotations

from typing import Mapping


def build_operational_observation(
    *,
    run_id: str,
    layer_id: str,
    phase: str,
    layout_digest: str,
    payload_roles: tuple[str, ...],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "run_id": str(run_id),
        "layer_id": str(layer_id),
        "phase": str(phase),
        "layout_digest": str(layout_digest),
        "payload_roles": [str(role) for role in payload_roles],
        "metadata": dict(metadata or {}),
    }
