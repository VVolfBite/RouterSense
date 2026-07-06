from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from rs.runtime.online.megatron_ep.phase.contracts import FutureDemandHint

from .p2_contracts import P2HintRequest


def _digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class P2HintProvider(Protocol):
    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        ...


class NoP2HintProvider:
    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        return FutureDemandHint(hint_mode="none", hint_digest="none", hint_source="no_p2_hint_provider")


class DeterministicStubP2HintProvider:
    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        payload = {
            "plan_key": request.plan_key,
            "layer_id": request.layer_id,
            "phase": request.phase,
            "global_rank": request.global_rank,
            "local_rank": request.local_rank,
            "ep_group_ranks": list(request.ep_group_ranks),
        }
        digest = _digest(payload)
        return FutureDemandHint(
            hint_mode="deterministic_stub",
            hint_digest=digest,
            hint_source="deterministic_stub_from_current_plan_key",
            metadata={"tag": f"stub:{request.phase}:{request.layer_id}:{request.global_rank}", "digest": digest},
        )


class CalibratedArtifactP2HintProvider:
    def __init__(self, *, shared_state: dict[str, Any]) -> None:
        self._shared_state = shared_state

    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        prepared_plan = self._shared_state.get("prepared_plan")
        if prepared_plan is None:
            return FutureDemandHint(
                hint_mode="none",
                hint_digest="none",
                hint_source="no_prepared_plan_available",
                metadata={"requested_layer_id": str(request.layer_id)},
            )
        plan_source_layer = str(self._shared_state.get("plan_source_layer", ""))
        plan_created_at_us = int(self._shared_state.get("plan_created_at_us", 0) or 0)
        forecast_digest = str(getattr(prepared_plan, "forecast_digest", ""))
        digest = hashlib.sha256(f"{forecast_digest}:{request.layer_id}".encode("utf-8")).hexdigest()[:16]
        return FutureDemandHint(
            hint_mode="calibrated_artifact",
            hint_digest=digest,
            hint_source=f"calibrated_artifact_from_layer_{plan_source_layer}",
            metadata={
                "source_layer": plan_source_layer,
                "window_key": str(getattr(prepared_plan, "window_key", "")),
                "plan_created_at_us": plan_created_at_us,
            },
        )


def build_p2_hint_provider(mode: str, *, shared_state: dict[str, Any] | None = None) -> P2HintProvider:
    if mode == "none":
        return NoP2HintProvider()
    if mode == "deterministic_stub":
        return DeterministicStubP2HintProvider()
    if mode == "calibrated_artifact":
        if shared_state is None:
            raise ValueError("p2_hint_mode='calibrated_artifact' requires shared_state")
        return CalibratedArtifactP2HintProvider(shared_state=shared_state)
    raise ValueError(f"Unsupported p2_hint_mode={mode!r}")
