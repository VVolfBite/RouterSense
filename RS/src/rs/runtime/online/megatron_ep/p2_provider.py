from __future__ import annotations

import hashlib
import json
from typing import Protocol

from rs.runtime.online.megatron_ep.phase.contracts import FutureDemandHint

from .p2_contracts import P2HintRequest


def _digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class P2HintProvider(Protocol):
    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        ...


class UnsupportedP2Predictor(RuntimeError):
    pass


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
    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        raise UnsupportedP2Predictor(
            "p2_hint_mode='calibrated_artifact' is not implemented in the frozen runtime; "
            "only 'none' and 'deterministic_stub' are currently supported"
        )


def build_p2_hint_provider(mode: str) -> P2HintProvider:
    if mode == "none":
        return NoP2HintProvider()
    if mode == "deterministic_stub":
        return DeterministicStubP2HintProvider()
    if mode == "calibrated_artifact":
        return CalibratedArtifactP2HintProvider()
    raise ValueError(f"Unsupported p2_hint_mode={mode!r}")
