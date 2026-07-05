from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

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
    def __init__(self, *, artifact_path: str) -> None:
        self.artifact_path = str(artifact_path)

    def build_hint(self, request: P2HintRequest) -> FutureDemandHint:
        path = Path(self.artifact_path)
        if not self.artifact_path or not path.exists():
            raise ValueError("incompatible_p2_artifact")
        digest = _digest(
            {
                "artifact_path": str(path.resolve()),
                "size": path.stat().st_size,
                "plan_key": request.plan_key,
                "layer_id": request.layer_id,
                "phase": request.phase,
            }
        )
        return FutureDemandHint(
            hint_mode="calibrated_artifact",
            hint_digest=digest,
            hint_source="calibrated_artifact",
            metadata={"artifact_path": str(path.resolve()), "artifact_sha256": digest},
        )


def build_p2_hint_provider(mode: str) -> P2HintProvider:
    if mode == "none":
        return NoP2HintProvider()
    if mode == "deterministic_stub":
        return DeterministicStubP2HintProvider()
    if mode == "calibrated_artifact":
        return CalibratedArtifactP2HintProvider(artifact_path=os.environ.get("ROUTERSENSE_P2_HINT_ARTIFACT", ""))
    raise ValueError(f"Unsupported p2_hint_mode={mode!r}")
