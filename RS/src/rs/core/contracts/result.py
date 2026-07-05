"""Result-envelope contracts for RouteSense experiments and runtime outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


OFFLINE_PIPELINE = "offline"
ONLINE_PIPELINE = "online"
LEGACY_TRACE_REPLAY_PIPELINE = "legacy"


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    pipeline: str
    claim_scope: str
    trace_origin: str
    future_information_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_result_envelope(
    *,
    run_id: str,
    pipeline: str,
    claim_scope: str,
    trace_origin: str,
    future_information_mode: str,
    is_real_ep_runtime: bool,
    source_ownership_mode: str,
    expert_residency_mode: str,
    transport_backend: str,
    correctness_status: str,
    performance_claim_eligible: bool,
    execution_mode: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "run_identity": RunIdentity(
            run_id=run_id,
            pipeline=pipeline,
            claim_scope=claim_scope,
            trace_origin=trace_origin,
            future_information_mode=future_information_mode,
        ).to_dict(),
        "pipeline": pipeline,
        "claim_scope": claim_scope,
        "trace_origin": trace_origin,
        "future_information_mode": future_information_mode,
        "is_real_ep_runtime": is_real_ep_runtime,
        "source_ownership_mode": source_ownership_mode,
        "expert_residency_mode": expert_residency_mode,
        "transport_backend": transport_backend,
        "correctness_status": correctness_status,
        "performance_claim_eligible": performance_claim_eligible,
    }
    if execution_mode is not None:
        payload["execution_mode"] = execution_mode
    if extra:
        payload.update(extra)
    return payload


__all__ = [
    "LEGACY_TRACE_REPLAY_PIPELINE",
    "OFFLINE_PIPELINE",
    "ONLINE_PIPELINE",
    "RunIdentity",
    "build_result_envelope",
]
