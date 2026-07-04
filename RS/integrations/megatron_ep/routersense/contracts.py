from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NativeEPSummary:
    pipeline: str = "host_runtime_native_ep"
    host_runtime: str = "megatron_core"
    model_family: str = "olmoe"
    ep_size: int = 0
    dispatcher: str = "alltoall"
    backend: str = "nccl"
    forward_completed: bool = False
    remote_dispatch_exercised: bool = False
    remote_combine_exercised: bool = False
    is_legacy_harness: bool = False
    performance_claim_eligible: bool = False
    status: str = "blocked_environment"
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
