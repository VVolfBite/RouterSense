"""Adapters that consume offline/shadow U-family priority artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class PairedUPriorityArtifact:
    source_u_policy: str
    paired_b_policy: str
    predictor_name: str
    p2_source: str
    granularity_mode: str = "dynamic_bucket_current"
    priority_table: tuple[dict[str, Any], ...] = ()
    generated_offline_or_shadow: bool = True
    heavy_solver_used_offline: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def priority_digest(self) -> str:
        return _stable_digest(self.to_dict())


class PairedUPhaseSyncAdapter:
    """Translate U-family artifact summaries into phase-sync ordering hints."""

    def build_ordering_hint(self, artifact: PairedUPriorityArtifact) -> dict[str, Any]:
        return {
            "source_u_policy": artifact.source_u_policy,
            "paired_b_policy": artifact.paired_b_policy,
            "predictor_name": artifact.predictor_name,
            "p2_source": artifact.p2_source,
            "granularity_mode": "dynamic_bucket_current",
            "priority_digest": artifact.priority_digest,
            "generated_offline_or_shadow": bool(artifact.generated_offline_or_shadow),
            "heavy_solver_used_offline": bool(artifact.heavy_solver_used_offline),
            "ordering_hint": tuple(artifact.priority_table),
        }


class PairedUAsyncReleaseAdapter:
    """Translate U-family artifact summaries into async-release shadow decisions."""

    def build_release_priority(
        self,
        artifact: PairedUPriorityArtifact,
        *,
        fallback_required: bool = False,
    ) -> dict[str, Any]:
        return {
            "source_u_policy": artifact.source_u_policy,
            "paired_b_policy": artifact.paired_b_policy,
            "predictor_name": artifact.predictor_name,
            "p2_source": artifact.p2_source,
            "granularity_mode": "dynamic_bucket_current",
            "priority_digest": artifact.priority_digest,
            "release_priority": tuple(artifact.priority_table),
            "fallback_decision": "fallback_phase_sync" if fallback_required else "release_ready_tasks",
            "generated_offline_or_shadow": bool(artifact.generated_offline_or_shadow),
            "heavy_solver_used_offline": bool(artifact.heavy_solver_used_offline),
        }
