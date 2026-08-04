"""Adapters that consume offline/shadow U-family priority artifacts."""

from __future__ import annotations

from typing import Any

from .priority_artifact import PairedUPriorityArtifact, build_priority_artifact_from_plan


class PairedUPhaseSyncAdapter:
    """Translate U-family artifact summaries into phase-sync ordering hints."""

    def build_ordering_hint(self, artifact: PairedUPriorityArtifact) -> dict[str, Any]:
        return {
            "source_safe_policy": artifact.source_safe_policy,
            "source_u_policy": artifact.raw_u_policy,
            "paired_b_policy": artifact.paired_b_policy,
            "selected_policy": artifact.selected_policy,
            "fallback_to_paired_b": bool(artifact.fallback_to_paired_b),
            "heuristic_family": artifact.heuristic_family,
            "predictor_name": artifact.predictor_name,
            "p2_source": artifact.p2_source,
            "granularity_mode": "dynamic_bucket_current",
            "priority_digest": artifact.priority_digest,
            "generated_offline_or_shadow": True,
            "heavy_solver_used_offline": bool(artifact.heavy_solver_used_offline),
            "ordering_hint": tuple(entry.to_dict() for entry in artifact.priority_entries),
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
            "source_safe_policy": artifact.source_safe_policy,
            "source_u_policy": artifact.raw_u_policy,
            "paired_b_policy": artifact.paired_b_policy,
            "selected_policy": artifact.selected_policy,
            "fallback_to_paired_b": bool(artifact.fallback_to_paired_b),
            "heuristic_family": artifact.heuristic_family,
            "predictor_name": artifact.predictor_name,
            "p2_source": artifact.p2_source,
            "granularity_mode": "dynamic_bucket_current",
            "priority_digest": artifact.priority_digest,
            "release_priority": tuple(entry.to_dict() for entry in artifact.priority_entries),
            "fallback_decision": "fallback_phase_sync" if fallback_required else "release_ready_tasks",
            "generated_offline_or_shadow": True,
            "heavy_solver_used_offline": bool(artifact.heavy_solver_used_offline),
        }


__all__ = [
    "PairedUPriorityArtifact",
    "PairedUAsyncReleaseAdapter",
    "PairedUPhaseSyncAdapter",
    "build_priority_artifact_from_plan",
]
