"""Adapters for execution-priority artifacts produced by formal plans."""
from __future__ import annotations
from typing import Any
from .plan_priority import PlanPriorityArtifact, build_priority_artifact_from_plan

class PlanPriorityPhaseSyncAdapter:
    def build_ordering_hint(self, artifact: PlanPriorityArtifact) -> dict[str, Any]:
        return {
            "source_policy": artifact.source_policy,
            "joint_policy": artifact.joint_policy,
            "local_policy": artifact.local_policy,
            "selected_policy": artifact.selected_policy,
            "fallback_to_local": bool(artifact.fallback_to_local),
            "planner_family": artifact.heuristic_family,
            "predictor_name": artifact.predictor_name,
            "p2_source": artifact.p2_source,
            "granularity_mode": "dynamic_bucket_current",
            "priority_digest": artifact.priority_digest,
            "generated_offline_or_shadow": True,
            "heavy_solver_used_offline": bool(artifact.heavy_solver_used_offline),
            "ordering_hint": tuple(entry.to_dict() for entry in artifact.priority_entries),
        }

class PlanPriorityAsyncReleaseAdapter:
    def build_release_priority(self, artifact: PlanPriorityArtifact, *, fallback_required: bool = False) -> dict[str, Any]:
        return {
            "source_policy": artifact.source_policy,
            "joint_policy": artifact.joint_policy,
            "local_policy": artifact.local_policy,
            "selected_policy": artifact.selected_policy,
            "fallback_to_local": bool(artifact.fallback_to_local),
            "planner_family": artifact.heuristic_family,
            "predictor_name": artifact.predictor_name,
            "p2_source": artifact.p2_source,
            "granularity_mode": "dynamic_bucket_current",
            "priority_digest": artifact.priority_digest,
            "release_priority": tuple(entry.to_dict() for entry in artifact.priority_entries),
            "fallback_decision": "fallback_phase_sync" if fallback_required else "release_ready_tasks",
            "generated_offline_or_shadow": True,
            "heavy_solver_used_offline": bool(artifact.heavy_solver_used_offline),
        }

__all__ = ["PlanPriorityArtifact", "PlanPriorityAsyncReleaseAdapter", "PlanPriorityPhaseSyncAdapter", "build_priority_artifact_from_plan"]
