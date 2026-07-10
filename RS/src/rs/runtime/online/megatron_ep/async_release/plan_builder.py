"""CPU-testable async-release execution-plan skeleton.

This builder does not invoke NCCL or real executor paths. It materializes the
dependency/release contract that a future async-release online executor would
consume, and fails closed into phase_sync when execution integration is absent.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rs.scheduling.online_adapters.priority_artifact import PairedUPriorityArtifact

from .contracts import AsyncReleaseExecutionPlan


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class AsyncReleasePlanBuilder:
    def __init__(self, *, executor_available: bool = False) -> None:
        self.executor_available = bool(executor_available)

    def build(
        self,
        *,
        priority_artifact: PairedUPriorityArtifact,
        observed_context: dict[str, Any],
        runtime_line: str = "async_release",
    ) -> AsyncReleaseExecutionPlan:
        phase_tasks: list[dict[str, Any]] = []
        dependency_edges: list[tuple[str, str]] = []
        release_conditions: dict[str, dict[str, Any]] = {}
        for entry in priority_artifact.priority_entries:
            task_id = f"{entry.phase}:{entry.src_rank}->{entry.dst_rank}"
            release_dependency = str(entry.release_dependency or "none")
            phase_tasks.append(
                {
                    "task_id": task_id,
                    "global_order_index": int(len(phase_tasks)),
                    "phase": str(entry.phase),
                    "src_rank": int(entry.src_rank),
                    "dst_rank": int(entry.dst_rank),
                    "byte_count": int(entry.byte_count),
                    "priority_score": float(entry.priority_score),
                    "wave_id": int(entry.wave_id),
                    "bucket_hint": int(entry.bucket_hint),
                    "release_dependency": release_dependency,
                    "participating_ranks": tuple(sorted({int(entry.src_rank), int(entry.dst_rank)})),
                    "dependency_task_ids": (),
                    "transfer_key": f"{entry.phase}:{entry.src_rank}:{entry.dst_rank}:{entry.wave_id}",
                    "metadata": {
                        "heuristic_family": str(priority_artifact.heuristic_family),
                        "predictor_name": str(priority_artifact.predictor_name),
                        "p2_source": str(priority_artifact.p2_source),
                    },
                }
            )
            if release_dependency == "wait_p0_complete":
                dependency_edges.append((f"p0_complete:{entry.src_rank}", task_id))
                release_conditions[task_id] = {"requires": "p0_inbound_completion", "rank": int(entry.src_rank)}
            elif release_dependency == "wait_p1_materialized":
                dependency_edges.append((f"p1_materialized:{entry.src_rank}", task_id))
                release_conditions[task_id] = {"requires": "p1_materialized", "rank": int(entry.src_rank)}
            else:
                release_conditions[task_id] = {"requires": "none", "rank": int(entry.src_rank)}
        payload = {
            "source_safe_policy": priority_artifact.source_safe_policy,
            "priority_artifact_digest": priority_artifact.priority_digest,
            "runtime_line": runtime_line,
            "observed_context": observed_context,
            "phase_tasks": phase_tasks,
            "dependency_edges": dependency_edges,
        }
        return AsyncReleaseExecutionPlan(
            plan_id=_stable_digest(payload),
            source_safe_policy=str(priority_artifact.source_safe_policy),
            priority_artifact_digest=str(priority_artifact.priority_digest),
            phase_tasks=tuple(phase_tasks),
            dependency_edges=tuple(dependency_edges),
            release_conditions=release_conditions,
            event_table={},
            fallback_to_phase_sync=not self.executor_available,
            online_executor_eligible=bool(self.executor_available and runtime_line == "async_release"),
            debug_replay_only=not self.executor_available,
        )


def validate_async_release_execution_plan(plan: AsyncReleaseExecutionPlan) -> dict[str, Any]:
    errors: list[str] = []
    task_node_ids = {task["task_id"] for task in plan.phase_tasks}
    event_node_ids = {left for left, _right in plan.dependency_edges if left not in task_node_ids}
    node_ids = task_node_ids | event_node_ids
    seen_edges: set[tuple[str, str]] = set()
    for left, right in plan.dependency_edges:
        if left == right:
            errors.append(f"self_cycle:{left}")
        if (left, right) in seen_edges:
            errors.append(f"duplicate_edge:{left}->{right}")
        seen_edges.add((left, right))
        if right not in task_node_ids:
            errors.append(f"unknown_task_target:{right}")
    indegree: dict[str, int] = {task_id: 0 for task_id in node_ids}
    graph: dict[str, list[str]] = {task_id: [] for task_id in node_ids}
    for left, right in plan.dependency_edges:
        if right in indegree:
            indegree[right] += 1
        graph.setdefault(left, []).append(right)
    queue = [task_id for task_id, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for nxt in graph.get(current, ()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if visited != len(node_ids):
        errors.append("dependency_cycle_detected")
    return {"valid": not errors, "errors": errors, "warnings": []}


__all__ = ["AsyncReleasePlanBuilder", "validate_async_release_execution_plan"]
