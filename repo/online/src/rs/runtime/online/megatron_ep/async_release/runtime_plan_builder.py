"""AR0 runtime async-release plan builder.

This builder uses real phase tasks from a phase-sync execution plan instead of
offline priority artifacts. It still remains fail-closed and does not invoke
real collectives.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rs.scheduling.phase_execution import BucketTask, PhaseExecutionPlan, PhaseReadyContext

from .contracts import AsyncReleaseExecutionPlan


EVENT_CODES = {
    "P0_PLAN_READY": 1,
    "P0_TRANSFER_COMPLETE": 2,
    "LOCAL_COMPUTE_COMPLETE": 3,
    "P1_MATERIALIZED": 4,
    "P1_TRANSFER_COMPLETE": 5,
    "P2_FORECAST_READY": 6,
    "FALLBACK_REQUIRED": 7,
}


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_runtime_async_release_task_id(
    *,
    layer_id: str,
    phase: str,
    src_rank: int,
    dst_rank: int,
    bundle_id: str,
    segment_ordinal: int,
    bucket_ordinal: int,
    wave_id: int,
) -> str:
    return (
        f"{layer_id}:{phase}:{src_rank}:{dst_rank}:{bundle_id}:"
        f"seg{segment_ordinal}:bucket{bucket_ordinal}:wave{wave_id}"
    )


class AsyncReleaseRuntimePlanBuilder:
    def __init__(self, *, executor_available: bool = False) -> None:
        self.executor_available = bool(executor_available)

    def build(
        self,
        *,
        local_context: PhaseReadyContext,
        execution_plan: PhaseExecutionPlan,
        transport_bundles: tuple[Any, ...] = (),
    ) -> AsyncReleaseExecutionPlan:
        phase_tasks: list[dict[str, Any]] = []
        dependency_edges: list[tuple[str, str]] = []
        release_conditions: dict[str, dict[str, Any]] = {}
        event_table: dict[str, dict[str, Any]] = {
            name: {"event_code": code, "name": name}
            for name, code in EVENT_CODES.items()
        }
        seen_task_ids: set[str] = set()
        for wave in execution_plan.waves:
            for ordinal, task in enumerate(wave.bucket_tasks):
                task_id = build_runtime_async_release_task_id(
                    layer_id=str(local_context.layer_id),
                    phase=str(task.phase),
                    src_rank=int(task.src_rank),
                    dst_rank=int(task.dst_rank),
                    bundle_id=str(task.bundle_id),
                    segment_ordinal=int(task.segment_ordinal),
                    bucket_ordinal=int(task.bucket_ordinal),
                    wave_id=int(wave.wave_id),
                )
                if task_id in seen_task_ids:
                    raise ValueError(f"async release task id collision: {task_id}")
                seen_task_ids.add(task_id)
                dependency_event_ids: list[str] = []
                if str(task.phase) == "P1":
                    dependency_event_ids.append(f"P0_TRANSFER_COMPLETE:{int(task.dst_rank)}:{str(task.bundle_id)}")
                release_event_ids = [f"{str(task.phase).upper()}_PLAN_READY:{int(task.src_rank)}:{str(task.bundle_id)}"]
                task_payload = _task_payload(
                    task=task,
                    task_id=task_id,
                    global_order_index=len(phase_tasks),
                    wave_id=int(wave.wave_id),
                    dependency_event_ids=tuple(dependency_event_ids),
                    release_event_ids=tuple(release_event_ids),
                    layer_id=str(local_context.layer_id),
                )
                phase_tasks.append(task_payload)
                release_conditions[task_id] = {
                    "dependency_event_ids": list(dependency_event_ids),
                    "release_event_ids": list(release_event_ids),
                    "transport_bundle_count": len(transport_bundles),
                }
        payload = {
            "layer_id": str(local_context.layer_id),
            "phase": str(local_context.phase),
            "plan_hash": str(execution_plan.plan_hash),
            "task_ids": [task["task_id"] for task in phase_tasks],
        }
        return AsyncReleaseExecutionPlan(
            plan_id=_stable_digest(payload),
            source_safe_policy=str(execution_plan.policy_name),
            priority_artifact_digest=str(execution_plan.plan_hash),
            phase_tasks=tuple(phase_tasks),
            dependency_edges=tuple(dependency_edges),
            release_conditions=release_conditions,
            event_table=event_table,
            fallback_to_phase_sync=not self.executor_available,
            online_executor_eligible=bool(self.executor_available),
            debug_replay_only=not self.executor_available,
        )


def _task_payload(
    *,
    task: BucketTask,
    task_id: str,
    global_order_index: int,
    wave_id: int,
    dependency_event_ids: tuple[str, ...],
    release_event_ids: tuple[str, ...],
    layer_id: str,
) -> dict[str, Any]:
    payload_roles = tuple(str(slice_.tensor_role) for slice_ in task.payload_slices)
    payload_specs = tuple(
        {
            "tensor_role": str(slice_.tensor_role),
            "row_start": int(slice_.sender_offset_rows),
            "row_count": int(slice_.row_count),
            "column_shape": tuple(int(v) for v in slice_.shape_suffix),
            "dtype": str(slice_.dtype),
        }
        for slice_ in task.payload_slices
    )
    return {
        "task_id": task_id,
        "window_id": layer_id,
        "layer_id": layer_id,
        "phase": str(task.phase),
        "global_order_index": int(global_order_index),
        "src_rank": int(task.src_rank),
        "dst_rank": int(task.dst_rank),
        "peer_rank": int(task.dst_rank),
        "transfer_key": f"{task.phase}:{task.src_rank}:{task.dst_rank}:{task.bundle_id}",
        "transport_bundle_id": str(task.bundle_id),
        "atomic_bundle_id": str(task.bundle_id),
        "bundle_id": str(task.bundle_id),
        "bucket_ordinal": int(task.bucket_ordinal),
        "segment_ordinal": int(task.segment_ordinal),
        "wave_id": int(wave_id),
        "send_offset_rows": int(task.sender_offset_rows),
        "recv_offset_rows": int(task.receiver_offset_rows),
        "row_count": int(task.row_count),
        "byte_count": int(task.byte_count),
        "payload_roles": payload_roles,
        "payload_specs": payload_specs,
        "dtype": payload_specs[0]["dtype"] if payload_specs else "",
        "shape_suffix": payload_specs[0]["column_shape"] if payload_specs else (),
        "dependency_event_ids": dependency_event_ids,
        "release_event_ids": release_event_ids,
        "dependency_task_ids": (),
        "release_dependency": "wait_p0_complete" if str(task.phase) == "P1" else "none",
        "participating_ranks": tuple(sorted({int(task.src_rank), int(task.dst_rank)})),
        "metadata": {
            "packed_send_layout_id": str(task.packed_send_layout_id),
            "canonical_receive_layout_id": str(task.canonical_receive_layout_id),
            "source_peer_index": int(task.source_peer_index),
            "destination_peer_index": int(task.destination_peer_index),
        },
    }


__all__ = ["AsyncReleaseRuntimePlanBuilder", "EVENT_CODES", "build_runtime_async_release_task_id"]
