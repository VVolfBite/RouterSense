from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs_sim.scheduler.errors import TaskizationError
from rs_sim.scheduler.planning.schema_api import SharedSchemaAdapter
from rs_sim.scheduler.stable import stable_digest, stable_id


@dataclass(frozen=True)
class TaskizationSpec:
    chunk_bytes: int
    alignment_bytes: int = 1
    version: str = "canonical_fixed_bytes_v1"
    identity_namespace: str | None = None

    def __post_init__(self) -> None:
        if int(self.chunk_bytes) <= 0:
            raise ValueError("chunk_bytes must be positive")
        if int(self.alignment_bytes) <= 0:
            raise ValueError("alignment_bytes must be positive")
        if self.identity_namespace is not None and not str(self.identity_namespace):
            raise ValueError("identity_namespace must be non-empty when provided")

    @property
    def effective_chunk_bytes(self) -> int:
        aligned = (int(self.chunk_bytes) // int(self.alignment_bytes)) * int(self.alignment_bytes)
        return max(int(self.alignment_bytes), aligned)

    def stable_payload(self) -> dict[str, Any]:
        return {
            "version": str(self.version),
            "chunk_bytes": int(self.chunk_bytes),
            "alignment_bytes": int(self.alignment_bytes),
            "effective_chunk_bytes": int(self.effective_chunk_bytes),
            "identity_namespace": (
                None if self.identity_namespace is None else str(self.identity_namespace)
            ),
        }


class CanonicalTaskizer:
    """The only producer of canonical transfer tasks in scheduler."""

    def __init__(self, *, adapter: SharedSchemaAdapter, spec: TaskizationSpec) -> None:
        self.adapter = adapter
        self.spec = spec

    def taskize(self, expectation: Any, *, registered_at_ns: int) -> tuple[Any, ...]:
        view = self.adapter.expectation_view(expectation)
        total = int(view.total_expected_payload_bytes)
        if total < 0:
            raise TaskizationError("expected payload bytes cannot be negative")
        if bool(view.zero_edge) != (total == 0):
            raise TaskizationError("zero_edge must exactly match total_expected_payload_bytes == 0")
        # Nonzero diagonal edges are exact workload truth, but they are local
        # assembly rather than EP network transfers.  They remain in the
        # ReceiveExpectation/closure contract and are completed by backend after
        # SourcePayloadReady; the canonical network catalogue excludes them.
        if int(view.src_rank) == int(view.dst_rank):
            return ()
        if total == 0:
            return ()

        taskization_payload = {
            "contract": "RS_SIM_CANONICAL_TASKIZATION",
            "spec": self.spec.stable_payload(),
            "phase_key": self.adapter.phase_payload(view.phase_key),
            "edge_key": self.adapter.edge_payload(view.edge_key),
            "src_rank": int(view.src_rank),
            "dst_rank": int(view.dst_rank),
            "total_expected_payload_bytes": total,
            "expectation_digest": str(view.expectation_digest),
        }
        taskization_digest = stable_digest(taskization_payload)
        step = int(self.spec.effective_chunk_bytes)
        offset = 0
        chunk_index = 0
        tasks: list[Any] = []
        while offset < total:
            payload_bytes = min(step, total - offset)
            if self.spec.identity_namespace is None:
                task_identity = {
                    "taskization_digest": taskization_digest,
                    "chunk_index": chunk_index,
                    "byte_offset": offset,
                    "payload_bytes": payload_bytes,
                }
            else:
                phase_key = view.phase_key
                task_identity = {
                    "schema_version": "PAIRED_CANONICAL_TASK_ID",
                    "identity_namespace": str(self.spec.identity_namespace),
                    "sample_id": str(getattr(phase_key, "sample_id")),
                    "layer_index": int(getattr(phase_key, "layer_index")),
                    "phase_kind": str(getattr(getattr(phase_key, "phase_kind"), "value", getattr(phase_key, "phase_kind"))),
                    "src_rank": int(view.src_rank),
                    "dst_rank": int(view.dst_rank),
                    "total_expected_payload_bytes": total,
                    "chunk_index": chunk_index,
                    "byte_offset": offset,
                    "payload_bytes": payload_bytes,
                }
            task_id = stable_id("task", task_identity)
            tasks.append(
                self.adapter.make_task(
                    task_id=task_id,
                    edge_key=view.edge_key,
                    phase_key=view.phase_key,
                    src_rank=int(view.src_rank),
                    dst_rank=int(view.dst_rank),
                    chunk_index=int(chunk_index),
                    byte_offset=int(offset),
                    payload_bytes=int(payload_bytes),
                    expectation_digest=str(view.expectation_digest),
                    taskization_digest=taskization_digest,
                    registered_at_ns=int(registered_at_ns),
                )
            )
            offset += payload_bytes
            chunk_index += 1
        self.validate_ranges(tasks, expected_total_bytes=total)
        return tuple(tasks)

    def validate_ranges(self, tasks: tuple[Any, ...] | list[Any], *, expected_total_bytes: int) -> None:
        views = [self.adapter.task_view(task) for task in tasks]
        views.sort(key=lambda item: (item.byte_offset, item.chunk_index, item.task_id))
        cursor = 0
        seen_ids: set[str] = set()
        seen_chunks: set[int] = set()
        for view in views:
            if view.task_id in seen_ids:
                raise TaskizationError(f"duplicate task_id {view.task_id}")
            if view.chunk_index in seen_chunks:
                raise TaskizationError(f"duplicate chunk_index {view.chunk_index}")
            if int(view.byte_offset) != cursor:
                raise TaskizationError(
                    f"task range gap/overlap: expected offset {cursor}, got {view.byte_offset}"
                )
            if int(view.payload_bytes) <= 0:
                raise TaskizationError("canonical tasks must have positive payload_bytes")
            seen_ids.add(view.task_id)
            seen_chunks.add(view.chunk_index)
            cursor += int(view.payload_bytes)
        if cursor != int(expected_total_bytes):
            raise TaskizationError(
                f"task ranges cover {cursor} bytes, expected {int(expected_total_bytes)}"
            )
