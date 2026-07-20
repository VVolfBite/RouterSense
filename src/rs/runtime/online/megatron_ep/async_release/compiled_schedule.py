"""Tensor-compiled async-release schedule.

The CPU-side plan remains a Python object, but any future runtime wire payload
must use a compact int64 tensor schema rather than Python object collectives.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from .contracts import AsyncReleaseExecutionPlan


_SCHEMA_VERSION = 1
_TASK_ROW_WIDTH = 9
_PHASE_CODES = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "p0_dispatch": 10,
    "p1_return": 11,
    "p2_next_dispatch": 12,
    "dispatch": 20,
    "return": 21,
}
_FLAGS_FALLBACK = 1 << 0
_FLAGS_DEBUG_ONLY = 1 << 1


def _task_code(task_id: str) -> int:
    digest = hashlib.sha256(str(task_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def _tensor_digest(payload: torch.Tensor) -> str:
    view = payload.detach().cpu().contiguous().view(torch.int64).numpy().tobytes()
    return hashlib.sha256(view).hexdigest()[:16]


@dataclass(frozen=True)
class CompiledAsyncReleaseSchedule:
    task_count: int
    tensor_payload: torch.Tensor
    schema_version: int
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_count": int(self.task_count),
            "schema_version": int(self.schema_version),
            "digest": str(self.digest),
            "tensor_len": int(self.tensor_payload.numel()),
        }


def compile_async_release_schedule(
    plan: AsyncReleaseExecutionPlan,
    *,
    device: str | torch.device = "cpu",
) -> CompiledAsyncReleaseSchedule:
    tasks = list(plan.phase_tasks)
    task_ids = [str(task["task_id"]) for task in tasks]
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    dependency_flat: list[int] = []
    task_rows: list[list[int]] = []
    for index, task in enumerate(tasks):
        dep_task_ids = tuple(str(value) for value in task.get("dependency_task_ids", ()) or ())
        release_dependency = str(task.get("release_dependency", "none"))
        if release_dependency == "wait_p0_complete":
            dep_task_ids = dep_task_ids or (f"p0_complete:{int(task['src_rank'])}",)
        elif release_dependency == "wait_p1_materialized":
            dep_task_ids = dep_task_ids or (f"p1_materialized:{int(task['src_rank'])}",)
        dep_start = len(dependency_flat)
        for dep in dep_task_ids:
            dependency_flat.append(int(task_index.get(dep, -1)))
        flags = 0
        if bool(plan.fallback_to_phase_sync):
            flags |= _FLAGS_FALLBACK
        if bool(plan.debug_replay_only):
            flags |= _FLAGS_DEBUG_ONLY
        task_rows.append(
            [
                int(task.get("global_order_index", index)),
                int(_task_code(str(task["task_id"]))),
                int(_PHASE_CODES.get(str(task.get("phase", "")), -1)),
                int(task.get("src_rank", -1)),
                int(task.get("dst_rank", -1)),
                int(task.get("byte_count", 0)),
                int(dep_start),
                int(len(dep_task_ids)),
                int(flags),
            ]
        )
    header = [
        int(_SCHEMA_VERSION),
        int(len(task_rows)),
        int(_TASK_ROW_WIDTH),
        int(len(dependency_flat)),
        int(bool(plan.fallback_to_phase_sync)),
        int(bool(plan.online_executor_eligible)),
        int(bool(plan.debug_replay_only)),
    ]
    flat = header + [value for row in task_rows for value in row] + dependency_flat
    tensor_payload = torch.tensor(flat, dtype=torch.int64, device=device)
    return CompiledAsyncReleaseSchedule(
        task_count=len(task_rows),
        tensor_payload=tensor_payload,
        schema_version=_SCHEMA_VERSION,
        digest=_tensor_digest(tensor_payload),
    )


def decode_compiled_async_release_schedule(schedule: CompiledAsyncReleaseSchedule) -> dict[str, Any]:
    values = [int(value) for value in schedule.tensor_payload.detach().cpu().tolist()]
    schema_version, task_count, row_width, dep_count = values[:4]
    cursor = 7
    rows: list[dict[str, Any]] = []
    for _ in range(task_count):
        row = values[cursor : cursor + row_width]
        cursor += row_width
        rows.append(
            {
                "global_order_index": int(row[0]),
                "task_id_code": int(row[1]),
                "phase_code": int(row[2]),
                "source_rank": int(row[3]),
                "target_rank": int(row[4]),
                "byte_count": int(row[5]),
                "dependency_start": int(row[6]),
                "dependency_count": int(row[7]),
                "flags": int(row[8]),
            }
        )
    dependency_codes = tuple(int(value) for value in values[cursor : cursor + dep_count])
    return {
        "schema_version": int(schema_version),
        "task_count": int(task_count),
        "task_row_width": int(row_width),
        "dependency_entry_count": int(dep_count),
        "rows": rows,
        "dependency_codes": dependency_codes,
        "digest": str(schedule.digest),
    }


__all__ = [
    "CompiledAsyncReleaseSchedule",
    "compile_async_release_schedule",
    "decode_compiled_async_release_schedule",
]
