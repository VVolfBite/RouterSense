"""Tensor-only async-release agreement helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from .compiled_schedule import CompiledAsyncReleaseSchedule


def build_async_release_order_digest(schedule: CompiledAsyncReleaseSchedule) -> str:
    return str(schedule.digest)


def validate_async_release_global_agreement(
    schedules: tuple[CompiledAsyncReleaseSchedule, ...],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not schedules:
        errors.append("no_schedules")
        return {"valid": False, "errors": errors, "warnings": warnings}
    digests = {str(schedule.digest) for schedule in schedules}
    if len(digests) != 1:
        errors.append("schedule_digest_mismatch")
    task_counts = {int(schedule.task_count) for schedule in schedules}
    if len(task_counts) != 1:
        errors.append("task_count_mismatch")
    reference = schedules[0].tensor_payload.detach().cpu()
    for index, schedule in enumerate(schedules[1:], start=1):
        current = schedule.tensor_payload.detach().cpu()
        if reference.shape != current.shape:
            errors.append(f"payload_shape_mismatch:{index}")
            continue
        if not torch.equal(reference, current):
            errors.append(f"payload_value_mismatch:{index}")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "digest": schedules[0].digest,
        "schedule_count": len(schedules),
    }


def gather_and_validate_async_release_schedule(
    local_schedule: CompiledAsyncReleaseSchedule,
    *,
    process_group: Any | None = None,
    gathered_schedules: tuple[CompiledAsyncReleaseSchedule, ...] | None = None,
) -> dict[str, Any]:
    if gathered_schedules is not None:
        return validate_async_release_global_agreement(gathered_schedules)
    if not dist.is_available() or not dist.is_initialized():
        return validate_async_release_global_agreement((local_schedule,))
    world_size = dist.get_world_size(group=process_group)
    payload = local_schedule.tensor_payload.detach()
    gather_buffers = [torch.empty_like(payload) for _ in range(world_size)]
    dist.all_gather(gather_buffers, payload, group=process_group)
    schedules = tuple(
        CompiledAsyncReleaseSchedule(
            task_count=int(local_schedule.task_count),
            tensor_payload=buffer,
            schema_version=int(local_schedule.schema_version),
            digest=str(local_schedule.digest),
        )
        for buffer in gather_buffers
    )
    return validate_async_release_global_agreement(schedules)


__all__ = [
    "build_async_release_order_digest",
    "gather_and_validate_async_release_schedule",
    "validate_async_release_global_agreement",
]
