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
    schema_versions = {int(schedule.schema_version) for schedule in schedules}
    if len(schema_versions) != 1:
        errors.append("schema_version_mismatch")
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
    payload = local_schedule.tensor_payload.detach().to(dtype=torch.int64)
    digest_code = int(str(local_schedule.digest), 16) & ((1 << 63) - 1)
    header = torch.tensor(
        [
            int(local_schedule.schema_version),
            int(local_schedule.task_count),
            int(payload.numel()),
            int(digest_code),
        ],
        dtype=torch.int64,
        device=payload.device,
    )
    gathered_headers = [torch.empty_like(header) for _ in range(world_size)]
    dist.all_gather(gathered_headers, header, group=process_group)
    payload_lengths = [int(item[2].item()) for item in gathered_headers]
    max_payload_len = max(payload_lengths, default=int(payload.numel()))
    padded = torch.zeros(max_payload_len, dtype=torch.int64, device=payload.device)
    padded[: payload.numel()] = payload
    gathered_payloads = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered_payloads, padded, group=process_group)
    schedules = []
    for current_header, current_payload in zip(gathered_headers, gathered_payloads, strict=True):
        payload_length = int(current_header[2].item())
        schedules.append(
            CompiledAsyncReleaseSchedule(
                task_count=int(current_header[1].item()),
                tensor_payload=current_payload[:payload_length].clone(),
                schema_version=int(current_header[0].item()),
                digest=f"{int(current_header[3].item()):016x}",
            )
        )
    schedules = tuple(schedules)
    return validate_async_release_global_agreement(schedules)


__all__ = [
    "build_async_release_order_digest",
    "gather_and_validate_async_release_schedule",
    "validate_async_release_global_agreement",
]
