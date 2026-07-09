"""Tensor-only async-release agreement helpers."""

from __future__ import annotations

from typing import Any

import torch

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


__all__ = ["build_async_release_order_digest", "validate_async_release_global_agreement"]
