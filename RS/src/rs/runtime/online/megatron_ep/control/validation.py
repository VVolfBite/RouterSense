from __future__ import annotations

from collections import Counter

from rs.runtime.online.megatron_ep.control.contracts import ControlCommand, PendingCommTask


def validate_plan_key_consistency(tasks: list[PendingCommTask]) -> None:
    keys = {task.bucket.plan_key.to_dict().__repr__() for task in tasks}
    if len(keys) > 1:
        raise ValueError("pending tasks have inconsistent plan keys")


def validate_unique_bucket_coverage(tasks: list[PendingCommTask]) -> None:
    counts = Counter(task.bucket.bucket_id for task in tasks)
    duplicates = sorted(bucket_id for bucket_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate bucket coverage detected: {duplicates}")


def validate_command_not_expired(command: ControlCommand, *, current_epoch: int) -> None:
    if current_epoch > command.expiry.expiry_epoch:
        raise ValueError(
            f"command expired command_id={command.command_id} current_epoch={current_epoch} expiry_epoch={command.expiry.expiry_epoch}"
        )
