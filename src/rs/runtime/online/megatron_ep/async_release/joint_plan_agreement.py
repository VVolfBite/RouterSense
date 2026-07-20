"""Global-plan agreement and local-schedule validation for async release.

The global abstract joint plan must match across ranks, but each rank's local
executable schedule may legitimately differ in task count and payload length.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GlobalJointPlanWire:
    window_key: str
    policy_name: str
    safe_selected_policy: str
    prediction_digest: str
    canonical_edge_order: tuple[tuple[str, int, int], ...]
    wave_metadata: tuple[tuple[int, tuple[tuple[str, int, int], ...]], ...]
    per_peer_sequence_digest: str

    @property
    def global_plan_digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


def agree_global_joint_plan(
    local_wire: GlobalJointPlanWire,
    *,
    gathered_wires: tuple[GlobalJointPlanWire, ...] | None = None,
) -> dict[str, Any]:
    wires = gathered_wires or (local_wire,)
    digests = {wire.global_plan_digest for wire in wires}
    return {
        "valid": len(digests) == 1,
        "global_plan_digest": local_wire.global_plan_digest,
        "wire_count": len(wires),
        "errors": [] if len(digests) == 1 else ["global_plan_digest_mismatch"],
    }


def validate_local_schedule_against_global_plan(
    local_schedule: tuple[dict[str, Any], ...],
    *,
    global_wire: GlobalJointPlanWire,
) -> dict[str, Any]:
    allowed_edges = {
        (str(phase), int(src_rank), int(dst_rank))
        for phase, src_rank, dst_rank in global_wire.canonical_edge_order
    }
    errors: list[str] = []
    for task in local_schedule:
        edge = (str(task.get("phase", "")), int(task.get("src_rank", -1)), int(task.get("dst_rank", -1)))
        if edge not in allowed_edges:
            errors.append(f"local_task_not_in_global_plan:{edge}")
    return {
        "valid": not errors,
        "errors": errors,
        "local_task_count": len(local_schedule),
        "global_edge_count": len(allowed_edges),
    }


def validate_pairwise_send_recv_contracts(
    local_schedules: tuple[tuple[dict[str, Any], ...], ...],
) -> dict[str, Any]:
    send_records: dict[tuple[str, int, int, str, int, tuple[int, ...]], int] = {}
    recv_records: dict[tuple[str, int, int, str, int, tuple[int, ...]], int] = {}
    errors: list[str] = []
    for schedule in local_schedules:
        for task in schedule:
            roles = tuple(str(role) for role in task.get("payload_roles", ()) or ())
            key = (
                str(task.get("phase", "")),
                int(task.get("src_rank", -1)),
                int(task.get("dst_rank", -1)),
                str(task.get("dtype", "")),
                int(task.get("row_count", 0)),
                roles,
            )
            owner = int(task.get("owner_global_rank", -1))
            if owner == int(task.get("src_rank", -1)):
                send_records[key] = send_records.get(key, 0) + 1
            elif owner == int(task.get("dst_rank", -1)):
                recv_records[key] = recv_records.get(key, 0) + 1
    all_keys = set(send_records) | set(recv_records)
    for key in sorted(all_keys):
        if send_records.get(key, 0) != recv_records.get(key, 0):
            errors.append(f"send_recv_mismatch:{key}:{send_records.get(key, 0)}!={recv_records.get(key, 0)}")
    return {"valid": not errors, "errors": errors}


__all__ = [
    "GlobalJointPlanWire",
    "agree_global_joint_plan",
    "validate_local_schedule_against_global_plan",
    "validate_pairwise_send_recv_contracts",
]
