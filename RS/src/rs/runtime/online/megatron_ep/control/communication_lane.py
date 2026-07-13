from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch.distributed as dist

from rs.runtime.online.megatron_ep.public_types import (
    ControlCommunicationLane,
    LocalPublicationCandidate,
    PublicationPollResult,
    PublicationPollStatus,
    PublicationSlot,
)


class GlooControlCommunicationLane(ControlCommunicationLane):
    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        root_rank: int,
        process_group: dist.ProcessGroup | None,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.root_rank = int(root_rank)
        self.process_group = process_group

    def poll(self, slot: PublicationSlot, local_candidate: LocalPublicationCandidate | None) -> PublicationPollResult:
        local_payload = self._local_status_payload(slot=slot, local_candidate=local_candidate)
        gathered = self._all_gather_status(local_payload)
        terminal = self._resolve_terminal(slot=slot, gathered=gathered)
        if terminal is not None:
            return terminal
        if not all(str(item.get("status")) == "READY" for item in gathered):
            return PublicationPollResult(
                slot=slot,
                status=PublicationPollStatus.NOT_READY,
                root_rank=int(self.root_rank),
                details={"gathered_statuses": tuple(str(item.get("status")) for item in gathered)},
            )
        root_payload = next((item for item in gathered if int(item.get("rank", -1)) == int(self.root_rank)), None)
        if root_payload is None:
            return PublicationPollResult(
                slot=slot,
                status=PublicationPollStatus.FAILED,
                root_rank=int(self.root_rank),
                details={"reason": "missing_root_payload"},
            )
        canonical_payload = dict(root_payload.get("candidate") or {})
        published_digest = str(canonical_payload.get("logical_plan_digest", ""))
        return PublicationPollResult(
            slot=slot,
            status=PublicationPollStatus.READY,
            root_rank=int(self.root_rank),
            published_plan_digest=published_digest,
            canonical_payload=canonical_payload,
            details={"gathered_statuses": tuple(str(item.get("status")) for item in gathered)},
        )

    def _local_status_payload(
        self,
        *,
        slot: PublicationSlot,
        local_candidate: LocalPublicationCandidate | None,
    ) -> dict[str, object]:
        if local_candidate is None:
            status = "NOT_SUBMITTED"
            candidate = {}
        else:
            status = str(local_candidate.status).upper()
            candidate = local_candidate.to_dict()
        return {
            "slot_digest": str(slot.semantic_digest()),
            "rank": int(self.rank),
            "status": str(status),
            "candidate": dict(candidate),
        }

    def _all_gather_status(self, local_payload: dict[str, object]) -> list[dict[str, object]]:
        if not dist.is_available() or not dist.is_initialized() or self.world_size <= 1:
            return [local_payload]
        gathered: list[dict[str, object] | None] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_payload, group=self.process_group)
        return [dict(item or {}) for item in gathered]

    @staticmethod
    def _resolve_terminal(
        *,
        slot: PublicationSlot,
        gathered: list[dict[str, object]],
    ) -> PublicationPollResult | None:
        statuses = {str(item.get("status")) for item in gathered}
        if "FAILED" in statuses:
            return PublicationPollResult(
                slot=slot,
                status=PublicationPollStatus.FAILED,
                details={"gathered_statuses": tuple(sorted(statuses))},
            )
        if "CANCELLED" in statuses:
            return PublicationPollResult(
                slot=slot,
                status=PublicationPollStatus.CANCELLED,
                details={"gathered_statuses": tuple(sorted(statuses))},
            )
        if "EXPIRED" in statuses:
            return PublicationPollResult(
                slot=slot,
                status=PublicationPollStatus.EXPIRED,
                details={"gathered_statuses": tuple(sorted(statuses))},
            )
        return None


def slot_from_request(
    *,
    run_id: str,
    forward_generation: int,
    microbatch_id: str,
    source_layer_id: str,
    target_layer_id: str,
) -> PublicationSlot:
    return PublicationSlot(
        run_id=str(run_id),
        forward_generation=int(forward_generation),
        microbatch_id=str(microbatch_id),
        source_layer_id=str(source_layer_id),
        target_layer_id=str(target_layer_id),
        planning_slot=f"{source_layer_id}->{target_layer_id}",
    )


__all__ = [
    "ControlCommunicationLane",
    "GlooControlCommunicationLane",
    "LocalPublicationCandidate",
    "PublicationPollResult",
    "PublicationPollStatus",
    "PublicationSlot",
    "slot_from_request",
]
