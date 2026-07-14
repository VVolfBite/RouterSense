from __future__ import annotations

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
        group_ranks: tuple[int, ...] | None = None,
    ) -> None:
        self.global_rank = int(rank)
        self.world_size = int(world_size)
        self.root_rank = int(root_rank)
        self.process_group = process_group
        self.group_ranks = tuple(int(value) for value in (group_ranks or tuple(range(int(world_size)))))
        self.group_world_size = len(self.group_ranks)
        if self.global_rank not in self.group_ranks:
            raise ValueError(f"global rank {self.global_rank} is not in group_ranks {self.group_ranks!r}")
        if self.root_rank not in self.group_ranks:
            raise ValueError(f"root rank {self.root_rank} is not in group_ranks {self.group_ranks!r}")
        self.group_rank = int(self.group_ranks.index(self.global_rank))
        self.root_group_rank = int(self.group_ranks.index(self.root_rank))

    def poll(self, slot: PublicationSlot, local_candidate: LocalPublicationCandidate | None) -> PublicationPollResult:
        local_payload = self._local_status_payload(slot=slot, local_candidate=local_candidate)
        gathered = self._all_gather_status(local_payload)
        slot_digests = {str(item.get("slot_digest", "")) for item in gathered}
        if slot_digests != {str(slot.semantic_digest())}:
            return PublicationPollResult(
                slot=slot,
                status=PublicationPollStatus.SLOT_MISMATCH,
                root_rank=int(self.root_rank),
                details={
                    "expected_slot_digest": str(slot.semantic_digest()),
                    "gathered_slot_digests": tuple(sorted(slot_digests)),
                },
            )
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
        root_payload = next((item for item in gathered if int(item.get("group_rank", -1)) == int(self.root_group_rank)), None)
        if root_payload is None:
            return PublicationPollResult(
                slot=slot,
                status=PublicationPollStatus.FAILED,
                root_rank=int(self.root_rank),
                details={"reason": "missing_root_payload"},
            )
        root_candidate = dict(root_payload.get("candidate") or {})
        published_digest = str(root_candidate.get("logical_plan_digest", ""))
        if any(str(dict(item.get("candidate") or {}).get("logical_plan_digest", "")) != published_digest for item in gathered):
            return PublicationPollResult(
                slot=slot,
                status=PublicationPollStatus.FAILED,
                root_rank=int(self.root_rank),
                details={"reason": "plan_digest_mismatch"},
            )
        canonical_payload = self._broadcast_root_plan(
            slot=slot,
            root_candidate=root_candidate,
        )
        if str(canonical_payload.get("status", "")).upper() == "FAILED":
            return PublicationPollResult(
                slot=slot,
                status=PublicationPollStatus.FAILED,
                root_rank=int(self.root_rank),
                details={"reason": str(canonical_payload.get("reason", "broadcast_failed"))},
            )
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
            "group_rank": int(self.group_rank),
            "global_rank": int(self.global_rank),
            "status": str(status),
            "candidate": dict(candidate),
        }

    def _all_gather_status(self, local_payload: dict[str, object]) -> list[dict[str, object]]:
        if not dist.is_available() or not dist.is_initialized() or self.group_world_size <= 1:
            return [local_payload]
        gathered: list[dict[str, object] | None] = [None for _ in range(self.group_world_size)]
        dist.all_gather_object(gathered, local_payload, group=self.process_group)
        return [dict(item or {}) for item in gathered]

    def _broadcast_root_plan(
        self,
        *,
        slot: PublicationSlot,
        root_candidate: dict[str, Any],
    ) -> dict[str, object]:
        payload: dict[str, object] | None
        if self.group_rank == self.root_group_rank:
            payload = dict(root_candidate)
        else:
            payload = None
        if not dist.is_available() or not dist.is_initialized() or self.group_world_size <= 1:
            return dict(payload or {})
        object_list: list[dict[str, object] | None] = [payload]
        dist.broadcast_object_list(object_list, src=int(self.root_group_rank), group=self.process_group)
        broadcast_payload = dict(object_list[0] or {})
        plan_payload = dict(broadcast_payload.get("plan") or {})
        if plan_payload:
            from rs.runtime.online.megatron_ep.target_planning.contracts import TargetLayerPreparedJointPlan

            plan = TargetLayerPreparedJointPlan.from_dict(plan_payload)
            if str(plan.logical_plan_digest) != str(broadcast_payload.get("logical_plan_digest", "")):
                return {
                    "slot": slot.semantic_payload(),
                    "status": "FAILED",
                    "reason": "broadcast_plan_digest_mismatch",
                }
        return broadcast_payload

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
