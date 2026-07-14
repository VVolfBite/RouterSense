from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from rs.core.contracts.execution import PublishedPlan
from rs.runtime.online.megatron_ep.control.rank_map import RankMap
from rs.scheduling.validation import stable_hash


def canonical_published_plan_digest(plan: PublishedPlan) -> str:
    plan.validate()
    return str(
        stable_hash(
            {
                "planner_id": str(plan.planner_id),
                "logical_plan_digest": str(plan.logical_plan_digest),
                "publication_slot_digest": str(plan.publication_slot_digest),
                "root_rank": int(plan.root_rank),
                "root_group_rank": int(plan.root_group_rank),
                "version": int(plan.version),
                "logical_plan": dict(plan.logical_plan),
                "metadata": dict(plan.metadata),
            }
        )
    )


class CanonicalPlanPublisher:
    def __init__(self, *, rank_map: RankMap) -> None:
        self._rank_map = rank_map
        self._rank_map.validate()

    def build(
        self,
        *,
        planner_id: str,
        logical_plan: Mapping[str, object],
        logical_plan_digest: str,
        publication_slot_digest: str,
        version: int = 1,
        metadata: Mapping[str, object] | None = None,
    ) -> PublishedPlan:
        draft = PublishedPlan(
            planner_id=str(planner_id),
            logical_plan_digest=str(logical_plan_digest),
            published_plan_digest="pending",
            publication_slot_digest=str(publication_slot_digest),
            root_rank=int(self._rank_map.root_rank),
            root_group_rank=int(self._rank_map.root_group_rank),
            version=int(version),
            logical_plan=dict(logical_plan),
            metadata=dict(metadata or {}),
        )
        return self.publish(draft)

    def publish(self, plan: PublishedPlan) -> PublishedPlan:
        if int(plan.root_rank) != int(self._rank_map.root_rank):
            raise ValueError("published plan root_rank does not match rank_map")
        if int(plan.root_group_rank) != int(self._rank_map.root_group_rank):
            raise ValueError("published plan root_group_rank does not match rank_map")
        digest = canonical_published_plan_digest(plan)
        return replace(plan, published_plan_digest=digest)
