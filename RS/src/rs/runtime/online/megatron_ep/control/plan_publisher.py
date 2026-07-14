from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from rs.core.contracts.execution import PublishedPlan
from rs.core.contracts.planning import PlanWave, PlannedFlow, WindowPlan
from rs.runtime.online.megatron_ep.control.rank_map import RankMap


def _window_plan_from_payload(payload: Mapping[str, object]) -> WindowPlan:
    return WindowPlan(
        planner_id=str(payload["planner_id"]),
        planner_family=str(payload["planner_family"]),
        request_digest=str(payload["request_digest"]),
        waves=tuple(
            PlanWave(
                wave_id=int(wave["wave_id"]),
                flows=tuple(
                    PlannedFlow(
                        flow_id=str(flow["flow_id"]),
                        phase=str(flow["phase"]),
                        src_rank=int(flow["src_rank"]),
                        dst_rank=int(flow["dst_rank"]),
                        row_count=int(flow["row_count"]),
                        release_state=str(flow["release_state"]),
                        executable=bool(flow["executable"]),
                    )
                    for flow in wave.get("flows", ())
                ),
                estimated_duration=float(wave.get("estimated_duration", 0.0)),
            )
            for wave in payload.get("waves", ())
        ),
        metadata=dict(payload.get("metadata", {})),
    )


class CanonicalPlanPublisher:
    def __init__(self, *, rank_map: RankMap) -> None:
        self._rank_map = rank_map
        self._rank_map.validate()

    def build(
        self,
        *,
        publication_slot: Mapping[str, object],
        window_plan: WindowPlan | Mapping[str, object],
        version: int = 1,
        metadata: Mapping[str, object] | None = None,
    ) -> PublishedPlan:
        material_window_plan = _window_plan_from_payload(window_plan) if isinstance(window_plan, Mapping) else window_plan
        material_window_plan.validate()
        draft = PublishedPlan(
            publication_slot=dict(publication_slot),
            window_plan=material_window_plan,
            logical_plan_digest=str(material_window_plan.semantic_digest()),
            published_plan_digest="pending",
            root_global_rank=int(self._rank_map.root_rank),
            root_group_rank=int(self._rank_map.root_group_rank),
            version=int(version),
            metadata=dict(metadata or {}),
        )
        return self.publish(draft)

    def publish(self, plan: PublishedPlan) -> PublishedPlan:
        if int(plan.root_global_rank) != int(self._rank_map.root_rank):
            raise ValueError("published plan root_global_rank does not match rank_map")
        if int(plan.root_group_rank) != int(self._rank_map.root_group_rank):
            raise ValueError("published plan root_group_rank does not match rank_map")
        plan.validate()
        return replace(plan, published_plan_digest=plan.recompute_published_plan_digest())
