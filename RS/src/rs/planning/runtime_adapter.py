from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import PlanningRequest
from rs.scheduling.unified_interface import build_policy as build_legacy_policy

from ._legacy_runtime import to_legacy_request
from .api import Planner, from_logical_plan


@dataclass(frozen=True)
class FormalRuntimePlanner(Planner):
    _planner_id: str
    _planner_family: str

    @property
    def planner_id(self) -> str:
        return self._planner_id

    @property
    def planner_family(self) -> str:
        return self._planner_family

    def plan(self, request: PlanningRequest):
        legacy_request = to_legacy_request(request)
        policy = build_legacy_policy(self._planner_id, legacy_request.policy_options)
        logical_plan = policy.plan(legacy_request)
        return from_logical_plan(
            planner_id=self._planner_id,
            planner_family=self._planner_family,
            request=request,
            logical_plan=logical_plan,
        )


__all__ = ["FormalRuntimePlanner"]
