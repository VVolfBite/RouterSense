from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rs.core.contracts import PlanScore, PlanningRequest, WindowPlan

from .api import Planner
from .estimation import CommonCorePlanEstimator, PlanningCostModel


class PlannerSelectionMode(Enum):
    LOCAL = "local"
    JOINT = "joint"
    COMPARE = "compare"


@dataclass(frozen=True)
class SelectedPlan:
    selected_plan: WindowPlan
    selected_score: PlanScore
    local_plan: WindowPlan | None = None
    local_score: PlanScore | None = None
    joint_plan: WindowPlan | None = None
    joint_score: PlanScore | None = None
    selection_reason: str = ""


class PlannerSelector:
    def __init__(
        self,
        *,
        local_planner: Planner,
        joint_planner: Planner,
        estimator: CommonCorePlanEstimator | None = None,
        cost_model: PlanningCostModel | None = None,
    ) -> None:
        self._local_planner = local_planner
        self._joint_planner = joint_planner
        self._estimator = estimator or CommonCorePlanEstimator()
        self._cost_model = cost_model or PlanningCostModel()

    def select(
        self,
        request: PlanningRequest,
        *,
        mode: PlannerSelectionMode,
    ) -> SelectedPlan:
        if mode is PlannerSelectionMode.LOCAL:
            local_plan = self._local_planner.plan(request)
            local_score = self._estimator.estimate(local_plan, request, self._cost_model)
            return SelectedPlan(
                selected_plan=local_plan,
                selected_score=local_score,
                local_plan=local_plan,
                local_score=local_score,
                selection_reason="mode=local",
            )
        if mode is PlannerSelectionMode.JOINT:
            joint_plan = self._joint_planner.plan(request)
            joint_score = self._estimator.estimate(joint_plan, request, self._cost_model)
            return SelectedPlan(
                selected_plan=joint_plan,
                selected_score=joint_score,
                joint_plan=joint_plan,
                joint_score=joint_score,
                selection_reason="mode=joint",
            )
        local_plan = self._local_planner.plan(request)
        joint_plan = self._joint_planner.plan(request)
        local_score = self._estimator.estimate(local_plan, request, self._cost_model)
        joint_score = self._estimator.estimate(joint_plan, request, self._cost_model)
        selected_plan = joint_plan if float(joint_score.estimated_makespan) < float(local_score.estimated_makespan) else local_plan
        selected_score = joint_score if selected_plan is joint_plan else local_score
        reason = "compare:joint_better" if selected_plan is joint_plan else "compare:local_better_or_equal"
        return SelectedPlan(
            selected_plan=selected_plan,
            selected_score=selected_score,
            local_plan=local_plan,
            local_score=local_score,
            joint_plan=joint_plan,
            joint_score=joint_score,
            selection_reason=reason,
        )


__all__ = ["PlannerSelectionMode", "PlannerSelector", "SelectedPlan"]
