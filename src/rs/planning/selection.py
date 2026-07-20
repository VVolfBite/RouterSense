from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rs.core.contracts import PlanScore, PlanningRequest, WindowPlan

from .api import Planner
from .estimation import CommonCorePlanEstimator, PlanningCostModel
from .validation import validate_window_plan_for_request


class PlannerSelectionMode(Enum):
    LOCAL = "local"
    JOINT = "joint"
    COMPARE = "compare"


class PlanningSelectionError(RuntimeError):
    pass


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

    @staticmethod
    def _invalid_score(*, reason: str) -> PlanScore:
        return PlanScore(
            estimated_makespan=float("inf"),
            estimator_id="common_core_estimator",
            cost_model_id="formal_cost_model",
            valid=False,
            reason=str(reason),
        )

    def _estimate_validated(self, *, plan: WindowPlan, request: PlanningRequest) -> PlanScore:
        try:
            validate_window_plan_for_request(plan, request)
        except Exception as exc:
            return self._invalid_score(reason=str(exc))
        return self._estimator.estimate(plan, request, self._cost_model)

    def select(
        self,
        request: PlanningRequest,
        *,
        mode: PlannerSelectionMode,
    ) -> SelectedPlan:
        if mode is PlannerSelectionMode.LOCAL:
            return self.select_prebuilt(
                request=request,
                local_plan=self._local_planner.plan(request),
                joint_plan=None,
                mode=mode,
            )
        if mode is PlannerSelectionMode.JOINT:
            return self.select_prebuilt(
                request=request,
                local_plan=None,
                joint_plan=self._joint_planner.plan(request),
                mode=mode,
            )
        return self.select_prebuilt(
            request=request,
            local_plan=self._local_planner.plan(request),
            joint_plan=self._joint_planner.plan(request),
            mode=mode,
        )

    def select_prebuilt(
        self,
        *,
        request: PlanningRequest,
        local_plan: WindowPlan | None,
        joint_plan: WindowPlan | None,
        mode: PlannerSelectionMode,
    ) -> SelectedPlan:
        if mode is PlannerSelectionMode.LOCAL:
            if local_plan is None:
                raise ValueError("LOCAL selection requires local_plan")
            local_score = self._estimate_validated(plan=local_plan, request=request)
            if not local_score.valid:
                raise PlanningSelectionError(f"local_plan_invalid:{local_score.reason or 'unknown'}")
            return SelectedPlan(
                selected_plan=local_plan,
                selected_score=local_score,
                local_plan=local_plan,
                local_score=local_score,
                selection_reason="mode=local",
            )
        if mode is PlannerSelectionMode.JOINT:
            if joint_plan is None:
                raise ValueError("JOINT selection requires joint_plan")
            joint_score = self._estimate_validated(plan=joint_plan, request=request)
            if not joint_score.valid:
                raise PlanningSelectionError(f"joint_plan_invalid:{joint_score.reason or 'unknown'}")
            return SelectedPlan(
                selected_plan=joint_plan,
                selected_score=joint_score,
                joint_plan=joint_plan,
                joint_score=joint_score,
                selection_reason="mode=joint",
            )
        if local_plan is None or joint_plan is None:
            raise ValueError("COMPARE selection requires both local_plan and joint_plan")
        local_score = self._estimate_validated(plan=local_plan, request=request)
        joint_score = self._estimate_validated(plan=joint_plan, request=request)
        if local_score.valid and joint_score.valid:
            selected_plan = joint_plan if float(joint_score.estimated_makespan) < float(local_score.estimated_makespan) else local_plan
            selected_score = joint_score if selected_plan is joint_plan else local_score
            reason = "compare:joint_better" if selected_plan is joint_plan else "compare:local_better_or_equal"
        elif local_score.valid and not joint_score.valid:
            selected_plan = local_plan
            selected_score = local_score
            reason = f"compare:joint_invalid:{joint_score.reason or 'unknown'}"
        elif joint_score.valid and not local_score.valid:
            selected_plan = joint_plan
            selected_score = joint_score
            reason = f"compare:local_invalid:{local_score.reason or 'unknown'}"
        else:
            raise PlanningSelectionError(
                f"compare:no_valid_plan local={local_score.reason or 'unknown'} joint={joint_score.reason or 'unknown'}"
            )
        return SelectedPlan(
            selected_plan=selected_plan,
            selected_score=selected_score,
            local_plan=local_plan,
            local_score=local_score,
            joint_plan=joint_plan,
            joint_score=joint_score,
            selection_reason=reason,
        )

__all__ = ["PlannerSelectionMode", "PlannerSelector", "PlanningSelectionError", "SelectedPlan"]
