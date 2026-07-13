from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rs.core.contracts import PlanScore, PlanningRequest, WindowPlan


@dataclass(frozen=True)
class PlanningCostModel:
    cost_model_id: str = "common_core_v1"
    row_transfer_cost: float = 1.0
    launch_cost: float = 0.0
    full_duplex: bool = True
    max_outgoing_per_rank_per_wave: int = 1
    max_incoming_per_rank_per_wave: int = 1
    expert_compute_delay: float = 0.0


class PlanEstimator(Protocol):
    @property
    def estimator_id(self) -> str:
        ...

    def estimate(
        self,
        plan: WindowPlan,
        request: PlanningRequest,
        cost_model: PlanningCostModel,
    ) -> PlanScore:
        ...


class CommonCorePlanEstimator:
    @property
    def estimator_id(self) -> str:
        return "common_core_estimator_v1"

    def estimate(
        self,
        plan: WindowPlan,
        request: PlanningRequest,
        cost_model: PlanningCostModel,
    ) -> PlanScore:
        legacy_makespan = plan.metadata.get("legacy_makespan")
        if legacy_makespan is not None:
            estimated = float(legacy_makespan)
        else:
            estimated = 0.0
            for wave in plan.waves:
                if float(wave.estimated_duration) > 0.0:
                    estimated += float(wave.estimated_duration)
                    continue
                flow_rows = [int(flow.row_count) for flow in wave.flows]
                wave_rows = max(flow_rows, default=0)
                estimated += float(cost_model.launch_cost) + float(cost_model.row_transfer_cost) * float(wave_rows)
        return PlanScore(
            estimated_makespan=float(estimated),
            estimator_id=self.estimator_id,
            cost_model_id=str(cost_model.cost_model_id),
            valid=True,
            reason=None,
        )


__all__ = ["CommonCorePlanEstimator", "PlanEstimator", "PlanningCostModel"]
