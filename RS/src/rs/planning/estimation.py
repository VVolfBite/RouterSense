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
        estimated = 0.0
        invalid_reason: str | None = None
        for wave in plan.waves:
            wave_duration = self._estimate_wave_duration(wave=wave, request=request, cost_model=cost_model)
            if wave_duration is None:
                invalid_reason = f"non_executable_flow_in_wave:{wave.wave_id}"
                break
            estimated += float(wave_duration)
        return PlanScore(
            estimated_makespan=float(estimated),
            estimator_id=self.estimator_id,
            cost_model_id=str(cost_model.cost_model_id),
            valid=invalid_reason is None,
            reason=invalid_reason,
        )

    def _estimate_wave_duration(
        self,
        *,
        wave,
        request: PlanningRequest,
        cost_model: PlanningCostModel,
    ) -> float | None:
        send_loads: dict[int, int] = {}
        recv_loads: dict[int, int] = {}
        combined_rank_loads: dict[int, int] = {}
        max_flow_rows = 0
        has_expert_release = False
        for flow in wave.flows:
            if not bool(flow.executable):
                return None
            rows = int(flow.row_count)
            max_flow_rows = max(max_flow_rows, rows)
            send_loads[int(flow.src_rank)] = int(send_loads.get(int(flow.src_rank), 0)) + rows
            recv_loads[int(flow.dst_rank)] = int(recv_loads.get(int(flow.dst_rank), 0)) + rows
            combined_rank_loads[int(flow.src_rank)] = int(combined_rank_loads.get(int(flow.src_rank), 0)) + rows
            combined_rank_loads[int(flow.dst_rank)] = int(combined_rank_loads.get(int(flow.dst_rank), 0)) + rows
            if str(flow.release_state) not in {"ready", "none"}:
                has_expert_release = True
        outgoing_ports = max(1, int(cost_model.max_outgoing_per_rank_per_wave))
        incoming_ports = max(1, int(cost_model.max_incoming_per_rank_per_wave))
        send_bound = max((float(rows) / float(outgoing_ports) for rows in send_loads.values()), default=0.0)
        recv_bound = max((float(rows) / float(incoming_ports) for rows in recv_loads.values()), default=0.0)
        if bool(cost_model.full_duplex):
            port_bound_rows = max(send_bound, recv_bound)
        else:
            port_bound_rows = max((float(rows) for rows in combined_rank_loads.values()), default=max(send_bound, recv_bound))
        computed = float(cost_model.launch_cost) + float(cost_model.row_transfer_cost) * float(port_bound_rows)
        if has_expert_release:
            computed += float(cost_model.expert_compute_delay)
        if float(wave.estimated_duration) > 0.0:
            computed = max(float(wave.estimated_duration), computed)
        if max_flow_rows == 0:
            computed = max(float(wave.estimated_duration), float(cost_model.launch_cost))
        return float(computed)


__all__ = ["CommonCorePlanEstimator", "PlanEstimator", "PlanningCostModel"]
