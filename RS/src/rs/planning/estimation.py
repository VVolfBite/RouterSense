from __future__ import annotations

from dataclasses import dataclass
import math
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

    def validate(self) -> None:
        if not str(self.cost_model_id):
            raise ValueError("cost_model_id must be non-empty")
        for name, value in {
            "row_transfer_cost": self.row_transfer_cost,
            "launch_cost": self.launch_cost,
            "expert_compute_delay": self.expert_compute_delay,
        }.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if int(self.max_outgoing_per_rank_per_wave) != 1:
            raise ValueError("formal cost model requires max_outgoing_per_rank_per_wave == 1")
        if int(self.max_incoming_per_rank_per_wave) != 1:
            raise ValueError("formal cost model requires max_incoming_per_rank_per_wave == 1")


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
        try:
            request.validate()
            plan.validate()
            cost_model.validate()
        except ValueError as exc:
            return PlanScore(
                estimated_makespan=float("inf"),
                estimator_id=self.estimator_id,
                cost_model_id=str(cost_model.cost_model_id),
                valid=False,
                reason=str(exc),
            )
        if str(plan.request_digest) != str(request.semantic_digest()):
            return PlanScore(
                estimated_makespan=float("inf"),
                estimator_id=self.estimator_id,
                cost_model_id=str(cost_model.cost_model_id),
                valid=False,
                reason="request_digest_mismatch",
            )
        estimated = 0.0
        invalid_reason: str | None = None
        for wave in plan.waves:
            wave_duration = self._estimate_wave_duration(wave=wave, request=request, cost_model=cost_model)
            if wave_duration is None:
                invalid_reason = f"invalid_wave:{wave.wave_id}"
                break
            estimated += float(wave_duration)
        return PlanScore(
            estimated_makespan=float("inf") if invalid_reason is not None else float(estimated),
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
        seen_flow_ids: set[str] = set()
        send_loads: dict[int, int] = {}
        recv_loads: dict[int, int] = {}
        combined_rank_loads: dict[int, int] = {}
        has_expert_release = False
        used_src: set[int] = set()
        used_dst: set[int] = set()
        world_size = int(request.topology.world_size)
        for flow in wave.flows:
            if flow.flow_id in seen_flow_ids:
                return None
            seen_flow_ids.add(flow.flow_id)
            if int(flow.src_rank) < 0 or int(flow.src_rank) >= world_size:
                return None
            if int(flow.dst_rank) < 0 or int(flow.dst_rank) >= world_size:
                return None
            rows = int(flow.row_count)
            if rows < 0:
                return None
            if str(flow.phase) not in {"p0_dispatch", "p1_return", "p2_next_dispatch_forecast", "p2_next_dispatch"}:
                return None
            if str(flow.release_state) not in {"ready", "none", "blocked", "after_p1", "advisory_only"}:
                return None
            if not bool(flow.executable) and str(flow.release_state) in {"ready", "none"}:
                return None
            if bool(flow.executable) and str(flow.release_state) == "advisory_only":
                return None
            if int(flow.src_rank) in used_src or int(flow.dst_rank) in used_dst:
                return None
            used_src.add(int(flow.src_rank))
            used_dst.add(int(flow.dst_rank))
            send_loads[int(flow.src_rank)] = int(send_loads.get(int(flow.src_rank), 0)) + rows
            recv_loads[int(flow.dst_rank)] = int(recv_loads.get(int(flow.dst_rank), 0)) + rows
            combined_rank_loads[int(flow.src_rank)] = int(combined_rank_loads.get(int(flow.src_rank), 0)) + rows
            combined_rank_loads[int(flow.dst_rank)] = int(combined_rank_loads.get(int(flow.dst_rank), 0)) + rows
            if str(flow.release_state) not in {"ready", "none"}:
                has_expert_release = True
        outgoing_ports = int(cost_model.max_outgoing_per_rank_per_wave)
        incoming_ports = int(cost_model.max_incoming_per_rank_per_wave)
        send_bound = max((float(rows) / float(outgoing_ports) for rows in send_loads.values()), default=0.0)
        recv_bound = max((float(rows) / float(incoming_ports) for rows in recv_loads.values()), default=0.0)
        if bool(cost_model.full_duplex):
            port_bound_rows = max(send_bound, recv_bound)
        else:
            port_bound_rows = max((float(rows) for rows in combined_rank_loads.values()), default=max(send_bound, recv_bound))
        computed = float(cost_model.launch_cost) + float(cost_model.row_transfer_cost) * float(port_bound_rows)
        if has_expert_release:
            computed += float(cost_model.expert_compute_delay)
        if not math.isfinite(computed) or computed < 0.0:
            return None
        return float(computed)


__all__ = ["CommonCorePlanEstimator", "PlanEstimator", "PlanningCostModel"]
