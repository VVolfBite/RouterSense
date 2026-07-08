"""Window-state contracts for online joint-shadow scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rs.runtime.online.megatron_ep.contracts import RuntimeObservation
from rs.runtime.online.megatron_ep.observation import parse_layer_id
from rs.scheduling.contracts import FlowDemand, FlowWindow, ForecastPressure, GlobalReadySetOptions, LogicalTopology, MultiPhaseSchedulingProblem, PreparedWindowPlan, ReleaseConstraint


def _zero_matrix(num_gpus: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in range(num_gpus)) for _ in range(num_gpus))


def _observation_row_matrix(observation: RuntimeObservation | None, *, num_gpus: int) -> tuple[tuple[int, ...], ...]:
    if observation is None:
        return _zero_matrix(num_gpus)
    row = [0 for _ in range(num_gpus)]
    for dst_rank, value in enumerate(tuple(int(v) for v in observation.per_peer_bytes)[:num_gpus]):
        if dst_rank == int(observation.global_rank):
            continue
        row[dst_rank] = int(value)
    return tuple(tuple(row if src_rank == int(observation.global_rank) else [0 for _ in range(num_gpus)]) for src_rank in range(num_gpus))


def _matrix_from_prepared_plan(prepared_plan: PreparedWindowPlan | None, *, phase: str, num_gpus: int) -> tuple[tuple[int, ...], ...]:
    if prepared_plan is None:
        return _zero_matrix(num_gpus)
    if phase == "p2_next_dispatch_forecast":
        forecast_matrix = tuple(
            tuple(int(value) for value in row)
            for row in getattr(prepared_plan, "forecast_matrix", ())
        )
        if forecast_matrix:
            return forecast_matrix
    matrix = [[0 for _ in range(num_gpus)] for _ in range(num_gpus)]
    for wave in prepared_plan.logical_plan.waves:
        for flow in wave.flows:
            if str(flow.phase) != phase:
                continue
            src_rank = int(flow.src_rank)
            dst_rank = int(flow.dst_rank)
            if 0 <= src_rank < num_gpus and 0 <= dst_rank < num_gpus and src_rank != dst_rank:
                matrix[src_rank][dst_rank] += int(flow.byte_count)
    return tuple(tuple(row) for row in matrix)


def _flows_from_matrix(
    matrix: tuple[tuple[int, ...], ...],
    *,
    phase: str,
    release_state: str,
    executable: bool,
    source: str,
) -> tuple[FlowDemand, ...]:
    flows: list[FlowDemand] = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, value in enumerate(row):
            if src_rank == dst_rank or int(value) <= 0:
                continue
            flows.append(
                FlowDemand(
                    flow_id=f"{phase}:{src_rank}->{dst_rank}",
                    phase=phase,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    byte_count=int(value),
                    release_state=release_state,
                    is_executable=executable,
                    dependency_metadata={"source": source},
                )
            )
    return tuple(flows)


@dataclass(frozen=True)
class WindowReleaseState:
    p0_dispatch_completed_ranks: tuple[int, ...] = ()
    p1_return_materialized_ranks: tuple[int, ...] = ()
    p1_return_completed_ranks: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedPlanBinding:
    source_layer_name: str
    source_layer_id: str
    target_layer_id: str
    window_key: str
    forecast_digest: str
    source_logical_plan_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineWindowState:
    window_key: str
    layer_name: str
    layer_id: str
    ep_group_ranks: tuple[int, ...]
    local_rank: int
    p0_observation: RuntimeObservation | None = None
    p1_observation: RuntimeObservation | None = None
    prepared_plan_binding: PreparedPlanBinding | None = None
    prepared_plan: PreparedWindowPlan | None = None
    release_state: WindowReleaseState = field(default_factory=WindowReleaseState)

    def to_record(self) -> dict[str, Any]:
        payload = {
            "window_key": self.window_key,
            "layer_name": self.layer_name,
            "layer_id": self.layer_id,
            "ep_group_ranks": list(self.ep_group_ranks),
            "local_rank": self.local_rank,
            "has_p0_observation": self.p0_observation is not None,
            "has_p1_observation": self.p1_observation is not None,
            "prepared_plan_binding": None if self.prepared_plan_binding is None else self.prepared_plan_binding.to_dict(),
            "release_state": self.release_state.to_dict(),
        }
        if self.p0_observation is not None:
            payload["p0_per_peer_bytes"] = list(int(v) for v in self.p0_observation.per_peer_bytes)
        if self.p1_observation is not None:
            payload["p1_per_peer_bytes"] = list(int(v) for v in self.p1_observation.per_peer_bytes)
        return payload


def bind_prepared_plan(*, layer_name: str, prepared_plan: PreparedWindowPlan | None, source_layer_name: str, source_logical_plan_hash: str) -> PreparedPlanBinding | None:
    if prepared_plan is None:
        return None
    layer_id = parse_layer_id(layer_name)
    if str(prepared_plan.applies_from_layer_id) != str(layer_id):
        return None
    return PreparedPlanBinding(
        source_layer_name=source_layer_name,
        source_layer_id=str(prepared_plan.created_at_layer_id),
        target_layer_id=str(prepared_plan.applies_from_layer_id),
        window_key=str(prepared_plan.window_key),
        forecast_digest=str(prepared_plan.forecast_digest),
        source_logical_plan_hash=source_logical_plan_hash,
    )


def build_window_state(
    *,
    layer_name: str,
    ep_group_ranks: tuple[int, ...],
    local_rank: int,
    p0_observation: RuntimeObservation | None,
    p1_observation: RuntimeObservation | None,
    prepared_plan: PreparedWindowPlan | None,
    prepared_plan_binding: PreparedPlanBinding | None,
    release_state: WindowReleaseState,
) -> OnlineWindowState:
    layer_id = parse_layer_id(layer_name)
    window_key = (
        str(prepared_plan_binding.window_key)
        if prepared_plan_binding is not None
        else f"window:{layer_id}:{','.join(str(rank) for rank in ep_group_ranks)}"
    )
    return OnlineWindowState(
        window_key=window_key,
        layer_name=layer_name,
        layer_id=layer_id,
        ep_group_ranks=ep_group_ranks,
        local_rank=local_rank,
        p0_observation=p0_observation,
        p1_observation=p1_observation,
        prepared_plan_binding=prepared_plan_binding,
        prepared_plan=prepared_plan,
        release_state=release_state,
    )


def build_shadow_problem(
    *,
    state: OnlineWindowState,
    p0_weight: float,
    p1_reservation_weight: float,
    p2_hint_weight: float,
) -> MultiPhaseSchedulingProblem:
    num_gpus = len(state.ep_group_ranks)
    predicted_p0 = _matrix_from_prepared_plan(state.prepared_plan, phase="p0_dispatch", num_gpus=num_gpus)
    predicted_p1 = _matrix_from_prepared_plan(state.prepared_plan, phase="p1_return", num_gpus=num_gpus)
    predicted_p2 = _matrix_from_prepared_plan(state.prepared_plan, phase="p2_next_dispatch_forecast", num_gpus=num_gpus)
    actual_p0 = _observation_row_matrix(state.p0_observation, num_gpus=num_gpus)
    actual_p1 = _observation_row_matrix(state.p1_observation, num_gpus=num_gpus)

    ready_p0_matrix = actual_p0 if any(any(value > 0 for value in row) for row in actual_p0) else predicted_p0
    release_state = state.release_state
    local_rank = int(state.local_rank)
    p0_completed = local_rank in set(int(rank) for rank in release_state.p0_dispatch_completed_ranks)
    p1_materialized = state.p1_observation is not None or local_rank in set(
        int(rank) for rank in release_state.p1_return_materialized_ranks
    )
    if p1_materialized and p0_completed:
        ready_p1_matrix = actual_p1
        blocked_p1_matrix = _zero_matrix(num_gpus)
        ready_p1_source = "actual_p1_materialized_and_released"
        blocked_p1_source = "none"
    elif p1_materialized:
        ready_p1_matrix = _zero_matrix(num_gpus)
        blocked_p1_matrix = actual_p1
        ready_p1_source = "none"
        blocked_p1_source = "actual_p1_materialized_waiting_p0_release"
    else:
        ready_p1_matrix = _zero_matrix(num_gpus)
        blocked_p1_matrix = predicted_p1
        ready_p1_source = "none"
        blocked_p1_source = "prepared_plan_p1_waiting_materialization"

    ready_flows = [
        *_flows_from_matrix(
            ready_p0_matrix,
            phase="p0_dispatch",
            release_state="ready",
            executable=True,
            source="actual_p0" if state.p0_observation is not None else "prepared_plan_p0",
        ),
        *_flows_from_matrix(
            ready_p1_matrix,
            phase="p1_return",
            release_state="ready",
            executable=True,
            source=ready_p1_source,
        ),
    ]
    blocked_flows = _flows_from_matrix(
        blocked_p1_matrix,
        phase="p1_return",
        release_state="blocked",
        executable=False,
        source=blocked_p1_source,
    )
    forecast_flows = _flows_from_matrix(
        predicted_p2,
        phase="p2_next_dispatch_forecast",
        release_state="advisory_only",
        executable=False,
        source="prepared_plan_p2",
    )
    has_forecast = any(any(value > 0 for value in row) for row in predicted_p2)
    has_blocked = bool(blocked_flows)
    information_mode = "p0_p1_p2" if has_forecast else "p0_p1" if has_blocked else "p0_only"
    forecast = (
        ForecastPressure(
            source="prepared_window_plan",
            digest=str(state.prepared_plan_binding.forecast_digest),
            oracle=False,
            evaluation_eligible=True,
            matrix_shape=(num_gpus, num_gpus),
            matrix_total_bytes=sum(sum(int(value) for value in row) for row in predicted_p2),
            matrix=predicted_p2,
            metadata={"window_key": state.window_key, "source_layer": state.prepared_plan_binding.source_layer_name},
        )
        if state.prepared_plan_binding is not None and has_forecast
        else None
    )
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=tuple(ready_flows),
            blocked_flows=blocked_flows,
            forecast_pressure=forecast_flows,
        ),
        topology=LogicalTopology(num_gpus=num_gpus),
        release_model=ReleaseConstraint(
            phase="p1_return",
            rank=int(state.local_rank),
            release_after_phase="p0_dispatch",
            expert_compute_delay=0.0,
        ),
        forecast=forecast,
        options=GlobalReadySetOptions(
            scheduling_mode="runtime_lookahead",
            information_mode=information_mode,
            prediction_confidence=1.0 if has_forecast else 0.0,
            p0_weight=float(p0_weight),
            p1_reservation_weight=float(p1_reservation_weight),
            p2_hint_weight=float(p2_hint_weight),
            max_waves=256,
        ),
        p0_dispatch_matrix=ready_p0_matrix,
        p1_return_matrix=ready_p1_matrix if p1_materialized else blocked_p1_matrix,
        p2_next_dispatch_forecast_matrix=predicted_p2,
    )


__all__ = [
    "OnlineWindowState",
    "PreparedPlanBinding",
    "WindowReleaseState",
    "bind_prepared_plan",
    "build_shadow_problem",
    "build_window_state",
]
