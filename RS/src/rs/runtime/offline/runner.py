"""Formal offline flow-window study API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rs.runtime.offline.prediction import UnsupportedP2Predictor, build_dispatch_forecast
from rs.runtime.offline.traffic.matrix_builder import (
    build_owner_by_expert,
    build_sample_layer_matrices,
    combine_matrix_from_dispatch,
    load_trace_jsonl,
)
from rs.scheduling import (
    FlowDemand,
    FlowWindow,
    GlobalReadySetOptions,
    LogicalSchedulePlan,
    LogicalTopology,
    MultiPhaseSchedulingProblem,
    ReleaseConstraint,
    resolve_policy,
)
from rs.scheduling.multiphase.global_ready_set import replay_and_audit_schedule


@dataclass(frozen=True)
class PlacementConfig:
    mode: str = "round_robin"


@dataclass(frozen=True)
class FlowWindowSelector:
    sample_selector: str = "first"
    start_layer_selector: str = "first"


@dataclass(frozen=True)
class P2ForecastSource:
    mode: str


@dataclass(frozen=True)
class OfflineFlowStudyRequest:
    trace_artifact_dir: Path
    logical_topology: LogicalTopology
    placement: PlacementConfig
    window: FlowWindowSelector
    p2_source: P2ForecastSource
    policy_names: tuple[str, ...]
    expert_compute_delay: float
    scheduling_mode: str
    p0_weight: float = 1.0
    p1_reservation_weight: float = 1.0
    p2_hint_weight: float = 1.0


def _resolve_trace_path(request: OfflineFlowStudyRequest) -> Path:
    trace_path = request.trace_artifact_dir / "trace.jsonl"
    return trace_path if trace_path.exists() else request.trace_artifact_dir


def _matrix_to_flows(
    matrix: tuple[tuple[int, ...], ...],
    *,
    phase: str,
    release_state: str,
    executable: bool,
) -> tuple[FlowDemand, ...]:
    flows: list[FlowDemand] = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            flows.append(
                FlowDemand(
                    flow_id=f"{phase}:{src_rank}->{dst_rank}",
                    phase=phase,
                    src_rank=int(src_rank),
                    dst_rank=int(dst_rank),
                    byte_count=int(byte_count),
                    release_state=release_state,
                    is_executable=executable,
                )
            )
    return tuple(flows)


def build_flow_window(request: OfflineFlowStudyRequest) -> tuple[FlowWindow, dict[str, Any]]:
    trace_path = _resolve_trace_path(request)
    records = load_trace_jsonl(trace_path)
    owner_by_expert = build_owner_by_expert(records, placement=request.placement.mode, num_gpus=request.logical_topology.num_gpus)
    sample_layer_matrices = build_sample_layer_matrices(records, owner_by_expert=owner_by_expert, num_gpus=request.logical_topology.num_gpus)
    sample_ids = sorted(sample_layer_matrices)
    if not sample_ids:
        raise ValueError(f"no trace samples found in {trace_path}")
    sample_id = sample_ids[0] if request.window.sample_selector == "first" else sample_ids[0]
    layer_ids = sorted(sample_layer_matrices[sample_id])
    start_layer = layer_ids[0] if request.window.start_layer_selector == "first" else layer_ids[0]
    dispatch_matrix = tuple(tuple(int(v) for v in row) for row in sample_layer_matrices[sample_id][start_layer])
    p1_return_matrix = tuple(tuple(int(v) for v in row) for row in combine_matrix_from_dispatch([list(row) for row in dispatch_matrix]))
    actual_next_dispatch = (
        tuple(tuple(int(v) for v in row) for row in sample_layer_matrices[sample_id][layer_ids[1]])
        if len(layer_ids) >= 2
        else tuple(tuple(0 for _ in range(request.logical_topology.num_gpus)) for _ in range(request.logical_topology.num_gpus))
    )
    forecast = build_dispatch_forecast(
        mode=request.p2_source.mode,
        current_dispatch_matrix=dispatch_matrix,
        actual_next_dispatch_matrix=actual_next_dispatch,
    )
    flow_window = FlowWindow(
        ready_flows=_matrix_to_flows(dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True),
        blocked_flows=_matrix_to_flows(p1_return_matrix, phase="p1_return", release_state="blocked", executable=False),
        forecast_pressure=_matrix_to_flows(forecast.matrix, phase="p2_next_dispatch_forecast", release_state="advisory_only", executable=False),
    )
    metadata = {
        "trace_path": str(trace_path),
        "sample_id": sample_id,
        "layer_ids": layer_ids,
        "start_layer": start_layer,
        "placement": {"mode": request.placement.mode, "owner_by_expert": owner_by_expert},
        "dispatch_matrix": dispatch_matrix,
        "p1_return_matrix": p1_return_matrix,
        "actual_next_dispatch_matrix": actual_next_dispatch,
        "p2_next_dispatch_forecast_matrix": forecast.matrix,
        "forecast_digest": forecast.digest,
        "forecast_source": forecast.source,
        "forecast_oracle": forecast.oracle,
        "forecast_evaluation_eligible": forecast.evaluation_eligible,
    }
    return flow_window, metadata


def build_scheduling_problem(request: OfflineFlowStudyRequest) -> MultiPhaseSchedulingProblem:
    flow_window, metadata = build_flow_window(request)
    forecast = build_dispatch_forecast(
        mode=request.p2_source.mode,
        current_dispatch_matrix=metadata["dispatch_matrix"],
        actual_next_dispatch_matrix=metadata["actual_next_dispatch_matrix"],
    )
    prediction_confidence = 1.0 if any(any(value > 0 for value in row) for row in forecast.matrix) else 0.0
    return MultiPhaseSchedulingProblem(
        flow_window=flow_window,
        topology=request.logical_topology,
        release_model=ReleaseConstraint(
            phase="p1_return",
            rank=0,
            release_after_phase="p0_dispatch",
            expert_compute_delay=float(request.expert_compute_delay),
        ),
        forecast=forecast,
        options=GlobalReadySetOptions(
            scheduling_mode=request.scheduling_mode,
            information_mode="p0_p1_p2",
            prediction_confidence=prediction_confidence,
            p0_weight=float(request.p0_weight),
            p1_reservation_weight=float(request.p1_reservation_weight),
            p2_hint_weight=float(request.p2_hint_weight),
        ),
        p0_dispatch_matrix=metadata["dispatch_matrix"],
        p1_return_matrix=metadata["p1_return_matrix"],
        p2_next_dispatch_forecast_matrix=forecast.matrix,
    )


def build_policy_logical_plan(
    *,
    problem: MultiPhaseSchedulingProblem,
    policy_name: str,
    bucket_rows: int = 0,
    p0_weight: float = 1.0,
    p1_reservation_weight: float = 1.0,
    p2_hint_weight: float = 1.0,
) -> LogicalSchedulePlan:
    policy = resolve_policy(
        policy_name=policy_name,
        bucket_rows=bucket_rows,
        p0_weight=p0_weight,
        p1_reservation_weight=p1_reservation_weight,
        p2_hint_weight=p2_hint_weight,
    )
    return policy.build_logical_plan(problem)


def schedule_global_ready_set(problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
    return build_policy_logical_plan(problem=problem, policy_name="routersense_multiphase_lookahead:p0_p1_p2")


def schedule_greedy(problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
    return build_policy_logical_plan(problem=problem, policy_name="greedy_ready_set")


def replay_and_audit_logical_plan(problem: MultiPhaseSchedulingProblem, plan: LogicalSchedulePlan) -> dict[str, Any]:
    raw_schedule = list(plan.diagnostics.get("raw_schedule", []))
    if raw_schedule:
        return replay_and_audit_schedule(
            schedule=raw_schedule,
            dispatch_matrix=[list(row) for row in problem.p0_dispatch_matrix],
            combine_matrix=[list(row) for row in problem.p1_return_matrix],
            next_dispatch_matrix=[list(row) for row in problem.p2_next_dispatch_forecast_matrix],
            num_gpus=problem.topology.num_gpus,
            expert_compute_delay=problem.release_model.expert_compute_delay,
            mode=problem.options.scheduling_mode,
            scheduler_name=plan.policy_name,
            planning_time_ms=float(plan.diagnostics.get("solve_time_ms", 0.0)),
            reported_makespan=float(plan.diagnostics.get("makespan", 0.0)),
            prediction_used=bool(plan.diagnostics.get("prediction_used", False)),
        )
    flow_remaining = {flow.flow_id: int(flow.byte_count) for wave in plan.waves for flow in wave.flows}
    seen: set[str] = set()
    for wave in plan.waves:
        used_src: set[int] = set()
        used_dst: set[int] = set()
        for flow in wave.flows:
            if flow.src_rank in used_src or flow.dst_rank in used_dst:
                return {"valid": False, "validation_errors": [f"wave {wave.wave_id} violates full-duplex legality"], "makespan": None}
            used_src.add(flow.src_rank)
            used_dst.add(flow.dst_rank)
            seen.add(flow.flow_id)
            flow_remaining[flow.flow_id] = max(0, flow_remaining.get(flow.flow_id, 0) - int(flow.byte_count))
    incomplete = [flow_id for flow_id, remaining in flow_remaining.items() if remaining != 0]
    return {
        "valid": not incomplete,
        "validation_errors": [f"incomplete flow coverage: {incomplete!r}"] if incomplete else [],
        "makespan": float(sum(float(wave.duration) for wave in plan.waves)),
        "wave_count": len(plan.waves),
        "replay_makespan": float(sum(float(wave.duration) for wave in plan.waves)),
    }


__all__ = [
    "FlowWindowSelector",
    "LogicalTopology",
    "OfflineFlowStudyRequest",
    "P2ForecastSource",
    "PlacementConfig",
    "UnsupportedP2Predictor",
    "build_flow_window",
    "build_policy_logical_plan",
    "build_scheduling_problem",
    "replay_and_audit_logical_plan",
    "schedule_global_ready_set",
    "schedule_greedy",
]
