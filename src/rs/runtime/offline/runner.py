"""Formal offline flow-window study API."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from rs.runtime.offline.prediction import UnsupportedP2Predictor, build_dispatch_forecast
from rs.planning.runtime_bridge import PlannerPolicyConfig, build_runtime_policy, build_runtime_request_from_problem
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
)
from rs.scheduling.multiphase.replay import replay_and_audit_schedule


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
    request = build_runtime_request_from_problem(
        request_id=f"offline:{policy_name}",
        problem=problem,
        bucket_rows=int(bucket_rows),
        policy_options=PlannerPolicyConfig(
            p0_weight=float(p0_weight),
            p1_weight=float(p1_reservation_weight),
            p2_hint_weight=float(p2_hint_weight),
        ),
        hint_type=str(getattr(problem.forecast, "source", "none") if problem.forecast is not None else "none"),
        confidence=float(problem.options.prediction_confidence),
        layer_id=None,
    )
    policy = build_runtime_policy(policy_name, PlannerPolicyConfig(
        p0_weight=float(p0_weight),
        p1_weight=float(p1_reservation_weight),
        p2_hint_weight=float(p2_hint_weight),
    ))
    return policy.plan(request)


def schedule_global_ready_set(problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
    return build_policy_logical_plan(problem=problem, policy_name="birkhoff_bucket_phase_local")


def schedule_greedy(problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
    return build_policy_logical_plan(problem=problem, policy_name="greedy_bucket")


def replay_and_audit_logical_plan(problem: MultiPhaseSchedulingProblem, plan: LogicalSchedulePlan) -> dict[str, Any]:
    raw_schedule = list(plan.diagnostics.get("raw_schedule", []))
    if not raw_schedule:
        cursor = 0.0
        phase_map = {
            "p0_dispatch": 0,
            "p1_return": 1,
            "p2_next_dispatch": 2,
            "P0": 0,
            "P1": 1,
            "P2": 2,
        }
        for wave in sorted(plan.waves, key=lambda item: int(item.wave_id)):
            start = float(cursor)
            end = start + float(wave.duration)
            for flow in wave.flows:
                raw_schedule.append(
                    {
                        "chunk_id": str(flow.flow_id),
                        "flow_id": str(flow.flow_id),
                        "phase": int(phase_map[str(flow.phase)]),
                        "src_gpu": int(flow.src_rank),
                        "dst_gpu": int(flow.dst_rank),
                        "start": start,
                        "end": end,
                        "served_volume": float(flow.byte_count),
                        "wave_id": int(wave.wave_id),
                    }
                )
            cursor = end
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
        reported_makespan=float(plan.diagnostics.get("makespan", 0.0)) if plan.diagnostics.get("makespan") is not None else None,
        prediction_used=bool(plan.diagnostics.get("prediction_used", False)),
    )


def summarize_schedule_tail_metrics(
    *,
    problem: MultiPhaseSchedulingProblem,
    plan: LogicalSchedulePlan,
    audit: dict[str, Any],
) -> dict[str, Any]:
    raw_schedule = list(plan.diagnostics.get("raw_schedule", ()))
    if not raw_schedule:
        wave_durations = [float(wave.duration) for wave in plan.waves]
        makespan = float(audit.get("makespan", sum(wave_durations)))
        return {
            "active_wave_count": len(wave_durations),
            "wave_duration_p50": _percentile(wave_durations, 0.50),
            "wave_duration_p95": _percentile(wave_durations, 0.95),
            "wave_duration_p99": _percentile(wave_durations, 0.99),
            "wave_duration_max": max(wave_durations, default=0.0),
            "first_p1_release_time": None,
            "first_p1_start_time": None,
            "first_p1_release_wait": None,
            "mean_p1_release_wait": None,
            "max_p1_release_wait": None,
            "first_p2_release_time": None,
            "first_p2_start_time": None,
            "first_p2_release_wait": None,
            "mean_p2_release_wait": None,
            "max_p2_release_wait": None,
            "p0_inbound_completion_p50": None,
            "p0_inbound_completion_p95": None,
            "p0_inbound_completion_p99": None,
            "p0_inbound_completion_max": None,
            "p1_inbound_completion_p50": None,
            "p1_inbound_completion_p95": None,
            "p1_inbound_completion_p99": None,
            "p1_inbound_completion_max": None,
            "p0_inbound_tail_gap": None,
            "p1_inbound_tail_gap": None,
            "bottleneck_send_busy_share": _busy_share(audit.get("send_busy_time", ()), makespan),
            "bottleneck_recv_busy_share": _busy_share(audit.get("recv_busy_time", ()), makespan),
        }

    num_gpus = int(problem.topology.num_gpus)
    entries = sorted(
        raw_schedule,
        key=lambda item: (
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
            int(item.get("phase", 0)),
            int(item.get("src_gpu", 0)),
            int(item.get("dst_gpu", 0)),
        ),
    )
    p0_inbound_completion = [0.0] * num_gpus
    p1_inbound_completion = [0.0] * num_gpus
    p1_release_waits: list[float] = []
    p2_release_waits: list[float] = []
    p1_release_times: list[float] = []
    p1_start_times: list[float] = []
    p2_release_times: list[float] = []
    p2_start_times: list[float] = []
    wave_bounds: dict[int, tuple[float, float]] = {}

    for entry in entries:
        phase = int(entry["phase"])
        dst = int(entry["dst_gpu"])
        start = float(entry.get("start", 0.0))
        end = float(entry.get("end", 0.0))
        wave_id = int(entry.get("wave_id", 0))
        bounds = wave_bounds.get(wave_id)
        if bounds is None:
            wave_bounds[wave_id] = (start, end)
        else:
            wave_bounds[wave_id] = (min(bounds[0], start), max(bounds[1], end))
        if phase == 0:
            p0_inbound_completion[dst] = max(p0_inbound_completion[dst], end)
        elif phase == 1:
            p1_inbound_completion[dst] = max(p1_inbound_completion[dst], end)

    for entry in entries:
        phase = int(entry["phase"])
        src = int(entry["src_gpu"])
        start = float(entry.get("start", 0.0))
        if phase == 1:
            required = float(p0_inbound_completion[src]) + float(problem.release_model.expert_compute_delay)
            p1_release_times.append(required)
            p1_start_times.append(start)
            p1_release_waits.append(start - required)
        elif phase == 2 and problem.options.scheduling_mode == "execution_window":
            required = float(p1_inbound_completion[src])
            p2_release_times.append(required)
            p2_start_times.append(start)
            p2_release_waits.append(start - required)

    active_p0 = [value for value in p0_inbound_completion if value > 0.0]
    active_p1 = [value for value in p1_inbound_completion if value > 0.0]
    wave_durations = [max(0.0, float(end) - float(start)) for start, end in wave_bounds.values()]
    makespan = float(plan.diagnostics.get("makespan", audit.get("makespan", 0.0)) or 0.0)
    return {
        "active_wave_count": len(wave_durations),
        "wave_duration_p50": _percentile(wave_durations, 0.50),
        "wave_duration_p95": _percentile(wave_durations, 0.95),
        "wave_duration_p99": _percentile(wave_durations, 0.99),
        "wave_duration_max": max(wave_durations, default=0.0),
        "first_p1_release_time": min(p1_release_times) if p1_release_times else None,
        "first_p1_start_time": min(p1_start_times) if p1_start_times else None,
        "first_p1_release_wait": min(p1_release_waits) if p1_release_waits else None,
        "mean_p1_release_wait": _mean(p1_release_waits),
        "max_p1_release_wait": max(p1_release_waits, default=None),
        "first_p2_release_time": min(p2_release_times) if p2_release_times else None,
        "first_p2_start_time": min(p2_start_times) if p2_start_times else None,
        "first_p2_release_wait": min(p2_release_waits) if p2_release_waits else None,
        "mean_p2_release_wait": _mean(p2_release_waits),
        "max_p2_release_wait": max(p2_release_waits, default=None),
        "p0_inbound_completion_p50": _percentile(active_p0, 0.50),
        "p0_inbound_completion_p95": _percentile(active_p0, 0.95),
        "p0_inbound_completion_p99": _percentile(active_p0, 0.99),
        "p0_inbound_completion_max": max(active_p0, default=None),
        "p1_inbound_completion_p50": _percentile(active_p1, 0.50),
        "p1_inbound_completion_p95": _percentile(active_p1, 0.95),
        "p1_inbound_completion_p99": _percentile(active_p1, 0.99),
        "p1_inbound_completion_max": max(active_p1, default=None),
        "p0_inbound_tail_gap": _tail_gap(active_p0),
        "p1_inbound_tail_gap": _tail_gap(active_p1),
        "bottleneck_send_busy_share": _busy_share(audit.get("send_busy_time", ()), makespan),
        "bottleneck_recv_busy_share": _busy_share(audit.get("recv_busy_time", ()), makespan),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _tail_gap(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values) - min(values))


def _busy_share(values: Any, makespan: float) -> float | None:
    if makespan <= 0.0:
        return None
    busy = [float(value) for value in values]
    if not busy:
        return None
    return float(max(busy) / makespan)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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
    "summarize_schedule_tail_metrics",
]
