"""Formal offline flow-window study API."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rs.runtime.offline.traffic.matrix_builder import (
    build_owner_by_expert,
    build_sample_layer_matrices,
    combine_matrix_from_dispatch,
    load_trace_jsonl,
)
from rs.scheduling import FlowDemand, FlowWindow, LogicalSchedulePlan, LogicalWave
from rs.scheduling.multiphase.global_ready_set import (
    RUNTIME_LOOKAHEAD_MODE,
    replay_and_audit_schedule,
    schedule_global_ready_set as schedule_global_ready_set_impl,
    schedule_greedy as schedule_greedy_impl,
)


class UnsupportedP2Predictor(RuntimeError):
    pass


@dataclass(frozen=True)
class LogicalTopology:
    num_gpus: int


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
class ReleaseModel:
    expert_compute_delay: float = 0.0


@dataclass(frozen=True)
class GlobalReadySetOptions:
    scheduling_mode: str = RUNTIME_LOOKAHEAD_MODE
    prediction_confidence: float = 0.0


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


@dataclass(frozen=True)
class MultiPhaseSchedulingProblem:
    flow_window: FlowWindow
    topology: LogicalTopology
    release_model: ReleaseModel
    forecast: dict[str, Any] | None
    options: GlobalReadySetOptions
    dispatch_matrix: list[list[int]]
    p1_return_matrix: list[list[int]]
    p2_next_dispatch_forecast_matrix: list[list[int]]


def _resolve_trace_path(request: OfflineFlowStudyRequest) -> Path:
    trace_path = request.trace_artifact_dir / "trace.jsonl"
    return trace_path if trace_path.exists() else request.trace_artifact_dir


def _zero_matrix(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def _shuffle_matrix(matrix: list[list[int]]) -> list[list[int]]:
    flat = [value for row in matrix for value in row]
    rng = random.Random(42)
    rng.shuffle(flat)
    width = len(matrix[0]) if matrix else 0
    return [flat[index:index + width] for index in range(0, len(flat), width)] if width else []


def _matrix_to_flows(matrix: list[list[int]], *, phase: str, release_state: str, executable: bool) -> tuple[FlowDemand, ...]:
    flows: list[FlowDemand] = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            flows.append(
                FlowDemand(
                    flow_id=f"{phase}:{src_rank}->{dst_rank}",
                    phase=phase,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
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
    dispatch_matrix = sample_layer_matrices[sample_id][start_layer]
    p1_return_matrix = combine_matrix_from_dispatch(dispatch_matrix)
    perfect_p2 = sample_layer_matrices[sample_id][layer_ids[1]] if len(layer_ids) >= 2 else _zero_matrix(request.logical_topology.num_gpus)
    if request.p2_source.mode == "perfect_trace":
        p2_matrix = perfect_p2
    elif request.p2_source.mode == "zero_hint":
        p2_matrix = _zero_matrix(request.logical_topology.num_gpus)
    elif request.p2_source.mode == "shuffled_hint":
        p2_matrix = _shuffle_matrix(perfect_p2)
    elif request.p2_source.mode == "calibrated_artifact":
        raise UnsupportedP2Predictor("calibrated_artifact is not implemented in the frozen offline API")
    else:
        raise ValueError(f"unsupported p2_source={request.p2_source.mode!r}")
    flow_window = FlowWindow(
        ready_flows=_matrix_to_flows(dispatch_matrix, phase="p0_dispatch", release_state="ready", executable=True),
        blocked_flows=_matrix_to_flows(p1_return_matrix, phase="p1_return", release_state="blocked", executable=False),
        forecast_pressure=_matrix_to_flows(p2_matrix, phase="p2_next_dispatch_forecast", release_state="advisory_only", executable=False),
    )
    metadata = {
        "trace_path": str(trace_path),
        "sample_id": sample_id,
        "layer_ids": layer_ids,
        "start_layer": start_layer,
        "placement": {"mode": request.placement.mode, "owner_by_expert": owner_by_expert},
        "dispatch_matrix": dispatch_matrix,
        "p1_return_matrix": p1_return_matrix,
        "p2_next_dispatch_forecast_matrix": p2_matrix,
        "forecast_digest": hashlib.sha256(json.dumps(p2_matrix).encode("utf-8")).hexdigest()[:16],
        "forecast_source": request.p2_source.mode,
    }
    return flow_window, metadata


def build_scheduling_problem(request: OfflineFlowStudyRequest) -> MultiPhaseSchedulingProblem:
    flow_window, metadata = build_flow_window(request)
    prediction_confidence = 1.0 if any(any(value > 0 for value in row) for row in metadata["p2_next_dispatch_forecast_matrix"]) else 0.0
    return MultiPhaseSchedulingProblem(
        flow_window=flow_window,
        topology=request.logical_topology,
        release_model=ReleaseModel(expert_compute_delay=request.expert_compute_delay),
        forecast={
            "source": metadata["forecast_source"],
            "digest": metadata["forecast_digest"],
            "summary": {"nonzero_rows": sum(1 for row in metadata["p2_next_dispatch_forecast_matrix"] if any(value > 0 for value in row))},
        },
        options=GlobalReadySetOptions(scheduling_mode=request.scheduling_mode, prediction_confidence=prediction_confidence),
        dispatch_matrix=metadata["dispatch_matrix"],
        p1_return_matrix=metadata["p1_return_matrix"],
        p2_next_dispatch_forecast_matrix=metadata["p2_next_dispatch_forecast_matrix"],
    )


def _schedule_result_to_logical_plan(result: dict[str, Any]) -> LogicalSchedulePlan:
    waves: list[LogicalWave] = []
    by_wave: dict[int, list[FlowDemand]] = {}
    for entry in result["schedule"]:
        wave_id = int(entry["wave_id"])
        by_wave.setdefault(wave_id, []).append(
            FlowDemand(
                flow_id=str(entry["flow_id"]),
                phase=f"phase{int(entry['phase'])}",
                src_rank=int(entry["src_gpu"]),
                dst_rank=int(entry["dst_gpu"]),
                byte_count=int(round(float(entry["served_volume"]))),
                release_state="ready",
                is_executable=True,
            )
        )
    for wave_id in sorted(by_wave):
        waves.append(LogicalWave(wave_id=wave_id, flows=tuple(by_wave[wave_id])))
    return LogicalSchedulePlan(
        policy_name=str(result["strategy"]),
        waves=tuple(waves),
        diagnostics={
            "mode": result["mode"],
            "prediction_used": result["prediction_used"],
            "makespan": result["makespan"],
            "solve_time_ms": result["solve_time_ms"],
            "audit": result["audit"],
            "raw_schedule": result["schedule"],
        },
    )


def schedule_global_ready_set(problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
    result = schedule_global_ready_set_impl(
        problem.dispatch_matrix,
        problem.p1_return_matrix,
        problem.p2_next_dispatch_forecast_matrix,
        problem.topology.num_gpus,
        scheduling_mode=problem.options.scheduling_mode,
        prediction_confidence=problem.options.prediction_confidence,
        expert_compute_delay=problem.release_model.expert_compute_delay,
    )
    return _schedule_result_to_logical_plan(result)


def schedule_greedy(problem: MultiPhaseSchedulingProblem) -> LogicalSchedulePlan:
    result = schedule_greedy_impl(
        problem.dispatch_matrix,
        problem.p1_return_matrix,
        problem.p2_next_dispatch_forecast_matrix,
        problem.topology.num_gpus,
        scheduling_mode=problem.options.scheduling_mode,
        prediction_confidence=problem.options.prediction_confidence,
        expert_compute_delay=problem.release_model.expert_compute_delay,
    )
    return _schedule_result_to_logical_plan(result)


def replay_and_audit_logical_plan(problem: MultiPhaseSchedulingProblem, plan: LogicalSchedulePlan) -> dict[str, Any]:
    raw_schedule = list(plan.diagnostics.get("raw_schedule", []))
    return replay_and_audit_schedule(
        schedule=raw_schedule,
        dispatch_matrix=problem.dispatch_matrix,
        combine_matrix=problem.p1_return_matrix,
        next_dispatch_matrix=problem.p2_next_dispatch_forecast_matrix,
        num_gpus=problem.topology.num_gpus,
        expert_compute_delay=problem.release_model.expert_compute_delay,
        mode=problem.options.scheduling_mode,
        scheduler_name=plan.policy_name,
        planning_time_ms=float(plan.diagnostics.get("solve_time_ms", 0.0)),
        reported_makespan=float(plan.diagnostics.get("makespan", 0.0)),
        prediction_used=bool(plan.diagnostics.get("prediction_used", False)),
    )


__all__ = [
    "FlowWindowSelector",
    "GlobalReadySetOptions",
    "LogicalTopology",
    "MultiPhaseSchedulingProblem",
    "OfflineFlowStudyRequest",
    "P2ForecastSource",
    "PlacementConfig",
    "ReleaseModel",
    "UnsupportedP2Predictor",
    "build_flow_window",
    "build_scheduling_problem",
    "replay_and_audit_logical_plan",
    "schedule_global_ready_set",
    "schedule_greedy",
]
