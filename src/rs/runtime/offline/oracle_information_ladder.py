"""Exact information ladder for tiny/reduced P0/P1/P2 windows.

The ladder separates future-information value from heuristic quality:

* ``exact_joint_p01_reactive`` sees true P0/P1 initially and reveals each true
  P2 source row only when that rank completes its P1 inbound barrier;
* ``exact_joint_p012_predicted`` solves the same rolling exact problem with a
  forecast row for every unrevealed P2 source, while executing only true bytes;
* ``oracle_joint_p012_perfect`` sees true P0/P1/P2 from the start and is the
  clairvoyant exact upper bound for the supported tiny atomic-edge model.

Predicted P2 is advisory planning geometry. It never becomes executable traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
import time
from typing import Any, Literal

from rs.scheduling.contracts import (
    FlowDemand,
    FlowWindow,
    ForecastPressure,
    GlobalReadySetOptions,
    LogicalTopology,
    MultiPhaseSchedulingProblem,
    ReleaseConstraint,
)
from rs.scheduling.reference.exact_small_instance import solve_problem_exact_with_scope
from rs.prediction.traffic_envelope import evaluate_traffic_forecast
from rs.scheduling.traffic_matrix import matrix_digest_remote, matrix_remote_bytes

from .traffic_dataset import TrafficInstanceRecord


Matrix = tuple[tuple[int, ...], ...]
InformationLevel = Literal[
    "exact_joint_p01_reactive",
    "exact_joint_p012_predicted",
    "oracle_joint_p012_perfect",
]


@dataclass(frozen=True)
class ExactReduction:
    source_instance_id: str
    selected_original_ranks: tuple[int, ...]
    p0: Matrix
    p1: Matrix
    p2: Matrix
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rank_count(self) -> int:
        return len(self.p0)


@dataclass(frozen=True)
class ExactInformationResult:
    information_level: InformationLevel
    objective: float | None
    solver_status: str
    certified_optimal_steps: bool
    planning_runtime_ms: float
    replan_count: int
    revealed_rows: int
    schedule: tuple[dict[str, Any], ...]
    audit: dict[str, Any]

    @property
    def valid(self) -> bool:
        return bool(self.audit.get("valid", False))

    def to_dict(self, *, include_schedule: bool = False) -> dict[str, Any]:
        payload = {
            "information_level": self.information_level,
            "objective": self.objective,
            "solver_status": self.solver_status,
            "certified_optimal_steps": self.certified_optimal_steps,
            "planning_runtime_ms": self.planning_runtime_ms,
            "replan_count": self.replan_count,
            "revealed_rows": self.revealed_rows,
            "valid": self.valid,
            "audit": self.audit,
        }
        if include_schedule:
            payload["schedule"] = list(self.schedule)
        return payload


@dataclass(frozen=True)
class ExactInformationLadder:
    reactive: ExactInformationResult
    predicted: ExactInformationResult | None
    perfect: ExactInformationResult
    metrics: dict[str, float | None]
    reduction: ExactReduction

    def to_dict(self, *, include_schedule: bool = False) -> dict[str, Any]:
        return {
            "schema_version": "exact_p2_information_ladder.v1",
            "reactive": self.reactive.to_dict(include_schedule=include_schedule),
            "predicted": None if self.predicted is None else self.predicted.to_dict(include_schedule=include_schedule),
            "perfect": self.perfect.to_dict(include_schedule=include_schedule),
            "metrics": self.metrics,
            "reduction": {
                "source_instance_id": self.reduction.source_instance_id,
                "selected_original_ranks": list(self.reduction.selected_original_ranks),
                "p0": [list(row) for row in self.reduction.p0],
                "p1": [list(row) for row in self.reduction.p1],
                "p2": [list(row) for row in self.reduction.p2],
                "metadata": self.reduction.metadata,
            },
        }


def _matrix(rows: Any) -> Matrix:
    matrix = tuple(tuple(int(value) for value in row) for row in rows)
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be non-empty and square")
    if any(value < 0 for row in matrix for value in row):
        raise ValueError("matrix must be non-negative")
    return matrix


def _zero(size: int) -> Matrix:
    return tuple(tuple(0 for _ in range(size)) for _ in range(size))


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(tuple(int(matrix[destination][source]) for destination in range(len(matrix))) for source in range(len(matrix)))


def _nonzero_remote_count(matrix: Matrix) -> int:
    return sum(1 for source, row in enumerate(matrix) for destination, value in enumerate(row) if source != destination and value > 0)


def _select_rank_subset(instance: TrafficInstanceRecord, rank_count: int) -> tuple[int, ...]:
    size = instance.world_size
    target = min(max(2, int(rank_count)), 4, size)
    if size <= target:
        return tuple(range(size))
    best_subset: tuple[int, ...] | None = None
    best_key: tuple[float, float, tuple[int, ...]] | None = None
    for subset in combinations(range(size), target):
        selected = set(subset)
        internal = 0.0
        incident = 0.0
        for matrix in (instance.p0, instance.p2):
            for source, row in enumerate(matrix):
                for destination, value in enumerate(row):
                    if source == destination or value <= 0:
                        continue
                    if source in selected and destination in selected:
                        internal += float(value)
                    if source in selected or destination in selected:
                        incident += float(value)
        key = (internal, incident, tuple(-rank for rank in subset))
        if best_key is None or key > best_key:
            best_key = key
            best_subset = subset
    assert best_subset is not None
    return tuple(best_subset)


def _submatrix(matrix: Matrix, ranks: tuple[int, ...]) -> Matrix:
    return tuple(tuple(int(matrix[source][destination]) for destination in ranks) for source in ranks)


def _quantized_top_edges(matrix: Matrix, *, edge_budget: int, duration_levels: int) -> Matrix:
    size = len(matrix)
    edges = sorted(
        (
            (int(value), source, destination)
            for source, row in enumerate(matrix)
            for destination, value in enumerate(row)
            if source != destination and int(value) > 0
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )[: max(0, int(edge_budget))]
    output = [[0 for _ in range(size)] for _ in range(size)]
    maximum = max((value for value, _, _ in edges), default=1)
    levels = max(1, int(duration_levels))
    for value, source, destination in edges:
        duration = max(1, int(math.ceil(levels * float(value) / float(maximum))))
        output[source][destination] = duration
    return _matrix(output)


def reduce_traffic_instance_for_exact(
    instance: TrafficInstanceRecord,
    *,
    rank_count: int = 4,
    p0_edge_budget: int = 4,
    total_task_budget: int = 12,
    duration_levels: int = 3,
) -> ExactReduction:
    """Deterministically reduce a real traffic window to the certified scale."""

    instance.validate()
    ranks = _select_rank_subset(instance, rank_count)
    p0_source = _submatrix(instance.p0, ranks)
    p2_source = _submatrix(instance.p2, ranks)
    p0 = _quantized_top_edges(p0_source, edge_budget=min(int(p0_edge_budget), int(total_task_budget) // 2), duration_levels=duration_levels)
    p1 = _transpose(p0)
    p0_edges = _nonzero_remote_count(p0)
    p2_budget = max(0, int(total_task_budget) - 2 * p0_edges)
    p2 = _quantized_top_edges(p2_source, edge_budget=p2_budget, duration_levels=duration_levels)
    task_count = 2 * _nonzero_remote_count(p0) + _nonzero_remote_count(p2)
    if task_count > int(total_task_budget):
        raise AssertionError("exact reduction exceeded task budget")
    return ExactReduction(
        source_instance_id=instance.traffic_instance_id,
        selected_original_ranks=ranks,
        p0=p0,
        p1=p1,
        p2=p2,
        metadata={
            "reduction_policy": "max_induced_remote_load_top_edges_v1",
            "source_world_size": instance.world_size,
            "reduced_world_size": len(ranks),
            "p0_edge_budget": int(p0_edge_budget),
            "total_task_budget": int(total_task_budget),
            "duration_levels": int(duration_levels),
            "task_count": task_count,
            "p2_edge_budget": p2_budget,
        },
    )


def reduce_forecast_for_exact(
    forecast_matrix: Matrix,
    reduction: ExactReduction,
) -> Matrix:
    """Project a full-rank forecast through the same deterministic reduction."""

    forecast = _matrix(forecast_matrix)
    ranks = reduction.selected_original_ranks
    if max(ranks, default=-1) >= len(forecast):
        raise ValueError("forecast matrix is smaller than selected original ranks")
    source = _submatrix(forecast, ranks)
    return _quantized_top_edges(
        source,
        edge_budget=int(reduction.metadata.get("p2_edge_budget", _nonzero_remote_count(reduction.p2))),
        duration_levels=int(reduction.metadata.get("duration_levels", 3)),
    )


def _flows(matrix: Matrix, *, phase: str, release_state: str) -> tuple[FlowDemand, ...]:
    return tuple(
        FlowDemand(
            flow_id=f"{phase}:{source}->{destination}",
            phase=phase,
            src_rank=source,
            dst_rank=destination,
            byte_count=int(value),
            release_state=release_state,
            is_executable=True,
        )
        for source, row in enumerate(matrix)
        for destination, value in enumerate(row)
        if source != destination and int(value) > 0
    )


def _problem(p0: Matrix, p1: Matrix, p2: Matrix, *, information_mode: str) -> MultiPhaseSchedulingProblem:
    size = len(p0)
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows(p0, phase="p0_dispatch", release_state="ready"),
            blocked_flows=_flows(p1, phase="p1_return", release_state="blocked"),
            forecast_pressure=_flows(p2, phase="p2_next_dispatch", release_state="blocked"),
        ),
        topology=LogicalTopology(num_gpus=size),
        release_model=ReleaseConstraint(phase="rank_phase_release", rank=0, release_after_phase="p0_dispatch", expert_compute_delay=0.0),
        forecast=ForecastPressure(
            source=information_mode,
            digest=matrix_digest_remote(p2),
            oracle=information_mode == "p012_perfect",
            evaluation_eligible=information_mode != "p012_perfect",
            matrix_shape=(size, size),
            matrix_total_bytes=int(matrix_remote_bytes(p2)),
            matrix=p2,
            metadata={"information_mode": information_mode},
        ),
        options=GlobalReadySetOptions(
            scheduling_mode="execution_window",
            information_mode=information_mode,
            prediction_confidence=1.0 if any(value for row in p2 for value in row) else 0.0,
            max_waves=256,
        ),
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_forecast_matrix=p2,
    )


def _first_wave(result: dict[str, Any]) -> dict[str, Any] | None:
    schedule = list(result.get("schedule", ()))
    return None if not schedule else dict(schedule[0])


def _column_zero(matrix: list[list[int]], column: int) -> bool:
    return all(int(matrix[source][column]) == 0 for source in range(len(matrix)))


def _bounded_planning_p2(
    *,
    p0: list[list[int]],
    p1: list[list[int]],
    truth: list[list[int]],
    forecast: list[list[int]],
    revealed: set[int],
    total_task_budget: int = 12,
) -> tuple[list[list[int]], int]:
    """Build a task-budget-safe advisory P2 view for the tiny exact model.

    All actually revealed P2 edges are retained.  Unrevealed forecast edges are
    advisory only and are deterministically truncated by descending predicted
    duration when the mixed true/forecast view would exceed the exact solver's
    total task budget.  This keeps each rolling solve within the same certified
    tiny-instance model without dropping executable truth.
    """

    size = len(p0)
    current_edges = sum(
        1
        for matrix in (p0, p1)
        for source, row in enumerate(matrix)
        for destination, value in enumerate(row)
        if source != destination and int(value) > 0
    )
    available = max(0, int(total_task_budget) - current_edges)
    output = [[0 for _ in range(size)] for _ in range(size)]
    actual_edges = []
    forecast_edges = []
    for source in range(size):
        row = truth[source] if source in revealed else forecast[source]
        for destination, value in enumerate(row):
            amount = int(value)
            if source == destination or amount <= 0:
                continue
            item = (source, destination, amount)
            if source in revealed:
                actual_edges.append(item)
            else:
                forecast_edges.append(item)
    if len(actual_edges) > available:
        raise ValueError(
            "revealed exact P2 edges exceed the tiny solver task budget; "
            "the reduction contract is inconsistent"
        )
    for source, destination, amount in actual_edges:
        output[source][destination] = amount
    remaining = available - len(actual_edges)
    ordered_forecast = sorted(
        forecast_edges,
        key=lambda item: (-item[2], item[0], item[1]),
    )
    for source, destination, amount in ordered_forecast[:remaining]:
        output[source][destination] = amount
    return output, max(0, len(ordered_forecast) - remaining)


def _rolling_exact(
    reduction: ExactReduction,
    *,
    forecast: Matrix | None,
    time_limit_ms: int,
) -> ExactInformationResult:
    predicted = forecast is not None
    p0 = [list(row) for row in reduction.p0]
    p1 = [list(row) for row in reduction.p1]
    truth = [list(row) for row in reduction.p2]
    forecast_rows = [list(row) for row in (forecast or _zero(len(p0)))]
    if len(forecast_rows) != len(p0) or any(len(row) != len(p0) for row in forecast_rows):
        raise ValueError("forecast shape must match reduced instance")
    revealed: set[int] = set()
    schedule: list[dict[str, Any]] = []
    current_time = 0.0
    planning_runtime_ms = 0.0
    replan_count = 0
    certified = True
    status = "optimal"
    total_truncated_forecast_edges = 0

    def reveal_ready_rows() -> None:
        for rank in range(len(p0)):
            if rank not in revealed and _column_zero(p1, rank):
                revealed.add(rank)

    reveal_ready_rows()
    while any(value for matrix in (p0, p1, truth) for row in matrix for value in row):
        if predicted:
            planning_p2, truncated_forecast_edges = _bounded_planning_p2(
                p0=p0,
                p1=p1,
                truth=truth,
                forecast=forecast_rows,
                revealed=revealed,
            )
        else:
            planning_p2 = [
                list(truth[source]) if source in revealed else [0 for _ in range(len(p0))]
                for source in range(len(p0))
            ]
            truncated_forecast_edges = 0
        total_truncated_forecast_edges += int(truncated_forecast_edges)
        started = time.perf_counter()
        result = solve_problem_exact_with_scope(
            _problem(_matrix(p0), _matrix(p1), _matrix(planning_p2), information_mode="p012_predicted" if predicted else "p01_reactive"),
            time_limit_ms=int(time_limit_ms),
            scope="joint",
        )
        planning_runtime_ms += (time.perf_counter() - started) * 1000.0
        replan_count += 1
        if str(result.get("solver_status")) != "optimal" or not bool(result.get("certified_optimal", False)):
            status = str(result.get("solver_status", "unknown"))
            certified = False
            break
        wave = _first_wave(result)
        if wave is None:
            status = "stalled"
            certified = False
            break
        wave_duration = float(wave.get("duration", 0.0))
        executed_flows: list[dict[str, Any]] = []
        for flow in wave.get("flows", ()):
            phase = str(flow["phase"])
            source = int(flow["src_rank"])
            destination = int(flow["dst_rank"])
            if phase == "p0_dispatch":
                matrix = p0
                phase_index = 0
            elif phase == "p1_return":
                matrix = p1
                phase_index = 1
            elif phase == "p2_next_dispatch":
                if source not in revealed:
                    status = "forecast_flow_became_executable"
                    certified = False
                    break
                matrix = truth
                phase_index = 2
            else:
                status = f"unknown_phase:{phase}"
                certified = False
                break
            actual = int(matrix[source][destination])
            if actual <= 0:
                status = "planned_edge_missing_from_actual_state"
                certified = False
                break
            matrix[source][destination] = 0
            executed_flows.append({
                "flow_id": str(flow["flow_id"]),
                "phase": phase,
                "phase_index": phase_index,
                "src_rank": source,
                "dst_rank": destination,
                "byte_count": actual,
            })
        if not certified:
            break
        schedule.append({
            "wave_id": len(schedule),
            "start": current_time,
            "end": current_time + wave_duration,
            "duration": wave_duration,
            "flows": executed_flows,
        })
        current_time += wave_duration
        reveal_ready_rows()
        if len(schedule) > 256:
            status = "max_wave_limit_exceeded"
            certified = False
            break

    all_served = not any(value for matrix in (p0, p1, truth) for row in matrix for value in row)
    valid = certified and all_served and len(revealed) == len(p0)
    level: InformationLevel = "exact_joint_p012_predicted" if predicted else "exact_joint_p01_reactive"
    return ExactInformationResult(
        information_level=level,
        objective=current_time if valid else None,
        solver_status=status,
        certified_optimal_steps=certified,
        planning_runtime_ms=planning_runtime_ms,
        replan_count=replan_count,
        revealed_rows=len(revealed),
        schedule=tuple(schedule),
        audit={
            "valid": valid,
            "all_actual_traffic_served": all_served,
            "all_p2_rows_revealed": len(revealed) == len(p0),
            "predicted_bytes_executed": False,
            "forecast_advisory_edges_truncated": int(total_truncated_forecast_edges),
            "rolling_policy": "replan_exact_execute_first_wave_v1",
            "note": "Each replan is exact for the information available at that decision; the reactive policy is not clairvoyant-global optimal.",
        },
    )


def _perfect_exact(reduction: ExactReduction, *, time_limit_ms: int) -> ExactInformationResult:
    started = time.perf_counter()
    result = solve_problem_exact_with_scope(
        _problem(reduction.p0, reduction.p1, reduction.p2, information_mode="p012_perfect"),
        time_limit_ms=int(time_limit_ms),
        scope="joint",
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    valid = str(result.get("solver_status")) == "optimal" and bool(result.get("certified_optimal", False))
    return ExactInformationResult(
        information_level="oracle_joint_p012_perfect",
        objective=(None if result.get("objective_logical_makespan") is None else float(result["objective_logical_makespan"])),
        solver_status=str(result.get("solver_status", "unknown")),
        certified_optimal_steps=bool(result.get("certified_optimal", False)),
        planning_runtime_ms=elapsed,
        replan_count=1,
        revealed_rows=reduction.rank_count,
        schedule=tuple(result.get("schedule", ())),
        audit={
            "valid": valid,
            "clairvoyant": True,
            "scope": "joint",
            "reference_model": result.get("reference_model"),
            "search_nodes": result.get("search_nodes"),
        },
    )


def _safe_gain(base: float | None, value: float | None) -> float | None:
    if base is None or value is None or base <= 0.0:
        return None
    return (base - value) / base


def evaluate_exact_information_ladder(
    reduction: ExactReduction,
    *,
    p2_forecast: Matrix | None = None,
    time_limit_ms: int = 5000,
) -> ExactInformationLadder:
    reactive = _rolling_exact(reduction, forecast=None, time_limit_ms=time_limit_ms)
    predicted = None if p2_forecast is None else _rolling_exact(reduction, forecast=_matrix(p2_forecast), time_limit_ms=time_limit_ms)
    perfect = _perfect_exact(reduction, time_limit_ms=time_limit_ms)
    reactive_value = reactive.objective
    perfect_value = perfect.objective
    predicted_value = None if predicted is None else predicted.objective
    perfect_gain = _safe_gain(reactive_value, perfect_value)
    predicted_gain = _safe_gain(reactive_value, predicted_value)
    capture = None
    if perfect_gain is not None and predicted_gain is not None and perfect_gain > 1e-12:
        capture = predicted_gain / perfect_gain
    regret = None
    if predicted_value is not None and perfect_value is not None and perfect_value > 0.0:
        regret = (predicted_value - perfect_value) / perfect_value
    forecast_quality = (
        {} if p2_forecast is None
        else evaluate_traffic_forecast(p2_forecast, reduction.p2)
    )
    metrics = {
        "perfect_p2_information_value": perfect_gain,
        "predicted_p2_information_value": predicted_gain,
        "prediction_capture_ratio": capture,
        "prediction_regret_to_perfect": regret,
        **{f"forecast_{key}": value for key, value in forecast_quality.items()},
    }
    return ExactInformationLadder(
        reactive=reactive,
        predicted=predicted,
        perfect=perfect,
        metrics=metrics,
        reduction=reduction,
    )


__all__ = [
    "ExactInformationLadder",
    "ExactInformationResult",
    "ExactReduction",
    "evaluate_exact_information_ladder",
    "reduce_traffic_instance_for_exact",
    "reduce_forecast_for_exact",
]
