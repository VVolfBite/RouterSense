from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from rs.core.hashing import stable_hash_dict

from .planning import PlanningRequest, WindowPlan
from .prediction import MatrixRows


class TrafficProvenance(str, Enum):
    REAL_EP_OBSERVED = "REAL_EP_OBSERVED"
    ROUTE_RECONSTRUCTED = "ROUTE_RECONSTRUCTED"
    SYNTHETIC_SOURCE_MAPPING = "SYNTHETIC_SOURCE_MAPPING"
    GENERATED_SYNTHETIC = "GENERATED_SYNTHETIC"


def _matrix_payload(matrix: MatrixRows) -> list[list[int]]:
    return [list(int(value) for value in row) for row in matrix]


def _validate_matrix(name: str, matrix: MatrixRows, *, world_size: int | None = None) -> None:
    if world_size is not None and int(world_size) <= 0:
        raise ValueError("world_size must be > 0")
    if world_size is not None and len(matrix) != int(world_size):
        raise ValueError(f"{name} row count {len(matrix)} does not match world_size {world_size}")
    widths = {len(row) for row in matrix}
    if world_size is not None:
        if widths != {int(world_size)}:
            raise ValueError(f"{name} column widths {sorted(widths)} do not match world_size {world_size}")
    elif len(widths) > 1:
        raise ValueError(f"{name} has ragged row widths {sorted(widths)}")
    for row in matrix:
        for value in row:
            if int(value) < 0:
                raise ValueError(f"{name} values must be non-negative")


@dataclass(frozen=True)
class OfflineWindow:
    window_identity: str
    source_layer: str
    target_layer: str
    p0_actual: MatrixRows
    p1_actual: MatrixRows
    p2_actual: MatrixRows
    placement_snapshot: Mapping[str, object]
    traffic_provenance: TrafficProvenance
    matrix_unit: str
    return_model: str
    raw_token_count: int
    used_token_count: int
    dropped_token_count: int
    drop_reason: str | None
    trace_digest: str

    def validate(self) -> None:
        if not str(self.window_identity):
            raise ValueError("window_identity must be non-empty")
        if not str(self.source_layer):
            raise ValueError("source_layer must be non-empty")
        if not str(self.target_layer):
            raise ValueError("target_layer must be non-empty")
        world_size = len(self.p0_actual)
        _validate_matrix("p0_actual", self.p0_actual, world_size=world_size)
        _validate_matrix("p1_actual", self.p1_actual, world_size=world_size)
        _validate_matrix("p2_actual", self.p2_actual, world_size=world_size)
        if str(self.matrix_unit) != "rows":
            raise ValueError("offline window currently supports matrix_unit='rows' only")
        if not str(self.return_model):
            raise ValueError("return_model must be non-empty")
        if int(self.raw_token_count) < 0 or int(self.used_token_count) < 0 or int(self.dropped_token_count) < 0:
            raise ValueError("token counts must be >= 0")
        if int(self.used_token_count) + int(self.dropped_token_count) > int(self.raw_token_count):
            raise ValueError("used_token_count + dropped_token_count must not exceed raw_token_count")
        if self.drop_reason is not None and not str(self.drop_reason):
            raise ValueError("drop_reason must be non-empty when provided")
        if not str(self.trace_digest):
            raise ValueError("trace_digest must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "window_identity": str(self.window_identity),
            "source_layer": str(self.source_layer),
            "target_layer": str(self.target_layer),
            "p0_actual": _matrix_payload(self.p0_actual),
            "p1_actual": _matrix_payload(self.p1_actual),
            "p2_actual": _matrix_payload(self.p2_actual),
            "placement_snapshot": dict(self.placement_snapshot),
            "traffic_provenance": str(self.traffic_provenance.value),
            "matrix_unit": str(self.matrix_unit),
            "return_model": str(self.return_model),
            "raw_token_count": int(self.raw_token_count),
            "used_token_count": int(self.used_token_count),
            "dropped_token_count": int(self.dropped_token_count),
            "drop_reason": self.drop_reason,
            "trace_digest": str(self.trace_digest),
        }


@dataclass(frozen=True)
class EvaluationSpec:
    track: str
    world_size: int
    task_granularity: str
    matrix_unit: str
    time_unit: str
    cost_model_id: str
    release_model: str
    return_model: str
    full_duplex: bool
    launch_cost: float
    bytes_per_row: int
    bandwidth: float
    compute_delay: float
    p2_semantics: str
    residual_policy: str
    schema_version: str = "offline_eval_v1"

    def validate(self) -> None:
        if str(self.track) not in {"runtime_lookahead", "execution_window"}:
            raise ValueError(f"unsupported track {self.track!r}")
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be > 0")
        if str(self.task_granularity) not in {"matrix_cell"}:
            raise ValueError("task_granularity must be 'matrix_cell'")
        if str(self.matrix_unit) != "rows":
            raise ValueError("matrix_unit must be 'rows'")
        if not str(self.time_unit):
            raise ValueError("time_unit must be non-empty")
        if not str(self.cost_model_id):
            raise ValueError("cost_model_id must be non-empty")
        if not str(self.release_model):
            raise ValueError("release_model must be non-empty")
        if not str(self.return_model):
            raise ValueError("return_model must be non-empty")
        if float(self.launch_cost) < 0.0:
            raise ValueError("launch_cost must be >= 0")
        if int(self.bytes_per_row) <= 0:
            raise ValueError("bytes_per_row must be > 0")
        if float(self.bandwidth) <= 0.0:
            raise ValueError("bandwidth must be > 0")
        if float(self.compute_delay) < 0.0:
            raise ValueError("compute_delay must be >= 0")
        if not str(self.p2_semantics):
            raise ValueError("p2_semantics must be non-empty")
        if not str(self.residual_policy):
            raise ValueError("residual_policy must be non-empty")
        if not str(self.schema_version):
            raise ValueError("schema_version must be non-empty")

    def semantic_digest(self) -> str:
        return stable_hash_dict(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "track": str(self.track),
            "world_size": int(self.world_size),
            "task_granularity": str(self.task_granularity),
            "matrix_unit": str(self.matrix_unit),
            "time_unit": str(self.time_unit),
            "cost_model_id": str(self.cost_model_id),
            "release_model": str(self.release_model),
            "return_model": str(self.return_model),
            "full_duplex": bool(self.full_duplex),
            "launch_cost": float(self.launch_cost),
            "bytes_per_row": int(self.bytes_per_row),
            "bandwidth": float(self.bandwidth),
            "compute_delay": float(self.compute_delay),
            "p2_semantics": str(self.p2_semantics),
            "residual_policy": str(self.residual_policy),
            "schema_version": str(self.schema_version),
        }


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    phase: str
    src_rank: int
    dst_rank: int
    row_count: int
    byte_count: int
    release_dependencies: tuple[str, ...]
    release_time: float
    provenance: str

    def validate(self, *, world_size: int) -> None:
        if not str(self.task_id):
            raise ValueError("task_id must be non-empty")
        if str(self.phase) not in {"p0_dispatch", "p1_return", "p2_next_dispatch"}:
            raise ValueError(f"unsupported task phase {self.phase!r}")
        if int(self.src_rank) < 0 or int(self.src_rank) >= int(world_size):
            raise ValueError("src_rank outside world_size")
        if int(self.dst_rank) < 0 or int(self.dst_rank) >= int(world_size):
            raise ValueError("dst_rank outside world_size")
        if int(self.row_count) < 0 or int(self.byte_count) < 0:
            raise ValueError("row_count and byte_count must be >= 0")
        if float(self.release_time) < 0.0:
            raise ValueError("release_time must be >= 0")
        if not str(self.provenance):
            raise ValueError("provenance must be non-empty")
        for dependency in self.release_dependencies:
            if not str(dependency):
                raise ValueError("release_dependencies must not contain empty values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "phase": str(self.phase),
            "src_rank": int(self.src_rank),
            "dst_rank": int(self.dst_rank),
            "row_count": int(self.row_count),
            "byte_count": int(self.byte_count),
            "release_dependencies": list(self.release_dependencies),
            "release_time": float(self.release_time),
            "provenance": str(self.provenance),
        }


@dataclass(frozen=True)
class EvaluationTaskSet:
    task_set_digest: str
    tasks: tuple[EvaluationTask, ...]
    p0_tasks: tuple[str, ...]
    p1_tasks: tuple[str, ...]
    p2_tasks: tuple[str, ...]
    world_size: int
    task_granularity: str
    coverage_summary: Mapping[str, object]

    def validate(self) -> None:
        if not str(self.task_set_digest):
            raise ValueError("task_set_digest must be non-empty")
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be > 0")
        if str(self.task_granularity) not in {"matrix_cell"}:
            raise ValueError("task_granularity must be 'matrix_cell'")
        task_ids = set()
        by_phase = {"p0_dispatch": set(), "p1_return": set(), "p2_next_dispatch": set()}
        for task in self.tasks:
            task.validate(world_size=int(self.world_size))
            if task.task_id in task_ids:
                raise ValueError(f"duplicate task_id {task.task_id!r}")
            task_ids.add(task.task_id)
            by_phase[str(task.phase)].add(str(task.task_id))
        if by_phase["p0_dispatch"] != set(self.p0_tasks):
            raise ValueError("p0_tasks do not match tasks")
        if by_phase["p1_return"] != set(self.p1_tasks):
            raise ValueError("p1_tasks do not match tasks")
        if by_phase["p2_next_dispatch"] != set(self.p2_tasks):
            raise ValueError("p2_tasks do not match tasks")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "task_set_digest": str(self.task_set_digest),
            "tasks": [task.to_dict() for task in self.tasks],
            "p0_tasks": list(self.p0_tasks),
            "p1_tasks": list(self.p1_tasks),
            "p2_tasks": list(self.p2_tasks),
            "world_size": int(self.world_size),
            "task_granularity": str(self.task_granularity),
            "coverage_summary": dict(self.coverage_summary),
        }


@dataclass(frozen=True)
class ExecutionTruth:
    task_set: EvaluationTaskSet
    actual_matrices: Mapping[str, MatrixRows]
    actual_release_dependencies: Mapping[str, tuple[str, ...]]
    truth_digest: str
    provenance: TrafficProvenance

    def validate(self) -> None:
        self.task_set.validate()
        if not str(self.truth_digest):
            raise ValueError("truth_digest must be non-empty")
        normalized = {str(key): value for key, value in self.actual_matrices.items()}
        for phase_name in ("p0_dispatch", "p1_return", "p2_next_dispatch"):
            if phase_name not in normalized:
                raise ValueError(f"actual_matrices missing {phase_name}")
            _validate_matrix(phase_name, normalized[phase_name], world_size=int(self.task_set.world_size))
        task_ids = {task.task_id for task in self.task_set.tasks}
        for task_id, dependencies in self.actual_release_dependencies.items():
            if task_id not in task_ids:
                raise ValueError(f"release dependency references unknown task {task_id!r}")
            for dependency in dependencies:
                if dependency not in task_ids:
                    raise ValueError(f"release dependency {dependency!r} not found in task set")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "task_set": self.task_set.to_dict(),
            "actual_matrices": {key: _matrix_payload(value) for key, value in self.actual_matrices.items()},
            "actual_release_dependencies": {key: list(value) for key, value in self.actual_release_dependencies.items()},
            "truth_digest": str(self.truth_digest),
            "provenance": str(self.provenance.value),
        }


@dataclass(frozen=True)
class PlanEvaluation:
    valid: bool
    reason: str | None
    realized_makespan: float | None
    completed_tasks: tuple[str, ...] = ()
    unresolved_tasks: tuple[str, ...] = ()
    dependency_violations: tuple[str, ...] = ()
    coverage_valid: bool = False
    port_valid: bool = False
    metrics: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "reason": self.reason,
            "realized_makespan": self.realized_makespan,
            "completed_tasks": list(self.completed_tasks),
            "unresolved_tasks": list(self.unresolved_tasks),
            "dependency_violations": list(self.dependency_violations),
            "coverage_valid": bool(self.coverage_valid),
            "port_valid": bool(self.port_valid),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class OfflineEvaluationRecord:
    window_identity: str
    evaluation_spec_digest: str
    task_set_digest: str
    planning_request_digest: str
    prediction_digest: str
    logical_plan_digest: str
    execution_truth_digest: str
    planner_id: str
    planner_family: str
    predictor_id: str
    track: str
    realized_makespan: float | None
    planner_reported_makespan: float | None
    audit_status: str
    coverage_status: str
    fallback_status: str
    oracle_status: str
    eligibility: Mapping[str, object]
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class OfflineEvaluationBundle:
    schema_version: str
    evaluation_spec: EvaluationSpec
    records: tuple[OfflineEvaluationRecord, ...]
    oracle_records: tuple[Mapping[str, object], ...] = ()
    prediction_records: tuple[Mapping[str, object], ...] = ()
    parity_records: tuple[Mapping[str, object], ...] = ()
    paired_aggregates: tuple[Mapping[str, object], ...] = ()
    eligibility_summary: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.schema_version):
            raise ValueError("schema_version must be non-empty")
        self.evaluation_spec.validate()


__all__ = [
    "EvaluationSpec",
    "EvaluationTask",
    "EvaluationTaskSet",
    "ExecutionTruth",
    "OfflineEvaluationBundle",
    "OfflineEvaluationRecord",
    "OfflineWindow",
    "PlanEvaluation",
    "PlanningRequest",
    "TrafficProvenance",
    "WindowPlan",
]
