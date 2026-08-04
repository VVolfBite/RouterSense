from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rs.core.contracts import (
    EvaluationSpec,
    EvaluationTask,
    EvaluationTaskSet,
    ExecutionTruth,
    OfflineWindow,
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
    PredictionResult,
    TrafficProvenance,
)
from rs.planning.request_builder import build_window_planning_request
from rs.core.hashing import stable_hash_dict


def _task_phase_name(phase_name: str) -> str:
    if phase_name in {"p2_next_dispatch", "p2_next_dispatch_forecast"}:
        return "p2_next_dispatch"
    return phase_name


def _matrix_payload(matrix: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [list(int(value) for value in row) for row in matrix]


def _tasks_from_matrix(
    matrix: tuple[tuple[int, ...], ...],
    *,
    phase: str,
    bytes_per_row: int,
    provenance: TrafficProvenance,
) -> tuple[EvaluationTask, ...]:
    tasks: list[EvaluationTask] = []
    canonical_phase = _task_phase_name(phase)
    for src_rank, row in enumerate(matrix):
        for dst_rank, row_count in enumerate(row):
            if src_rank == dst_rank or int(row_count) <= 0:
                continue
            task_id = f"{canonical_phase}:{src_rank}->{dst_rank}"
            tasks.append(
                EvaluationTask(
                    task_id=task_id,
                    phase=canonical_phase,
                    src_rank=int(src_rank),
                    dst_rank=int(dst_rank),
                    row_count=int(row_count),
                    byte_count=int(row_count) * int(bytes_per_row),
                    release_dependencies=(),
                    release_time=0.0,
                    provenance=str(provenance.value),
                )
            )
    return tuple(tasks)


def _release_dependencies(
    p0_tasks: tuple[EvaluationTask, ...],
    p1_tasks: tuple[EvaluationTask, ...],
    p2_tasks: tuple[EvaluationTask, ...],
) -> dict[str, tuple[str, ...]]:
    p0_inbound_by_rank: dict[int, list[str]] = {}
    p1_inbound_by_rank: dict[int, list[str]] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    for task in p0_tasks:
        p0_inbound_by_rank.setdefault(int(task.dst_rank), []).append(str(task.task_id))
        dependencies[str(task.task_id)] = ()
    for task in p1_tasks:
        deps = tuple(sorted(p0_inbound_by_rank.get(int(task.src_rank), ())))
        dependencies[str(task.task_id)] = deps
        p1_inbound_by_rank.setdefault(int(task.dst_rank), []).append(str(task.task_id))
    for task in p2_tasks:
        dependencies[str(task.task_id)] = tuple(sorted(p1_inbound_by_rank.get(int(task.src_rank), ())))
    return dependencies


def build_evaluation_task_set(window: OfflineWindow, spec: EvaluationSpec) -> EvaluationTaskSet:
    window.validate()
    spec.validate()
    if len(window.p0_actual) != int(spec.world_size):
        raise ValueError("offline window world size does not match evaluation spec world_size")
    if str(window.return_model) != str(spec.return_model):
        raise ValueError("offline window return_model does not match evaluation spec return_model")
    p0_tasks = _tasks_from_matrix(
        window.p0_actual,
        phase="p0_dispatch",
        bytes_per_row=int(spec.bytes_per_row),
        provenance=window.traffic_provenance,
    )
    p1_tasks = _tasks_from_matrix(
        window.p1_actual,
        phase="p1_return",
        bytes_per_row=int(spec.bytes_per_row),
        provenance=window.traffic_provenance,
    )
    p2_tasks = _tasks_from_matrix(
        window.p2_actual,
        phase="p2_next_dispatch",
        bytes_per_row=int(spec.bytes_per_row),
        provenance=window.traffic_provenance,
    )
    dependencies = _release_dependencies(p0_tasks, p1_tasks, p2_tasks)
    tasks = tuple(
        EvaluationTask(
            task_id=str(task.task_id),
            phase=str(task.phase),
            src_rank=int(task.src_rank),
            dst_rank=int(task.dst_rank),
            row_count=int(task.row_count),
            byte_count=int(task.byte_count),
            release_dependencies=tuple(dependencies.get(str(task.task_id), ())),
            release_time=0.0 if task.phase == "p0_dispatch" else float(spec.compute_delay if task.phase == "p1_return" else 0.0),
            provenance=str(task.provenance),
        )
        for task in (*p0_tasks, *p1_tasks, *p2_tasks)
    )
    task_set = EvaluationTaskSet(
        task_set_digest="pending",
        tasks=tasks,
        p0_tasks=tuple(task.task_id for task in tasks if task.phase == "p0_dispatch"),
        p1_tasks=tuple(task.task_id for task in tasks if task.phase == "p1_return"),
        p2_tasks=tuple(task.task_id for task in tasks if task.phase == "p2_next_dispatch"),
        world_size=int(spec.world_size),
        task_granularity=str(spec.task_granularity),
        coverage_summary={
            "p0_task_count": len(p0_tasks),
            "p1_task_count": len(p1_tasks),
            "p2_task_count": len(p2_tasks),
            "total_row_count": int(sum(task.row_count for task in tasks)),
        },
    )
    task_set = EvaluationTaskSet(
        task_set_digest=task_set.recompute_digest(),
        tasks=task_set.tasks,
        p0_tasks=task_set.p0_tasks,
        p1_tasks=task_set.p1_tasks,
        p2_tasks=task_set.p2_tasks,
        world_size=task_set.world_size,
        task_granularity=task_set.task_granularity,
        coverage_summary=task_set.coverage_summary,
    )
    task_set.validate()
    return task_set


def build_execution_truth(window: OfflineWindow, spec: EvaluationSpec) -> ExecutionTruth:
    task_set = build_evaluation_task_set(window, spec)
    dependencies = {task.task_id: task.release_dependencies for task in task_set.tasks}
    truth = ExecutionTruth(
        task_set=task_set,
        actual_matrices={
            "p0_dispatch": window.p0_actual,
            "p1_return": window.p1_actual,
            "p2_next_dispatch": window.p2_actual,
        },
        actual_release_dependencies=dependencies,
        truth_digest="pending",
        provenance=window.traffic_provenance,
    )
    truth = ExecutionTruth(
        task_set=truth.task_set,
        actual_matrices=truth.actual_matrices,
        actual_release_dependencies=truth.actual_release_dependencies,
        truth_digest=truth.recompute_digest(),
        provenance=truth.provenance,
    )
    truth.validate()
    return truth


@dataclass(frozen=True)
class OfflinePlanningRequestBuilder:
    bucket_rows: int = 0
    max_waves: int = 256
    information_mode: str = "p0_p1_p2"

    def build(
        self,
        window: OfflineWindow,
        prediction: PredictionResult,
        spec: EvaluationSpec,
    ) -> PlanningRequest:
        window.validate()
        spec.validate()
        if len(window.p0_actual) != int(spec.world_size):
            raise ValueError("window world size does not match spec.world_size")
        if len(prediction.hint.target_dispatch_rows) != int(spec.world_size):
            raise ValueError("prediction world size does not match spec.world_size")
        prediction.validate(world_size=int(spec.world_size))
        if str(window.source_layer) != str(prediction.identity.source_layer_id):
            raise ValueError("prediction source_layer_id does not match offline window")
        if str(window.target_layer) != str(prediction.identity.target_layer_id):
            raise ValueError("prediction target_layer_id does not match offline window")
        if str(window.return_model) != str(spec.return_model):
            raise ValueError("window return_model does not match evaluation spec")
        hint = prediction.hint
        if tuple(tuple(int(value) for value in row) for row in hint.target_dispatch_rows) == ():
            raise ValueError("prediction hint must not be empty")
        return build_window_planning_request(
            identity=PlanningIdentity(
                request_id=str(window.window_identity),
                run_id=str(window.trace_digest),
                forward_id="offline",
                window_id=str(window.window_identity),
                source_layer_id=str(window.source_layer),
                target_layer_id=str(window.target_layer),
            ),
            p0_dispatch_rows=window.p0_actual,
            p1_return_rows=window.p1_actual,
            p2_hint_rows=hint.target_dispatch_rows,
            predictor_id=str(hint.predictor_id),
            confidence=float(hint.confidence),
            topology=PlanningTopology(world_size=int(spec.world_size), full_duplex=bool(spec.full_duplex)),
            constraints=PlanningConstraints(
                bucket_rows=int(self.bucket_rows),
                max_waves=int(self.max_waves),
                expert_compute_delay=float(spec.compute_delay),
                phase_release_model="p1_return",
            ),
            weights=PlanningWeights(),
            information_mode=str(self.information_mode),
            hint_type=str(hint.hint_type),
            oracle=bool(hint.oracle),
        )


def prediction_digest(prediction: PredictionResult) -> str:
    prediction.validate()
    return stable_hash_dict(
        {
            "prediction_result_version": "offline_prediction_v1",
            "identity": prediction.identity.to_dict(),
            "hint": prediction.hint.to_dict(),
            "expert_route": None if prediction.expert_route is None else prediction.expert_route.to_dict(),
            "auxiliary": dict(prediction.auxiliary),
        }
    )


__all__ = [
    "OfflinePlanningRequestBuilder",
    "build_evaluation_task_set",
    "build_execution_truth",
    "prediction_digest",
]
