from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.scheduling.unified_interface import PolicyOptions, build_policy, build_request_from_replay_window
from rs.scheduling.validation import stable_hash
from rs.scheduling import (
    FlowDemand,
    FlowWindow,
    ForecastPressure,
    GlobalReadySetOptions,
    LogicalTopology,
    MultiPhaseSchedulingProblem,
    ReleaseConstraint,
)
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_digest_remote, matrix_remote_bytes


Matrix = tuple[tuple[int, ...], ...]
MatrixUnit = Literal["rows"]


def _matrix(value: Any) -> Matrix:
    return canonicalize_remote_matrix(value)


def _flows_from_matrix(
    matrix: Matrix,
    *,
    phase: str,
    release_state: str,
    executable: bool,
) -> tuple[FlowDemand, ...]:
    flows: list[FlowDemand] = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, row_count in enumerate(row):
            if src_rank == dst_rank or int(row_count) <= 0:
                continue
            flows.append(
                FlowDemand(
                    flow_id=f"{phase}:{src_rank}->{dst_rank}",
                    phase=phase,
                    src_rank=int(src_rank),
                    dst_rank=int(dst_rank),
                    byte_count=int(row_count),
                    release_state=release_state,
                    is_executable=bool(executable),
                )
            )
    return tuple(flows)


def _prediction_confidence(hint_type: str, confidence: float) -> float:
    if hint_type == "zero_hint":
        return 0.0
    return float(confidence)


@dataclass(frozen=True)
class ReplayWindow:
    fixture_id: str
    window_id: str
    layer_id: int
    p0_truth_rows: Matrix
    p1_truth_rows: Matrix
    p2_truth_rows: Matrix
    matrix_unit: MatrixUnit
    group_size: int
    payload_row_bytes_by_phase: dict[str, int]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningHint:
    hint_type: str
    p2_hint_rows: Matrix
    confidence: float
    source_layer: int | None
    target_layer: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningProblem:
    replay_window: ReplayWindow
    planning_hint: PlanningHint


@dataclass(frozen=True)
class ExecutionTruth:
    p0_truth_rows: Matrix
    p1_truth_rows: Matrix
    p2_truth_rows: Matrix


@dataclass(frozen=True)
class CanonicalBucketTask:
    phase: str
    src_group_rank: int
    dst_group_rank: int
    row_offset: int
    row_count: int
    task_id: str
    release_dependency: str

    def to_tuple(self) -> tuple[Any, ...]:
        return (
            str(self.phase),
            int(self.src_group_rank),
            int(self.dst_group_rank),
            int(self.row_offset),
            int(self.row_count),
            str(self.task_id),
            str(self.release_dependency),
        )


class CanonicalBucketizer:
    def __init__(self, *, bucket_rows: int) -> None:
        self.bucket_rows = int(bucket_rows)

    def bucketize(self, replay_window: ReplayWindow) -> tuple[CanonicalBucketTask, ...]:
        tasks: list[CanonicalBucketTask] = []
        for phase, matrix, dependency in (
            ("P0", replay_window.p0_truth_rows, "none"),
            ("P1", replay_window.p1_truth_rows, "after_p0"),
            ("P2", replay_window.p2_truth_rows, "after_p1"),
        ):
            for src_rank, row in enumerate(matrix):
                for dst_rank, value in enumerate(row):
                    row_count = int(value)
                    if src_rank == dst_rank or row_count <= 0:
                        continue
                    step = row_count if self.bucket_rows <= 0 else self.bucket_rows
                    offset = 0
                    bucket_ordinal = 0
                    while offset < row_count:
                        current = min(step, row_count - offset)
                        tasks.append(
                            CanonicalBucketTask(
                                phase=phase,
                                src_group_rank=int(src_rank),
                                dst_group_rank=int(dst_rank),
                                row_offset=int(offset),
                                row_count=int(current),
                                task_id=f"{phase}:{src_rank}->{dst_rank}:bucket:{bucket_ordinal}",
                                release_dependency=dependency,
                            )
                        )
                        offset += current
                        bucket_ordinal += 1
        return tuple(tasks)

    @staticmethod
    def digest(tasks: tuple[CanonicalBucketTask, ...]) -> str:
        digest = hashlib.sha256()
        for task in tasks:
            digest.update(repr(task.to_tuple()).encode("utf-8"))
        return digest.hexdigest()


def build_planning_problem(*, replay_window: ReplayWindow, planning_hint: PlanningHint) -> PlanningProblem:
    if replay_window.matrix_unit != "rows":
        raise ValueError(f"ReplayWindow currently supports rows only, got {replay_window.matrix_unit!r}")
    if planning_hint.target_layer != int(replay_window.layer_id) + 1:
        raise ValueError(
            f"planning hint target_layer={planning_hint.target_layer} does not match replay window next layer {int(replay_window.layer_id) + 1}"
        )
    return PlanningProblem(replay_window=replay_window, planning_hint=planning_hint)


def build_execution_truth(replay_window: ReplayWindow) -> ExecutionTruth:
    return ExecutionTruth(
        p0_truth_rows=replay_window.p0_truth_rows,
        p1_truth_rows=replay_window.p1_truth_rows,
        p2_truth_rows=replay_window.p2_truth_rows,
    )


def build_multiphase_problem(
    *,
    planning_problem: PlanningProblem,
    execution_truth: ExecutionTruth,
    scheduling_mode: str,
    expert_compute_delay: float,
) -> MultiPhaseSchedulingProblem:
    replay_window = planning_problem.replay_window
    hint = planning_problem.planning_hint
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows_from_matrix(
                replay_window.p0_truth_rows,
                phase="p0_dispatch",
                release_state="ready",
                executable=True,
            ),
            blocked_flows=_flows_from_matrix(
                replay_window.p1_truth_rows,
                phase="p1_return",
                release_state="blocked",
                executable=False,
            ),
            forecast_pressure=_flows_from_matrix(
                hint.p2_hint_rows,
                phase="p2_next_dispatch_forecast",
                release_state="advisory_only",
                executable=False,
            ),
        ),
        topology=LogicalTopology(num_gpus=int(replay_window.group_size)),
        release_model=ReleaseConstraint(
            phase="p1_return",
            rank=0,
            release_after_phase="p0_dispatch",
            expert_compute_delay=float(expert_compute_delay),
        ),
        forecast=ForecastPressure(
            source=str(hint.hint_type),
            digest=matrix_digest_remote(hint.p2_hint_rows),
            oracle=bool(hint.hint_type == "perfect_trace_hint"),
            evaluation_eligible=bool(hint.hint_type != "shuffled_control"),
            matrix_shape=(len(hint.p2_hint_rows), len(hint.p2_hint_rows[0]) if hint.p2_hint_rows else 0),
            matrix_total_bytes=int(matrix_remote_bytes(hint.p2_hint_rows)),
            matrix=hint.p2_hint_rows,
            metadata={
                "planning_hint_matrix": [list(row) for row in hint.p2_hint_rows],
                "planning_hint_digest": matrix_digest_remote(hint.p2_hint_rows),
                "execution_truth_digest": matrix_digest_remote(execution_truth.p2_truth_rows),
                "replay_window_id": replay_window.window_id,
                "replay_matrix_unit": replay_window.matrix_unit,
            },
        ),
        options=GlobalReadySetOptions(
            scheduling_mode=str(scheduling_mode),
            information_mode="p0_p1_p2",
            prediction_confidence=_prediction_confidence(hint.hint_type, hint.confidence),
            max_waves=256,
        ),
        p0_dispatch_matrix=replay_window.p0_truth_rows,
        p1_return_matrix=replay_window.p1_truth_rows,
        p2_next_dispatch_forecast_matrix=execution_truth.p2_truth_rows,
    )


class ReplayEngine:
    def __init__(self, *, scheduling_mode: str, expert_compute_delay: float, bucket_rows: int) -> None:
        self.scheduling_mode = str(scheduling_mode)
        self.expert_compute_delay = float(expert_compute_delay)
        self.bucket_rows = int(bucket_rows)

    def execute(
        self,
        *,
        replay_window: ReplayWindow,
        planning_hint: PlanningHint,
        policy_name: str,
    ) -> dict[str, Any]:
        planning_problem = build_planning_problem(replay_window=replay_window, planning_hint=planning_hint)
        execution_truth = build_execution_truth(replay_window)
        problem = build_multiphase_problem(
            planning_problem=planning_problem,
            execution_truth=execution_truth,
            scheduling_mode=self.scheduling_mode,
            expert_compute_delay=self.expert_compute_delay,
        )
        request = build_request_from_replay_window(
            replay_window=replay_window,
            p2_hint_rows=planning_hint.p2_hint_rows,
            hint_type=str(planning_hint.hint_type),
            confidence=float(planning_hint.confidence),
            bucket_rows=int(self.bucket_rows),
            policy_options=PolicyOptions(
                p0_weight=1.0,
                p1_weight=1.0,
                p2_hint_weight=float(planning_hint.confidence),
            ),
        )
        policy = build_policy(str(policy_name), request.policy_options)
        logical_plan = policy.plan(request)
        audit = replay_and_audit_logical_plan(problem, logical_plan)
        canonical_tasks = CanonicalBucketizer(bucket_rows=self.bucket_rows).bucketize(replay_window)
        return {
            "policy_name": str(policy_name),
            "bucket_rows": int(self.bucket_rows),
            "planning_hint": planning_hint.to_dict(),
            "replay_window": replay_window.to_dict(),
            "input_task_count": len(canonical_tasks),
            "input_total_rows": int(sum(task.row_count for task in canonical_tasks)),
            "input_task_digest": CanonicalBucketizer.digest(canonical_tasks),
            "logical_plan_policy_name": str(logical_plan.policy_name),
            "logical_plan_digest": stable_hash(logical_plan.to_dict()),
            "makespan": float(audit.get("replay_makespan", audit.get("makespan", 0.0)) or 0.0),
            "audit_valid": bool(audit.get("valid", False)),
            "audit": audit,
        }


__all__ = [
    "CanonicalBucketTask",
    "CanonicalBucketizer",
    "ExecutionTruth",
    "Matrix",
    "PlanningHint",
    "PlanningProblem",
    "ReplayEngine",
    "ReplayWindow",
    "build_execution_truth",
    "build_multiphase_problem",
    "build_planning_problem",
]
