from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint as FormalPredictionHint,
)
from rs.planning import PlannerRegistry
from rs.runtime.online.megatron_ep.target_planning.contracts import _compat_logical_plan_from_window_plan
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.scheduling.bucketizer import CanonicalBucketTask, CanonicalBucketizer
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
from rs.core.hashing import stable_hash_dict


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


def _planning_mode_contract(*, scheduling_mode: str, has_p2: bool) -> tuple[str, str]:
    mode = str(scheduling_mode)
    if mode == "execution_window":
        return ("execution_window", "executable_actual" if has_p2 else "absent")
    if mode == "runtime_lookahead":
        return ("runtime_lookahead", "advisory_hint" if has_p2 else "absent")
    raise ValueError(f"unsupported scheduling_mode {scheduling_mode!r}")


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
    max_waves: int = 256,
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
            max_waves=int(max_waves),
        ),
        p0_dispatch_matrix=replay_window.p0_truth_rows,
        p1_return_matrix=replay_window.p1_truth_rows,
        p2_next_dispatch_forecast_matrix=hint.p2_hint_rows,
    )


@dataclass(frozen=True)
class _PlanningBucketWindow:
    p0_truth_rows: Matrix
    p1_truth_rows: Matrix
    p2_truth_rows: Matrix


def bucketize_planning_request(request: PlanningRequest) -> tuple[CanonicalBucketTask, ...]:
    planning_window = _PlanningBucketWindow(
        p0_truth_rows=request.traffic.p0_dispatch_rows,
        p1_truth_rows=request.traffic.p1_return_rows,
        p2_truth_rows=request.prediction_hint.target_dispatch_rows,
    )
    return CanonicalBucketizer(bucket_rows=int(request.constraints.bucket_rows)).bucketize(planning_window)


def execution_truth_digest(execution_truth: ExecutionTruth) -> str:
    return stable_hash_dict(
        {
            "execution_truth_version": "v1",
            "p0_truth_rows": [list(row) for row in execution_truth.p0_truth_rows],
            "p1_truth_rows": [list(row) for row in execution_truth.p1_truth_rows],
            "p2_truth_rows": [list(row) for row in execution_truth.p2_truth_rows],
        }
    )


class ReplayEngine:
    def __init__(self, *, scheduling_mode: str, expert_compute_delay: float, bucket_rows: int, max_waves: int = 256) -> None:
        self.scheduling_mode = str(scheduling_mode)
        self.expert_compute_delay = float(expert_compute_delay)
        self.bucket_rows = int(bucket_rows)
        self.max_waves = int(max_waves)

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
            max_waves=self.max_waves,
        )
        has_p2 = any(int(value) > 0 for row in planning_hint.p2_hint_rows for value in row)
        planning_track, p2_semantics = _planning_mode_contract(
            scheduling_mode=self.scheduling_mode,
            has_p2=has_p2,
        )
        request = PlanningRequest(
            identity=PlanningIdentity(
                request_id=f"{replay_window.fixture_id}:{replay_window.window_id}:{policy_name}",
                run_id=str(replay_window.fixture_id),
                window_id=str(replay_window.window_id),
                source_layer_id=str(replay_window.layer_id),
                target_layer_id=str(planning_hint.target_layer),
            ),
            traffic=PlanningTraffic(
                p0_dispatch_rows=replay_window.p0_truth_rows,
                p1_return_rows=replay_window.p1_truth_rows,
            ),
            prediction_hint=FormalPredictionHint(
                predictor_id=str(planning_hint.hint_type),
                hint_type=str(planning_hint.hint_type),
                target_dispatch_rows=planning_hint.p2_hint_rows,
                confidence=float(planning_hint.confidence),
                oracle=bool(planning_hint.hint_type == "perfect_trace_hint"),
                source_layer_id=None if planning_hint.source_layer is None else str(planning_hint.source_layer),
                target_layer_id=str(planning_hint.target_layer),
            ),
            topology=PlanningTopology(world_size=int(replay_window.group_size)),
            constraints=PlanningConstraints(
                bucket_rows=int(self.bucket_rows),
                max_waves=int(self.max_waves),
                expert_compute_delay=float(self.expert_compute_delay),
                phase_release_model="p1_return",
            ),
            weights=PlanningWeights(
                p0_weight=1.0,
                p1_weight=1.0,
                p2_weight=float(planning_hint.confidence),
            ),
            information_mode="p0_p1_p2",
            planning_track=str(planning_track),
            p2_semantics=str(p2_semantics),
        )
        planner = PlannerRegistry.create(str(policy_name), None)
        formal_plan = planner.plan(request)
        logical_plan = _compat_logical_plan_from_window_plan(formal_plan)
        audit = replay_and_audit_logical_plan(problem, logical_plan)
        planning_tasks = bucketize_planning_request(request)
        truth_digest = execution_truth_digest(execution_truth)
        return {
            "policy_name": str(policy_name),
            "bucket_rows": int(self.bucket_rows),
            "max_waves": int(self.max_waves),
            "planning_hint": planning_hint.to_dict(),
            "replay_window": replay_window.to_dict(),
            "planning_task_count": len(planning_tasks),
            "planning_task_total_rows": int(sum(task.row_count for task in planning_tasks)),
            "planning_task_digest": CanonicalBucketizer.digest(planning_tasks),
            "execution_truth_digest": truth_digest,
            "input_task_count": len(planning_tasks),
            "input_total_rows": int(sum(task.row_count for task in planning_tasks)),
            "input_task_digest": CanonicalBucketizer.digest(planning_tasks),
            "input_task_digest_deprecated": True,
            "logical_plan_policy_name": str(logical_plan.policy_name),
            "logical_plan_digest": str(formal_plan.semantic_digest()),
            "logical_plan_audit_digest": str(formal_plan.audit_digest()),
            "planner_family": str(formal_plan.planner_family),
            "plan_metadata": dict(formal_plan.metadata),
            "planning_track": str(request.planning_track),
            "p2_semantics": str(request.p2_semantics),
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
    "bucketize_planning_request",
    "build_execution_truth",
    "build_multiphase_problem",
    "build_planning_problem",
    "execution_truth_digest",
]
