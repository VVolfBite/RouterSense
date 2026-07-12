from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rs.scheduling.bucketizer import CanonicalBucketTask, CanonicalBucketizer
from rs.scheduling.contracts import (
    FlowDemand,
    FlowWindow,
    ForecastPressure,
    GlobalReadySetOptions,
    LogicalSchedulePlan,
    LogicalTopology,
    MultiPhaseSchedulingProblem,
    ReleaseConstraint,
)
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_digest_remote, matrix_remote_bytes
from rs.scheduling.registry import resolve_policy


MatrixRows = tuple[tuple[int, ...], ...]


class ReplayWindowLike(Protocol):
    fixture_id: str
    window_id: str
    layer_id: int
    p0_truth_rows: MatrixRows
    p1_truth_rows: MatrixRows
    p2_truth_rows: MatrixRows
    group_size: int


@dataclass(frozen=True)
class _BucketizerReplayWindow:
    fixture_id: str
    window_id: str
    layer_id: int
    p0_truth_rows: MatrixRows
    p1_truth_rows: MatrixRows
    p2_truth_rows: MatrixRows
    group_size: int


@dataclass(frozen=True)
class SchedulingTopology:
    group_size: int


@dataclass(frozen=True)
class PlanningHintMetadata:
    hint_type: str
    confidence: float
    source_layer: int | None = None
    target_layer: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyOptions:
    p0_weight: float = 1.0
    p1_weight: float = 1.0
    p2_hint_weight: float = 1.0
    residual_weight: float = 0.75
    barrier_weight: float = 1.75
    age_weight: float = 0.15
    prediction_weight: float = 0.35
    criticality_weight: float = 0.0
    maxweight_enabled: bool = False
    pricing: dict[str, Any] | None = None
    iteration_budget: int | None = None
    debug_trace: bool = False


@dataclass(frozen=True)
class SchedulingRequest:
    request_id: str
    tasks: tuple["CanonicalBucketTask", ...]
    p0_truth_rows: MatrixRows
    p1_truth_rows: MatrixRows
    p2_hint_rows: MatrixRows
    topology: SchedulingTopology
    release_model: ReleaseConstraint
    policy_options: PolicyOptions
    hint_metadata: PlanningHintMetadata
    scheduling_mode: str = "runtime_lookahead"
    information_mode: str = "p0_p1_p2"
    max_waves: int = 256
    task_quantum_rows: int = 0
    fixture_id: str | None = None
    window_id: str | None = None
    layer_id: int | None = None


@dataclass(frozen=True)
class ReplayExecutionTruth:
    p0_truth_rows: MatrixRows
    p1_truth_rows: MatrixRows
    p2_truth_rows: MatrixRows


class SchedulingPolicy(Protocol):
    policy_id: str

    def plan(self, request: SchedulingRequest) -> LogicalSchedulePlan:
        ...


def _matrix(value: Any) -> MatrixRows:
    return canonicalize_remote_matrix(value)


def _flows_from_matrix(
    matrix: MatrixRows,
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


def request_to_legacy_problem(request: SchedulingRequest) -> MultiPhaseSchedulingProblem:
    confidence = 0.0 if request.hint_metadata.hint_type == "zero_hint" else float(request.hint_metadata.confidence)
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows_from_matrix(request.p0_truth_rows, phase="p0_dispatch", release_state="ready", executable=True),
            blocked_flows=_flows_from_matrix(request.p1_truth_rows, phase="p1_return", release_state="blocked", executable=False),
            forecast_pressure=_flows_from_matrix(
                request.p2_hint_rows,
                phase="p2_next_dispatch_forecast",
                release_state="advisory_only",
                executable=False,
            ),
        ),
        topology=LogicalTopology(num_gpus=int(request.topology.group_size)),
        release_model=request.release_model,
        forecast=ForecastPressure(
            source=str(request.hint_metadata.hint_type),
            digest=matrix_digest_remote(request.p2_hint_rows),
            oracle=bool(request.hint_metadata.hint_type == "perfect_trace_hint"),
            evaluation_eligible=bool(request.hint_metadata.hint_type != "shuffled_control"),
            matrix_shape=(len(request.p2_hint_rows), len(request.p2_hint_rows[0]) if request.p2_hint_rows else 0),
            matrix_total_bytes=int(matrix_remote_bytes(request.p2_hint_rows)),
            matrix=request.p2_hint_rows,
            metadata={
                "request_id": str(request.request_id),
                "fixture_id": request.fixture_id,
                "window_id": request.window_id,
                "layer_id": request.layer_id,
                **dict(request.hint_metadata.metadata),
            },
        ),
        options=GlobalReadySetOptions(
            scheduling_mode=str(request.scheduling_mode),
            information_mode=str(request.information_mode),
            prediction_confidence=float(confidence),
            p0_weight=float(request.policy_options.p0_weight),
            p1_reservation_weight=float(request.policy_options.p1_weight),
            p2_hint_weight=float(request.policy_options.p2_hint_weight),
            max_waves=(
                int(request.max_waves)
                if request.policy_options.iteration_budget is None
                else int(request.policy_options.iteration_budget)
            ),
        ),
        p0_dispatch_matrix=request.p0_truth_rows,
        p1_return_matrix=request.p1_truth_rows,
        p2_next_dispatch_forecast_matrix=request.p2_hint_rows,
    )


class LegacyLogicalPolicyAdapter:
    def __init__(self, *, canonical_policy_id: str, builder_key: str, options: PolicyOptions) -> None:
        self.policy_id = str(canonical_policy_id)
        self._builder_key = str(builder_key)
        self._options = options

    def plan(self, request: SchedulingRequest) -> LogicalSchedulePlan:
        legacy_problem = request_to_legacy_problem(request)
        bucket_rows = int(request.task_quantum_rows or 0)
        legacy_policy = resolve_policy(
            policy_name=self._builder_key,
            bucket_rows=bucket_rows,
            p0_weight=float(self._options.p0_weight),
            p1_reservation_weight=float(self._options.p1_weight),
            p2_hint_weight=float(self._options.p2_hint_weight),
            residual_weight=float(self._options.residual_weight),
            barrier_weight=float(self._options.barrier_weight),
            age_weight=float(self._options.age_weight),
            prediction_weight=float(self._options.prediction_weight),
        )
        plan = legacy_policy.build_logical_plan(legacy_problem)
        diagnostics = dict(getattr(plan, "diagnostics", {}) or {})
        diagnostics.setdefault("unified_policy_interface", True)
        diagnostics.setdefault("requested_policy_id", self.policy_id)
        diagnostics.setdefault("builder_key", self._builder_key)
        diagnostics.setdefault("request_id", str(request.request_id))
        diagnostics.setdefault("task_digest", CanonicalBucketizer.digest(request.tasks))
        return LogicalSchedulePlan(
            policy_name=str(plan.policy_name),
            waves=tuple(plan.waves),
            diagnostics=diagnostics,
        )


def build_policy(policy_id: str, options: PolicyOptions) -> SchedulingPolicy:
    from rs.scheduling.catalog import resolve_algorithm_id

    resolved = resolve_algorithm_id(policy_id)
    if str(resolved.canonical_name) == "barrier_criticality_posthoc_best":
        raise ValueError(
            "barrier_criticality_posthoc_best is a reference-only posthoc selector and does not map to a unified planning-time policy"
        )
    if str(resolved.canonical_name) == "oracle_local_cp_sat":
        raise ValueError(
            "oracle_local_cp_sat remains a reference-only reporting alias and does not map to a unified planning-time policy"
        )
    return LegacyLogicalPolicyAdapter(
        canonical_policy_id=str(resolved.canonical_name),
        builder_key=str(resolved.builder_key),
        options=options,
    )


def build_request_from_replay_window(
    *,
    replay_window: ReplayWindowLike,
    p2_hint_rows: MatrixRows,
    hint_type: str,
    confidence: float,
    bucket_rows: int,
    policy_options: PolicyOptions,
) -> SchedulingRequest:
    tasks = CanonicalBucketizer(bucket_rows=bucket_rows).bucketize(replay_window)
    return SchedulingRequest(
        request_id=f"{replay_window.fixture_id}:{replay_window.window_id}:bucket={bucket_rows}:hint={hint_type}",
        tasks=tasks,
        p0_truth_rows=_matrix(replay_window.p0_truth_rows),
        p1_truth_rows=_matrix(replay_window.p1_truth_rows),
        p2_hint_rows=_matrix(p2_hint_rows),
        topology=SchedulingTopology(group_size=int(replay_window.group_size)),
        release_model=ReleaseConstraint(
            phase="p1_return",
            rank=0,
            release_after_phase="p0_dispatch",
            expert_compute_delay=0.0,
        ),
        policy_options=policy_options,
        hint_metadata=PlanningHintMetadata(
            hint_type=str(hint_type),
            confidence=float(confidence),
            source_layer=int(replay_window.layer_id),
            target_layer=int(replay_window.layer_id) + 1,
        ),
        scheduling_mode="execution_window",
        information_mode="p0_p1_p2",
        max_waves=256,
        task_quantum_rows=int(bucket_rows),
        fixture_id=str(replay_window.fixture_id),
        window_id=str(replay_window.window_id),
        layer_id=int(replay_window.layer_id),
    )


def build_request_from_problem(
    *,
    request_id: str,
    problem: MultiPhaseSchedulingProblem,
    bucket_rows: int,
    policy_options: PolicyOptions,
    hint_type: str,
    confidence: float,
    fixture_id: str | None = None,
    window_id: str | None = None,
    layer_id: int | None = None,
    hint_metadata: dict[str, Any] | None = None,
) -> SchedulingRequest:
    replay_window = _BucketizerReplayWindow(
        fixture_id=str(fixture_id or "runtime"),
        window_id=str(window_id or request_id),
        layer_id=int(0 if layer_id is None else layer_id),
        p0_truth_rows=_matrix(problem.p0_dispatch_matrix),
        p1_truth_rows=_matrix(problem.p1_return_matrix),
        p2_truth_rows=_matrix(problem.p2_next_dispatch_forecast_matrix),
        group_size=int(problem.topology.num_gpus),
    )
    tasks = CanonicalBucketizer(bucket_rows=int(bucket_rows)).bucketize(replay_window)
    return SchedulingRequest(
        request_id=str(request_id),
        tasks=tasks,
        p0_truth_rows=_matrix(problem.p0_dispatch_matrix),
        p1_truth_rows=_matrix(problem.p1_return_matrix),
        p2_hint_rows=_matrix(problem.p2_next_dispatch_forecast_matrix),
        topology=SchedulingTopology(group_size=int(problem.topology.num_gpus)),
        release_model=problem.release_model,
        policy_options=policy_options,
        hint_metadata=PlanningHintMetadata(
            hint_type=str(hint_type),
            confidence=float(confidence),
            source_layer=None if layer_id is None else int(layer_id),
            target_layer=None if layer_id is None else int(layer_id) + 1,
            metadata=dict(hint_metadata or {}),
        ),
        scheduling_mode=str(problem.options.scheduling_mode),
        information_mode=str(problem.options.information_mode),
        max_waves=int(problem.options.max_waves),
        task_quantum_rows=int(bucket_rows),
        fixture_id=None if fixture_id is None else str(fixture_id),
        window_id=None if window_id is None else str(window_id),
        layer_id=None if layer_id is None else int(layer_id),
    )


__all__ = [
    "MatrixRows",
    "PlanningHintMetadata",
    "PolicyOptions",
    "ReplayExecutionTruth",
    "SchedulingPolicy",
    "SchedulingRequest",
    "SchedulingTopology",
    "build_policy",
    "build_request_from_problem",
    "build_request_from_replay_window",
    "request_to_legacy_problem",
]
