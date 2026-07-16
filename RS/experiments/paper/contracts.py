from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from rs.core.contracts.execution import ExecutionOutcome, MaterializedPlan, PublishedPlan
from rs.core.contracts.planning import PlanningRequest, WindowPlan


@dataclass(frozen=True)
class RecordMetadata:
    branch: str
    commit: str
    config_digest: str
    run_id: str
    seed: int
    model_id: str
    model_revision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TraceSample:
    schema_version: str
    model_id: str
    model_revision: str
    prompt_id: str
    batch_id: str
    sequence_length: int
    layer_id: str
    num_experts: int
    top_k: int
    router_logits_digest: str | None
    selected_experts_digest: str
    routing_weights_digest: str | None
    compact_route_counts: tuple[tuple[int, ...], ...]
    capture_timestamp: str
    metadata: RecordMetadata
    source_kind: str = "real_router_trace"
    trace_sample_id: str = ""
    trace_bundle_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload


@dataclass(frozen=True)
class TrafficInstance:
    instance_id: str
    trace_sample_id: str
    virtual_ep_size: int
    expert_to_rank_mapping: tuple[int, ...]
    mapping_digest: str
    P0_matrix: tuple[tuple[int, ...], ...]
    P1_matrix: tuple[tuple[int, ...], ...]
    P2_truth_matrix: tuple[tuple[int, ...], ...]
    flow_granularity: str
    bucketization: str
    cost_model_id: str
    traffic_digest: str
    metadata: RecordMetadata
    physical_world_size: int = 1
    source_trace_bundle: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload


@dataclass(frozen=True)
class ScheduleEvaluationRecord:
    instance_id: str
    policy_id: str
    policy_family: str
    scope: Literal["local", "joint"]
    is_exact: bool
    oracle_like: bool
    reference_model: str | None
    heuristic: bool
    solver_supported: bool | None
    solver_status: str
    certified_optimal: bool | None
    objective_logical_makespan: float | None
    best_bound: float | None
    optimality_gap: float | None
    planner_runtime_ms: float | None
    validation_runtime_ms: float | None
    replay_evaluation_runtime_ms: float | None
    record_construction_runtime_ms: float | None
    evaluation_total_runtime_ms: float | None
    objective: float | None
    coverage_valid: bool
    plan_digest: str | None
    fallback_count: int | None
    comparable: bool
    comparable_reason: str
    validation_errors: tuple[str, ...]
    cost_model_id: str
    runtime_info: dict[str, Any]
    metadata: RecordMetadata

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload


@dataclass(frozen=True)
class PredictionEvaluationRecord:
    instance_id: str
    predictor_id: str | None
    input_digest: str
    prediction_digest: str | None
    truth_digest: str
    no_future_leakage: bool | None
    allowed_input_fields: tuple[str, ...]
    prediction_ready_at: int | None
    truth_available_at: int | None
    raw_prediction_metrics: dict[str, Any]
    perfect_plan_metrics: dict[str, Any]
    predicted_plan_metrics: dict[str, Any]
    zero_plan_metrics: dict[str, Any]
    shuffled_plan_metrics: dict[str, Any]
    prediction_regret: float | None
    gain_over_zero: float | None
    metadata: RecordMetadata

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload


@dataclass(frozen=True)
class HidingTimelineRecord:
    model_id: str
    prompt_id: str
    layer_id: str
    current_dispatch_visible_ns: int | None
    observation_ready_ns: int | None
    prediction_ready_ns: int | None
    planning_ready_ns: int | None
    publication_ready_ns: int | None
    store_ready_ns: int | None
    target_dispatch_started_ns: int | None
    plan_consumed_ns: int | None
    available_window_us: float | None
    total_prepare_us: float | None
    ready_margin_us: float | None
    plan_source: str
    fallback_count: int | None
    metadata: RecordMetadata
    status: str = "PARTIAL"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload


@dataclass(frozen=True)
class RuntimeEvaluationRecord:
    instance_id: str
    requested_policy_id: str | None
    selected_policy_id: str | None
    published_plan_digest: str | None
    materialized_plan_digest: str | None
    executed_plan_digest: str | None
    execution_backend_id: str | None
    submitted_tasks: int | None
    completed_tasks: int | None
    unresolved_tasks: int | None
    fallback_count: int | None
    reference_output_digest: str | None
    executed_output_digest: str | None
    parity_status: str
    communication_makespan_ms: float | None
    visible_control_ms: float | None
    runtime_status: str
    metadata: RecordMetadata
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload


__all__ = [
    "ExecutionOutcome",
    "HidingTimelineRecord",
    "MaterializedPlan",
    "PlanningRequest",
    "PredictionEvaluationRecord",
    "PublishedPlan",
    "RecordMetadata",
    "RuntimeEvaluationRecord",
    "ScheduleEvaluationRecord",
    "TraceSample",
    "TrafficInstance",
    "WindowPlan",
]
