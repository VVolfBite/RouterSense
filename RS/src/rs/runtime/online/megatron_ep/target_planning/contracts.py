from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from rs.scheduling.contracts import LogicalSchedulePlan


MatrixRows = tuple[tuple[int, ...], ...]
ReconciliationStatus = Literal["exact", "repaired", "rejected"]
PlanOrigin = Literal["current_window", "prepared_priority_hint", "target_prepared", "provisional", "late_spliced"]
TargetPlanState = Literal[
    "LOGICAL_READY",
    "CLAIMED",
    "BOUND",
    "EXECUTING",
    "COMPLETED",
    "FAILED",
    "EXPIRED",
    "CANCELLED",
    "CONSUMED",
    "REJECTED",
]
TargetPlanFinalStatus = Literal["COMPLETED", "FAILED", "EXPIRED", "CANCELLED", "CONSUMED", "REJECTED"]


@dataclass(frozen=True)
class CurrentWindowJointPlan:
    source_layer_id: str
    execution_layer_id: str
    forecast_target_layer_id: str
    logical_plan: LogicalSchedulePlan
    logical_plan_digest: str
    actual_p0_rows: MatrixRows
    inferred_p1_rows: MatrixRows
    forecast_h1_rows: MatrixRows
    created_at_ns: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actual_p0_rows"] = [list(row) for row in self.actual_p0_rows]
        payload["inferred_p1_rows"] = [list(row) for row in self.inferred_p1_rows]
        payload["forecast_h1_rows"] = [list(row) for row in self.forecast_h1_rows]
        return payload


@dataclass(frozen=True)
class PreparedPriorityHint:
    source_layer_id: str
    target_layer_id: str
    priority_digest: str
    preferred_edges: tuple[tuple[str, int, int], ...]
    created_at_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetLayerPreparedJointPlan:
    source_layer_id: str
    target_layer_id: str
    run_id: str
    forward_epoch: int
    microbatch_id: str
    h1_prediction_digest: str
    h2_prediction_digest: str
    target_problem_digest: str
    logical_plan: LogicalSchedulePlan
    logical_plan_digest: str
    policy: str
    weights: dict[str, float]
    bucket_contract_digest: str
    topology_digest: str
    h1_rows: MatrixRows
    derived_p1_rows: MatrixRows
    h2_rows: MatrixRows
    created_at_ns: int
    ready_at_ns: int
    safe_projection_mode: str = "disabled"
    selected_variant: str = "raw_u"
    raw_logical_plan_digest: str = ""
    paired_b_logical_plan_digest: str = ""
    selected_logical_plan_digest: str = ""
    raw_u_estimated_makespan: float = 0.0
    paired_b_estimated_makespan: float = 0.0
    raw_u_build_us: float = 0.0
    paired_b_build_us: float = 0.0
    safe_selection_us: float = 0.0
    raw_u_plan_was_built: bool = True
    raw_u_plan_was_scored: bool = True
    raw_u_plan_was_selected: bool = True
    paired_b_plan_was_built: bool = False
    paired_b_plan_was_scored: bool = False
    paired_b_plan_was_selected: bool = False
    plan_origin: PlanOrigin = "target_prepared"
    plan_version: int = 1
    parent_plan_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["h1_rows"] = [list(row) for row in self.h1_rows]
        payload["derived_p1_rows"] = [list(row) for row in self.derived_p1_rows]
        payload["h2_rows"] = [list(row) for row in self.h2_rows]
        return payload


@dataclass(frozen=True)
class ProvisionalExecutionPlan:
    target_layer_id: str
    plan_origin: PlanOrigin
    plan_version: int
    parent_plan_version: int
    logical_plan: LogicalSchedulePlan
    logical_plan_digest: str
    created_at_ns: int
    execution_started_at_ns: int
    selected_variant: str = "raw_u"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TwoHorizonPrediction:
    forecast_horizon: int
    source_layer_id: str
    target_layer_id: str
    matrix_unit: str
    matrix_rows: MatrixRows
    matrix_digest: str
    predictor: str
    confidence: float
    created_at_ns: int
    prediction_us: float
    terminal: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matrix_rows"] = [list(row) for row in self.matrix_rows]
        return payload


@dataclass(frozen=True)
class TargetPlanKey:
    run_id: str
    forward_epoch: int
    microbatch_id: str
    target_layer_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationOutcome:
    status: ReconciliationStatus
    matched_edges: int
    removed_edges: int
    new_edges: int
    resized_edges: int
    preserved_order_ratio: float
    repair_us: float
    result_h1_rows: MatrixRows
    result_p1_rows: MatrixRows
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["result_h1_rows"] = [list(row) for row in self.result_h1_rows]
        payload["result_p1_rows"] = [list(row) for row in self.result_p1_rows]
        return payload


@dataclass(frozen=True)
class ReconciledExecutionPlan:
    status: ReconciliationStatus
    logical_plan: LogicalSchedulePlan | None
    logical_plan_digest: str | None
    preserved_edge_ratio: float
    inserted_edge_count: int
    removed_edge_count: int
    resized_edge_count: int
    repair_us: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanVersionLineage:
    old_version: int
    new_version: int
    plan_origin: PlanOrigin
    parent_plan_version: int
    frontier_digest: str
    replacement_suffix_digest: str
    switch_epoch: int
    all_rank_agreement: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetPlanTerminalRecord:
    key: TargetPlanKey
    plan_digest: str
    final_status: TargetPlanFinalStatus
    execution_origin: str
    terminal_at_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetPlanStateRecord:
    key: TargetPlanKey
    plan_digest: str
    state: TargetPlanState
    claim_owner: str = ""
    bound_owner: str = ""
    execution_origin: str = ""
    updated_at_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
