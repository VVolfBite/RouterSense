from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from rs.core.contracts import PlanWave, PlannedFlow, WindowPlan
from rs.planning.api import to_logical_plan
from rs.scheduling.contracts import FlowDemand, LogicalSchedulePlan, LogicalWave
from rs.scheduling.validation import stable_hash


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
    window_plan: WindowPlan | None
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
    legacy_logical_plan_digest: str = ""
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

    def validate(self) -> None:
        if not str(self.source_layer_id):
            raise ValueError("source_layer_id must be non-empty")
        if not str(self.target_layer_id):
            raise ValueError("target_layer_id must be non-empty")
        if not str(self.run_id):
            raise ValueError("run_id must be non-empty")
        if int(self.forward_epoch) < 0:
            raise ValueError("forward_epoch must be >= 0")
        if not str(self.microbatch_id):
            raise ValueError("microbatch_id must be non-empty")
        if str(self.plan_origin) not in {"current_window", "prepared_priority_hint", "target_prepared", "provisional", "late_spliced"}:
            raise ValueError(f"unsupported plan_origin {self.plan_origin!r}")
        recomputed_digest = str(stable_hash(self.logical_plan.to_dict()))
        if self.window_plan is not None:
            self.window_plan.validate()
            recomputed_window_digest = str(self.window_plan.semantic_digest())
            if recomputed_window_digest != str(self.logical_plan_digest):
                raise ValueError("logical_plan_digest must match window_plan.semantic_digest()")
            canonical_legacy = to_logical_plan(self.window_plan)
            canonical_legacy_digest = str(stable_hash(canonical_legacy.to_dict()))
            if recomputed_digest != canonical_legacy_digest:
                raise ValueError("logical_plan payload must match compatibility projection of window_plan")
            if str(self.legacy_logical_plan_digest) and str(self.legacy_logical_plan_digest) != canonical_legacy_digest:
                raise ValueError("legacy_logical_plan_digest does not match logical_plan payload")
        elif recomputed_digest != str(self.logical_plan_digest):
            raise ValueError("logical_plan_digest does not match logical_plan payload")
        if str(self.selected_logical_plan_digest) and str(self.selected_logical_plan_digest) != str(self.logical_plan_digest):
            raise ValueError("selected_logical_plan_digest must match logical_plan_digest")
        for matrix_name, matrix in {
            "h1_rows": self.h1_rows,
            "derived_p1_rows": self.derived_p1_rows,
            "h2_rows": self.h2_rows,
        }.items():
            widths = {len(row) for row in matrix}
            if matrix and len(widths) != 1:
                raise ValueError(f"{matrix_name} must not be ragged")
            for row in matrix:
                for value in row:
                    if int(value) < 0:
                        raise ValueError(f"{matrix_name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        if self.window_plan is not None:
            payload["window_plan"] = self.window_plan.to_dict()
        payload["h1_rows"] = [list(row) for row in self.h1_rows]
        payload["derived_p1_rows"] = [list(row) for row in self.derived_p1_rows]
        payload["h2_rows"] = [list(row) for row in self.h2_rows]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TargetLayerPreparedJointPlan":
        window_plan_payload = payload.get("window_plan")
        window_plan = None
        if isinstance(window_plan_payload, dict):
            window_plan = WindowPlan(
                planner_id=str(window_plan_payload["planner_id"]),
                planner_family=str(window_plan_payload["planner_family"]),
                request_digest=str(window_plan_payload["request_digest"]),
                waves=tuple(
                    PlanWave(
                        wave_id=int(wave["wave_id"]),
                        flows=tuple(
                            PlannedFlow(
                                flow_id=str(flow["flow_id"]),
                                phase=str(flow["phase"]),
                                src_rank=int(flow["src_rank"]),
                                dst_rank=int(flow["dst_rank"]),
                                row_count=int(flow["row_count"]),
                                release_state=str(flow["release_state"]),
                                executable=bool(flow["executable"]),
                            )
                            for flow in wave.get("flows", ())
                        ),
                        estimated_duration=float(wave.get("estimated_duration", 0.0)),
                    )
                    for wave in window_plan_payload.get("waves", ())
                ),
                metadata=dict(window_plan_payload.get("metadata", {})),
            )
        logical_plan_payload = dict(payload["logical_plan"])
        logical_plan = LogicalSchedulePlan(
            policy_name=str(logical_plan_payload["policy_name"]),
            waves=tuple(
                LogicalWave(
                    wave_id=int(wave["wave_id"]),
                    flows=tuple(
                        FlowDemand(
                            flow_id=str(flow["flow_id"]),
                            phase=str(flow["phase"]),
                            src_rank=int(flow["src_rank"]),
                            dst_rank=int(flow["dst_rank"]),
                            byte_count=int(flow["byte_count"]),
                            release_state=str(flow["release_state"]),
                            is_executable=bool(flow["is_executable"]),
                            dependency_metadata=dict(flow.get("dependency_metadata", {})),
                        )
                        for flow in wave.get("flows", ())
                    ),
                    duration=float(wave.get("duration", 0.0)),
                )
                for wave in logical_plan_payload.get("waves", ())
            ),
            diagnostics=dict(logical_plan_payload.get("diagnostics", {})),
        )
        return cls(
            source_layer_id=str(payload["source_layer_id"]),
            target_layer_id=str(payload["target_layer_id"]),
            run_id=str(payload["run_id"]),
            forward_epoch=int(payload["forward_epoch"]),
            microbatch_id=str(payload["microbatch_id"]),
            h1_prediction_digest=str(payload["h1_prediction_digest"]),
            h2_prediction_digest=str(payload["h2_prediction_digest"]),
            target_problem_digest=str(payload["target_problem_digest"]),
            window_plan=window_plan,
            logical_plan=logical_plan,
            logical_plan_digest=str(payload["logical_plan_digest"]),
            legacy_logical_plan_digest=str(payload.get("legacy_logical_plan_digest", "")),
            policy=str(payload["policy"]),
            weights={str(key): float(value) for key, value in dict(payload.get("weights", {})).items()},
            bucket_contract_digest=str(payload["bucket_contract_digest"]),
            topology_digest=str(payload["topology_digest"]),
            h1_rows=tuple(tuple(int(value) for value in row) for row in payload.get("h1_rows", ())),
            derived_p1_rows=tuple(tuple(int(value) for value in row) for row in payload.get("derived_p1_rows", ())),
            h2_rows=tuple(tuple(int(value) for value in row) for row in payload.get("h2_rows", ())),
            created_at_ns=int(payload["created_at_ns"]),
            ready_at_ns=int(payload["ready_at_ns"]),
            safe_projection_mode=str(payload.get("safe_projection_mode", "disabled")),
            selected_variant=str(payload.get("selected_variant", "raw_u")),
            raw_logical_plan_digest=str(payload.get("raw_logical_plan_digest", "")),
            paired_b_logical_plan_digest=str(payload.get("paired_b_logical_plan_digest", "")),
            selected_logical_plan_digest=str(payload.get("selected_logical_plan_digest", "")),
            raw_u_estimated_makespan=float(payload.get("raw_u_estimated_makespan", 0.0)),
            paired_b_estimated_makespan=float(payload.get("paired_b_estimated_makespan", 0.0)),
            raw_u_build_us=float(payload.get("raw_u_build_us", 0.0)),
            paired_b_build_us=float(payload.get("paired_b_build_us", 0.0)),
            safe_selection_us=float(payload.get("safe_selection_us", 0.0)),
            raw_u_plan_was_built=bool(payload.get("raw_u_plan_was_built", True)),
            raw_u_plan_was_scored=bool(payload.get("raw_u_plan_was_scored", True)),
            raw_u_plan_was_selected=bool(payload.get("raw_u_plan_was_selected", True)),
            paired_b_plan_was_built=bool(payload.get("paired_b_plan_was_built", False)),
            paired_b_plan_was_scored=bool(payload.get("paired_b_plan_was_scored", False)),
            paired_b_plan_was_selected=bool(payload.get("paired_b_plan_was_selected", False)),
            plan_origin=str(payload.get("plan_origin", "target_prepared")),
            plan_version=int(payload.get("plan_version", 1)),
            parent_plan_version=int(payload.get("parent_plan_version", 0)),
        )


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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TwoHorizonPrediction":
        return cls(
            forecast_horizon=int(payload["forecast_horizon"]),
            source_layer_id=str(payload["source_layer_id"]),
            target_layer_id=str(payload["target_layer_id"]),
            matrix_unit=str(payload["matrix_unit"]),
            matrix_rows=tuple(tuple(int(value) for value in row) for row in payload.get("matrix_rows", ())),
            matrix_digest=str(payload["matrix_digest"]),
            predictor=str(payload["predictor"]),
            confidence=float(payload["confidence"]),
            created_at_ns=int(payload["created_at_ns"]),
            prediction_us=float(payload["prediction_us"]),
            terminal=bool(payload.get("terminal", False)),
        )


@dataclass(frozen=True)
class TargetPlanKey:
    run_id: str
    forward_epoch: int
    microbatch_id: str
    target_layer_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TargetPlanKey":
        return cls(
            run_id=str(payload["run_id"]),
            forward_epoch=int(payload["forward_epoch"]),
            microbatch_id=str(payload["microbatch_id"]),
            target_layer_id=str(payload["target_layer_id"]),
        )


@dataclass(frozen=True)
class PreparationToken:
    service_session_id: int
    forward_generation: int
    target_key: TargetPlanKey
    task_version: int
    publish_sequence: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_key"] = self.target_key.to_dict()
        return payload


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
