from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from rs.runtime.guards import INVARIANT_MODE_DIAGNOSTIC, InvariantFailure, RuntimeStateFieldError, normalize_invariant_mode


MatrixRows = tuple[tuple[int, ...], ...]


@dataclass
class RuntimeExecutionMetrics:
    selected_layer_match_count: int = 0
    selected_p0_hook_count: int = 0
    selected_p1_hook_count: int = 0
    prediction_source_p0_hook_count: int = 0
    none_heavy_hook_count: int = 0
    selected_transport_execution_count: int = 0
    real_p0_execution_count: int = 0
    real_p1_execution_count: int = 0
    shadow_dispatch_execution_count: int = 0
    shadow_combine_execution_count: int = 0
    observation_finalize_dispatch_count: int = 0
    observation_finalize_combine_count: int = 0
    shadow_policy_agreement_count: int = 0
    shadow_plan_build_count: int = 0
    shadow_control_collective_count: int = 0

    p0_traffic_matrix_gather_count: int = 0
    prediction_extra_collective_count: int = 0
    p1_planning_collective_count: int = 0

    before_async_p2p_phase_count: int = 0
    after_async_p2p_phase_count: int = 0
    selected_layer_before_async_p2p_phase_count: int = 0
    selected_layer_after_async_p2p_phase_count: int = 0
    all_layer_async_phase_count: int = 0

    compiler_id: str = ""
    logical_plan_digest: str = ""
    compiled_plan_digest: str = ""
    secondary_policy_invocation_count: int = 0
    secondary_policy_call_count: int = 0
    direct_compiler_selected_count: int = 0
    compiler_shadow_compare_count: int = 0

    dispatch_transport_start_ns: int = 0
    dispatch_transport_end_ns: int = 0
    rank_release_ns: int = 0
    expert_compute_start_ns: int = 0
    expert_compute_end_ns: int = 0
    combine_transport_start_ns: int = 0
    combine_transport_end_ns: int = 0
    forward_start_ns: int = 0
    forward_end_ns: int = 0


@dataclass
class PreparedWindowRuntimeState:
    invariant_mode: str = INVARIANT_MODE_DIAGNOSTIC
    prepared_plan: Any | None = None
    plan_source_layer: str = ""
    plan_created_at_us: int = 0

    stored_p1_plan_digest: str = ""
    consumed_p1_plan_digest: str = ""
    stored_p1_logical_plan_digest: str = ""
    consumed_p1_logical_plan_digest: str = ""
    stored_p1_compile_input_digest: str = ""
    consumed_p1_compile_input_digest: str = ""

    global_joint_window_plan: Any | None = None
    global_joint_plan_wire: Any | None = None
    global_joint_plan_agreement: Any | None = None
    prepared_priority_cache: Any | None = None

    actual_dispatch_by_layer: dict[str, Any] = field(default_factory=dict)
    predicted_dispatch_by_layer: dict[str, Any] = field(default_factory=dict)

    active_next_dispatch_prediction: Any | None = None
    prediction_consumption_records: list[Any] = field(default_factory=list)

    planning_traffic_source: str = ""
    pre_transport_observation_valid: bool = False
    captured_before_transport: bool = False

    dispatcher_send_splits: tuple[int, ...] = ()
    dispatcher_recv_splits: tuple[int, ...] = ()
    local_p0_row: tuple[int, ...] = ()

    p0_truth_rows: MatrixRows = ()
    p1_truth_rows: MatrixRows = ()
    actual_p0_total_rows: int = 0
    latest_predictor_name: str = ""
    latest_prediction_digest: str = ""
    latest_prediction_target_layer_id: str = ""
    latest_prediction_matrix_source: str = ""
    latest_prediction_row_sums: list[int] = field(default_factory=list)
    latest_prediction_col_sums: list[int] = field(default_factory=list)
    predictor_name: str = ""
    prediction_digest: str = ""
    prediction_confidence: float = 0.0
    predicted_row_sums: list[int] = field(default_factory=list)
    predicted_col_sums: list[int] = field(default_factory=list)
    latest_prediction_audit: dict[str, Any] = field(default_factory=dict)
    p2_matrix_source: str = ""
    p2_matrix_total_bytes: int = 0
    p2_matrix_row_sums: list[int] = field(default_factory=list)
    p2_matrix_col_sums: list[int] = field(default_factory=list)
    p2_matrix_is_replicated_local_row: bool = False
    p2_matrix_shape: list[int] = field(default_factory=list)
    p2_matrix_gather_time_us: float = 0.0
    p2_matrix_gather_status: str = ""
    p2_matrix_gather_call_count: int = 0
    prepared_priority_mode: str = ""
    has_real_p1_reservation: bool = False
    p1_reservation_row_sums: list[int] = field(default_factory=list)
    p1_reservation_col_sums: list[int] = field(default_factory=list)
    p1_inferred_from_p0: list[list[int]] = field(default_factory=list)
    ideal_joint_candidate_makespan: float = 0.0
    ideal_local_fallback_makespan: float = 0.0
    host_projected_joint_candidate_makespan: float = 0.0
    host_projected_local_fallback_makespan: float = 0.0
    host_projected_estimated_makespan: float = 0.0
    ideal_estimated_makespan: float = 0.0
    joint_candidate_plan_digest: str = ""
    local_plan_digest: str = ""
    selected_plan_digest: str = ""
    local_build_count: int = 0
    host_projection_count: int = 0
    requested_bucket_mode: str = ""
    effective_bucket_mode: str = ""
    requested_bucket_rows: int = 0
    effective_bucket_rows: int = 0
    canonical_task_digest: str = ""
    canonical_task_count: int = 0
    canonical_task_total_rows: int = 0
    dedicated_p2p_group_initialized: bool = False
    p2p_group_ranks: list[int] = field(default_factory=list)
    p2p_group_warmup_passed: bool = False
    hotpath_new_group_count: int = 0
    dedicated_p2p_groups_created: list[list[int]] = field(default_factory=list)
    local_dedicated_group_ranks: list[int] = field(default_factory=list)
    new_group_call_order: list[list[int]] = field(default_factory=list)
    compiler_shadow_status: str = ""
    compiler_shadow_plan_hash_matches_legacy: bool = False
    compiler_shadow_plan_hash: str = ""
    compiler_shadow_missing_task_count: int = 0
    compiler_shadow_extra_task_count: int = 0
    compiler_shadow_execution_order_matches_legacy: bool = False
    prepared_plan_found: bool = False
    execution_origin: str = ""
    reconciliation_count: int = 0
    full_u_replan_count: int = 0
    suffix_splice_count: int = 0
    prepared_target_logical_plan_digest: str = ""
    prepared_target_selected_variant: str = ""
    prepared_target_safe_projection_mode: str = ""
    provisional_plan_digest: str = ""
    total_model_moe_layers: int = 0
    selected_layer_ids: list[str] = field(default_factory=list)
    prediction_source_layer_ids: list[str] = field(default_factory=list)
    none_layer_ids: list[str] = field(default_factory=list)
    wrapped_selected_layer_ids: list[str] = field(default_factory=list)
    wrapped_prediction_source_layer_ids: list[str] = field(default_factory=list)
    unwrapped_none_layer_ids: list[str] = field(default_factory=list)
    effective_policy_name: str = ""
    effective_planner_id: str = ""
    effective_planner_family: str = ""
    requested_preflight_mode: str = ""
    effective_preflight_mode: str = ""
    joint_build_count_by_layer: dict[str, int] = field(default_factory=dict)
    local_build_count_by_layer: dict[str, int] = field(default_factory=dict)
    predict_count_by_layer: dict[str, int] = field(default_factory=dict)
    target_plan_enqueue_count_by_source_target: dict[str, int] = field(default_factory=dict)
    window_state_count_by_layer: dict[str, int] = field(default_factory=dict)
    shadow_plan_count_by_layer: dict[str, int] = field(default_factory=dict)
    selected_layer_timing_records: list[dict[str, Any]] = field(default_factory=list)
    expert_module_timing_records: list[dict[str, Any]] = field(default_factory=list)
    attribution_boundary_status: dict[str, Any] = field(default_factory=dict)
    dispatch_finalize_shape: list[int] | None = None
    dispatch_finalize_dispatcher: str = ""
    combine_finalize_shape: list[int] | None = None
    combine_finalize_dispatcher: str = ""
    dtoh_callsite_count: dict[str, int] = field(default_factory=dict)
    dtoh_callsite_wall_us: dict[str, float] = field(default_factory=dict)
    dtoh_callsite_bytes: dict[str, int] = field(default_factory=dict)

    metrics: RuntimeExecutionMetrics = field(default_factory=RuntimeExecutionMetrics)
    extras: dict[str, Any] = field(default_factory=dict)

    def set_invariant_mode(self, mode: str) -> None:
        self.invariant_mode = normalize_invariant_mode(mode)

    def _field_names(self) -> set[str]:
        return {item.name for item in fields(self)}

    def _metric_names(self) -> set[str]:
        return {item.name for item in fields(RuntimeExecutionMetrics)}

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._field_names():
            return getattr(self, key)
        if key in self._metric_names():
            return getattr(self.metrics, key)
        if self.invariant_mode != INVARIANT_MODE_DIAGNOSTIC:
            raise RuntimeStateFieldError(
                InvariantFailure(
                    error_code="RS-STATE-001",
                    stage="state",
                    message=f"unknown runtime state field read: {key}",
                    actual=key,
                )
            )
        return self.extras.get(key, default)

    def read(self, key: str, default: Any = None) -> Any:
        return self.get(key, default)

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, None)
        if value is None and key not in self._field_names() and key not in self._metric_names() and key not in self.extras:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._field_names():
            setattr(self, key, value)
            return
        if key in self._metric_names():
            setattr(self.metrics, key, value)
            return
        if self.invariant_mode != INVARIANT_MODE_DIAGNOSTIC:
            raise RuntimeStateFieldError(
                InvariantFailure(
                    error_code="RS-STATE-002",
                    stage="state",
                    message=f"unknown runtime state field write: {key}",
                    actual=key,
                )
            )
        self.extras[key] = value

    def write(self, key: str, value: Any) -> None:
        self[key] = value

    def pop(self, key: str, default: Any = None) -> Any:
        if key in self._field_names():
            current = getattr(self, key)
            setattr(self, key, default)
            return current
        if key in self._metric_names():
            current = getattr(self.metrics, key)
            setattr(self.metrics, key, default if default is not None else 0)
            return current
        if self.invariant_mode != INVARIANT_MODE_DIAGNOSTIC:
            raise RuntimeStateFieldError(
                InvariantFailure(
                    error_code="RS-STATE-003",
                    stage="state",
                    message=f"unknown runtime state field pop: {key}",
                    actual=key,
                )
            )
        return self.extras.pop(key, default)

    def remove(self, key: str, default: Any = None) -> Any:
        return self.pop(key, default)

    def update(self, payload: dict[str, Any]) -> None:
        for key, value in payload.items():
            self[key] = value

    def merge(self, payload: dict[str, Any]) -> None:
        self.update(payload)

    def to_legacy_artifact_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        metric_payload = payload.pop("metrics", {})
        extras = payload.pop("extras", {})
        return {
            **payload,
            **metric_payload,
            **extras,
        }
