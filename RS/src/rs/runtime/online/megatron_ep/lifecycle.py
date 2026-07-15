"""Megatron EP 正式执行链路的 P0/P1 生命周期主线。

这个文件是在线运行时的核心编排器，主要负责：
- before/after token_dispatch
- before/after token_combine
- phase context 构建、计划协商、transport 激活/清理
- prepared plan、release state、pending-window shadow 的记录
如果想看“运行时一层里到底发生了什么”，优先看这里。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rs.core.layer_ids import stable_layer_ids
from rs.core.layer_selection import layer_selected, resolve_layer_selector
from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
    PredictionIdentity,
    TrafficHistoryContext,
)
from rs.core.contracts.execution import ActualPhaseContext
from rs.core.contracts.measurement import MeasurementEvent
from rs.core.contracts.observation import RuntimeObservationConfig
from rs.core.contracts.result import EligibilityResult, ResultBundle, RunIdentity
from rs.prediction import PredictionRegistry, resolve_predictor_id
from rs.planning import (
    CommonCorePlanEstimator,
    PlannerPolicyConfig,
    PlannerRegistry,
    PlannerSelectionMode,
    PlannerSelector,
    PlanningCostModel,
)
from rs.planning.api import to_logical_plan
from rs.planning.request_builder import build_window_planning_request
from rs.planning.runtime_compat import resolve_phase_policy
from rs.runtime.online.megatron_ep.contracts import (
    HookExecutionMode,
    InjectionDecision,
    PlanAgreement,
    PolicyContext,
    RouterSenseInjectionConfig,
    RouterSensePlan,
    RuntimeObservation,
)
from rs.runtime.online.megatron_ep.control.agreement_wire import compute_ep_group_hash, run_policy_agreement
from rs.runtime.online.megatron_ep.control.p2_matrix import TrafficMatrixBundle, build_traffic_matrix_bundle
from rs.runtime.online.megatron_ep.control.plan_agreement import run_phase_plan_agreement
from rs.runtime.online.megatron_ep.control.p2_contracts import P2HintRequest
from rs.runtime.online.megatron_ep.control.p2_provider import build_p2_hint_provider
from rs.runtime.online.megatron_ep.config import resolve_online_policy_config
from rs.runtime.online.megatron_ep.observation import (
    PolicyRuntimeRecord,
    RouterSenseObserver,
    RuntimeObservationRecorder,
    build_runtime_observation,
    control_replay_trace_row,
    digest_text,
    extract_int_tuple,
    parse_layer_id,
    phase_context_artifact,
    scheduled_plan_artifact,
    transport_bundle_artifact,
)
from rs.runtime.online.megatron_ep.state.window_runtime_state import (
    PreparedPlanBinding,
    WindowReleaseState,
    bind_prepared_plan,
)
from rs.runtime.online.megatron_ep.async_release.joint_plan_agreement import GlobalJointPlanWire
from rs.runtime.online.megatron_ep.compiler_facade import (
    CompilationOptions,
    PlanCompilationRequest,
    build_phase_canonical_tasks,
    compile_schedule,
)
from rs.runtime.online.megatron_ep.execution.release_frontier import ReleaseBatchTask
from rs.runtime.online.megatron_ep.state import PreparedWindowRuntimeState
from rs.runtime.online.megatron_ep.planning.window_shadow_service import (
    advance_window_release,
    build_window_state_record,
    maybe_build_window_shadow,
)
from rs.runtime.online.megatron_ep.observation.runtime_export import build_prepared_plan_summary
from rs.runtime.online.megatron_ep.phase import (
    DispatcherSnapshot,
    PhaseContextBuildRequest,
    PhaseExecutionPlan,
    PhasePayloadContract,
    PhaseReadyContext,
    PreTransportTrafficObservation,
    RuntimeIdentity,
    build_phase_ready_context,
    reconstruct_global_phase_contexts_from_byte_matrix,
)
from rs.runtime.online.megatron_ep.public_types import (
    CombineFailedEvent,
    CombineCompleteEvent,
    CombineReadyEvent,
    ControlGroupHandle,
    DispatchFailedEvent,
    DispatchCompleteEvent,
    DispatchReadyEvent,
    ForwardBeginEvent,
    ForwardEndEvent,
    ForwardFailedEvent,
    PublicationPollStatus,
    RuntimeDecision,
    RuntimeEvent,
    SelectedLayerStop,
    UnsupportedSchedulerMode,
)
from rs.runtime.online.megatron_ep.control.communication_lane import GlooControlCommunicationLane, slot_from_request
from rs.runtime.online.megatron_ep.control.shadow_policy.joint_shadow import JointShadowP0P1Policy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_order import NativeOrderPolicy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_passthrough_identity import NativePassthroughIdentityPolicy
from rs.runtime.online.megatron_ep.prediction import (
    ActiveNextDispatchPrediction,
    compare_predicted_to_actual,
    maybe_capture_expert_route_trace,
)
from rs.runtime.online.megatron_ep.target_planning import (
    ProvisionalExecutionPlan,
    TargetLayerPlannerService,
    TargetLayerPlanningRequest,
    TargetLayerPreparedJointPlan,
    TargetPlanKey,
    TargetPlanStore,
    reconcile_once,
)
from rs.runtime.online.megatron_ep.target_planning.planner_service import PreparationSubmitStatus
from rs.scheduling.contracts import PreparedWindowPlan
from rs.scheduling.bucketizer import (
    BUCKET_MODE_DYNAMIC_CURRENT,
    BUCKET_MODE_FIXED_ROWS,
    bucket_mode_for_rows,
    summarize_bucket_tasks,
)
from rs.scheduling.capabilities import PolicyCapabilities
from rs.scheduling.traffic_matrix import (
    canonicalize_remote_matrix,
    matrix_col_sums_remote,
    matrix_digest_remote,
    matrix_nonzero_remote_edge_count,
    matrix_remote_bytes,
    matrix_row_sums_remote,
)
from rs.scheduling.validation import stable_hash


@dataclass
class ReleaseStateLedger:
    run_id: str
    forward_generation: int
    microbatch_id: str
    completed_payload_roles_by_phase: dict[tuple[str, str, int], set[str]] = field(default_factory=dict)
    satisfied_release_ids: set[str] = field(default_factory=set)

    def reset(self, *, run_id: str, forward_generation: int, microbatch_id: str) -> None:
        self.run_id = str(run_id)
        self.forward_generation = int(forward_generation)
        self.microbatch_id = str(microbatch_id)
        self.completed_payload_roles_by_phase.clear()
        self.satisfied_release_ids.clear()

    def record_payload_completion(
        self,
        *,
        layer_id: str,
        phase: str,
        local_group_rank: int,
        payload_role: str,
        required_payload_roles: tuple[str, ...],
    ) -> tuple[str, ...]:
        key = (str(layer_id), str(phase), int(local_group_rank))
        completed = self.completed_payload_roles_by_phase.setdefault(key, set())
        completed.add(str(payload_role))
        required = {str(item) for item in required_payload_roles}
        if not required.issubset(completed):
            return ()
        if str(phase) == "P0":
            release_id = f"release:{str(layer_id)}:p0_inbound_complete:{int(local_group_rank)}"
        elif str(phase) == "P1":
            release_id = f"release:{str(layer_id)}:p1_inbound_complete:{int(local_group_rank)}"
        else:
            return ()
        if release_id in self.satisfied_release_ids:
            return ()
        self.satisfied_release_ids.add(release_id)
        return (release_id,)


@dataclass(frozen=True)
class RuntimePredictionCompatResult:
    predictor_id: str
    matrix: tuple[tuple[int, ...], ...]
    matrix_digest: str
    confidence: float
    predictor_version: str = "v1"
    evaluation_eligible: bool = True
    is_oracle: bool = False
    valid: bool = True
    error: str = ""
    fallback: bool = False

    @property
    def predictor_name(self) -> str:
        return self.predictor_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": [list(row) for row in self.matrix],
            "matrix_digest": str(self.matrix_digest),
            "predictor_name": str(self.predictor_name),
            "predictor_id": str(self.predictor_id),
            "predictor_version": str(self.predictor_version),
            "confidence": float(self.confidence),
            "evaluation_eligible": bool(self.evaluation_eligible),
            "is_oracle": bool(self.is_oracle),
            "valid": bool(self.valid),
            "error": str(self.error),
            "fallback": bool(self.fallback),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimePredictionCompatResult":
        return cls(
            predictor_id=str(payload.get("predictor_id", payload.get("predictor_name", ""))),
            matrix=tuple(tuple(int(value) for value in row) for row in payload.get("matrix", [])),
            matrix_digest=str(payload.get("matrix_digest", "")),
            predictor_version=str(payload.get("predictor_version", "v1")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            evaluation_eligible=bool(payload.get("evaluation_eligible", True)),
            is_oracle=bool(payload.get("is_oracle", False)),
            valid=bool(payload.get("valid", False)),
            error=str(payload.get("error", "")),
            fallback=bool(payload.get("fallback", False)),
        )


@dataclass
class RouterSenseInjectionRuntime:
    config: RouterSenseInjectionConfig
    rank: int
    local_rank: int
    run_id: str
    step_id: str
    microbatch_id: str
    model_revision_hash: str
    request_table_hash: str
    hostname: str
    observer: RouterSenseObserver | None = None
    ep_group_ranks: tuple[int, ...] = ()
    ep_group_root_global_rank: int = 0
    ep_process_group: Any | None = None
    completed: list[PolicyRuntimeRecord] = field(default_factory=list)
    _pending_p0: dict[str, RuntimeObservation] = field(default_factory=dict)
    _pending_p1: dict[str, RuntimeObservation] = field(default_factory=dict)
    _runtime_state: PreparedWindowRuntimeState = field(default_factory=PreparedWindowRuntimeState)
    plan_arrival_records: list[dict[str, Any]] = field(default_factory=list)
    window_state_records: list[dict[str, Any]] = field(default_factory=list)
    prepared_plan_bindings: list[dict[str, Any]] = field(default_factory=list)
    release_events: list[dict[str, Any]] = field(default_factory=list)
    window_schedule_shadows: list[dict[str, Any]] = field(default_factory=list)
    prepared_phase_plan_shadows: list[dict[str, Any]] = field(default_factory=list)
    pending_window_driver_records: list[dict[str, Any]] = field(default_factory=list)
    planning_timing_records: list[dict[str, Any]] = field(default_factory=list)
    control_replay_traces: list[dict[str, Any]] = field(default_factory=list)
    prediction_audits: list[dict[str, Any]] = field(default_factory=list)
    control_timeline: list[dict[str, Any]] = field(default_factory=list)
    control_commands: list[dict[str, Any]] = field(default_factory=list)
    assertion_state: dict[str, Any] = field(default_factory=dict)
    _active_plan_versions: dict[str, int] = field(default_factory=dict)
    _active_plan_hashes: dict[str, str] = field(default_factory=dict)
    _window_states: dict[str, Any] = field(default_factory=dict)
    _selected_layer_matches_seen: set[str] = field(default_factory=set)
    _target_plan_reconciled_keys: set[tuple[str, int, str, str]] = field(default_factory=set)
    _forward_epoch: int = 0
    observation_recorder: RuntimeObservationRecorder | None = None
    _active_transport: dict[str, Any] | None = None
    _p2_hint_provider: Any | None = None
    _pending_window_adapter_instance: Any | None = None
    perf_counters: dict[str, dict[str, float]] = field(default_factory=dict)
    target_plan_store: TargetPlanStore | None = None
    target_planner_service: TargetLayerPlannerService | None = None
    target_plan_control_group: Any | None = None
    target_plan_control_group_handle: ControlGroupHandle | None = None
    control_communication_lane: Any | None = None
    _ready_target_plan_candidates: dict[str, Any] = field(default_factory=dict)
    _expected_publication_slots: dict[tuple[str, int, str, str], Any] = field(default_factory=dict)
    _terminal_publication_slots: set[str] = field(default_factory=set)
    _published_publication_slots: set[str] = field(default_factory=set)
    _poll_attempts: set[tuple[str, str]] = field(default_factory=set)
    _available_moe_layer_ids: tuple[str, ...] = ()
    _resolved_schedule_selector: Any | None = None
    _selected_layer_id_set: frozenset[str] = field(default_factory=frozenset)
    _prediction_source_layer_id_set: frozenset[str] = field(default_factory=frozenset)
    _none_layer_id_set: frozenset[str] = field(default_factory=frozenset)
    _effective_phase_policy_name_cache: str = ""
    _resolved_policy_capabilities_cache: Any | None = None
    _joint_window_enabled_cache: bool = False
    _cross_layer_prediction_enabled_cache: bool = False
    _target_preplanning_enabled_cache: bool = False
    _current_plan_build_keys: set[tuple[int, int, str, str, str]] = field(default_factory=set)
    _selected_layer_active_ns: dict[tuple[int, str], int] = field(default_factory=dict)
    _expert_module_active_ns: dict[tuple[int, str], int] = field(default_factory=dict)
    release_state_ledger: ReleaseStateLedger = field(
        default_factory=lambda: ReleaseStateLedger(
            run_id="",
            forward_generation=0,
            microbatch_id="",
        )
    )
    _latest_execution_outcomes: list[dict[str, Any]] = field(default_factory=list)
    _latest_result_bundle: ResultBundle | None = None

    # Configuration and policy selection

    def __post_init__(self) -> None:
        self.release_state_ledger.reset(
            run_id=str(self.run_id),
            forward_generation=int(self._forward_epoch),
            microbatch_id=str(self.microbatch_id),
        )
        self._runtime_state.set_invariant_mode(str(getattr(self.config, "invariant_mode", "diagnostic")))
        if self.observation_recorder is None:
            self.observation_recorder = RuntimeObservationRecorder(
                config=RuntimeObservationConfig(
                    profile=str(getattr(self.config, "observation_profile", "minimal")),
                    invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
                    capture_enabled=bool(getattr(self.config, "capture_phase_tensors", False)),
                    capture_expert_trace=bool(getattr(self.config, "capture_expert_trace", False)),
                    capture_layer_selector=str(getattr(self.config, "capture_layer_selector", "")),
                    capture_phase_selector=str(getattr(self.config, "capture_phase_selector", "")),
                    heartbeat_enabled=bool(getattr(self.config, "heartbeat_enabled", False)),
                    per_wave_timing_enabled=bool(getattr(self.config, "per_wave_timing_enabled", False)),
                    replay_trace_enabled=bool(getattr(self.config, "replay_trace_enabled", False)),
                )
            )
        if self.config.p2_hint_mode == "calibrated_artifact":
            self._p2_hint_provider = build_p2_hint_provider(
                self.config.p2_hint_mode,
                shared_state=self._runtime_state,
            )
        self._refresh_policy_caches()
        self._ensure_target_planner_runtime()

    def _refresh_policy_caches(self) -> None:
        resolved = resolve_online_policy_config(self.config)
        if resolved is None:
            self._effective_phase_policy_name_cache = ""
            self._resolved_policy_capabilities_cache = None
            self._joint_window_enabled_cache = False
            self._cross_layer_prediction_enabled_cache = False
            self._target_preplanning_enabled_cache = False
            return
        self._effective_phase_policy_name_cache = str(resolved.builder_key)
        spec = resolved.spec
        scope = str(spec.scheduling_scope)
        execution_model = str(spec.execution_model)
        base = PolicyCapabilities(
            supports_offline=bool(spec.offline_eligible),
            supports_online_phase_local_execution=bool(spec.online_eligible and spec.phase_local_eligible),
            supports_online_multiphase_execution=bool(spec.online_eligible and ("joint" in scope or "multiphase" in scope or "global" in execution_model)),
            uses_current_ready_flows=True,
            uses_blocked_p1_dependency=bool("joint" in scope or "multiphase" in scope),
            uses_p2_forecast=bool(spec.supports_p2_hint),
            requires_fixed_placement=False,
            evaluation_eligible=bool(spec.offline_eligible),
        )
        predictor_name = self._online_p2_predictor_name()
        has_prediction = predictor_name not in {"none", "zero_hint"}
        is_joint_window = str(self.config.execution_mode) == "joint_window_async_p2p"
        safe_projection_mode = str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select")
        supports_target_preplanning = bool(
            has_prediction
            and is_joint_window
            and base.uses_p2_forecast
            and (safe_projection_mode == "disabled" or safe_projection_mode == "host_select")
        )
        self._resolved_policy_capabilities_cache = base.with_runtime_flags(
            supports_current_window_joint_planning=bool(
                is_joint_window and base.supports_online_multiphase_execution
            ),
            supports_cross_layer_prediction=bool(has_prediction and base.uses_p2_forecast),
            supports_two_horizon_prediction=bool(has_prediction and base.uses_p2_forecast),
            supports_target_layer_preplanning=bool(supports_target_preplanning),
            supports_p1_plan_reuse=bool(
                is_joint_window and base.supports_online_multiphase_execution
            ),
            supports_late_suffix_splice=False,
            supports_rank_release_batch=bool(is_joint_window),
        )
        self._joint_window_enabled_cache = bool(
            self._resolved_policy_capabilities_cache.supports_current_window_joint_planning
        )
        self._cross_layer_prediction_enabled_cache = bool(
            self._resolved_policy_capabilities_cache.supports_cross_layer_prediction
        )
        self._target_preplanning_enabled_cache = bool(
            self._resolved_policy_capabilities_cache.supports_target_layer_preplanning
        )
        self._runtime_state.write("effective_policy_name", str(self._effective_phase_policy_name_cache))
        self._runtime_state.write("requested_preflight_mode", str(getattr(self.config, "preflight_mode", "full")))
        self._runtime_state.write("effective_preflight_mode", str(getattr(self.config, "preflight_mode", "full")))

    def configure_hook_scope(self, *, available_layer_names: tuple[str, ...]) -> None:
        available_layer_ids: list[str] = []
        for layer_name in available_layer_names:
            layer_id = str(parse_layer_id(layer_name))
            if layer_id not in available_layer_ids:
                available_layer_ids.append(layer_id)
        self._available_moe_layer_ids = tuple(available_layer_ids)
        resolved = resolve_layer_selector(
            str(self.config.schedule_layer_selector),
            selected_layer_ids=tuple(str(item) for item in getattr(self.config, "selected_layer_ids", ()) or ()),
            available_layer_ids=self._available_moe_layer_ids,
            invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
        )
        self._resolved_schedule_selector = resolved
        if resolved.matches_all:
            selected = frozenset(self._available_moe_layer_ids)
        else:
            selected = frozenset(str(item) for item in resolved.resolved_layer_ids)
        prediction_source: set[str] = set()
        if self._target_preplanning_enabled_cache:
            for layer_id in selected:
                if str(layer_id).isdigit() and int(layer_id) > 0:
                    candidate = str(int(layer_id) - 1)
                    if candidate not in selected:
                        prediction_source.add(candidate)
        self._selected_layer_id_set = selected
        self._prediction_source_layer_id_set = frozenset(prediction_source)
        self._none_layer_id_set = frozenset(
            layer_id
            for layer_id in self._available_moe_layer_ids
            if layer_id not in self._selected_layer_id_set and layer_id not in self._prediction_source_layer_id_set
        )
        self._runtime_state.write("total_model_moe_layers", int(len(self._available_moe_layer_ids)))
        self._runtime_state.write("selected_layer_ids", stable_layer_ids(self._selected_layer_id_set))
        self._runtime_state.write("prediction_source_layer_ids", stable_layer_ids(self._prediction_source_layer_id_set))
        self._runtime_state.write("none_layer_ids", stable_layer_ids(self._none_layer_id_set))
        self._runtime_state.write("wrapped_selected_layer_ids", stable_layer_ids(self._selected_layer_id_set))
        self._runtime_state.write("wrapped_prediction_source_layer_ids", stable_layer_ids(self._prediction_source_layer_id_set))
        self._runtime_state.write("unwrapped_none_layer_ids", stable_layer_ids(self._none_layer_id_set))

    def layer_role_for_name(self, layer_name: str) -> str:
        layer_id = str(parse_layer_id(layer_name))
        if self._resolved_schedule_selector is None:
            fallback_selector = resolve_layer_selector(
                str(self.config.schedule_layer_selector),
                selected_layer_ids=tuple(str(item) for item in getattr(self.config, "selected_layer_ids", ()) or ()),
                invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
            )
            if fallback_selector.matches_all or layer_selected(layer_id, selector=fallback_selector):
                self._selected_layer_matches_seen.add(layer_id)
                self._runtime_state.metrics.selected_layer_match_count = int(len(self._selected_layer_matches_seen))
                return "selected"
            return "none"
        if layer_id in self._selected_layer_id_set:
            self._selected_layer_matches_seen.add(layer_id)
            self._runtime_state.metrics.selected_layer_match_count = int(len(self._selected_layer_matches_seen))
            return "selected"
        if layer_id in self._prediction_source_layer_id_set:
            return "prediction_source"
        return "none"

    def _layer_id_selected(self, layer_id: str) -> bool:
        normalized = str(layer_id)
        if self._resolved_schedule_selector is None:
            fallback_selector = resolve_layer_selector(
                str(self.config.schedule_layer_selector),
                selected_layer_ids=tuple(str(item) for item in getattr(self.config, "selected_layer_ids", ()) or ()),
                invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
            )
            return bool(fallback_selector.matches_all or layer_selected(normalized, selector=fallback_selector))
        return normalized in self._selected_layer_id_set

    @property
    def _prepared_plan_state(self) -> PreparedWindowRuntimeState:
        return self._runtime_state

    def _artifact_profile(self) -> str:
        return str(getattr(self.config, "observation_profile", "minimal"))

    def _is_perf_profile(self) -> bool:
        return self._artifact_profile() in {"perf", "timeline_light", "attribution_light"}

    def _is_debug_profile(self) -> bool:
        return self._artifact_profile() == "debug"

    def _allow_shadow_artifacts(self) -> bool:
        return not self._is_perf_profile()

    def _replay_trace_enabled(self) -> bool:
        return bool(getattr(self.config, "replay_trace_enabled", False))

    def _record_control_replay_trace(self, *, phase_ctx: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
        if not self._replay_trace_enabled():
            return
        self.control_replay_traces.append(
            control_replay_trace_row(
                run_id=self.run_id,
                ep_group_size=int(len(self.ep_group_ranks) or 1),
                bucket_rows=int(self.config.bucket_rows),
                phase_ctx=phase_ctx,
                plan=plan,
            )
        )

    def _effective_phase_policy_name(self) -> str:
        return str(self._effective_phase_policy_name_cache)

    def _phase_policy(self):
        phase_policy_name = self._effective_phase_policy_name()
        if phase_policy_name:
            return resolve_phase_policy(
                policy_name=phase_policy_name,
                bucket_rows=self.config.bucket_rows,
                p0_weight=self.config.p0_weight,
                p1_reservation_weight=self.config.p1_reservation_weight,
                p2_hint_weight=self.config.p2_hint_weight,
                p2_hint_artifact=self.config.p2_hint_artifact,
            )
        if self.config.scheduler_mode == "native_passthrough_identity":
            return NativePassthroughIdentityPolicy()
        if self.config.scheduler_mode == "native_order":
            return NativeOrderPolicy()
        if self.config.scheduler_mode == "joint_shadow_p0p1":
            return JointShadowP0P1Policy()
        raise UnsupportedSchedulerMode(f"Unsupported scheduler_mode={self.config.scheduler_mode!r}")

    def _pending_window_adapter(self) -> Any:
        from rs.runtime.online.megatron_ep.pending_window import MultiphasePendingWindowAdapter

        phase_policy_name = self._effective_phase_policy_name()
        if not phase_policy_name:
            raise UnsupportedSchedulerMode("multiphase_pending_window requires a resolved phase policy name")
        if self._pending_window_adapter_instance is None:
            self._pending_window_adapter_instance = MultiphasePendingWindowAdapter(
                shared_state=self._runtime_state,
                phase_policy_name=phase_policy_name,
                bucket_rows=self.config.bucket_rows,
                p0_weight=self.config.p0_weight,
                p1_reservation_weight=self.config.p1_reservation_weight,
                p2_hint_weight=self.config.p2_hint_weight,
                fast_path_enabled=self._is_perf_profile(),
            )
        return self._pending_window_adapter_instance

    def _layer_selected(self, layer_name: str) -> bool:
        return self.layer_role_for_name(layer_name) == "selected"

    def _layer_is_prediction_source(self, layer_name: str) -> bool:
        return self.layer_role_for_name(layer_name) == "prediction_source"

    def _phase_selected(self, phase: str) -> bool:
        selector = str(self.config.schedule_phase_selector).lower()
        if selector in {"", "both", "all"}:
            return True
        return selector == str(phase).lower()

    def _should_schedule_phase(self, *, layer_name: str, phase: str) -> bool:
        return (
            bool(self._effective_phase_policy_name_cache)
            and self.config.execution_mode in {"phase_sync_wave", "multiphase_pending_window", "joint_window_async_p2p"}
            and self.config.control_mode == "sync_before_phase"
            and self.layer_role_for_name(layer_name) == "selected"
            and self._phase_selected(phase)
        )

    def _is_joint_window_async_mode(self) -> bool:
        return bool(self._joint_window_enabled_cache)

    def _runtime_safe_joint_pair(self) -> tuple[str, str]:
        policy_name = str(self.config.policy or "")
        if "gated_greedy" in policy_name:
            return ("U_gated_greedy_maximal", "B_gated_greedy_maximal")
        return ("U_barrier_criticality_global_matching", "B_barrier_criticality_core_independent")

    def _effective_bucket_mode(self) -> str:
        return bucket_mode_for_rows(int(self.config.bucket_rows))

    def _requested_bucket_mode(self) -> str:
        requested = str(getattr(self.config, "bucket_mode", "") or "").strip()
        if requested:
            return requested
        return self._effective_bucket_mode()

    def _assert_bucket_mode_consistency(self) -> None:
        requested = self._requested_bucket_mode()
        effective = self._effective_bucket_mode()
        if requested != effective:
            raise RuntimeError(
                f"bucket mode mismatch: requested={requested!r} effective={effective!r} "
                f"bucket_rows={int(self.config.bucket_rows)}"
            )

    def _should_stop_after_layer(self, *, layer_name: str, phase: str) -> bool:
        if not (
            self.config.stop_after_selected_layer
            and self._layer_selected(layer_name)
            and self._phase_selected(phase)
        ):
            return False
        selector = str(self.config.schedule_phase_selector).lower()
        if selector in {"", "both", "all"}:
            return str(phase).upper() == "P1"
        return True

    # Transport activation and timing

    def _activate_transport(self, *, layer_name: str, phase: str, context: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
        start_ns = time.monotonic_ns()
        effective_preflight_mode = str(getattr(self.config, "preflight_mode", "full") or "full")
        plan_metrics = dict(plan.metrics or {})
        plan_preflight_mode = str(plan_metrics.get("preflight_mode", "") or "")
        if plan_preflight_mode and plan_preflight_mode != effective_preflight_mode:
            raise RuntimeError(
                f"preflight mode mismatch before transport activation: "
                f"plan={plan_preflight_mode!r} effective={effective_preflight_mode!r}"
            )
        if plan_preflight_mode != effective_preflight_mode:
            plan = replace(plan, metrics={**plan_metrics, "preflight_mode": effective_preflight_mode})
        if self._layer_selected(layer_name):
            self._runtime_state.metrics.selected_transport_execution_count = int(
                self._runtime_state.metrics.selected_transport_execution_count
            ) + 1
        prepared_execution = None
        if self._is_joint_window_async_mode():
            prepared_execution = self._prepared_execution_cache().get(self.target_plan_store._key(self._target_plan_key(layer_name=layer_name))) if self.target_plan_store is not None else None
        self._active_transport = {
            "layer_name": layer_name,
            "phase": phase,
            "context": context,
            "plan": plan,
            "prepared_execution": prepared_execution,
        }
        adapter = getattr(self, "transport_adapter", None)
        if adapter is not None:
            if hasattr(adapter, "set_effective_preflight_mode"):
                adapter.set_effective_preflight_mode(effective_preflight_mode)
            adapter.activate(
                layer_name=layer_name,
                phase=phase,
                context=context,
                plan=plan,
                prepared_execution=prepared_execution,
                execution_pipeline=getattr(self, "execution_pipeline", None),
                runtime=self,
            )
        end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase=phase,
            stage="activate_transport",
            start_ns=start_ns,
            end_ns=end_ns,
            wave_count=int(len(plan.waves)),
            bucket_count=int(sum(len(wave.bucket_tasks) for wave in plan.waves)),
        )

    def current_transport(self) -> dict[str, Any] | None:
        return self._active_transport

    def _execution_plan_cache(self) -> dict[tuple[str, int, str, str], Any]:
        cache = getattr(self, "_published_execution_plans", None)
        if cache is None:
            cache = {}
            setattr(self, "_published_execution_plans", cache)
        return cache

    def _prepared_execution_cache(self) -> dict[tuple[str, int, str, str], Any]:
        cache = getattr(self, "_prepared_executions", None)
        if cache is None:
            cache = {}
            setattr(self, "_prepared_executions", cache)
        return cache

    def _local_group_rank(self) -> int:
        ranks = tuple(int(value) for value in self.ep_group_ranks)
        return int(ranks.index(int(self.rank))) if int(self.rank) in ranks else 0

    def _required_payload_roles_for_phase(self, phase: str) -> tuple[str, ...]:
        if str(phase) == "P0":
            return ("hidden_states", "routing_probs")
        if str(phase) == "P1":
            return ("hidden_states",)
        return ()

    def record_phase_payload_completion(
        self,
        *,
        layer_id: str,
        phase: str,
        payload_role: str,
    ) -> tuple[str, ...]:
        releases = self.release_state_ledger.record_payload_completion(
            layer_id=str(layer_id),
            phase=str(phase),
            local_group_rank=self._local_group_rank(),
            payload_role=str(payload_role),
            required_payload_roles=self._required_payload_roles_for_phase(str(phase)),
        )
        if not releases:
            return ()
        self.release_state_ledger.satisfied_release_ids.update(str(item) for item in releases)
        return tuple(str(item) for item in releases)

    def satisfied_release_dependency_ids_for(
        self,
        *,
        layer_id: str,
        phase: str,
        local_group_rank: int | None = None,
    ) -> tuple[str, ...]:
        layer_id = str(layer_id)
        phase = str(phase)
        rank = self._local_group_rank() if local_group_rank is None else int(local_group_rank)
        if phase == "P1":
            prefix = f"release:{layer_id}:p0_inbound_complete:{rank}"
        elif phase == "P2":
            prefix = f"release:{layer_id}:p1_inbound_complete:{rank}"
        else:
            return ()
        return tuple(
            sorted(
                str(item)
                for item in self.release_state_ledger.satisfied_release_ids
                if str(item) == prefix
            )
        )

    def record_execution_outcome(
        self,
        *,
        layer_id: str,
        phase: str,
        payload_role: str,
        outcome: dict[str, object],
    ) -> None:
        self._latest_execution_outcomes.append(
            {
                "layer_id": str(layer_id),
                "phase": str(phase),
                "payload_role": str(payload_role),
                "outcome": dict(outcome),
                "release_ids": list(sorted(self.release_state_ledger.satisfied_release_ids)),
            }
        )

    def _finalize_result_bundle(self) -> ResultBundle | None:
        instrumentation = getattr(self, "runtime_instrumentation", None)
        if instrumentation is None:
            return None
        measurement_sink = getattr(instrumentation, "measurement_sink", None)
        measurement_complete = hasattr(measurement_sink, "snapshot")
        measurement_snapshot = measurement_sink.snapshot() if measurement_complete else None
        mode = str(getattr(self, "_instrumentation_mode", "off") or "off")
        commit_sha = str(getattr(self, "_commit_sha", "") or "")
        git_clean = bool(getattr(self, "_git_clean", False))
        outcomes = list(self._latest_execution_outcomes)
        formal_execution_expected = bool(
            any(
                str(row.get("state", "")).upper() == "EXECUTING"
                for row in list(getattr(self, "prepared_plan_bindings", ()))
            )
            or bool(getattr(self, "_prepared_executions", {}))
        )
        if formal_execution_expected and not outcomes:
            all_work_completed = False
            correctness_status = "invalid"
            status = "failure"
            failure_reason = "missing_execution_outcomes"
        elif outcomes:
            all_work_completed = all(
                bool(dict(item.get("outcome", {})).get("success", False))
                and bool(dict(item.get("outcome", {})).get("all_work_completed", False))
                and not tuple(dict(item.get("outcome", {})).get("unresolved_task_ids", ()))
                for item in outcomes
            )
            correctness_status = "valid" if all_work_completed else "invalid"
            status = "success" if all_work_completed else "failure"
            failure_reason = "" if all_work_completed else "execution_incomplete"
        else:
            all_work_completed = True
            correctness_status = "valid"
            status = "success"
            failure_reason = ""
        summary = {
            "formal_execution_expected": bool(formal_execution_expected),
            "all_work_completed": bool(all_work_completed),
            "fallback_count": 0,
            "timeout_count": 0,
            "check_failure_count": 0,
            "execution_outcome_count": int(len(outcomes)),
            "missing_execution_outcome_count": int(1 if formal_execution_expected and not outcomes else 0),
            "release_id_count": int(len(self.release_state_ledger.satisfied_release_ids)),
            "measurement_event_count": int(getattr(measurement_snapshot, "event_count", 0) or 0) if measurement_snapshot is not None else 0,
        }
        bundle = ResultBundle(
            run_identity=RunIdentity(
                run_id=str(self.run_id),
                pipeline="online",
                claim_scope="formal",
                trace_origin="runtime",
                future_information_mode=str(getattr(self.config, "future_hint_mode", "runtime")),
            ),
            status=status,
            correctness_status=correctness_status,
            performance_status="unknown",
            pipeline="online",
            commit_sha=commit_sha or "unknown",
            git_clean=bool(git_clean),
            instrumentation_mode=mode,
            audit_evidence_level="summary_only",
            measurement_complete=bool(measurement_complete),
            eligibility=EligibilityResult(
                correctness_eligible=bool(all_work_completed),
                performance_eligible=False,
                prediction_evaluation_eligible=False,
                offline_replay_eligible=False,
                reasons=() if all_work_completed else (failure_reason or "execution_incomplete",),
            ),
            summary=summary,
            details={
                "latest_execution_outcomes": outcomes,
                "measurement_summary": {} if measurement_snapshot is None else dict(measurement_snapshot.summary),
                "failure_reason": str(failure_reason),
            },
        )
        self._latest_result_bundle = bundle
        instrumentation.record_result(bundle)
        return bundle

    def _record_instrumentation_measurement(
        self,
        *,
        event_type: str,
        layer_id: str | None,
        phase: str | None,
        started_at_ns: int,
        ended_at_ns: int,
        details: dict[str, object] | None = None,
    ) -> None:
        instrumentation = getattr(self, "runtime_instrumentation", None)
        if instrumentation is None:
            return
        instrumentation.record_measurement(
            MeasurementEvent(
                event_type=str(event_type),
                started_at_ns=int(started_at_ns),
                ended_at_ns=int(ended_at_ns),
                layer_id=None if layer_id is None else str(layer_id),
                phase=None if phase is None else str(phase),
                details=dict(details or {}),
            )
        )

    def _actual_phase_context_from_ready_context(self, *, phase_ctx: PhaseReadyContext) -> ActualPhaseContext:
        return ActualPhaseContext(
            layer_id=str(phase_ctx.layer_id),
            phase=str(phase_ctx.phase),
            world_size=int(len(phase_ctx.ep_group_ranks)),
            rank_space="global",
            layout_digest=str(phase_ctx.canonical_receive_layout_id),
            metadata={"phase_ready_context": phase_ctx.to_dict()},
        )

    def clear_transport(self, *, layer_name: str, phase: str) -> None:
        adapter = getattr(self, "transport_adapter", None)
        if adapter is not None:
            adapter.deactivate(layer_name=layer_name, phase=phase)
        if self._active_transport is None:
            return
        if self._active_transport.get("layer_name") == layer_name and self._active_transport.get("phase") == phase:
            self._active_transport = None

    def record_transport_execution(self, payload: dict[str, Any]) -> None:
        if self.observation_recorder is not None:
            self.observation_recorder.record_transport_execution(dict(payload))

    def _append_heartbeat(self, payload: dict[str, Any]) -> None:
        if self._is_perf_profile():
            return
        if not self.config.executor_heartbeat_path:
            return
        heartbeat_dir = Path(self.config.executor_heartbeat_path)
        heartbeat_dir.mkdir(parents=True, exist_ok=True)
        target = heartbeat_dir / f"heartbeat-rank{self.rank}.jsonl"
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            handle.flush()

    def _timeline(self, event: str, *, layer_name: str, **detail: Any) -> None:
        if self._is_perf_profile():
            return
        row = {
            "ts_us": int(time.time() * 1e6),
            "monotonic_ns": time.monotonic_ns(),
            "event_seq": len(self.control_timeline) + 1,
            "event": event,
            "run_id": self.run_id,
            "forward_epoch": int(self._forward_epoch),
            "step_id": self.step_id,
            "microbatch_id": self.microbatch_id,
            "layer_id": parse_layer_id(layer_name),
            "phase": "P0" if ("p0" in event or "dispatch" in event) else "P1" if ("p1" in event or "combine" in event) else "control",
            "layer": layer_name,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "control_mode": self.config.control_mode,
            "scheduler_mode": self.config.scheduler_mode,
            **detail,
        }
        self.control_timeline.append(row)
        if event in {
            "before_phase_plan",
            "after_phase_plan",
            "before_wave",
            "after_wave",
            "before_payload_collective",
            "after_payload_collective",
            "after_phase",
            "p0_pre_transport_observation_ready",
            "p1_pre_transport_observation_ready",
            "p0_native_dispatch_committed",
        }:
            self._append_heartbeat(row)
            if self.observation_recorder is not None:
                self.observation_recorder.record_heartbeat(row)

    def _record_planning_timing(
        self,
        *,
        layer_name: str,
        phase: str,
        stage: str,
        start_ns: int,
        end_ns: int,
        **detail: Any,
    ) -> float:
        duration_us = max(0.0, float(end_ns - start_ns) / 1000.0)
        if str(stage).startswith("materialize"):
            measurement_event_type = "materialization"
        elif str(stage).startswith("validate"):
            measurement_event_type = "validation"
        elif "publish" in str(stage):
            measurement_event_type = "publish"
        elif "executor" in str(stage) or "transport" in str(stage):
            measurement_event_type = "active_transport"
        else:
            measurement_event_type = "planning"
        self._record_instrumentation_measurement(
            event_type=measurement_event_type,
            layer_id=parse_layer_id(layer_name),
            phase=str(phase),
            started_at_ns=int(start_ns),
            ended_at_ns=int(end_ns),
            details={"stage": str(stage), **detail},
        )
        if self._is_perf_profile():
            counter = self.perf_counters.setdefault(
                str(stage),
                {"count": 0.0, "total_us": 0.0, "max_us": 0.0},
            )
            counter["count"] += 1.0
            counter["total_us"] += float(duration_us)
            counter["max_us"] = max(float(counter["max_us"]), float(duration_us))
            return duration_us
        record = {
            "ts_us": int(time.time() * 1e6),
            "layer_name": layer_name,
            "layer_id": parse_layer_id(layer_name),
            "phase": phase,
            "stage": stage,
            "duration_us": duration_us,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "execution_mode": self.config.execution_mode,
            "control_mode": self.config.control_mode,
            **detail,
        }
        self.planning_timing_records.append(record)
        self._timeline(
            "planning_stage_timing",
            layer_name=layer_name,
            phase_name=phase,
            stage=stage,
            duration_us=duration_us,
            **detail,
        )
        return duration_us

    def _record_hook_timing(
        self,
        *,
        layer_name: str,
        phase: str,
        hook_name: str,
        start_ns: int,
        end_ns: int,
        **detail: Any,
    ) -> float:
        return self._record_planning_timing(
            layer_name=layer_name,
            phase=phase,
            stage=f"hook_{hook_name}",
            start_ns=start_ns,
            end_ns=end_ns,
            **detail,
        )

    def _increment_state_counter_map(self, key: str, item: str) -> None:
        payload = dict(self._runtime_state.read(key, {}) or {})
        payload[str(item)] = int(payload.get(str(item), 0) or 0) + 1
        self._runtime_state.write(key, payload)

    def _register_current_plan_build(self, *, layer_name: str, phase: str, plan_origin: str) -> None:
        build_key = (
            int(self.rank),
            int(self._forward_epoch),
            str(parse_layer_id(layer_name)),
            str(phase),
            str(plan_origin),
        )
        if build_key in self._current_plan_build_keys:
            raise RuntimeError(f"duplicate current plan build detected for {build_key}")
        self._current_plan_build_keys.add(build_key)

    def _record_none_heavy_hook(self, *, layer_name: str, phase: str, hook_name: str, start_ns: int) -> None:
        self._runtime_state.metrics.none_heavy_hook_count = int(self._runtime_state.metrics.none_heavy_hook_count) + 1
        end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase=phase,
            hook_name=hook_name,
            start_ns=start_ns,
            end_ns=end_ns,
            scheduled=False,
            reason="layer_role_none_defensive_entry",
        )
        self._timeline(
            f"{hook_name}_none_heavy_defensive_exit",
            layer_name=layer_name,
            phase_name=phase,
            scheduled=False,
        )

    def _hook_execution_mode(self, *, layer_name: str) -> HookExecutionMode:
        layer_role = self.layer_role_for_name(layer_name)
        if layer_role == "none":
            return "DISABLED"
        if layer_role == "prediction_source":
            return "OBSERVATION_ONLY"
        if layer_role != "selected":
            return "DISABLED"
        if self.config.scheduler_mode in {"native_passthrough_identity", "native_order", "joint_shadow_p0p1"}:
            return "LEGACY_SHADOW"
        if str(self.config.execution_mode) in {"joint_window_async_p2p", "phase_sync_wave"}:
            return "REAL_EXECUTION_WITH_OBSERVATION"
        if self.config.scheduler_mode == "disabled":
            return "OBSERVATION_ONLY"
        return "OBSERVATION_ONLY"

    def _record_dtoh_callsite(
        self,
        *,
        callsite_id: str,
        start_ns: int,
        end_ns: int,
        bytes_if_known: int | None = None,
    ) -> None:
        count_map = dict(self._runtime_state.read("dtoh_callsite_count", {}) or {})
        wall_map = dict(self._runtime_state.read("dtoh_callsite_wall_us", {}) or {})
        byte_map = dict(self._runtime_state.read("dtoh_callsite_bytes", {}) or {})
        count_map[str(callsite_id)] = int(count_map.get(str(callsite_id), 0) or 0) + 1
        wall_map[str(callsite_id)] = float(wall_map.get(str(callsite_id), 0.0) or 0.0) + max(
            0.0, float(end_ns - start_ns) / 1000.0
        )
        if bytes_if_known is not None:
            byte_map[str(callsite_id)] = int(byte_map.get(str(callsite_id), 0) or 0) + int(bytes_if_known)
        self._runtime_state.write("dtoh_callsite_count", count_map)
        self._runtime_state.write("dtoh_callsite_wall_us", wall_map)
        self._runtime_state.write("dtoh_callsite_bytes", byte_map)

    def _finalize_dispatch_observation(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        self._runtime_state.metrics.observation_finalize_dispatch_count = int(
            self._runtime_state.metrics.observation_finalize_dispatch_count
        ) + 1
        self._runtime_state.write(
            "dispatch_finalize_shape",
            list(hidden_states.shape) if isinstance(hidden_states, torch.Tensor) else None,
        )
        self._runtime_state.write(
            "dispatch_finalize_dispatcher",
            str(type(dispatcher).__name__) if dispatcher is not None else "",
        )

    def _finalize_combine_observation(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        self._runtime_state.metrics.observation_finalize_combine_count = int(
            self._runtime_state.metrics.observation_finalize_combine_count
        ) + 1
        self._runtime_state.write(
            "combine_finalize_shape",
            list(hidden_states.shape) if isinstance(hidden_states, torch.Tensor) else None,
        )
        self._runtime_state.write(
            "combine_finalize_dispatcher",
            str(type(dispatcher).__name__) if dispatcher is not None else "",
        )

    def before_prediction_source_dispatch(
        self,
        *,
        layer_name: str,
        dispatcher: Any,
        packed_hidden_states: Any,
        packed_probs: Any,
    ) -> None:
        hook_start_ns = time.monotonic_ns()
        if self.layer_role_for_name(layer_name) != "prediction_source":
            return
        self._runtime_state.metrics.prediction_source_p0_hook_count = int(
            self._runtime_state.metrics.prediction_source_p0_hook_count
        ) + 1
        sync_fn = getattr(dispatcher, "_maybe_dtoh_and_synchronize", None)
        if callable(sync_fn):
            try:
                tokens_per_expert = getattr(dispatcher, "tokens_per_expert", None)
                dtoh_start_ns = time.monotonic_ns()
                synchronized = sync_fn("before_ep_alltoall", tokens_per_expert)
                dtoh_end_ns = time.monotonic_ns()
                self._record_dtoh_callsite(
                    callsite_id="DTOH_P0_DISPATCHER_SYNC",
                    start_ns=dtoh_start_ns,
                    end_ns=dtoh_end_ns,
                )
                if synchronized is not None:
                    dispatcher.tokens_per_expert = synchronized
            except Exception:
                pass
        phase_ctx = self._build_phase_ready_context_from_dispatcher(
            layer_name=layer_name,
            phase="P0",
            dispatcher=dispatcher,
            packed_tensors=tuple(
                tensor for tensor in (packed_hidden_states, packed_probs) if isinstance(tensor, torch.Tensor)
            ),
        )
        pretransport = self._capture_pretransport_traffic_observation(phase_ctx=phase_ctx)
        actual_p0_full_row_matrix = self._gather_actual_p0_full_row_matrix(
            layer_name=layer_name,
            observation=pretransport,
            device=self._matrix_device(packed_hidden_states),
        )
        if self._should_generate_runtime_prediction():
            self._record_prediction_for_dispatch(
                layer_name=layer_name,
                phase_ctx=phase_ctx,
                observation=pretransport,
                actual_p0_full_row_matrix=actual_p0_full_row_matrix,
                device=self._matrix_device(packed_hidden_states),
            )
        hook_end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase="P0",
            hook_name="before_token_dispatch_total",
            start_ns=hook_start_ns,
            end_ns=hook_end_ns,
            scheduled=False,
            reason="prediction_source_only",
        )

    def _matrix_device(self, candidate: Any) -> torch.device:
        if isinstance(candidate, torch.Tensor):
            return candidate.device
        return torch.device("cpu")

    def _runtime_topology_dict(self) -> dict[str, Any]:
        return {
            "global_rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "node_index": -1,
            "hostname_digest": digest_text(self.hostname),
            "device_index": int(self.local_rank),
            "ep_group_rank": int(tuple(int(v) for v in self.ep_group_ranks).index(int(self.rank))) if int(self.rank) in tuple(int(v) for v in self.ep_group_ranks) else 0,
        }

    def _dispatcher_expert_placement_hash(self, dispatcher: Any) -> str:
        return digest_text(
            stable_hash(
                {
                    "placement_mode": "megatron_native_ep",
                    "ep_group_ranks": list(int(v) for v in self.ep_group_ranks),
                    "ep_group_size": len(self.ep_group_ranks),
                    "dispatcher_class": type(dispatcher).__name__,
                }
            )
        )

    def _build_phase_ready_context_from_dispatcher(
        self,
        *,
        layer_name: str,
        phase: str,
        dispatcher: Any,
        packed_tensors: tuple[torch.Tensor, ...],
        p2_hint: Any | None = None,
    ) -> PhaseReadyContext:
        return build_phase_ready_context(
            PhaseContextBuildRequest(
                plan_key=self._plan_key(layer_name, phase),
                runtime_identity=RuntimeIdentity(
                    run_id=self.run_id,
                    forward_epoch=int(self._forward_epoch),
                    layer_id=parse_layer_id(layer_name),
                    layer_name=layer_name,
                    global_rank=self.rank,
                    local_rank=self.local_rank,
                    ep_group_ranks=self.ep_group_ranks,
                    ep_group_root_rank=self.ep_group_root_global_rank,
                ),
                topology=self._runtime_topology_dict(),
                dispatcher_snapshot=DispatcherSnapshot(
                    dispatcher_class=type(dispatcher).__name__,
                    dispatcher_fingerprint={"dispatcher_class": type(dispatcher).__name__},
                    expert_placement_hash=self._dispatcher_expert_placement_hash(dispatcher),
                    input_splits=tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "input_splits", None))[: len(self.ep_group_ranks)]),
                    output_splits=tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "output_splits", None))[: len(self.ep_group_ranks)]),
                ),
                payload_contract=PhasePayloadContract(
                    phase=phase,
                    payload_roles=("hidden_states", "routing_probs") if phase == "P0" else ("hidden_states",),
                    atomic_submit=(phase == "P0"),
                ),
                packed_tensors=packed_tensors,
                control_mode=self.config.control_mode,
                release_state="ready",
                demand_known_at="router_ready",
                payload_exists=True,
                p2_hint=p2_hint,
            )
        )

    def _capture_pretransport_traffic_observation(
        self,
        *,
        phase_ctx: PhaseReadyContext,
    ) -> PreTransportTrafficObservation:
        group_ranks = tuple(int(v) for v in phase_ctx.ep_group_ranks)
        group_rank = group_ranks.index(int(phase_ctx.global_rank)) if int(phase_ctx.global_rank) in group_ranks else 0
        send_splits_rows = tuple(int(v) for v in phase_ctx.send_splits)
        recv_splits_rows = tuple(int(v) for v in phase_ctx.recv_splits)
        valid = str(phase_ctx.phase) == "P0" and len(send_splits_rows) == len(group_ranks) and len(recv_splits_rows) == len(group_ranks)
        error = None if valid else "invalid_phase_or_split_shape"
        return PreTransportTrafficObservation(
            run_id=str(self.run_id),
            forward_epoch=int(phase_ctx.forward_epoch),
            microbatch_id=str(self.microbatch_id),
            layer_id=int(parse_layer_id(phase_ctx.layer_name)) if str(parse_layer_id(phase_ctx.layer_name)).isdigit() else -1,
            phase=str(phase_ctx.phase),
            global_rank=int(phase_ctx.global_rank),
            group_rank=int(group_rank),
            group_global_ranks=group_ranks,
            send_splits_rows=send_splits_rows,
            recv_splits_rows=recv_splits_rows,
            local_p0_row=send_splits_rows,
            local_send_rows=int(sum(send_splits_rows)),
            local_recv_rows=int(sum(recv_splits_rows)),
            source="phase_ready_context_dispatcher_splits",
            captured_before_transport=True,
            valid=bool(valid),
            error=error,
        )

    def _bundle_bytes_per_row(self, *, phase_ctx: PhaseReadyContext) -> int:
        max_row_count = max((int(bundle.outgoing_segment.row_count) for bundle in phase_ctx.transport_bundles if int(bundle.outgoing_segment.row_count) > 0), default=0)
        if max_row_count <= 0:
            return 1
        for bundle in phase_ctx.transport_bundles:
            row_count = int(bundle.outgoing_segment.row_count)
            if row_count <= 0:
                continue
            total_bytes = int(sum(int(payload.payload_byte_count) for payload in bundle.payload_slices))
            if total_bytes > 0:
                return max(1, int(round(total_bytes / row_count)))
        return 1

    def _gather_actual_p0_full_row_matrix(
        self,
        *,
        layer_name: str,
        observation: PreTransportTrafficObservation,
        device: torch.device,
    ) -> tuple[tuple[int, ...], ...]:
        local_prepare_start_ns = time.monotonic_ns()
        local_row = tuple(int(v) for v in observation.local_p0_row)
        local_total = int(sum(local_row))
        if local_total != int(sum(observation.send_splits_rows)):
            raise RuntimeError(f"pre-transport local send mismatch for {layer_name}: local_row={local_row} send_splits={observation.send_splits_rows}")
        row_tensor = torch.tensor(local_row, dtype=torch.int64, device=device)
        local_prepare_end_ns = time.monotonic_ns()
        if len(local_row) <= 1:
            matrix = (local_row,)
            gather_count = 0
            collective_start_ns = local_prepare_end_ns
            collective_end_ns = local_prepare_end_ns
            dtoh_decode_start_ns = local_prepare_end_ns
            dtoh_decode_end_ns = local_prepare_end_ns
        elif dist.is_available() and dist.is_initialized():
            collective_start_ns = time.monotonic_ns()
            gathered = [torch.empty_like(row_tensor) for _ in range(len(local_row))]
            dist.all_gather(gathered, row_tensor, group=self.ep_process_group)
            collective_end_ns = time.monotonic_ns()
            dtoh_decode_start_ns = time.monotonic_ns()
            matrix = tuple(tuple(int(v) for v in item.detach().cpu().tolist()) for item in gathered)
            dtoh_decode_end_ns = time.monotonic_ns()
            gather_count = 1
        else:
            matrix = tuple(local_row for _ in range(len(local_row)))
            gather_count = 0
            collective_start_ns = local_prepare_end_ns
            collective_end_ns = local_prepare_end_ns
            dtoh_decode_start_ns = local_prepare_end_ns
            dtoh_decode_end_ns = local_prepare_end_ns
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="p0_matrix_local_prepare",
            start_ns=local_prepare_start_ns,
            end_ns=local_prepare_end_ns,
        )
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="p0_matrix_collective",
            start_ns=collective_start_ns,
            end_ns=collective_end_ns,
            collective_count=int(gather_count),
        )
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="p0_matrix_dtoh_decode",
            start_ns=dtoh_decode_start_ns,
            end_ns=dtoh_decode_end_ns,
        )
        if dtoh_decode_end_ns > dtoh_decode_start_ns:
            self._record_dtoh_callsite(
                callsite_id="DTOH_P0_MATRIX_DECODE",
                start_ns=dtoh_decode_start_ns,
                end_ns=dtoh_decode_end_ns,
                bytes_if_known=int(row_tensor.numel() * row_tensor.element_size() * max(1, len(local_row))),
            )
        matrix_total = int(sum(sum(int(v) for v in row) for row in matrix))
        self._runtime_state.write("planning_traffic_source", "pre_transport_phase_ready_context")
        self._runtime_state.write("pre_transport_observation_valid", bool(observation.valid))
        self._runtime_state.write("captured_before_transport", bool(observation.captured_before_transport))
        self._runtime_state.write("dispatcher_send_splits", tuple(int(v) for v in observation.send_splits_rows))
        self._runtime_state.write("dispatcher_recv_splits", tuple(int(v) for v in observation.recv_splits_rows))
        self._runtime_state.write("local_p0_row", local_row)
        self._runtime_state.write("actual_p0_total_rows", int(matrix_total))
        self._runtime_state.write("p0_traffic_matrix_gather_count", int(gather_count))
        self._runtime_state.write("prediction_extra_collective_count", 0)
        if (int(sum(observation.send_splits_rows)) > 0 or int(sum(observation.recv_splits_rows)) > 0) and matrix_total <= 0:
            self._write_traffic_source_mismatch(
                layer_name=layer_name,
                observation=observation,
                global_matrix=matrix,
                transport_started=False,
            )
            raise RuntimeError(f"traffic_source_mismatch for {layer_name}: nonzero dispatcher splits but zero actual_p0_full_row_matrix")
        local_col_total = int(sum(int(matrix[src][observation.group_rank]) for src in range(len(matrix)))) if matrix else 0
        if int(sum(observation.recv_splits_rows)) != local_col_total:
            raise RuntimeError(
                f"pre-transport recv mismatch for {layer_name}: recv_total={sum(observation.recv_splits_rows)} col_total={local_col_total} group_rank={observation.group_rank}"
            )
        return matrix

    def _write_traffic_source_mismatch(
        self,
        *,
        layer_name: str,
        observation: PreTransportTrafficObservation,
        global_matrix: tuple[tuple[int, ...], ...],
        transport_started: bool,
    ) -> None:
        target_dir = Path(self.config.executor_heartbeat_path) if self.config.executor_heartbeat_path else Path("outputs/distributed/runtime_traffic_source_mismatch")
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "forward_epoch": int(self._forward_epoch),
            "microbatch_id": self.microbatch_id,
            "layer_name": layer_name,
            "layer_id": parse_layer_id(layer_name),
            "global_rank": int(self.rank),
            "group_rank": int(observation.group_rank),
            "dispatcher_send_splits": list(observation.send_splits_rows),
            "dispatcher_recv_splits": list(observation.recv_splits_rows),
            "phase_ready_context_send_splits": list(observation.send_splits_rows),
            "phase_ready_context_recv_splits": list(observation.recv_splits_rows),
            "local_p0_row": list(observation.local_p0_row),
            "global_p0_matrix": [list(row) for row in global_matrix],
            "runtime_observation_p0": (
                self._pending_p0.get(layer_name).to_dict() if self._pending_p0.get(layer_name) is not None else None
            ),
            "planning_stage": "before_token_dispatch",
            "transport_started": bool(transport_started),
        }
        (target_dir / f"traffic_source_mismatch_rank{self.rank}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _next_layer_id(self, layer_name: str) -> str:
        layer_id = parse_layer_id(layer_name)
        try:
            return str(int(layer_id) + 1)
        except ValueError:
            return layer_id

    def _online_p2_predictor_name(self) -> str:
        return str(getattr(self.config, "online_p2_predictor", "copy_current_dispatch") or "copy_current_dispatch")

    def _build_online_predictor(self):
        return PredictionRegistry.create(self._online_p2_predictor_name(), {"alpha": 0.5}, usage="runtime")

    def _predict_dispatch_matrix(
        self,
        *,
        layer_id: str,
        next_layer_id: str,
        current_dispatch_matrix: tuple[tuple[int, ...], ...],
        previous_dispatch_matrix: tuple[tuple[int, ...], ...] | None,
        fallback: bool = False,
    ):
        predictor = self._build_online_predictor()
        context = TrafficHistoryContext(
            identity=PredictionIdentity(
                request_id=f"{self.run_id}:{self.microbatch_id}:{layer_id}:{next_layer_id}",
                run_id=self.run_id,
                forward_id=str(self._forward_epoch),
                source_layer_id=str(layer_id),
                target_layer_id=str(next_layer_id),
            ),
            current_dispatch_rows=current_dispatch_matrix,
            current_return_rows=tuple(
                tuple(int(current_dispatch_matrix[col][row]) for col in range(len(current_dispatch_matrix)))
                for row in range(len(current_dispatch_matrix))
            ),
            history_dispatch_rows=(() if previous_dispatch_matrix is None else (previous_dispatch_matrix,)),
            world_size=len(current_dispatch_matrix),
        )
        prediction = predictor.predict(context)
        return RuntimePredictionCompatResult(
            predictor_id=str(prediction.hint.predictor_id),
            matrix=prediction.hint.target_dispatch_rows,
            matrix_digest=stable_hash([list(row) for row in prediction.hint.target_dispatch_rows]),
            predictor_version="v1",
            confidence=float(prediction.hint.confidence or 0.0),
            evaluation_eligible=not bool(prediction.hint.oracle),
            is_oracle=bool(prediction.hint.oracle),
            valid=True,
            error="",
            fallback=bool(fallback),
        )

    def _resolved_online_policy_family(self) -> str:
        resolved = resolve_online_policy_config(self.config)
        if resolved is None:
            return ""
        return str(getattr(resolved.spec, "family", ""))

    def _resolved_online_policy_capabilities(self):
        return self._resolved_policy_capabilities_cache

    def _policy_supports_runtime_prediction(self) -> bool:
        return bool(self._cross_layer_prediction_enabled_cache)

    def _policy_uses_joint_window_plan(self) -> bool:
        return bool(self._joint_window_enabled_cache)

    def _should_generate_runtime_prediction(self) -> bool:
        return self._policy_supports_runtime_prediction()

    def _policy_supports_target_layer_preplanning(self) -> bool:
        return bool(self._target_preplanning_enabled_cache)

    def _target_plan_key(self, *, layer_name: str) -> TargetPlanKey:
        return TargetPlanKey(
            run_id=str(self.run_id),
            forward_epoch=int(self._forward_epoch),
            microbatch_id=str(self.microbatch_id),
            target_layer_id=str(parse_layer_id(layer_name)),
        )

    def _ensure_target_planner_runtime(self) -> None:
        if not self._policy_supports_target_layer_preplanning():
            return
        if self.target_plan_store is None:
            self.target_plan_store = TargetPlanStore()
        if self.control_communication_lane is None:
            self.control_communication_lane = GlooControlCommunicationLane(
                rank=int(self.rank),
                world_size=int(len(self.ep_group_ranks) or 1),
                root_rank=int(self.ep_group_root_global_rank),
                process_group=self.target_plan_control_group,
                group_ranks=tuple(int(v) for v in self.ep_group_ranks),
            )
        if self.target_planner_service is None:
            self.target_planner_service = TargetLayerPlannerService(
                store=self.target_plan_store,
            )
            self.target_planner_service.start()

    def _cleanup_target_plan_runtime(self) -> None:
        if self.target_planner_service is not None:
            self.target_planner_service.shutdown()
        if self.target_plan_store is not None:
            self.target_plan_store.shutdown()
        self.control_communication_lane = None
        self.target_plan_control_group = None
        self.target_plan_control_group_handle = None
        self._ready_target_plan_candidates.clear()
        self._expected_publication_slots.clear()
        self._terminal_publication_slots.clear()
        self._published_publication_slots.clear()
        self._poll_attempts.clear()
        self._execution_plan_cache().clear()
        self._prepared_execution_cache().clear()

    @staticmethod
    def _target_plan_key_from_slot(slot: Any) -> TargetPlanKey:
        return TargetPlanKey(
            run_id=str(slot.run_id),
            forward_epoch=int(slot.forward_generation),
            microbatch_id=str(slot.microbatch_id),
            target_layer_id=str(slot.target_layer_id),
        )

    def _pump_target_planner_publications(self) -> None:
        if self.target_planner_service is None:
            return
        for ready in self.target_planner_service.drain_ready_publications():
            candidate = self.target_planner_service.local_publication_candidate(ready)
            if candidate is None:
                continue
            self._ready_target_plan_candidates[str(candidate.slot.semantic_digest())] = (ready, candidate)
            if len(self.ep_group_ranks) <= 1 or not dist.is_available() or not dist.is_initialized():
                self._poll_target_plan_slot(target_layer_id=str(ready.request.target_layer_id), safe_point="single_rank_autopublish")

    def _poll_target_plan_slot(self, *, target_layer_id: str, safe_point: str | None = None) -> None:
        if self.control_communication_lane is None or self.target_plan_store is None:
            return
        slot_key = (str(self.run_id), int(self._forward_epoch), str(self.microbatch_id), str(target_layer_id))
        slot = self._expected_publication_slots.get(slot_key)
        if slot is None:
            return
        slot_digest = str(slot.semantic_digest())
        if slot_digest in self._terminal_publication_slots or slot_digest in self._published_publication_slots:
            return
        if safe_point is not None and (slot_digest, str(safe_point)) in self._poll_attempts:
            return
        if safe_point is not None:
            self._poll_attempts.add((slot_digest, str(safe_point)))
        ready_pair = self._ready_target_plan_candidates.get(slot_digest)
        local_candidate = None if ready_pair is None else ready_pair[1]
        if local_candidate is None and self.target_planner_service is not None:
            local_candidate = self.target_planner_service.publication_state_for_slot(slot)
        poll_result = self.control_communication_lane.poll(slot, local_candidate)
        if poll_result.status is PublicationPollStatus.NOT_READY:
            return
        if poll_result.status in {PublicationPollStatus.CANCELLED, PublicationPollStatus.EXPIRED, PublicationPollStatus.FAILED, PublicationPollStatus.SLOT_MISMATCH}:
            self._terminal_publication_slots.add(slot_digest)
            target_key = self._target_plan_key_from_slot(slot)
            if self.target_planner_service is not None:
                self.target_planner_service.cancel_slot(
                    slot,
                    final_status=str(poll_result.status.value).upper(),
                )
            self.target_plan_store.clear_expected_publication(target_key)
            self.target_plan_store.close_key_if_unclaimed(
                target_key,
                final_status="FAILED" if poll_result.status in {PublicationPollStatus.FAILED, PublicationPollStatus.SLOT_MISMATCH} else "CANCELLED",
                execution_origin=f"lane:{poll_result.status.value}",
            )
            self._ready_target_plan_candidates.pop(slot_digest, None)
            self._published_publication_slots.discard(slot_digest)
            return
        if ready_pair is None:
            return
        ready = ready_pair[0]
        canonical_payload = dict(poll_result.canonical_payload)
        metadata_payload = dict(canonical_payload.get("metadata") or {})
        plan_payload = dict(canonical_payload.get("plan") or metadata_payload.get("plan") or {})
        if not plan_payload:
            self.target_plan_store.close_key_if_unclaimed(
                ready.key,
                final_status="FAILED",
                execution_origin="lane:missing_plan_payload",
            )
            return
        published = TargetLayerPreparedJointPlan.from_dict(plan_payload)
        published_plan = None
        if getattr(self, "plan_publisher", None) is not None and published.window_plan is not None:
            published_plan = self.plan_publisher.build(
                publication_slot=slot.semantic_payload(),
                window_plan=published.window_plan,
            )
        publish_result = self.target_plan_store.publish_if_current(token=ready.token, plan=published)
        if publish_result.status not in {"PUBLISHED", "ALREADY_PUBLISHED_SAME"}:
            self._timeline(
                "target_plan_publish_rejected",
                target_layer_id=str(ready.key.target_layer_id),
                status=str(publish_result.status),
                logical_plan_digest=str(published.logical_plan_digest),
            )
            return
        if published_plan is not None:
            self._execution_plan_cache()[self.target_plan_store._key(ready.key)] = published_plan
        self._ready_target_plan_candidates.pop(slot_digest, None)
        self._published_publication_slots.add(slot_digest)
        self._store_target_planner_predictions(ready=ready)
        self._timeline(
            "target_plan_ready",
            layer_name=str(ready.request.target_layer_id),
            target_layer_id=str(ready.request.target_layer_id),
            logical_plan_digest=str(published.logical_plan_digest),
            h1_digest=str(ready.bundle.h1.matrix_digest),
            h2_digest=str(ready.bundle.h2.matrix_digest),
            planner_wall_us=float(ready.metrics.planner_wall_us),
            publish_status=str(publish_result.status),
        )

    def _store_target_planner_predictions(self, *, ready) -> None:
        from rs.runtime.online.megatron_ep.prediction.contracts import ActiveNextDispatchPrediction, PredictedTrafficMatrix

        predicted_dispatch_by_layer = dict(self._runtime_state.read("predicted_dispatch_by_layer", {}) or {})

        def _to_predicted(prediction) -> PredictedTrafficMatrix:
            matrix = tuple(tuple(int(value) for value in row) for row in prediction.matrix_rows)
            return PredictedTrafficMatrix(
                predictor_name=str(prediction.predictor),
                predictor_version="v1",
                source_layer_id=str(prediction.source_layer_id),
                predicted_layer_id=str(prediction.target_layer_id),
                matrix=matrix,
                matrix_digest=str(prediction.matrix_digest),
                total_bytes=int(matrix_remote_bytes(matrix)),
                nonzero_edge_count=int(matrix_nonzero_remote_edge_count(matrix)),
                confidence=float(prediction.confidence),
                is_oracle=False,
                evaluation_eligible=True,
                created_at_phase="P0",
                valid=True,
                error="",
            )

        h1_prediction = _to_predicted(ready.bundle.h1)
        predicted_dispatch_by_layer[str(ready.bundle.h1.target_layer_id)] = h1_prediction.to_dict()
        h2_prediction = _to_predicted(ready.bundle.h2)
        predicted_dispatch_by_layer[str(ready.bundle.h2.target_layer_id)] = h2_prediction.to_dict()
        self._runtime_state.write("predicted_dispatch_by_layer", predicted_dispatch_by_layer)
        self._increment_state_counter_map("predict_count_by_layer", str(ready.request.source_layer_id))

        active_prediction = ActiveNextDispatchPrediction(
            source_layer_id=str(ready.bundle.h1.source_layer_id),
            target_layer_id=str(ready.bundle.h1.target_layer_id),
            forecast_matrix=h1_prediction.matrix,
            matrix_digest=str(h1_prediction.matrix_digest),
            predictor_name=str(h1_prediction.predictor_name),
            predictor_version=str(h1_prediction.predictor_version),
            confidence=float(h1_prediction.confidence),
            evaluation_eligible=bool(h1_prediction.evaluation_eligible),
            is_oracle=bool(h1_prediction.is_oracle),
            created_at_phase="P0",
            created_at_stage="target_planner_worker",
            prediction_time_us=max(0.0, float(ready.bundle.h1.prediction_us)),
            valid=True,
            error="",
        )
        self._runtime_state.write("active_next_dispatch_prediction", active_prediction.to_dict())
        self._runtime_state.write("latest_predictor_name", str(h1_prediction.predictor_name))
        self._runtime_state.write("latest_prediction_digest", str(h1_prediction.matrix_digest))
        self._runtime_state.write("latest_prediction_target_layer_id", str(ready.bundle.h1.target_layer_id))
        self._runtime_state.write("latest_prediction_matrix_source", "target_planner_worker_h1")
        self._runtime_state.write("latest_prediction_row_sums", [int(sum(row)) for row in h1_prediction.matrix])
        self._runtime_state.write(
            "latest_prediction_col_sums",
            [
                int(sum(h1_prediction.matrix[row_idx][col_idx] for row_idx in range(len(h1_prediction.matrix))))
                for col_idx in range(len(h1_prediction.matrix[0]) if h1_prediction.matrix else 0)
            ],
        )

    def _agree_target_plan_payload(self, payload: dict[str, Any]) -> str:
        digest = str(payload.get("logical_plan_digest", ""))
        if not dist.is_available() or not dist.is_initialized() or len(self.ep_group_ranks) <= 1:
            return digest
        group = self.target_plan_control_group if self.target_plan_control_group is not None else self.ep_process_group
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        device = torch.device("cpu")
        local_len = torch.tensor([len(encoded)], dtype=torch.int64, device=device)
        world_size = int(len(self.ep_group_ranks) or dist.get_world_size(group=group))
        gathered_lens = [torch.empty_like(local_len) for _ in range(world_size)]
        dist.all_gather(gathered_lens, local_len, group=group)
        max_len = max(int(item.item()) for item in gathered_lens)
        padded = torch.zeros(max_len, dtype=torch.uint8, device=device)
        if encoded:
            padded[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
        gathered_payloads = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered_payloads, padded, group=group)
        decoded = []
        for length_tensor, bytes_tensor in zip(gathered_lens, gathered_payloads, strict=True):
            length = int(length_tensor.item())
            decoded.append(bytes(bytes_tensor[:length].tolist()).decode("utf-8"))
        if len(set(decoded)) != 1:
            raise RuntimeError(
                f"target plan agreement mismatch rank={self.rank} payloads={decoded}"
            )
        return digest

    def _build_release_batch_tasks_from_plan(
        self,
        *,
        plan: PhaseExecutionPlan,
        tensor_role: str,
    ) -> list[ReleaseBatchTask]:
        tasks: list[ReleaseBatchTask] = []
        previous_task_id = ""
        peer_sequence = 0
        for wave in plan.waves:
            for task in wave.bucket_tasks:
                payload = next((item for item in task.payload_slices if item.tensor_role == tensor_role), None)
                if payload is None or int(payload.row_count) <= 0 or int(task.src_rank) == int(task.dst_rank):
                    continue
                deps = (previous_task_id,) if previous_task_id else ()
                tasks.append(
                    ReleaseBatchTask(
                        task_id=str(task.task_id),
                        phase=str(plan.phase),
                        src_rank=int(task.src_rank),
                        dst_rank=int(task.dst_rank),
                        row_count=int(payload.row_count),
                        sender_offset=int(payload.sender_offset_rows),
                        receiver_offset=int(payload.receiver_offset_rows),
                        tensor_role=str(tensor_role),
                        peer_sequence=int(peer_sequence),
                        dependency_ids=deps,
                        plan_digest=str(plan.plan_hash),
                        plan_version=int((plan.metrics or {}).get("plan_version", 1) or 1),
                    )
                )
                previous_task_id = str(task.task_id)
                peer_sequence += 1
        return tasks

    @staticmethod
    def _residualize_suffix_tasks(
        *,
        candidate_tasks: list[ReleaseBatchTask],
        frozen_tasks: tuple[ReleaseBatchTask, ...],
    ) -> list[ReleaseBatchTask]:
        frozen_ends: dict[tuple[int, int], int] = {}
        for task in frozen_tasks:
            edge = (int(task.src_rank), int(task.dst_rank))
            frozen_ends[edge] = max(
                int(frozen_ends.get(edge, 0)),
                int(task.sender_offset) + int(task.row_count),
            )
        residual: list[ReleaseBatchTask] = []
        for task in candidate_tasks:
            edge = (int(task.src_rank), int(task.dst_rank))
            frozen_end = int(frozen_ends.get(edge, 0))
            start = int(task.sender_offset)
            end = int(task.sender_offset) + int(task.row_count)
            if end <= frozen_end:
                continue
            if start < frozen_end:
                shrink = int(frozen_end - start)
                task = replace(
                    task,
                    sender_offset=int(task.sender_offset) + shrink,
                    receiver_offset=int(task.receiver_offset) + shrink,
                    row_count=int(task.row_count) - shrink,
                )
            if int(task.row_count) > 0:
                residual.append(task)
        return residual

    def handle(self, event: RuntimeEvent) -> RuntimeDecision:
        if isinstance(event, ForwardBeginEvent):
            self.begin_forward(forward_epoch=event.forward_epoch)
            return RuntimeDecision(action="forward_begin")
        if isinstance(event, DispatchReadyEvent):
            if event.layer_role == "prediction_source":
                self.before_prediction_source_dispatch(
                    layer_name=event.layer_name,
                    dispatcher=event.dispatcher,
                    packed_hidden_states=event.packed_hidden_states,
                    packed_probs=event.packed_probs,
                )
            else:
                self.before_token_dispatch(
                    layer_name=event.layer_name,
                    dispatcher=event.dispatcher,
                    packed_hidden_states=event.packed_hidden_states,
                    packed_probs=event.packed_probs,
                )
                self.mark_token_dispatch_committed(layer_name=event.layer_name)
            return RuntimeDecision(action="dispatch_ready", details={"layer_role": event.layer_role})
        if isinstance(event, DispatchCompleteEvent):
            if event.layer_role != "prediction_source":
                self.capture_phase_transport_output(
                    layer_name=event.layer_name,
                    phase="P0",
                    result=event.result,
                    dispatcher=event.dispatcher,
                )
                self.after_token_dispatch(layer_name=event.layer_name)
            return RuntimeDecision(action="dispatch_complete", details={"layer_role": event.layer_role})
        if isinstance(event, DispatchFailedEvent):
            self._active_transport = None
            self.release_state_ledger.reset(
                run_id=str(self.run_id),
                forward_generation=int(self._forward_epoch),
                microbatch_id=str(self.microbatch_id),
            )
            return RuntimeDecision(action="dispatch_failed", details={"layer_role": event.layer_role, "error": type(event.error).__name__})
        if isinstance(event, CombineReadyEvent):
            self.before_token_combine(
                layer_name=event.layer_name,
                dispatcher=event.dispatcher,
                packed_hidden_states=event.packed_hidden_states,
            )
            return RuntimeDecision(action="combine_ready")
        if isinstance(event, CombineCompleteEvent):
            self.capture_phase_transport_output(
                layer_name=event.layer_name,
                phase="P1",
                result=event.result,
                dispatcher=event.dispatcher,
            )
            self.after_token_combine(layer_name=event.layer_name)
            return RuntimeDecision(action="combine_complete")
        if isinstance(event, CombineFailedEvent):
            self._active_transport = None
            self.release_state_ledger.reset(
                run_id=str(self.run_id),
                forward_generation=int(self._forward_epoch),
                microbatch_id=str(self.microbatch_id),
            )
            return RuntimeDecision(action="combine_failed", details={"error": type(event.error).__name__})
        if isinstance(event, ForwardEndEvent):
            self._finalize_result_bundle()
            self.end_forward()
            return RuntimeDecision(action="forward_end")
        if isinstance(event, ForwardFailedEvent):
            self._finalize_result_bundle()
            self.end_forward()
            return RuntimeDecision(action="forward_failed", details={"error": type(event.error).__name__})
        raise TypeError(f"unsupported runtime event: {type(event).__name__}")

    def _agree_late_suffix(
        self,
        *,
        key: TargetPlanKey,
        frontier: Any,
        residual_digest: str,
        replacement_tasks: list[ReleaseBatchTask],
        new_plan_digest: str,
        release_epoch: int,
    ) -> dict[str, Any]:
        payload = {
            "key": key.to_dict(),
            "release_epoch": int(release_epoch),
            "frontier_digest": str(frontier.frontier_digest()),
            "residual_digest": str(residual_digest),
            "replacement_suffix_digest": stable_hash(
                [
                    (
                        str(task.task_id),
                        int(task.src_rank),
                        int(task.dst_rank),
                        int(task.row_count),
                        int(task.sender_offset),
                        int(task.receiver_offset),
                        int(task.peer_sequence),
                    )
                    for task in replacement_tasks
                ]
            ),
            "new_plan_digest": str(new_plan_digest),
        }
        self._agree_target_plan_payload(payload)
        return {"agreed": True, "payload": payload}

    def _compile_async_phase_from_logical_plan(
        self,
        *,
        logical_plan: Any,
        layer_name: str,
        phase: str,
        local_context: PhaseReadyContext,
        matrix: tuple[tuple[int, ...], ...],
        plan_origin: str,
        plan_version: int,
    ) -> PhaseExecutionPlan:
        global_contexts = reconstruct_global_phase_contexts_from_byte_matrix(
            local_context=local_context,
            matrix=matrix,
            matrix_unit="rows",
        )
        compiled_local_context = next(
            (context for context in global_contexts if int(context.global_rank) == int(local_context.global_rank)),
            local_context,
        )
        canonical_tasks = build_phase_canonical_tasks(
            phase=str(phase),
            matrix_rows=matrix,
            bucket_rows=int(self.config.bucket_rows),
        )
        prepared_wrapper = PreparedWindowPlan(
            window_key=stable_hash({"layer_name": str(layer_name), "phase": str(phase), "plan_origin": str(plan_origin)})[:16],
            forecast_digest="",
            logical_plan=logical_plan,
            created_at_layer_id=str(parse_layer_id(layer_name)),
            applies_from_layer_id=str(parse_layer_id(layer_name)),
            execution_capability_required="joint_window_async_p2p",
            forecast_matrix=(),
        )
        compilation = compile_schedule(
            PlanCompilationRequest(
                logical_plan=logical_plan,
                local_context=compiled_local_context,
                global_contexts=global_contexts,
                canonical_tasks=canonical_tasks,
                phase=str(phase),
                tensor_role="hidden_states" if str(phase) == "P1" else "dispatch_bundle",
                rank_context={
                    "global_rank": int(compiled_local_context.global_rank),
                    "local_rank": int(compiled_local_context.local_rank),
                },
                compilation_options=CompilationOptions(
                    bucket_rows=int(self.config.bucket_rows),
                    p0_weight=float(self.config.p0_weight),
                    p1_reservation_weight=float(self.config.p1_reservation_weight),
                    p2_hint_weight=float(self.config.p2_hint_weight),
                    debug_trace=not self._is_perf_profile(),
                    invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
                    legacy_compiler_bridge=bool(getattr(self.config, "legacy_compiler_bridge", False)),
                ),
                prepared_plan=prepared_wrapper,
            )
        )
        compiled = compilation.execution_plan
        self._runtime_state.write("compiler_id", str(compilation.audit.compiler_id))
        self._runtime_state.write("logical_plan_digest", str(compilation.audit.logical_plan_digest))
        self._runtime_state.write("compiled_plan_digest", str(compilation.audit.compiled_plan_digest))
        self._runtime_state.write("canonical_task_digest", str(compilation.audit.task_digest))
        self._runtime_state.write("canonical_task_count", int(compilation.audit.task_count))
        self._runtime_state.write("canonical_task_total_rows", int(compilation.audit.total_rows))
        return replace(
            compiled,
            execution_mode="joint_window_async_p2p",
            metrics={
                **compiled.metrics,
                "plan_origin": str(plan_origin),
                "plan_version": int(plan_version),
                "requested_bucket_mode": str(self._requested_bucket_mode()),
                "effective_bucket_mode": str(self._effective_bucket_mode()),
                "requested_bucket_rows": int(self.config.bucket_rows),
                "effective_bucket_rows": int(self.config.bucket_rows),
                "max_inflight_release_batches": int(getattr(self.config, "max_inflight_release_batches", 1) or 1),
            },
        )

    def _record_prediction_for_dispatch(
        self,
        *,
        layer_name: str,
        phase_ctx: PhaseReadyContext,
        observation: PreTransportTrafficObservation,
        actual_p0_full_row_matrix: tuple[tuple[int, ...], ...],
        device: torch.device,
    ) -> None:
        stage_start_ns = time.monotonic_ns()
        layer_id = parse_layer_id(layer_name)
        next_layer_id = self._next_layer_id(layer_name)
        world_size = int(len(self.ep_group_ranks) or len(observation.local_p0_row) or 1)
        full_matrix = tuple(tuple(int(value) for value in row) for row in actual_p0_full_row_matrix)
        remote_matrix = canonicalize_remote_matrix(full_matrix)
        actual_dispatch_by_layer = dict(self._runtime_state.read("actual_dispatch_by_layer", {}) or {})
        actual_dispatch_by_layer[str(layer_id)] = {
            "matrix": [list(row) for row in remote_matrix],
            "full_matrix": [list(row) for row in full_matrix],
            "matrix_digest": matrix_digest_remote(remote_matrix),
            "matrix_source": "pre_transport_phase_ready_context",
            "row_sums": list(matrix_row_sums_remote(remote_matrix)),
            "col_sums": list(matrix_col_sums_remote(remote_matrix)),
            "total_bytes": int(matrix_remote_bytes(remote_matrix)),
            "nonzero_edge_count": int(matrix_nonzero_remote_edge_count(remote_matrix)),
        }
        self._runtime_state.write("actual_dispatch_by_layer", actual_dispatch_by_layer)

        predicted_dispatch_by_layer = dict(self._runtime_state.read("predicted_dispatch_by_layer", {}) or {})
        existing_prediction = predicted_dispatch_by_layer.get(str(layer_id))
        audit_start_ns = time.monotonic_ns()
        if isinstance(existing_prediction, dict) and existing_prediction:
            from rs.runtime.online.megatron_ep.prediction.contracts import PredictedTrafficMatrix

            predicted = PredictedTrafficMatrix(
                predictor_name=str(existing_prediction.get("predictor_name", "")),
                predictor_version=str(existing_prediction.get("predictor_version", "")),
                source_layer_id=str(existing_prediction.get("source_layer_id", "")),
                predicted_layer_id=str(existing_prediction.get("predicted_layer_id", "")),
                matrix=tuple(tuple(int(value) for value in row) for row in existing_prediction.get("matrix", [])),
                matrix_digest=str(existing_prediction.get("matrix_digest", "")),
                total_bytes=int(existing_prediction.get("total_bytes", 0) or 0),
                nonzero_edge_count=int(existing_prediction.get("nonzero_edge_count", 0) or 0),
                confidence=float(existing_prediction.get("confidence", 0.0) or 0.0),
                is_oracle=bool(existing_prediction.get("is_oracle", False)),
                evaluation_eligible=bool(existing_prediction.get("evaluation_eligible", False)),
                created_at_phase=str(existing_prediction.get("created_at_phase", "")),
            )
            audit = compare_predicted_to_actual(predicted, remote_matrix)
            audit_row = audit.to_dict()
            if str(audit_row.get("predictor_name", "")) == "copy_current" and self._online_p2_predictor_name() == "copy_current_dispatch":
                audit_row["predictor_name"] = "copy_current_dispatch"
            self.prediction_audits.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "actual_matrix_source": "pre_transport_phase_ready_context",
                    **audit_row,
                }
            )
            predicted_dispatch_by_layer.pop(str(layer_id), None)
        audit_end_ns = time.monotonic_ns()

        stage_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="predict_next_dispatch",
            start_ns=stage_start_ns,
            end_ns=stage_end_ns,
            matrix_source="pre_transport_phase_ready_context",
            matrix_total_bytes=int(matrix_remote_bytes(remote_matrix)),
            matrix_nonzero_edge_count=int(matrix_nonzero_remote_edge_count(remote_matrix)),
            p2_matrix_gather_time_us=0.0,
            p2_matrix_gather_call_count=0,
            predictor_name="target_planner_worker",
            predicted_layer_id=str(next_layer_id),
            prediction_confidence=0.0,
            prediction_valid=True,
            prediction_error="",
            prediction_time_us=0.0,
            audit_time_us=max(0.0, float(audit_end_ns - audit_start_ns) / 1000.0),
            prediction_audit_emitted=bool(existing_prediction is not None),
        )
        if self._policy_supports_target_layer_preplanning() and self._layer_id_selected(str(next_layer_id)):
            self._ensure_target_planner_runtime()
            previous_matrix = None
            previous_record = actual_dispatch_by_layer.get(str(int(layer_id) - 1)) if str(layer_id).isdigit() else None
            if isinstance(previous_record, dict):
                previous_matrix = tuple(tuple(int(value) for value in row) for row in previous_record.get("matrix", []))
            if self.target_planner_service is not None:
                raw_u_name = str(self._effective_phase_policy_name() or "")
                paired_b_name = ""
                if str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select") == "host_select":
                    raw_u_name, paired_b_name = self._runtime_safe_joint_pair()
                result = self.target_planner_service.submit(
                    TargetLayerPlanningRequest(
                        run_id=str(self.run_id),
                        forward_epoch=int(self._forward_epoch),
                        microbatch_id=str(self.microbatch_id),
                        source_layer_id=str(layer_id),
                        target_layer_id=str(next_layer_id),
                        current_p0_rows=remote_matrix,
                        previous_p0_rows=previous_matrix,
                        predictor_name=str(self._online_p2_predictor_name()),
                        policy_id=str(self._effective_phase_policy_name() or ""),
                        group_size=int(world_size),
                        bucket_rows=int(self.config.bucket_rows),
                        policy_options=PlannerPolicyConfig(
                            p0_weight=float(getattr(self.config, "p0_weight", 1.0)),
                            p1_weight=float(getattr(self.config, "p1_reservation_weight", 1.0)),
                            p2_hint_weight=float(getattr(self.config, "p2_hint_weight", 1.0)),
                            residual_weight=float(getattr(self.config, "residual_weight", 0.75)),
                            barrier_weight=float(getattr(self.config, "barrier_weight", 1.75)),
                            age_weight=float(getattr(self.config, "age_weight", 0.15)),
                            prediction_weight=float(getattr(self.config, "prediction_weight", 0.35)),
                        ),
                        topology_digest=digest_text(stable_hash({"ep_group_ranks": list(int(v) for v in self.ep_group_ranks)})),
                        bucket_contract_digest=str(self._effective_bucket_mode()),
                        raw_u_policy_id=str(raw_u_name),
                        paired_b_policy_id=str(paired_b_name),
                        safe_projection_mode=str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select"),
                    )
                )
                slot = slot_from_request(
                    run_id=str(self.run_id),
                    forward_generation=int(self._forward_epoch),
                    microbatch_id=str(self.microbatch_id),
                    source_layer_id=str(layer_id),
                    target_layer_id=str(next_layer_id),
                )
                self._expected_publication_slots[
                    (str(self.run_id), int(self._forward_epoch), str(self.microbatch_id), str(next_layer_id))
                ] = slot
                self._runtime_state.write("latest_target_plan_submit_status", str(result.status.value))
                self._runtime_state.write("latest_target_plan_submit_task_key", str(result.task_key))
                self._increment_state_counter_map("target_plan_submit_count_by_source_target", f"{layer_id}->{next_layer_id}")
                if result.status in {PreparationSubmitStatus.ACCEPTED, PreparationSubmitStatus.REPLACED_STALE}:
                    self._increment_state_counter_map(
                        "target_plan_enqueue_count_by_source_target",
                        f"{layer_id}->{next_layer_id}",
                    )
                elif result.status is PreparationSubmitStatus.DROPPED_OVERLOAD:
                    self._runtime_state.write("latest_target_plan_preparation_state", "MISSED_OVERLOAD")
                    self._terminal_publication_slots.add(str(slot.semantic_digest()))
                elif result.status is PreparationSubmitStatus.REJECTED_EXPIRED:
                    self._runtime_state.write("latest_target_plan_preparation_state", "EXPIRED")
                    self._terminal_publication_slots.add(str(slot.semantic_digest()))
                elif result.status is PreparationSubmitStatus.REJECTED_CLOSED:
                    raise RuntimeError(f"target_planner_submit_failed:{result.status.value}:{result.task_key}")

    # Hint, shadow, and pending-window state

    def _build_p2_hint(self, *, layer_name: str, phase: str):
        start_ns = time.monotonic_ns()
        if self.config.p2_hint_mode == "calibrated_artifact":
            if self._p2_hint_provider is None:
                self._p2_hint_provider = build_p2_hint_provider(
                    self.config.p2_hint_mode,
                    shared_state=self._runtime_state,
                )
            provider = self._p2_hint_provider
        else:
            provider = build_p2_hint_provider(self.config.p2_hint_mode)
        hint = provider.build_hint(
            P2HintRequest(
                plan_key=self._plan_key(layer_name, phase),
                layer_id=parse_layer_id(layer_name),
                phase=phase,
                global_rank=self.rank,
                local_rank=self.local_rank,
                ep_group_ranks=self.ep_group_ranks,
            )
        )
        end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase=phase,
            stage="build_p2_hint",
            start_ns=start_ns,
            end_ns=end_ns,
            hint_mode=str(hint.hint_mode),
            hint_source=str(hint.hint_source),
        )
        return hint

    def _record_plan_arrival(self, *, layer_name: str, phase: str) -> None:
        now_us = int(time.time() * 1e6)
        plan = self._runtime_state.read("prepared_plan")
        plan_created_at = int(self._runtime_state.read("plan_created_at_us", 0) or 0)
        source_layer = str(self._runtime_state.read("plan_source_layer", ""))
        if plan is None:
            arrival_status = "none"
            plan_age_us = 0
        else:
            plan_age_us = max(0, now_us - plan_created_at)
            if self.config.control_mode == "sync_before_phase":
                arrival_status = "before_commit"
            else:
                arrival_status = "before_commit" if plan_age_us > 100 else "in_flight"
        record = {
            "ts_us": now_us,
            "layer_name": layer_name,
            "phase": phase,
            "arrival_status": arrival_status,
            "plan_age_us": plan_age_us,
            "source_layer": source_layer,
            "control_mode": self.config.control_mode,
            "has_prepared_plan": plan is not None,
            "window_key": str(getattr(plan, "window_key", "")) if plan is not None else "",
            "forecast_digest": str(getattr(plan, "forecast_digest", "")) if plan is not None else "",
        }
        self.plan_arrival_records.append(record)
        self._timeline(
            "shadow_plan_arrival",
            layer_name=layer_name,
            phase_name=phase,
            arrival_status=arrival_status,
            plan_age_us=plan_age_us,
            source_layer=source_layer,
            has_prepared_plan=plan is not None,
        )

    def _current_prepared_plan_binding(self, *, layer_name: str) -> PreparedPlanBinding | None:
        prepared_plan = self._runtime_state.read("prepared_plan")
        if prepared_plan is None:
            return None
        source_logical_plan_hash = ""
        logical_plan = getattr(prepared_plan, "logical_plan", None)
        if logical_plan is not None:
            source_logical_plan_hash = stable_hash(logical_plan.to_dict())
        return bind_prepared_plan(
            layer_name=layer_name,
            prepared_plan=prepared_plan,
            source_layer_name=str(self._runtime_state.read("plan_source_layer", "")),
            source_logical_plan_hash=source_logical_plan_hash,
        )

    def _record_window_state(
        self,
        *,
        layer_name: str,
        p0_observation: RuntimeObservation | None = None,
        p1_observation: RuntimeObservation | None = None,
    ) -> None:
        start_ns = time.monotonic_ns()
        existing = self._window_states.get(layer_name)
        release_state = WindowReleaseState() if existing is None else existing.release_state
        state, record = build_window_state_record(
            layer_name=layer_name,
            ep_group_ranks=self.ep_group_ranks,
            local_rank=self.local_rank,
            p0_observation=p0_observation if p0_observation is not None else (None if existing is None else existing.p0_observation),
            p1_observation=p1_observation if p1_observation is not None else (None if existing is None else existing.p1_observation),
            prepared_plan=self._runtime_state.read("prepared_plan"),
            prepared_plan_binding=self._current_prepared_plan_binding(layer_name=layer_name),
            release_state=release_state,
        )
        self._window_states[layer_name] = state
        self.window_state_records.append(record)
        self._increment_state_counter_map("window_state_count_by_layer", str(parse_layer_id(layer_name)))
        if state.prepared_plan_binding is not None:
            self.prepared_plan_bindings.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "layer_name": layer_name,
                    **state.prepared_plan_binding.to_dict(),
                }
            )
        shadow = maybe_build_window_shadow(
            enabled=self._allow_shadow_artifacts(),
            state=state,
            p0_weight=float(self.config.p0_weight),
            p1_reservation_weight=float(self.config.p1_reservation_weight),
            p2_hint_weight=float(self.config.p2_hint_weight),
        )
        if shadow is not None:
            self.window_schedule_shadows.append(shadow)
        end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="control",
            stage="record_window_state",
            start_ns=start_ns,
            end_ns=end_ns,
            has_p0=bool(state.p0_observation is not None),
            has_p1=bool(state.p1_observation is not None),
            has_prepared_plan=bool(state.prepared_plan_binding is not None),
        )

    def _record_release_update(self, *, layer_name: str, event: str) -> None:
        state = self._window_states.get(layer_name)
        if state is None:
            state, _ = build_window_state_record(
                layer_name=layer_name,
                ep_group_ranks=self.ep_group_ranks,
                local_rank=self.local_rank,
                p0_observation=None,
                p1_observation=None,
                prepared_plan=self._runtime_state.read("prepared_plan"),
                prepared_plan_binding=self._current_prepared_plan_binding(layer_name=layer_name),
                release_state=WindowReleaseState(),
            )
        state, record, state_record = advance_window_release(state=state, event=event, rank=self.rank, layer_name=layer_name)
        self._window_states[layer_name] = state
        self.release_events.append(record)
        self.window_state_records.append(state_record)
        shadow = maybe_build_window_shadow(
            enabled=self._allow_shadow_artifacts(),
            state=state,
            p0_weight=float(self.config.p0_weight),
            p1_reservation_weight=float(self.config.p1_reservation_weight),
            p2_hint_weight=float(self.config.p2_hint_weight),
        )
        if shadow is not None:
            self.window_schedule_shadows.append(shadow)

    def _record_prepared_phase_plan_shadow(
        self,
        *,
        layer_name: str,
        phase: str,
        local_context: PhaseReadyContext,
        global_contexts: tuple[PhaseReadyContext, ...],
    ) -> None:
        if not self._allow_shadow_artifacts():
            return
        start_ns = time.monotonic_ns()
        prepared_plan = self._runtime_state.read("prepared_plan")
        if prepared_plan is None:
            end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase=phase,
                stage="prepared_phase_plan_shadow",
                start_ns=start_ns,
                end_ns=end_ns,
                status="skipped_no_prepared_plan",
            )
            return
        binding = self._current_prepared_plan_binding(layer_name=layer_name)
        if binding is None:
            end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase=phase,
                stage="prepared_phase_plan_shadow",
                start_ns=start_ns,
                end_ns=end_ns,
                status="skipped_no_binding",
            )
            return
        phase_policy_name = self._effective_phase_policy_name()
        if not phase_policy_name:
            end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase=phase,
                stage="prepared_phase_plan_shadow",
                start_ns=start_ns,
                end_ns=end_ns,
                status="skipped_no_policy",
            )
            return
        try:
            compilation = compile_schedule(
                PlanCompilationRequest(
                    logical_plan=getattr(prepared_plan, "logical_plan"),
                    local_context=local_context,
                    global_contexts=global_contexts,
                    canonical_tasks=(),
                    phase=str(phase),
                    tensor_role="shadow",
                    rank_context={
                        "global_rank": int(local_context.global_rank),
                        "local_rank": int(local_context.local_rank),
                    },
                    compilation_options=CompilationOptions(
                        bucket_rows=int(self.config.bucket_rows),
                        p0_weight=float(self.config.p0_weight),
                        p1_reservation_weight=float(self.config.p1_reservation_weight),
                        p2_hint_weight=float(self.config.p2_hint_weight),
                        debug_trace=not self._is_perf_profile(),
                        invariant_mode="diagnostic",
                        legacy_compiler_bridge=True,
                    ),
                    prepared_plan=prepared_plan,
                    prepared_priority_cache=self._runtime_state.read("prepared_priority_cache"),
                    legacy_phase_policy_name=str(phase_policy_name),
                )
            )
            compiled = compilation.execution_plan
        except Exception as exc:  # pragma: no cover
            self.prepared_phase_plan_shadows.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "layer_name": layer_name,
                    "phase": phase,
                    "prepared_window_key": binding.window_key,
                    "compile_status": "failed",
                    "exception": f"{type(exc).__name__}: {exc}",
                }
            )
            end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase=phase,
                stage="prepared_phase_plan_shadow",
                start_ns=start_ns,
                end_ns=end_ns,
                status="failed",
                exception=f"{type(exc).__name__}: {exc}",
            )
            return
        self.prepared_phase_plan_shadows.append(
            {
                "ts_us": int(time.time() * 1e6),
                "layer_name": layer_name,
                "phase": phase,
                "prepared_window_key": binding.window_key,
                "compile_status": "ok",
                "source_layer_name": binding.source_layer_name,
                "source_logical_plan_hash": binding.source_logical_plan_hash,
                "compiled_plan_hash": compiled.plan_hash,
                "compiled_wave_count": len(compiled.waves),
                "compiled_bucket_order": list(compiled.metrics.get("bucket_order", [])),
                "prepared_plan_order_preserved": bool(compiled.metrics.get("prepared_plan_order_preserved", False)),
                "hint_edges_consumed": int(compiled.metrics.get("hint_edges_consumed", 0) or 0),
                "hint_match_rate": float(compiled.metrics.get("hint_match_rate", 0.0) or 0.0),
            }
        )
        self._increment_state_counter_map("shadow_plan_count_by_layer", str(parse_layer_id(layer_name)))
        end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase=phase,
            stage="prepared_phase_plan_shadow",
            start_ns=start_ns,
            end_ns=end_ns,
            status="ok",
            wave_count=int(len(compiled.waves)),
            hint_edges_consumed=int(compiled.metrics.get("hint_edges_consumed", 0) or 0),
        )

    def _build_global_joint_plan_wire(self, *, prepared_plan: Any) -> GlobalJointPlanWire:
        logical_plan = getattr(prepared_plan, "logical_plan")
        canonical_edge_order: list[tuple[str, int, int]] = []
        wave_metadata: list[tuple[int, tuple[tuple[str, int, int], ...]]] = []
        per_peer_sequence_rows: list[str] = []
        for wave in getattr(logical_plan, "waves", ()):
            wave_edges: list[tuple[str, int, int]] = []
            for flow in getattr(wave, "flows", ()):
                edge = (str(flow.phase), int(flow.src_rank), int(flow.dst_rank))
                wave_edges.append(edge)
                canonical_edge_order.append(edge)
                per_peer_sequence_rows.append(
                    f"{getattr(prepared_plan, 'created_at_layer_id', '')}:{getattr(prepared_plan, 'applies_from_layer_id', '')}:"
                    f"{str(flow.phase)}:{int(flow.src_rank)}:{int(flow.dst_rank)}:{int(getattr(wave, 'wave_id', 0))}"
                )
            wave_metadata.append((int(getattr(wave, "wave_id", 0)), tuple(wave_edges)))
        per_peer_sequence_digest = stable_hash(per_peer_sequence_rows)
        return GlobalJointPlanWire(
            window_key=str(getattr(prepared_plan, "window_key", "")),
            policy_name=str(getattr(logical_plan, "policy_name", "")),
            safe_selected_policy=str(getattr(logical_plan, "policy_name", "")),
            prediction_digest=str(getattr(prepared_plan, "forecast_digest", "")),
            canonical_edge_order=tuple(canonical_edge_order),
            wave_metadata=tuple(wave_metadata),
            per_peer_sequence_digest=str(per_peer_sequence_digest),
        )

    def _agree_joint_plan_digest(self, *, layer_name: str, phase: str, prepared_plan: Any) -> dict[str, Any]:
        wire = self._build_global_joint_plan_wire(prepared_plan=prepared_plan)
        digest = str(wire.global_plan_digest)
        device = torch.device("cuda", self.local_rank) if (torch.cuda.is_available() and self.ep_process_group is not None) else torch.device("cpu")
        digest_value = int(digest[:16], 16)
        if digest_value >= (1 << 63):
            digest_value -= 1 << 64
        local = torch.tensor([digest_value], dtype=torch.long, device=device)
        gathered = [torch.empty_like(local) for _ in range(len(self.ep_group_ranks) or 1)]
        if len(gathered) > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_gather(gathered, local, group=self.ep_process_group)
        else:
            gathered = [local]
        gathered_values = [int(item.item()) for item in gathered]
        valid = len(set(gathered_values)) == 1
        agreement = {
            "valid": bool(valid),
            "global_plan_digest": digest,
            "gathered_plan_digests": [
                f"{int(value) & ((1 << 64) - 1):016x}"
                for value in gathered_values
            ],
            "per_peer_sequence_digest": str(wire.per_peer_sequence_digest),
            "window_key": str(wire.window_key),
            "policy_name": str(wire.policy_name),
        }
        self._runtime_state.write("global_joint_plan_wire", wire)
        self._runtime_state.write("global_joint_plan_agreement", agreement)
        self._timeline(
            "global_joint_plan_digest_agreed" if valid else "global_joint_plan_digest_mismatch",
            layer_name=layer_name,
            phase_name=phase,
            global_plan_digest=digest,
            per_peer_sequence_digest=str(wire.per_peer_sequence_digest),
        )
        return agreement

    def _build_formal_planning_request(
        self,
        *,
        request_id: str,
        source_layer_id: str,
        target_layer_id: str,
        p0_dispatch_rows: tuple[tuple[int, ...], ...],
        p1_return_rows: tuple[tuple[int, ...], ...],
        p2_hint_rows: tuple[tuple[int, ...], ...],
        predictor_name: str,
        prediction_confidence: float,
        information_mode: str = "p0_p1_p2",
        max_waves: int = 256,
        planning_track: str = "runtime_lookahead",
        p2_semantics: str | None = None,
    ) -> PlanningRequest:
        effective_p2_semantics = (
            str(p2_semantics)
            if p2_semantics is not None
            else (
                "absent"
                if str(information_mode) in {"p0_only", "p0_p1"}
                else "advisory_hint"
                if str(planning_track) == "runtime_lookahead"
                else "executable_actual"
            )
        )
        return build_window_planning_request(
            identity=PlanningIdentity(
                request_id=str(request_id),
                run_id=str(self.run_id),
                forward_id=str(self._forward_epoch),
                window_id=f"{self._forward_epoch}:{self.microbatch_id}:{source_layer_id}",
                source_layer_id=str(source_layer_id),
                target_layer_id=str(target_layer_id),
            ),
            p0_dispatch_rows=tuple(tuple(int(v) for v in row) for row in p0_dispatch_rows),
            p1_return_rows=tuple(tuple(int(v) for v in row) for row in p1_return_rows),
            p2_hint_rows=tuple(tuple(int(v) for v in row) for row in p2_hint_rows),
            predictor_id=str(predictor_name or "zero"),
            confidence=float(prediction_confidence),
            topology=PlanningTopology(world_size=int(len(p0_dispatch_rows)), full_duplex=True),
            constraints=PlanningConstraints(
                bucket_rows=int(self.config.bucket_rows),
                max_waves=int(max_waves),
                expert_compute_delay=float(getattr(self.config, "expert_compute_delay", 0.0) or 0.0),
                phase_release_model="p1_return",
            ),
            weights=PlanningWeights(
                p0_weight=float(self.config.p0_weight),
                p1_weight=float(self.config.p1_reservation_weight),
                p2_weight=float(self.config.p2_hint_weight),
                residual_weight=float(getattr(self.config, "residual_weight", 0.75)),
                barrier_weight=float(getattr(self.config, "barrier_weight", 1.75)),
                age_weight=float(getattr(self.config, "age_weight", 0.15)),
                prediction_weight=float(getattr(self.config, "prediction_weight", 0.35)),
            ),
            information_mode=str(information_mode),
            planning_track=str(planning_track),
            p2_semantics=str(effective_p2_semantics),
            hint_type=(
                "perfect_trace_hint"
                if str(predictor_name) == "perfect_trace_hint"
                else "copy_current_dispatch"
                if str(predictor_name) == "copy_current_dispatch"
                else "learned_prediction"
            ),
            oracle=bool(str(predictor_name) == "perfect_trace_hint"),
        )

    def _store_runtime_joint_plan_from_p0(
        self,
        *,
        layer_name: str,
        phase_ctx: PhaseReadyContext,
        observation_p0: PreTransportTrafficObservation,
        actual_p0_full_row_matrix: tuple[tuple[int, ...], ...],
        plan_origin: str = "current_raw_u",
    ) -> None:
        from rs.runtime.online.megatron_ep.async_release.runtime_projection import host_project_safe_selection

        self._assert_bucket_mode_consistency()
        self._register_current_plan_build(layer_name=layer_name, phase="P0", plan_origin=plan_origin)
        layer_id = parse_layer_id(layer_name)
        dispatch_matrix_full = tuple(tuple(int(value) for value in row) for row in actual_p0_full_row_matrix)
        dispatch_matrix = canonicalize_remote_matrix(dispatch_matrix_full)
        if not dispatch_matrix_full:
            return
        num_peers = len(dispatch_matrix_full)
        inferred_p1 = tuple(
            tuple(int(dispatch_matrix_full[col_idx][row_idx]) for col_idx in range(num_peers))
            for row_idx in range(num_peers)
        )
        remote_dispatch_matrix = canonicalize_remote_matrix(dispatch_matrix_full)
        num_peers = len(remote_dispatch_matrix)
        inferred_p1_remote = tuple(
            tuple(int(remote_dispatch_matrix[col_idx][row_idx]) for col_idx in range(num_peers))
            for row_idx in range(num_peers)
        )
        active_prediction = dict(self._runtime_state.read("active_next_dispatch_prediction") or {})
        forecast_matrix = tuple(
            tuple(int(value) for value in row)
            for row in active_prediction.get("forecast_matrix", ())
        ) if active_prediction and bool(active_prediction.get("valid", False)) else tuple(tuple(0 for _ in range(num_peers)) for _ in range(num_peers))
        predictor_name = str(active_prediction.get("predictor_name", "")) if active_prediction else ""
        prediction_digest = str(active_prediction.get("matrix_digest", "")) if active_prediction else ""
        prediction_confidence = float(active_prediction.get("confidence", 0.0) or 0.0) if active_prediction else 0.0
        next_layer_id = self._next_layer_id(layer_name)
        forecast_digest = stable_hash(
            {
                "forecast_matrix": [list(row) for row in forecast_matrix],
                "source_layer": str(layer_id),
                "target_layer": str(next_layer_id),
                "predictor_name": predictor_name,
                "prediction_digest": prediction_digest,
            }
        )
        effective_policy = str(self._effective_phase_policy_name() or "")
        phase_local_async_policies = {
            "bucketed_fifo",
            "greedy_ready_set",
            "birkhoff_phase_local",
            "phase_barrier_fifo",
            "B_barrier_criticality_core_independent",
        }
        policy_options = PlannerPolicyConfig(
            p0_weight=float(self.config.p0_weight),
            p1_weight=float(self.config.p1_reservation_weight),
            p2_hint_weight=float(self.config.p2_hint_weight),
            residual_weight=float(getattr(self.config, "residual_weight", 0.75)),
            barrier_weight=float(getattr(self.config, "barrier_weight", 1.75)),
            age_weight=float(getattr(self.config, "age_weight", 0.15)),
            prediction_weight=float(getattr(self.config, "prediction_weight", 0.35)),
        )
        formal_request = self._build_formal_planning_request(
            request_id=f"{self.run_id}:{self.microbatch_id}:{layer_id}:current_window",
            source_layer_id=str(layer_id),
            target_layer_id=str(next_layer_id),
            p0_dispatch_rows=remote_dispatch_matrix,
            p1_return_rows=inferred_p1_remote,
            p2_hint_rows=forecast_matrix,
            predictor_name=str(predictor_name or "zero"),
            prediction_confidence=float(prediction_confidence),
            information_mode="p0_p1_p2",
            max_waves=256,
        )
        formal_cost_model = PlanningCostModel(
            expert_compute_delay=float(formal_request.constraints.expert_compute_delay),
            full_duplex=bool(formal_request.topology.full_duplex),
            max_outgoing_per_rank_per_wave=int(formal_request.topology.max_outgoing_per_rank_per_wave),
            max_incoming_per_rank_per_wave=int(formal_request.topology.max_incoming_per_rank_per_wave),
        )
        if effective_policy in phase_local_async_policies:
            raw_u_name = effective_policy
            paired_b_name = effective_policy
            raw_u_start_ns = time.monotonic_ns()
            raw_u_window_plan = PlannerRegistry.create(raw_u_name, None, usage="runtime").plan(formal_request)
            raw_u_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="raw_u_build",
                start_ns=raw_u_start_ns,
                end_ns=raw_u_end_ns,
                policy_name=raw_u_name,
            )
            self._increment_state_counter_map("raw_u_build_count_by_layer", str(layer_id))
            raw_u_plan = to_logical_plan(raw_u_window_plan)
            consumed_weights = dict((raw_u_plan.diagnostics or {}).get("consumed_weights", {}))
            requested_weights = {
                "residual_weight": float(policy_options.residual_weight),
                "barrier_weight": float(policy_options.barrier_weight),
                "age_weight": float(policy_options.age_weight),
                "prediction_weight": float(policy_options.prediction_weight),
            }
            if raw_u_name.startswith("U_") and consumed_weights != requested_weights:
                raise RuntimeError(
                    f"async joint U weights were not consumed: requested={requested_weights} consumed={consumed_weights}"
                )
            paired_b_start_ns = time.monotonic_ns()
            paired_b_window_plan = raw_u_window_plan
            paired_b_plan = raw_u_plan
            paired_b_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="paired_b_build",
                start_ns=paired_b_start_ns,
                end_ns=paired_b_end_ns,
                policy_name=paired_b_name,
            )
            self._increment_state_counter_map("paired_b_build_count_by_layer", str(layer_id))
            selected_window_plan = raw_u_window_plan
            selected_plan = raw_u_plan
        else:
            raw_u_name, paired_b_name = self._runtime_safe_joint_pair()
            safe_projection_mode = str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select")
            raw_u_start_ns = time.monotonic_ns()
            raw_u_window_plan = PlannerRegistry.create(raw_u_name, None, usage="runtime").plan(formal_request)
            raw_u_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="raw_u_build",
                start_ns=raw_u_start_ns,
                end_ns=raw_u_end_ns,
                policy_name=raw_u_name,
            )
            self._increment_state_counter_map("raw_u_build_count_by_layer", str(layer_id))
            raw_u_plan = to_logical_plan(raw_u_window_plan)
            paired_b_start_ns = time.monotonic_ns()
            if safe_projection_mode == "disabled":
                paired_b_window_plan = raw_u_window_plan
                paired_b_plan = raw_u_plan
                paired_b_end_ns = paired_b_start_ns
            else:
                paired_b_window_plan = PlannerRegistry.create(paired_b_name, None, usage="runtime").plan(formal_request)
                paired_b_plan = to_logical_plan(paired_b_window_plan)
                paired_b_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="paired_b_build",
                start_ns=paired_b_start_ns,
                end_ns=paired_b_end_ns,
                policy_name=paired_b_name,
                skipped=bool(safe_projection_mode == "disabled"),
            )
            if safe_projection_mode != "disabled":
                self._increment_state_counter_map("paired_b_build_count_by_layer", str(layer_id))
            selector = PlannerSelector(
                local_planner=PlannerRegistry.create(paired_b_name, None, usage="runtime"),
                joint_planner=PlannerRegistry.create(raw_u_name, None, usage="runtime"),
                estimator=CommonCorePlanEstimator(),
                cost_model=formal_cost_model,
            )
            if safe_projection_mode == "disabled":
                selected_window_plan = raw_u_window_plan
                selected_plan = raw_u_plan
            else:
                selected = selector.select_prebuilt(
                    request=formal_request,
                    local_plan=paired_b_window_plan,
                    joint_plan=raw_u_window_plan,
                    mode=PlannerSelectionMode.COMPARE,
                )
                selected_window_plan = selected.selected_plan
                selected_plan = to_logical_plan(selected_window_plan)
        safe_projection_mode = str(getattr(self.config, "safe_projection_mode", "host_select") or "host_select")
        if safe_projection_mode == "disabled":
            raw_score = CommonCorePlanEstimator().estimate(raw_u_window_plan, formal_request, formal_cost_model)
            host_projection_start_ns = time.monotonic_ns()
            host_projection_end_ns = host_projection_start_ns
            safe_projection = {
                "ideal_raw_u_estimated_makespan": float(raw_score.estimated_makespan),
                "host_projected_raw_u_estimated_makespan": float(raw_score.estimated_makespan),
                "ideal_paired_b_estimated_makespan": float(raw_score.estimated_makespan),
                "host_projected_paired_b_estimated_makespan": float(raw_score.estimated_makespan),
                "host_projected_safe_selection": str(raw_u_plan.policy_name),
                "projection_mode": "disabled",
            }
        else:
            host_projection_start_ns = time.monotonic_ns()
            safe_projection = host_project_safe_selection(
                raw_u_plan=raw_u_plan,
                paired_b_plan=paired_b_plan,
            )
            host_projection_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="host_projection",
            start_ns=host_projection_start_ns,
            end_ns=host_projection_end_ns,
            safe_projection_mode=safe_projection_mode,
        )
        actual_p0_row_matrix = [[int(value) for value in row] for row in remote_dispatch_matrix]
        actual_p0_full_row_matrix_list = [[int(value) for value in row] for row in dispatch_matrix_full]
        inferred_p1_row_matrix = [[int(value) for value in row] for row in inferred_p1]
        inferred_p1_remote_row_matrix = [[int(value) for value in row] for row in inferred_p1_remote]
        safe_selection_start_ns = time.monotonic_ns()
        safe_selection_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="safe_selection",
            start_ns=safe_selection_start_ns,
            end_ns=safe_selection_end_ns,
            selected_policy=str(selected_plan.policy_name),
            safe_projection_mode=safe_projection_mode,
        )
        prepared = PreparedWindowPlan(
            window_key=stable_hash(
                {
                    "runtime_safe_joint": bool(safe_projection_mode != "disabled"),
                    "safe_projection_mode": safe_projection_mode,
                    "raw_u_policy": raw_u_name,
                    "paired_b_policy": paired_b_name,
                    "selected_policy": str(selected_plan.policy_name),
                    "created_at_layer_id": str(layer_id),
                    "applies_from_layer_id": str(next_layer_id),
                    "forecast_digest": forecast_digest,
                }
            )[:16],
            forecast_digest=forecast_digest,
            logical_plan=selected_plan,
            created_at_layer_id=str(layer_id),
            applies_from_layer_id=str(next_layer_id),
            execution_capability_required="multiphase_pending_window",
            forecast_matrix=forecast_matrix,
        )
        self._runtime_state.write("prepared_plan", prepared)
        self._runtime_state.write("plan_created_at_us", int(time.time() * 1e6))
        self._runtime_state.write("plan_source_layer", layer_name)
        stored_logical_digest = stable_hash(selected_plan.to_dict())
        stored_compile_input_digest = stable_hash(
            {
                "phase": "P1",
                "layer_name": str(layer_name),
                "forward_epoch": int(self._forward_epoch),
                "matrix": [list(row) for row in inferred_p1],
            }
        )
        self._runtime_state.write("stored_p1_plan_digest", stored_logical_digest)
        self._runtime_state.write("stored_p1_logical_plan_digest", stored_logical_digest)
        self._runtime_state.write("stored_p1_compile_input_digest", stored_compile_input_digest)
        self._runtime_state.write("consumed_p1_plan_digest", "")
        self._runtime_state.write("consumed_p1_logical_plan_digest", "")
        self._runtime_state.write("consumed_p1_compile_input_digest", "")
        self._runtime_state.write("predictor_name", predictor_name)
        self._runtime_state.write("prediction_digest", prediction_digest)
        self._runtime_state.write("prediction_confidence", float(prediction_confidence))
        self._runtime_state.write("requested_bucket_mode", str(self._requested_bucket_mode()))
        self._runtime_state.write("effective_bucket_mode", str(self._effective_bucket_mode()))
        self._runtime_state.write("requested_bucket_rows", int(self.config.bucket_rows))
        self._runtime_state.write("effective_bucket_rows", int(self.config.bucket_rows))
        self._runtime_state.write("predicted_row_sums", [int(sum(row)) for row in forecast_matrix])
        self._runtime_state.write(
            "predicted_col_sums",
            [
            int(sum(forecast_matrix[row_idx][col_idx] for row_idx in range(len(forecast_matrix))))
            for col_idx in range(len(forecast_matrix[0]) if forecast_matrix else 0)
            ],
        )
        self._runtime_state.write("p2_matrix_source", "active_next_dispatch_prediction" if active_prediction else "zero_hint")
        self._runtime_state.write("p2_matrix_total_bytes", int(sum(sum(int(v) for v in row) for row in forecast_matrix)))
        self._runtime_state.write("p1_inferred_from_p0", [list(row) for row in inferred_p1])
        self._runtime_state.write(
            "global_joint_window_plan",
            {
            "window_key": str(prepared.window_key),
            "source_layer_id": str(layer_id),
            "target_layer_id": str(next_layer_id),
            "predictor_name": predictor_name,
            "prediction_digest": prediction_digest,
            "prediction_confidence": float(prediction_confidence),
            "actual_p0_matrix": [list(row) for row in remote_dispatch_matrix],
            "actual_p0_row_matrix": actual_p0_row_matrix,
            "actual_p0_full_matrix": [list(row) for row in dispatch_matrix_full],
            "actual_p0_full_row_matrix": actual_p0_full_row_matrix_list,
            "inferred_p1_matrix": [list(row) for row in inferred_p1],
            "inferred_p1_row_matrix": inferred_p1_row_matrix,
            "inferred_p1_remote_matrix": [list(row) for row in inferred_p1_remote],
            "inferred_p1_remote_row_matrix": inferred_p1_remote_row_matrix,
            "predicted_p2_matrix": [list(row) for row in forecast_matrix],
            "created_stage": "after_p0_observation",
            "planning_traffic_source": str(observation_p0.source),
            "captured_before_transport": bool(observation_p0.captured_before_transport),
            "pre_transport_observation_valid": bool(observation_p0.valid),
            "dispatcher_send_splits": list(observation_p0.send_splits_rows),
            "dispatcher_recv_splits": list(observation_p0.recv_splits_rows),
            "local_p0_row": list(observation_p0.local_p0_row),
            "actual_p0_total_rows": int(sum(sum(int(v) for v in row) for row in dispatch_matrix_full)),
            "p1_is_exact_transpose": bool(tuple(tuple(int(v) for v in row) for row in inferred_p1) == tuple(tuple(int(dispatch_matrix_full[col][row]) for col in range(len(dispatch_matrix_full))) for row in range(len(dispatch_matrix_full)))),
            "raw_u_policy_name": raw_u_name,
            "paired_b_policy_name": paired_b_name,
            "safe_projection_mode": safe_projection_mode,
            "requested_bucket_mode": str(self._requested_bucket_mode()),
            "effective_bucket_mode": str(self._effective_bucket_mode()),
            "requested_bucket_rows": int(self.config.bucket_rows),
            "effective_bucket_rows": int(self.config.bucket_rows),
            "default_weights": dict((raw_u_plan.diagnostics or {}).get("default_weights", {})),
            "requested_weights": dict((raw_u_plan.diagnostics or {}).get("requested_weights", {})),
            "effective_weights": dict((raw_u_plan.diagnostics or {}).get("effective_weights", {})),
            "consumed_weights": dict((raw_u_plan.diagnostics or {}).get("consumed_weights", {})),
            "safe_selected_policy": str(selected_plan.policy_name),
            "safe_selection_margin": float(
                safe_projection["host_projected_paired_b_estimated_makespan"]
                - safe_projection["host_projected_raw_u_estimated_makespan"]
            ),
            "safe_comparison_is_strict_common_core": bool(
                dict((raw_u_plan.diagnostics or {}).get("common_core", {}))
                == dict((paired_b_plan.diagnostics or {}).get("common_core", {}))
            ),
            "common_core_metadata": dict((raw_u_plan.diagnostics or {}).get("common_core", {})),
            "raw_u_plan_policy": str(raw_u_plan.policy_name),
            "paired_b_plan_policy": str(paired_b_plan.policy_name),
            "raw_plan_digest": stable_hash(raw_u_plan.to_dict()),
            "paired_b_plan_digest": stable_hash(paired_b_plan.to_dict()),
            "selected_plan_digest": stable_hash(selected_plan.to_dict()),
            "paired_b_build_count": 0 if safe_projection_mode == "disabled" else 1,
            "host_projection_count": 0 if safe_projection_mode == "disabled" else 1,
            "runtime_policy_equivalent_of": effective_policy,
            "service_demand_model": "rows_from_pre_transport_phase_ready_context",
            "bundle_bytes_per_row": int(self._bundle_bytes_per_row(phase_ctx=phase_ctx)),
            },
        )
        global_joint_window_plan = dict(self._runtime_state.read("global_joint_window_plan") or {})
        global_joint_window_plan["host_projected_safe_selection"] = dict(safe_projection)
        self._runtime_state.write("global_joint_window_plan", global_joint_window_plan)
        self._runtime_state.write("ideal_raw_u_makespan", float(safe_projection["ideal_raw_u_estimated_makespan"]))
        self._runtime_state.write("ideal_paired_b_makespan", float(safe_projection["ideal_paired_b_estimated_makespan"]))
        self._runtime_state.write("host_projected_raw_u_makespan", float(safe_projection["host_projected_raw_u_estimated_makespan"]))
        self._runtime_state.write("host_projected_paired_b_makespan", float(safe_projection["host_projected_paired_b_estimated_makespan"]))
        self._runtime_state.write("raw_plan_digest", stable_hash(raw_u_plan.to_dict()))
        self._runtime_state.write("paired_b_plan_digest", stable_hash(paired_b_plan.to_dict()))
        self._runtime_state.write("selected_plan_digest", stable_hash(selected_plan.to_dict()))
        self._runtime_state.write("paired_b_build_count", 0 if safe_projection_mode == "disabled" else 1)
        self._runtime_state.write("host_projection_count", 0 if safe_projection_mode == "disabled" else 1)
        self._runtime_state.write(
            "prediction_consumption_records",
            [
                {
                "prediction_first_consumed_stage": "during_p0_joint_planning",
                "consumer_layer": str(layer_id),
                "consumer_phase": "P1",
                "consumed_before_p1": True,
                "source_layer_id": str(active_prediction.get("source_layer_id", "")) if active_prediction else str(layer_id),
                "target_layer_id": str(active_prediction.get("target_layer_id", "")) if active_prediction else str(next_layer_id),
                "predictor_name": predictor_name,
                "prediction_digest": prediction_digest,
                "prediction_confidence": float(prediction_confidence),
                "prediction_matrix_total": int(sum(sum(int(v) for v in row) for row in forecast_matrix)),
                "consumed_during_p0_joint_planning": True,
                }
            ],
        )
        self._runtime_state.write(
            "host_projected_estimated_makespan",
            float(
            safe_projection["host_projected_paired_b_estimated_makespan"]
            if str(selected_plan.policy_name) == str(paired_b_plan.policy_name)
            else safe_projection["host_projected_raw_u_estimated_makespan"]
            ),
        )
        self._runtime_state.write(
            "ideal_estimated_makespan",
            float(
            safe_projection["ideal_paired_b_estimated_makespan"]
            if str(selected_plan.policy_name) == str(paired_b_plan.policy_name)
            else safe_projection["ideal_raw_u_estimated_makespan"]
            ),
        )
        self._runtime_state.remove("prepared_priority_cache", None)
        global_joint_window_plan = dict(self._runtime_state.read("global_joint_window_plan") or {})
        self._timeline(
            "runtime_joint_window_plan_stored",
            layer_name=layer_name,
            source_layer_id=str(layer_id),
            target_layer_id=str(next_layer_id),
            planning_traffic_source=str(observation_p0.source),
            captured_before_transport=bool(observation_p0.captured_before_transport),
            pre_transport_observation_valid=bool(observation_p0.valid),
            actual_p0_total_rows=int(sum(sum(int(v) for v in row) for row in dispatch_matrix_full)),
            actual_p0_matrix_unit="rows",
            p1_is_exact_transpose=bool(global_joint_window_plan.get("p1_is_exact_transpose", False)),
            prediction_digest=prediction_digest,
            prediction_confidence=float(prediction_confidence),
            predictor_name=predictor_name or "zero_hint",
            prediction_matrix_total=int(sum(sum(int(v) for v in row) for row in forecast_matrix)),
            stored_p1_plan_digest=str(self._runtime_state.read("stored_p1_plan_digest", "")),
            consumed_during_p0_joint_planning=True,
            ideal_raw_u_makespan=float(safe_projection["ideal_raw_u_estimated_makespan"]),
            ideal_paired_b_makespan=float(safe_projection["ideal_paired_b_estimated_makespan"]),
            host_projected_raw_u_makespan=float(safe_projection["host_projected_raw_u_estimated_makespan"]),
            host_projected_paired_b_makespan=float(safe_projection["host_projected_paired_b_estimated_makespan"]),
            host_projected_estimated_makespan=float(self._runtime_state.read("host_projected_estimated_makespan", 0.0)),
            ideal_estimated_makespan=float(self._runtime_state.read("ideal_estimated_makespan", 0.0)),
            safe_selected_policy=str(selected_plan.policy_name),
            raw_u_policy_name=raw_u_name,
            paired_b_policy_name=paired_b_name,
        )

    def _try_prepared_target_plan_for_p0(
        self,
        *,
        layer_name: str,
        phase_ctx: PhaseReadyContext,
        actual_p0_full_row_matrix: tuple[tuple[int, ...], ...],
    ) -> PhaseExecutionPlan | None:
        if not self._policy_supports_target_layer_preplanning() or self.target_plan_store is None:
            return None
        key = self._target_plan_key(layer_name=layer_name)
        peeked = self.target_plan_store.peek(key)
        if peeked is None:
            self._runtime_state.write("prepared_plan_found", False)
            return None
        prepared_plan = self.target_plan_store.claim_for_reconciliation(key)
        reconcile_key = (key.run_id, key.forward_epoch, key.microbatch_id, key.target_layer_id)
        if reconcile_key in self._target_plan_reconciled_keys:
            raise RuntimeError(f"target plan reconcile_once double invocation for {reconcile_key}")
        outcome = reconcile_once(
            prepared_plan=prepared_plan,
            actual_p0_rows=canonicalize_remote_matrix(actual_p0_full_row_matrix),
        )
        inferred_p1_rows = tuple(
            tuple(int(actual_p0_full_row_matrix[col_idx][row_idx]) for col_idx in range(len(actual_p0_full_row_matrix)))
            for row_idx in range(len(actual_p0_full_row_matrix))
        )
        self._target_plan_reconciled_keys.add(reconcile_key)
        self._runtime_state.write("prepared_plan_found", True)
        self._runtime_state.write("reconciliation_count", 1)
        self._runtime_state.write("full_u_replan_count", 0)
        self._runtime_state.write("prepared_target_selected_variant", str(getattr(prepared_plan, "selected_variant", "")))
        self._runtime_state.write(
            "prepared_target_safe_projection_mode",
            str(getattr(prepared_plan, "safe_projection_mode", "disabled") or "disabled"),
        )
        if outcome.status == "rejected" or outcome.logical_plan is None:
            self.target_plan_store.fail(key, execution_origin="prepared_rejected")
            self._runtime_state.write("execution_origin", "prepared_rejected")
            return None
        target_matrix = (
            tuple(tuple(int(value) for value in row) for row in actual_p0_full_row_matrix)
            if str(phase_ctx.phase) == "P0"
            else tuple(tuple(int(value) for value in row) for row in (self._runtime_state.read("p1_inferred_from_p0") or []))
        )
        compiled = self._compile_async_phase_from_logical_plan(
            logical_plan=outcome.logical_plan,
            layer_name=layer_name,
            phase=str(phase_ctx.phase),
            local_context=phase_ctx,
            matrix=target_matrix,
            plan_origin="prepared_exact" if outcome.status == "exact" else "prepared_repaired",
            plan_version=1,
        )
        synthetic_prepared = PreparedWindowPlan(
            window_key=stable_hash({"target_layer": str(layer_name), "origin": str(outcome.status)})[:16],
            forecast_digest=str(prepared_plan.h1_prediction_digest),
            logical_plan=outcome.logical_plan,
            created_at_layer_id=str(prepared_plan.source_layer_id),
            applies_from_layer_id=str(prepared_plan.target_layer_id),
            execution_capability_required="joint_window_async_p2p",
            forecast_matrix=tuple(tuple(int(value) for value in row) for row in prepared_plan.h1_rows),
        )
        stored_logical_digest = str(outcome.logical_plan_digest or stable_hash(outcome.logical_plan.to_dict()))
        stored_compile_input_digest = stable_hash(
            {
                "phase": "P1",
                "layer_name": str(layer_name),
                "forward_epoch": int(self._forward_epoch),
                "matrix": [list(row) for row in inferred_p1_rows],
            }
        )
        self._runtime_state.write("prepared_plan", synthetic_prepared)
        self._runtime_state.write("stored_p1_plan_digest", stored_logical_digest)
        self._runtime_state.write("stored_p1_logical_plan_digest", stored_logical_digest)
        self._runtime_state.write("stored_p1_compile_input_digest", stored_compile_input_digest)
        self._runtime_state.write("p1_inferred_from_p0", [list(row) for row in inferred_p1_rows])
        execution_origin = "prepared_exact" if outcome.status == "exact" else "prepared_repaired"
        self.target_plan_store.bind(key, bound_owner="prepared_reconcile")
        self.target_plan_store.start_execution(key, execution_origin=execution_origin, claim_owner="prepared_reconcile")
        self._runtime_state.write("execution_origin", execution_origin)
        self._runtime_state.write("prepared_target_logical_plan_digest", str(outcome.logical_plan_digest or ""))
        published_execution_plan = self._execution_plan_cache().get(self.target_plan_store._key(key))
        execution_pipeline = getattr(self, "execution_pipeline", None)
        if published_execution_plan is not None and execution_pipeline is not None:
            prepare_start_ns = time.monotonic_ns()
            prepared_execution = execution_pipeline.prepare(
                published_execution_plan,
                self._actual_phase_context_from_ready_context(phase_ctx=phase_ctx),
            )
            prepare_end_ns = time.monotonic_ns()
            self._record_instrumentation_measurement(
                event_type="materialization",
                layer_id=str(phase_ctx.layer_id),
                phase=str(phase_ctx.phase),
                started_at_ns=prepare_start_ns,
                ended_at_ns=prepare_end_ns,
                details={"valid": bool(prepared_execution.validation.valid)},
            )
            if not prepared_execution.validation.valid:
                self.target_plan_store.fail(key, execution_origin="materialization_invalid")
                self._runtime_state.write("execution_origin", "materialization_invalid")
                return None
            self._prepared_execution_cache()[self.target_plan_store._key(key)] = prepared_execution
        return compiled

    def _build_provisional_async_plan(
        self,
        *,
        layer_name: str,
        phase_ctx: PhaseReadyContext,
        observation_p0: PreTransportTrafficObservation,
        actual_p0_full_row_matrix: tuple[tuple[int, ...], ...],
    ) -> PhaseExecutionPlan:
        self._store_runtime_joint_plan_from_p0(
            layer_name=layer_name,
            phase_ctx=phase_ctx,
            observation_p0=observation_p0,
            actual_p0_full_row_matrix=actual_p0_full_row_matrix,
            plan_origin="provisional_current_plan",
        )
        prepared_plan = self._runtime_state.read("prepared_plan")
        if prepared_plan is None:
            raise RuntimeError(f"missing provisional prepared plan for {layer_name}")
        compiled = self._compile_async_phase_from_logical_plan(
            logical_plan=getattr(prepared_plan, "logical_plan"),
            layer_name=layer_name,
            phase="P0",
            local_context=phase_ctx,
            matrix=tuple(tuple(int(value) for value in row) for row in actual_p0_full_row_matrix),
            plan_origin="provisional",
            plan_version=0,
        )
        self._runtime_state.write("execution_origin", "provisional_only")
        self._runtime_state.write("provisional_plan_digest", str(compiled.plan_hash))
        return compiled

    def _late_suffix_provider(
        self,
        *,
        context: PhaseReadyContext,
        plan: PhaseExecutionPlan,
        tensor_role: str,
        frontier: Any,
        release_epoch: int,
    ) -> dict[str, Any] | None:
        if not self._policy_supports_target_layer_preplanning() or self.target_plan_store is None:
            return None
        if str(tensor_role) != "hidden_states":
            return None
        layer_name = str(context.layer_name)
        key = self._target_plan_key(layer_name=layer_name)
        if self.target_plan_store.peek(key) is None:
            return None
        prepared_plan = self.target_plan_store.claim_for_reconciliation(key)
        self._runtime_state.write("prepared_target_selected_variant", str(getattr(prepared_plan, "selected_variant", "")))
        self._runtime_state.write(
            "prepared_target_safe_projection_mode",
            str(getattr(prepared_plan, "safe_projection_mode", "disabled") or "disabled"),
        )
        if getattr(frontier, "pending_count", lambda: 0)() <= 0:
            self.target_plan_store.expire_key(key, execution_origin="too_late_no_effect")
            return None
        actual_rows = tuple(
            tuple(int(value) for value in row)
            for row in (
                ((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix"))
                or []
            )
        )
        if not actual_rows:
            return None
        outcome = reconcile_once(
            prepared_plan=prepared_plan,
            actual_p0_rows=canonicalize_remote_matrix(actual_rows),
            frozen_frontier=set(frontier.immutable_prefix_ids()),
        )
        if outcome.status == "rejected" or outcome.logical_plan is None:
            self.target_plan_store.reject(key, execution_origin="late_rejected")
            return None
        compiled = self._compile_async_phase_from_logical_plan(
            logical_plan=outcome.logical_plan,
            layer_name=layer_name,
            phase=str(context.phase),
            local_context=context,
            matrix=actual_rows,
            plan_origin="late_spliced",
            plan_version=2,
        )
        compiled_tasks = self._build_release_batch_tasks_from_plan(plan=compiled, tensor_role=tensor_role)
        suffix_tasks = self._residualize_suffix_tasks(
            candidate_tasks=compiled_tasks,
            frozen_tasks=tuple(frontier.immutable_prefix()),
        )
        if not suffix_tasks:
            self.target_plan_store.expire_key(key, execution_origin="too_late_no_effect")
            return None
        agreement_token = self._agree_late_suffix(
            key=key,
            frontier=frontier,
            residual_digest=stable_hash(
                [
                    (int(task.src_rank), int(task.dst_rank), int(task.row_count), int(task.sender_offset), int(task.receiver_offset))
                    for task in suffix_tasks
                ]
            ),
            replacement_tasks=suffix_tasks,
            new_plan_digest=str(outcome.logical_plan_digest or compiled.plan_hash),
            release_epoch=int(release_epoch),
        )
        self.target_plan_store.consume_once(key, execution_origin="provisional_then_late_suffix")
        self._runtime_state.write("execution_origin", "provisional_then_late_suffix")
        self._runtime_state.write("suffix_splice_count", 1)
        return {
            "apply_suffix": True,
            "suffix_tasks": suffix_tasks,
            "new_plan_version": 2,
            "parent_plan_version": int((plan.metrics or {}).get("plan_version", 0) or 0),
            "agreement_token": agreement_token,
        }

    def _compile_async_local_phase_plan(
        self,
        *,
        layer_name: str,
        phase: str,
        local_context: PhaseReadyContext,
    ) -> PhaseExecutionPlan:
        prepared_plan = self._runtime_state.read("prepared_plan")
        if prepared_plan is None:
            raise RuntimeError(f"missing prepared runtime joint plan for {layer_name} {phase}")
        if str(phase) == "P0":
            matrix = tuple(
                tuple(int(value) for value in row)
                for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix")) or [])
            )
            matrix_unit = "rows"
        else:
            matrix = tuple(
                tuple(int(value) for value in row)
                for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("inferred_p1_row_matrix")) or [])
            )
            matrix_unit = "rows"
            if not matrix:
                matrix = tuple(
                    tuple(int(value) for value in row)
                    for row in (self._runtime_state.read("p1_inferred_from_p0") or [])
                )
        if not matrix:
            raise RuntimeError(f"missing global row matrix for async local materialization {layer_name} {phase}")
        global_contexts = reconstruct_global_phase_contexts_from_byte_matrix(
            local_context=local_context,
            matrix=matrix,
            matrix_unit="rows",
        )
        compiled_local_context = next(
            (context for context in global_contexts if int(context.global_rank) == int(local_context.global_rank)),
            local_context,
        )
        canonical_tasks = build_phase_canonical_tasks(
            phase=str(phase),
            matrix_rows=matrix,
            bucket_rows=int(self.config.bucket_rows),
        )
        bucket_summary = summarize_bucket_tasks(canonical_tasks)
        compilation = compile_schedule(
            PlanCompilationRequest(
                logical_plan=getattr(prepared_plan, "logical_plan"),
                local_context=compiled_local_context,
                global_contexts=global_contexts,
                canonical_tasks=canonical_tasks,
                phase=str(phase),
                tensor_role="hidden_states" if str(phase) == "P1" else "dispatch_bundle",
                rank_context={
                    "global_rank": int(compiled_local_context.global_rank),
                    "local_rank": int(compiled_local_context.local_rank),
                },
                compilation_options=CompilationOptions(
                    bucket_rows=int(self.config.bucket_rows),
                    p0_weight=float(self.config.p0_weight),
                    p1_reservation_weight=float(self.config.p1_reservation_weight),
                    p2_hint_weight=float(self.config.p2_hint_weight),
                    debug_trace=not self._is_perf_profile(),
                    invariant_mode=str(getattr(self.config, "invariant_mode", "diagnostic")),
                    legacy_compiler_bridge=bool(getattr(self.config, "legacy_compiler_bridge", False)),
                ),
                prepared_plan=prepared_plan,
                prepared_priority_cache=self._runtime_state.read("prepared_priority_cache"),
                legacy_phase_policy_name=str(self._effective_phase_policy_name() or "routersense_p0p1p2_hint"),
            )
        )
        compiled = compilation.execution_plan
        self._runtime_state.write("compiler_id", str(compilation.audit.compiler_id))
        self._runtime_state.write("logical_plan_digest", str(compilation.audit.logical_plan_digest))
        self._runtime_state.write("compiled_plan_digest", str(compilation.audit.compiled_plan_digest))
        self._runtime_state.write("canonical_task_digest", str(compilation.audit.task_digest))
        self._runtime_state.write("canonical_task_count", int(compilation.audit.task_count))
        self._runtime_state.write("canonical_task_total_rows", int(compilation.audit.total_rows))
        self._runtime_state.write(
            "legacy_secondary_policy_invocation_count",
            int(compilation.audit.metrics.get("legacy_secondary_policy_invocation_count", 0) or 0),
        )
        self._runtime_state.write(
            "legacy_secondary_policy_call_count",
            int(compilation.audit.metrics.get("legacy_secondary_policy_call_count", 0) or 0),
        )
        self._runtime_state.write(
            "direct_compiler_selected_count",
            int(compilation.audit.metrics.get("direct_compiler_selected_count", 0) or 0),
        )
        self._runtime_state.write(
            "compiler_shadow_compare_count",
            int(compilation.audit.metrics.get("compiler_shadow_compare_count", 0) or 0),
        )
        self._runtime_state.write("compiler_shadow_status", str(compilation.audit.metrics.get("shadow_status", "")))
        self._runtime_state.write(
            "compiler_shadow_plan_hash_matches_legacy",
            bool(compilation.audit.metrics.get("shadow_plan_hash_matches_legacy", False)),
        )
        self._runtime_state.write("compiler_shadow_plan_hash", str(compilation.audit.metrics.get("shadow_plan_hash", "")))
        self._runtime_state.write(
            "compiler_shadow_missing_task_count",
            int(compilation.audit.metrics.get("shadow_missing_task_count", 0) or 0),
        )
        self._runtime_state.write(
            "compiler_shadow_extra_task_count",
            int(compilation.audit.metrics.get("shadow_extra_task_count", 0) or 0),
        )
        self._runtime_state.write(
            "compiler_shadow_execution_order_matches_legacy",
            bool(compilation.audit.metrics.get("shadow_execution_order_matches_legacy", False)),
        )
        return replace(
            compiled,
            execution_mode="joint_window_async_p2p",
            metrics={
                **compiled.metrics,
                "requested_bucket_mode": str(self._requested_bucket_mode()),
                "effective_bucket_mode": str(self._effective_bucket_mode()),
                "requested_bucket_rows": int(self.config.bucket_rows),
                "effective_bucket_rows": int(self.config.bucket_rows),
                "canonical_bucket_task_summary": bucket_summary,
                "joint_window_async_local_materialization": True,
                "p1_planning_collective_count": 0 if str(phase) == "P1" else int(compiled.metrics.get("p1_planning_collective_count", 0) or 0),
                "prediction_extra_collective_count": 0,
                "preflight_mode": str(getattr(self.config, "preflight_mode", "full")),
                "emit_detailed_task_artifacts": not self._is_perf_profile(),
            },
        )

    def _store_prepared_plan(self, *, layer_name: str, observation_p1: RuntimeObservation) -> None:
        total_start_ns = time.monotonic_ns()
        from rs.scheduling.contracts import (
            FlowWindow,
            ForecastPressure,
            GlobalReadySetOptions,
            LogicalTopology,
            MultiPhaseSchedulingProblem,
            ReleaseConstraint,
        )
        from rs.runtime.online.megatron_ep.pending_window.policy_adapter import get_or_build_prepared_priority_cache
        from rs.scheduling.multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy
        from rs.scheduling.validation import stable_hash

        per_peer = tuple(int(value) for value in observation_p1.per_peer_bytes)
        num_peers = len(per_peer)
        if num_peers <= 0:
            return
        p1_bundle = build_traffic_matrix_bundle(
            per_peer_bytes=per_peer,
            world_size=max(int(len(self.ep_group_ranks) or 0), num_peers),
            device=torch.device(str(getattr(observation_p1, "device", "cpu"))),
            group=self.ep_process_group,
        )
        layer_id = parse_layer_id(layer_name)
        next_layer_id = self._next_layer_id(layer_name)
        actual_dispatch_by_layer = dict(self._runtime_state.read("actual_dispatch_by_layer", {}) or {})
        dispatch_entry = actual_dispatch_by_layer.get(str(layer_id), {})
        dispatch_matrix = tuple(
            tuple(int(value) for value in row)
            for row in dispatch_entry.get("matrix", p1_bundle.matrix)
        )
        predicted_dispatch_by_layer = dict(self._runtime_state.read("predicted_dispatch_by_layer", {}) or {})
        prediction_entry = predicted_dispatch_by_layer.get(str(next_layer_id))
        active_prediction = self._runtime_state.read("active_next_dispatch_prediction")
        predictor_name = ""
        prediction_digest = ""
        prediction_confidence = 0.0
        prediction_evaluation_eligible = True
        prediction_is_oracle = False
        if (
            isinstance(active_prediction, dict)
            and active_prediction
            and str(active_prediction.get("target_layer_id", "")) == str(next_layer_id)
        ):
            forecast_matrix = tuple(
                tuple(int(value) for value in row)
                for row in active_prediction.get("forecast_matrix", [])
            )
            p2_matrix_source = "active_next_dispatch_prediction"
            predictor_name = str(active_prediction.get("predictor_name", ""))
            prediction_digest = str(active_prediction.get("matrix_digest", ""))
            prediction_confidence = float(active_prediction.get("confidence", 0.0) or 0.0)
            prediction_evaluation_eligible = bool(active_prediction.get("evaluation_eligible", True))
            prediction_is_oracle = bool(active_prediction.get("is_oracle", False))
        elif isinstance(prediction_entry, dict) and prediction_entry:
            forecast_matrix = tuple(
                tuple(int(value) for value in row)
                for row in prediction_entry.get("matrix", [])
            )
            p2_matrix_source = "predicted_next_dispatch"
            predictor_name = str(prediction_entry.get("predictor_name", ""))
            prediction_digest = str(prediction_entry.get("matrix_digest", ""))
            prediction_confidence = float(prediction_entry.get("confidence", 0.0) or 0.0)
            prediction_evaluation_eligible = bool(prediction_entry.get("evaluation_eligible", True))
            prediction_is_oracle = bool(prediction_entry.get("is_oracle", False))
        else:
            forecast_matrix = tuple(tuple(int(value) for value in row) for row in dispatch_matrix)
            p2_matrix_source = "copy_current_dispatch_fallback"
            predictor_name = "copy_current_dispatch"
            prediction_digest = stable_hash([list(row) for row in forecast_matrix])
            prediction_confidence = 1.0
            prediction_evaluation_eligible = True
            prediction_is_oracle = False
        if predictor_name == "copy_current" and self._online_p2_predictor_name() == "copy_current_dispatch":
            predictor_name = "copy_current_dispatch"
        row_sums = [int(sum(row)) for row in forecast_matrix]
        col_sums = [
            int(sum(forecast_matrix[row_idx][col_idx] for row_idx in range(len(forecast_matrix))))
            for col_idx in range(len(forecast_matrix[0]) if forecast_matrix else 0)
        ]
        forecast_digest = stable_hash(
            {
                "forecast_matrix": [list(row) for row in forecast_matrix],
                "layer": layer_name,
                "source": p2_matrix_source,
                "predictor_name": predictor_name,
                "prediction_digest": prediction_digest,
            }
        )
        problem = MultiPhaseSchedulingProblem(
            flow_window=FlowWindow(ready_flows=(), blocked_flows=(), forecast_pressure=()),
            topology=LogicalTopology(num_gpus=num_peers),
            release_model=ReleaseConstraint(
                phase="p1_return",
                rank=0,
                release_after_phase="p0_dispatch",
                expert_compute_delay=0.0,
            ),
            forecast=ForecastPressure(
                source=p2_matrix_source,
                digest=forecast_digest,
                oracle=prediction_is_oracle,
                evaluation_eligible=prediction_evaluation_eligible,
                matrix_shape=(num_peers, num_peers),
                matrix_total_bytes=int(sum(row_sums)),
                matrix=forecast_matrix,
                metadata={
                    "p2_matrix_source": p2_matrix_source,
                    "predictor_name": predictor_name,
                    "prediction_digest": prediction_digest,
                    "p2_matrix_is_replicated_local_row": False,
                    "p2_matrix_row_sums": row_sums,
                    "p2_matrix_col_sums": col_sums,
                    "p2_matrix_total_bytes": int(sum(row_sums)),
                },
            ),
            options=GlobalReadySetOptions(
                scheduling_mode="runtime_lookahead",
                information_mode="p0_p1_p2",
                prediction_confidence=float(prediction_confidence),
                p0_weight=float(self.config.p0_weight),
                p1_reservation_weight=float(self.config.p1_reservation_weight),
                p2_hint_weight=float(self.config.p2_hint_weight),
                max_waves=256,
            ),
            p0_dispatch_matrix=dispatch_matrix,
            p1_return_matrix=p1_bundle.matrix,
            p2_next_dispatch_forecast_matrix=forecast_matrix,
        )
        policy = RouterSenseMultiphaseLookaheadPolicy(
            information_mode="p0_p1_p2",
            p0_weight=self.config.p0_weight,
            p1_reservation_weight=self.config.p1_reservation_weight,
            p2_hint_weight=self.config.p2_hint_weight,
        )
        try:
            applies_from_layer_id = str(int(layer_id) + 1)
        except ValueError:
            applies_from_layer_id = layer_id
        build_start_ns = time.monotonic_ns()
        prepared = policy.build_prepared_window_plan(
            problem=problem,
            created_at_layer_id=str(layer_id),
            applies_from_layer_id=applies_from_layer_id,
        )
        build_end_ns = time.monotonic_ns()
        self._runtime_state.write("prepared_plan", prepared)
        self._runtime_state.write("plan_created_at_us", int(time.time() * 1e6))
        self._runtime_state.write("plan_source_layer", layer_name)
        self._runtime_state.write("p2_matrix_source", p2_matrix_source)
        self._runtime_state.write("p2_matrix_is_replicated_local_row", False)
        self._runtime_state.write("predictor_name", predictor_name)
        self._runtime_state.write("prediction_digest", prediction_digest)
        self._runtime_state.write("prediction_confidence", float(prediction_confidence))
        self._runtime_state.write("predicted_row_sums", row_sums)
        self._runtime_state.write("predicted_col_sums", col_sums)
        self._runtime_state.write("p2_matrix_source", p2_matrix_source)
        self._runtime_state.write("p2_matrix_total_bytes", int(sum(row_sums)))
        self._runtime_state.write("p2_matrix_row_sums", row_sums)
        self._runtime_state.write("p2_matrix_col_sums", col_sums)
        self._runtime_state.write("p2_matrix_is_replicated_local_row", False)
        self._runtime_state.write("p2_matrix_shape", [num_peers, num_peers])
        self._runtime_state.write("p2_matrix_gather_time_us", float(p1_bundle.gather_time_us))
        self._runtime_state.write("p2_matrix_gather_status", str(p1_bundle.matrix_source))
        self._runtime_state.write("p2_matrix_gather_call_count", int(p1_bundle.gather_call_count))
        self._runtime_state.write("prepared_priority_mode", "mapped_p2_tiebreak")
        self._runtime_state.write("has_real_p1_reservation", False)
        self._runtime_state.write("p1_reservation_row_sums", [])
        self._runtime_state.write("p1_reservation_col_sums", [])
        self._runtime_state.write("predictor_name", predictor_name)
        self._runtime_state.write("prediction_digest", prediction_digest)
        self._runtime_state.write("prediction_confidence", float(prediction_confidence))
        self._runtime_state.remove("prepared_priority_cache", None)
        cache_build_start_ns = time.monotonic_ns()
        _, _, cache_build_time_us = get_or_build_prepared_priority_cache(
            shared_state=self._runtime_state,
            prepared_plan=prepared,
        )
        cache_build_end_ns = time.monotonic_ns()
        self._timeline(
            "prepared_window_plan_stored",
            layer_name=layer_name,
            window_key=prepared.window_key,
            forecast_digest=prepared.forecast_digest,
            applies_from_layer_id=prepared.applies_from_layer_id,
            p2_matrix_source=p2_matrix_source,
        )
        total_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P1",
            stage="store_prepared_plan",
            start_ns=total_start_ns,
            end_ns=total_end_ns,
            prepared_window_key=str(prepared.window_key),
            forecast_digest=str(prepared.forecast_digest),
            policy_name=str(prepared.logical_plan.policy_name),
            logical_build_time_us=max(0.0, float(build_end_ns - build_start_ns) / 1000.0),
            prepared_priority_cache_build_time_us=float(cache_build_time_us),
            prepared_priority_cache_total_time_us=(cache_build_end_ns - cache_build_start_ns) / 1000.0,
            p2_matrix_gather_time_us=float(p1_bundle.gather_time_us),
            p2_matrix_gather_status=str(p1_bundle.matrix_source),
            p2_matrix_gather_call_count=int(p1_bundle.gather_call_count),
            predictor_name=predictor_name,
            prediction_digest=prediction_digest,
            prediction_confidence=float(prediction_confidence),
        )

    def _record_pending_window_driver(
        self,
        *,
        layer_name: str,
        phase: str,
        plan: PhaseExecutionPlan,
    ) -> None:
        if self.config.execution_mode != "multiphase_pending_window":
            return
        metrics = dict(plan.metrics)
        record = {
            "ts_us": int(time.time() * 1e6),
            "layer_name": layer_name,
            "layer_id": parse_layer_id(layer_name),
            "phase": phase,
            "plan_hash": plan.plan_hash,
            "policy_name": plan.policy_name,
            "compiled_from_pending_window": bool(metrics.get("compiled_from_pending_window", False)),
            "pending_window_logical_policy_name": str(metrics.get("pending_window_logical_policy_name", "")),
            "pending_window_plan_hash": str(metrics.get("pending_window_plan_hash", "")),
            "pending_window_information_mode": str(metrics.get("pending_window_information_mode", "")),
            "pending_window_forecast_available": bool(metrics.get("pending_window_forecast_available", False)),
            "pending_window_p0_total_bytes": int(metrics.get("pending_window_p0_total_bytes", 0) or 0),
            "pending_window_p1_total_bytes": int(metrics.get("pending_window_p1_total_bytes", 0) or 0),
            "pending_window_p2_total_bytes": int(metrics.get("pending_window_p2_total_bytes", 0) or 0),
            "pending_window_p1_matrix_source": str(metrics.get("pending_window_p1_matrix_source", "")),
            "pending_window_p2_matrix_source": str(metrics.get("pending_window_p2_matrix_source", "")),
            "p2_matrix_source": str(self._runtime_state.read("p2_matrix_source", "")),
            "p2_matrix_total_bytes": int(self._runtime_state.read("p2_matrix_total_bytes", 0) or 0),
            "p2_matrix_row_sums": list(self._runtime_state.read("p2_matrix_row_sums", []) or []),
            "p2_matrix_col_sums": list(self._runtime_state.read("p2_matrix_col_sums", []) or []),
            "p2_matrix_is_replicated_local_row": bool(self._runtime_state.read("p2_matrix_is_replicated_local_row", False)),
            "predictor_name": str(self._runtime_state.read("predictor_name", "")),
            "prediction_digest": str(self._runtime_state.read("prediction_digest", "")),
            "prepared_window_key": str(metrics.get("prepared_window_key", "")),
            "source_logical_plan_hash": str(metrics.get("source_logical_plan_hash", "")),
            "wave_count": len(plan.waves),
            "bucket_count": sum(len(wave.bucket_tasks) for wave in plan.waves),
            "hint_edges_consumed": int(metrics.get("hint_edges_consumed", 0) or 0),
            "hint_match_rate": float(metrics.get("hint_match_rate", 0.0) or 0.0),
            "prepared_plan_order_preserved": bool(metrics.get("prepared_plan_order_preserved", False)),
        }
        if not self._is_perf_profile():
            record["bucket_order"] = list(metrics.get("bucket_order", []))
        self.pending_window_driver_records.append(record)

    # Tensor/debug capture and context builders

    def capture_phase_transport_output(
        self,
        *,
        layer_name: str,
        phase: str,
        result: Any,
        dispatcher: Any,
    ) -> None:
        recorder = self.observation_recorder
        if recorder is None:
            return
        layer_id = parse_layer_id(layer_name)
        if not recorder.should_capture_tensor(layer_id=layer_id, phase=phase):
            return
        tensors: list[tuple[str, torch.Tensor]] = []
        if isinstance(result, torch.Tensor):
            tensors.append(("hidden_states", result))
        elif isinstance(result, (list, tuple)):
            roles = ["hidden_states", "routing_probs"]
            for index, item in enumerate(result):
                if isinstance(item, torch.Tensor):
                    role = roles[index] if index < len(roles) else f"output_{index}"
                    tensors.append((role, item))
        input_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "input_splits", None)))
        output_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "output_splits", None)))
        for role, tensor in tensors:
            checksum = hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()
            row_digest = hashlib.sha256(
                tensor.detach().float().cpu().reshape(tensor.shape[0], -1).numpy().tobytes()
            ).hexdigest() if tensor.ndim >= 1 else checksum
            recorder.record_captured_tensor(
                {
                    "layer_name": layer_name,
                    "layer_id": layer_id,
                    "phase": phase,
                    "rank": self.rank,
                    "tensor_role": role,
                    "shape": [int(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                    "input_splits": list(input_splits),
                    "output_splits": list(output_splits),
                    "row_order_digest": row_digest,
                    "tensor_checksum": checksum,
                    "tensor": tensor.detach().cpu(),
                }
            )

    def _record_observer(self, **payload: Any) -> None:
        if self.observer is None:
            return
        try:
            self.observer.record(**payload)
        except Exception:
            pass

    def _context(self, layer_name: str) -> PolicyContext:
        layer_id = parse_layer_id(layer_name)
        ep_group_hash = compute_ep_group_hash(self.ep_group_ranks)
        return PolicyContext(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            layer_id=layer_id,
            run_id_digest=digest_text(self.run_id),
            step_id_digest=digest_text(self.step_id),
            microbatch_id_digest=digest_text(self.microbatch_id),
            request_table_hash=self.request_table_hash,
            model_revision_hash=self.model_revision_hash,
            expert_placement_hash="unknown",
            ep_group_ranks=self.ep_group_ranks,
            ep_group_size=len(self.ep_group_ranks),
            ep_group_hash=ep_group_hash,
            future_hint_mode=self.config.future_hint_mode,
            control_mode=self.config.control_mode,
        )

    def _plan_key(self, layer_name: str, phase: str) -> dict[str, Any]:
        return {
            "run_id_digest": digest_text(self.run_id),
            "forward_epoch": int(self._forward_epoch),
            "step_id": self.step_id,
            "microbatch_id": self.microbatch_id,
            "layer_id": parse_layer_id(layer_name),
            "phase": phase,
            "ep_group_hash": compute_ep_group_hash(self.ep_group_ranks),
            "ep_group_epoch": 0,
            "model_revision_hash": self.model_revision_hash,
            "expert_placement_hash": "unknown",
            "request_table_hash": self.request_table_hash,
        }

    def begin_forward(self, *, forward_epoch: int | None = None) -> None:
        previous_epoch = int(self._forward_epoch)
        if forward_epoch is None:
            self._forward_epoch += 1
        else:
            self._forward_epoch = int(forward_epoch)
        self._current_plan_build_keys.clear()
        self._selected_layer_active_ns.clear()
        self._expert_module_active_ns.clear()
        self._pending_p0.clear()
        self._pending_p1.clear()
        self._selected_layer_active_ns.clear()
        self._expert_module_active_ns.clear()
        self._active_transport = None
        self._runtime_state.write("active_next_dispatch_prediction", None)
        self._runtime_state.write("prediction_consumption_records", [])
        self._runtime_state.write("consumed_p1_plan_digest", "")
        self._runtime_state.remove("prepared_priority_cache", None)
        self._runtime_state.remove("global_joint_plan_wire", None)
        self._runtime_state.remove("global_joint_plan_agreement", None)
        self._runtime_state.remove("global_joint_window_plan", None)
        self._runtime_state.write("forward_start_ns", int(time.monotonic_ns()))
        self._runtime_state.write("forward_end_ns", 0)
        self._target_plan_reconciled_keys.clear()
        self._latest_execution_outcomes.clear()
        self._latest_result_bundle = None
        self.release_state_ledger.reset(
            run_id=str(self.run_id),
            forward_generation=int(self._forward_epoch),
            microbatch_id=str(self.microbatch_id),
        )
        self._ready_target_plan_candidates.clear()
        self._expected_publication_slots.clear()
        self._terminal_publication_slots.clear()
        self._published_publication_slots.clear()
        self._poll_attempts.clear()
        if self.target_planner_service is not None:
            self.target_planner_service.cancel_before_generation(
                run_id=str(self.run_id),
                microbatch_id=str(self.microbatch_id),
                current_generation=int(self._forward_epoch),
            )
        if self.target_plan_store is not None:
            self.target_plan_store.cleanup_before_generation(
                run_id=str(self.run_id),
                microbatch_id=str(self.microbatch_id),
                current_generation=int(self._forward_epoch),
            )
        if self.control_communication_lane is not None and hasattr(self.control_communication_lane, "cancel_before_generation"):
            self.control_communication_lane.cancel_before_generation(
                run_id=str(self.run_id),
                microbatch_id=str(self.microbatch_id),
                current_generation=int(self._forward_epoch),
            )

    def end_forward(self) -> dict[str, Any]:
        active_transport = self._active_transport is not None
        has_active_prediction = bool(self._runtime_state.read("active_next_dispatch_prediction"))
        self._runtime_state.write("forward_end_ns", int(time.monotonic_ns()))
        self._pending_p0.clear()
        self._pending_p1.clear()
        self._active_transport = None
        self._runtime_state.write("active_next_dispatch_prediction", None)
        self._runtime_state.write("prediction_consumption_records", [])
        self._runtime_state.write("consumed_p1_plan_digest", "")
        self._runtime_state.remove("prepared_priority_cache", None)
        self._runtime_state.remove("global_joint_plan_wire", None)
        self._runtime_state.remove("global_joint_plan_agreement", None)
        self._runtime_state.remove("global_joint_window_plan", None)
        self._ready_target_plan_candidates.clear()
        self._expected_publication_slots.clear()
        self._terminal_publication_slots.clear()
        self._published_publication_slots.clear()
        self._poll_attempts.clear()
        if self.target_planner_service is not None:
            self.target_planner_service.cancel_generation(
                run_id=str(self.run_id),
                forward_epoch=int(self._forward_epoch),
                microbatch_id=str(self.microbatch_id),
            )
        if self.target_plan_store is not None:
            self.target_plan_store.cleanup_epoch(
                run_id=str(self.run_id),
                forward_epoch=int(self._forward_epoch),
                microbatch_id=str(self.microbatch_id),
            )
        self.release_state_ledger.reset(
            run_id=str(self.run_id),
            forward_generation=int(self._forward_epoch),
            microbatch_id=str(self.microbatch_id),
        )
        return {
            "forward_epoch": int(self._forward_epoch),
            "active_transport_cleared": bool(active_transport),
            "stale_prediction_cleared": bool(has_active_prediction),
            "valid": not active_transport,
        }

    # Main lifecycle hooks

    def _append_runtime_state_record(self, key: str, record: dict[str, Any]) -> None:
        rows = list(self._runtime_state.read(key, []) or [])
        rows.append(dict(record))
        self._runtime_state.write(key, rows)

    def record_selected_layer_enter(self, *, layer_name: str) -> None:
        if self.layer_role_for_name(layer_name) != "selected":
            return
        layer_id = str(parse_layer_id(layer_name))
        self._selected_layer_active_ns[(int(self._forward_epoch), layer_id)] = int(time.perf_counter_ns())

    def record_selected_layer_exit(self, *, layer_name: str) -> None:
        if self.layer_role_for_name(layer_name) != "selected":
            return
        end_ns = int(time.perf_counter_ns())
        layer_id = str(parse_layer_id(layer_name))
        key = (int(self._forward_epoch), layer_id)
        start_ns = self._selected_layer_active_ns.pop(key, 0)
        if start_ns <= 0:
            self._append_runtime_state_record("selected_layer_timing_records", {"rank": int(self.rank), "forward_epoch": int(self._forward_epoch), "layer_name": str(layer_name), "layer_id": layer_id, "measurement_status": "unavailable", "reason": "missing_selected_layer_enter"})
            return
        self._append_runtime_state_record("selected_layer_timing_records", {"rank": int(self.rank), "forward_epoch": int(self._forward_epoch), "layer_name": str(layer_name), "layer_id": layer_id, "selected_layer_enter_ns": int(start_ns), "selected_layer_exit_ns": int(end_ns), "selected_layer_total_us": max(0.0, float(end_ns - start_ns) / 1000.0), "measurement_status": "measured"})

    def record_expert_module_enter(self, *, layer_name: str, expert_module_name: str = "") -> None:
        if self.layer_role_for_name(layer_name) != "selected":
            return
        layer_id = str(parse_layer_id(layer_name))
        self._expert_module_active_ns[(int(self._forward_epoch), layer_id)] = int(time.perf_counter_ns())
        status = dict(self._runtime_state.read("attribution_boundary_status", {}) or {})
        status[layer_id] = {**dict(status.get(layer_id, {}) or {}), "expert_boundary_status": "hook_registered", "expert_module_name": str(expert_module_name)}
        self._runtime_state.write("attribution_boundary_status", status)

    def record_expert_module_exit(self, *, layer_name: str, expert_module_name: str = "") -> None:
        if self.layer_role_for_name(layer_name) != "selected":
            return
        end_ns = int(time.perf_counter_ns())
        layer_id = str(parse_layer_id(layer_name))
        key = (int(self._forward_epoch), layer_id)
        start_ns = self._expert_module_active_ns.pop(key, 0)
        if start_ns <= 0:
            self._append_runtime_state_record("expert_module_timing_records", {"rank": int(self.rank), "forward_epoch": int(self._forward_epoch), "layer_name": str(layer_name), "layer_id": layer_id, "expert_module_name": str(expert_module_name), "measurement_status": "unavailable", "reason": "missing_expert_module_enter"})
            return
        self._append_runtime_state_record("expert_module_timing_records", {"rank": int(self.rank), "forward_epoch": int(self._forward_epoch), "layer_name": str(layer_name), "layer_id": layer_id, "expert_module_name": str(expert_module_name), "expert_module_enter_ns": int(start_ns), "expert_module_exit_ns": int(end_ns), "expert_module_wall_us": max(0.0, float(end_ns - start_ns) / 1000.0), "measurement_status": "measured"})

    def record_expert_boundary_unavailable(self, *, layer_name: str, reason: str) -> None:
        if self.layer_role_for_name(layer_name) != "selected":
            return
        layer_id = str(parse_layer_id(layer_name))
        status = dict(self._runtime_state.read("attribution_boundary_status", {}) or {})
        status[layer_id] = {**dict(status.get(layer_id, {}) or {}), "expert_boundary_status": "unavailable", "expert_boundary_reason": str(reason)}
        self._runtime_state.write("attribution_boundary_status", status)

    def before_token_dispatch(
        self,
        *,
        layer_name: str,
        dispatcher: Any,
        packed_hidden_states: Any,
        packed_probs: Any,
    ) -> None:
        hook_start_ns = time.monotonic_ns()
        layer_role = self.layer_role_for_name(layer_name)
        hook_mode = self._hook_execution_mode(layer_name=layer_name)
        if layer_role == "selected":
            self._runtime_state.metrics.selected_p0_hook_count = int(self._runtime_state.metrics.selected_p0_hook_count) + 1
            if hook_mode == "REAL_EXECUTION_WITH_OBSERVATION":
                self._runtime_state.metrics.real_p0_execution_count = int(self._runtime_state.metrics.real_p0_execution_count) + 1
        self._timeline("before_token_dispatch_enter", layer_name=layer_name, phase_name="P0")
        if layer_role == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P0",
                hook_name="before_token_dispatch_total",
                start_ns=hook_start_ns,
            )
            self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=False)
            return
        if layer_role == "prediction_source":
            self.before_prediction_source_dispatch(
                layer_name=layer_name,
                dispatcher=dispatcher,
                packed_hidden_states=packed_hidden_states,
                packed_probs=packed_probs,
            )
            self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=False)
            return
        self._pump_target_planner_publications()
        self._poll_target_plan_slot(target_layer_id=str(parse_layer_id(layer_name)), safe_point="target_dispatch_ready")
        sync_fn = getattr(dispatcher, "_maybe_dtoh_and_synchronize", None)
        if callable(sync_fn):
            try:
                tokens_per_expert = getattr(dispatcher, "tokens_per_expert", None)
                dtoh_start_ns = time.monotonic_ns()
                synchronized = sync_fn("before_ep_alltoall", tokens_per_expert)
                dtoh_end_ns = time.monotonic_ns()
                self._record_dtoh_callsite(
                    callsite_id="DTOH_P0_DISPATCHER_SYNC",
                    start_ns=dtoh_start_ns,
                    end_ns=dtoh_end_ns,
                )
                if synchronized is not None:
                    dispatcher.tokens_per_expert = synchronized
            except Exception:
                pass
        phase_ctx_start_ns = time.monotonic_ns()
        phase_ctx = self._build_phase_ready_context_from_dispatcher(
            layer_name=layer_name,
            phase="P0",
            dispatcher=dispatcher,
            packed_tensors=tuple(
                tensor for tensor in (packed_hidden_states, packed_probs) if isinstance(tensor, torch.Tensor)
            ),
        )
        phase_ctx_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="build_phase_ready_context",
            start_ns=phase_ctx_start_ns,
            end_ns=phase_ctx_end_ns,
            remote_rows=int(sum(int(v) for idx, v in enumerate(phase_ctx.send_splits) if idx != int(self._runtime_topology_dict()["ep_group_rank"]))),
            hint_mode="none",
        )
        pretransport = self._capture_pretransport_traffic_observation(phase_ctx=phase_ctx)
        matrix_device = self._matrix_device(packed_hidden_states)
        actual_p0_full_row_matrix = self._gather_actual_p0_full_row_matrix(
            layer_name=layer_name,
            observation=pretransport,
            device=matrix_device,
        )
        if self._should_generate_runtime_prediction():
            self._record_prediction_for_dispatch(
                layer_name=layer_name,
                phase_ctx=phase_ctx,
                observation=pretransport,
                actual_p0_full_row_matrix=actual_p0_full_row_matrix,
                device=matrix_device,
            )
        p2_hint = self._build_p2_hint(layer_name=layer_name, phase="P0")
        phase_ctx = replace(phase_ctx, p2_hint=p2_hint)
        observation_start_ns = time.monotonic_ns()
        ep_group_hash = compute_ep_group_hash(self.ep_group_ranks)
        observation = build_runtime_observation(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            model_revision_hash=self.model_revision_hash,
            request_table_hash=self.request_table_hash,
            hostname=self.hostname,
            layer_name=layer_name,
            rank=self.rank,
            local_rank=self.local_rank,
            ep_group_ranks=self.ep_group_ranks,
            ep_group_hash=ep_group_hash,
            dispatcher=dispatcher,
            phase="P0",
            hidden_states=packed_hidden_states,
        )
        observation_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P0",
            stage="build_runtime_observation",
            start_ns=observation_start_ns,
            end_ns=observation_end_ns,
            remote_rows=int(observation.remote_rows),
            remote_bytes=int(sum(int(v) for v in observation.per_peer_bytes)),
        )
        if self.observation_recorder is not None and bool(getattr(self.config, "capture_expert_trace", False)):
            bytes_per_token = 1
            if isinstance(packed_hidden_states, torch.Tensor) and packed_hidden_states.ndim >= 1:
                bytes_per_token = int(packed_hidden_states.shape[-1]) * int(packed_hidden_states.element_size())
            maybe_capture_expert_route_trace(
                recorder=self.observation_recorder,
                layer_id=int(parse_layer_id(layer_name)) if str(parse_layer_id(layer_name)).isdigit() else 0,
                rank=int(self.rank),
                source_rank=int(self.rank),
                dispatcher=dispatcher,
                selected_experts=getattr(getattr(dispatcher, "_comm_manager", None), "token_indices", None),
                routing_weights=getattr(getattr(dispatcher, "_comm_manager", None), "token_probs", None),
                top_k=int(getattr(dispatcher, "router_topk", getattr(getattr(dispatcher, "_comm_manager", None), "router_topk", 1)) or 1),
                token_count=int(packed_hidden_states.shape[0]) if isinstance(packed_hidden_states, torch.Tensor) and packed_hidden_states.ndim >= 1 else 0,
                hidden_shape=tuple(int(v) for v in packed_hidden_states.shape) if isinstance(packed_hidden_states, torch.Tensor) else None,
                bytes_per_token=bytes_per_token,
                per_peer_bytes=tuple(int(v) for v in observation.per_peer_bytes),
                ep_group_ranks=tuple(int(v) for v in self.ep_group_ranks),
                enabled=True,
            )
        self._record_plan_arrival(layer_name=layer_name, phase="P0")
        self._pending_p0[layer_name] = observation
        self._record_window_state(layer_name=layer_name, p0_observation=observation)
        if self.observation_recorder is not None:
            self.observation_recorder.record_phase_context(
                phase_context_artifact(context=phase_ctx, perf_profile=self._is_perf_profile())
            )
            for bundle in phase_ctx.transport_bundles:
                self.observation_recorder.record_transport_bundle(
                    transport_bundle_artifact(bundle=bundle, perf_profile=self._is_perf_profile())
                )
            self._record_prepared_phase_plan_shadow(
                layer_name=layer_name,
                phase="P0",
                local_context=phase_ctx,
                global_contexts=(
                    reconstruct_global_phase_contexts_from_byte_matrix(
                        local_context=phase_ctx,
                        matrix=tuple(
                            tuple(int(value) for value in row)
                            for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix")) or [])
                        ),
                        matrix_unit="rows",
                    )
                    if self._is_joint_window_async_mode()
                    and ((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix"))
                    else (phase_ctx,)
                ),
            )
        pre_input_splits = tuple(int(v) for v in phase_ctx.input_splits)
        pre_output_splits = tuple(int(v) for v in phase_ctx.output_splits)
        hidden_ptr = int(packed_hidden_states.data_ptr()) if isinstance(packed_hidden_states, torch.Tensor) else -1
        probs_ptr = int(packed_probs.data_ptr()) if isinstance(packed_probs, torch.Tensor) else -1
        self._timeline(
            "p0_pre_transport_observation_ready",
            layer_name=layer_name,
            input_splits=list(pre_input_splits),
            output_splits=list(pre_output_splits),
            planning_traffic_source="pre_transport_phase_ready_context",
            pre_transport_observation_valid=bool(pretransport.valid),
            local_p0_row=list(pretransport.local_p0_row),
            actual_p0_total_rows=int(sum(sum(int(v) for v in row) for row in actual_p0_full_row_matrix)),
            hidden_shape=list(packed_hidden_states.shape) if isinstance(packed_hidden_states, torch.Tensor) else None,
            probs_shape=list(packed_probs.shape) if isinstance(packed_probs, torch.Tensor) else None,
            p2_hint_mode=p2_hint.hint_mode,
            p2_hint_digest=p2_hint.hint_digest,
        )
        self._timeline("before_phase_plan", layer_name=layer_name, phase_name="P0")
        if self._should_schedule_phase(layer_name=layer_name, phase="P0"):
            if self._is_joint_window_async_mode():
                target_plan = self._try_prepared_target_plan_for_p0(
                    layer_name=layer_name,
                    phase_ctx=phase_ctx,
                    actual_p0_full_row_matrix=actual_p0_full_row_matrix,
                )
                if target_plan is None:
                    plan = self._build_provisional_async_plan(
                        layer_name=layer_name,
                        phase_ctx=phase_ctx,
                        observation_p0=pretransport,
                        actual_p0_full_row_matrix=actual_p0_full_row_matrix,
                    )
                else:
                    plan = target_plan
                if self.observation_recorder is not None:
                    self.observation_recorder.record_scheduled_plan(
                        scheduled_plan_artifact(plan=plan, perf_profile=self._is_perf_profile())
                    )
                self._activate_transport(layer_name=layer_name, phase="P0", context=phase_ctx, plan=plan)
                adapter = getattr(self, "transport_adapter", None)
                if adapter is not None:
                    setattr(adapter, "late_suffix_provider", None)
                self._runtime_state.write("before_async_p2p_phase_count", int(self._runtime_state.read("before_async_p2p_phase_count", 0) or 0) + 1)
                self._timeline(
                    "phase_execution_plan_agreed",
                    layer_name=layer_name,
                    phase_name="P0",
                    plan_hash=plan.plan_hash,
                    wave_count=len(plan.waves),
                    bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
                    execution_mode=plan.execution_mode,
                    execution_origin=str(self._runtime_state.read("execution_origin", "")),
                )
                hook_end_ns = time.monotonic_ns()
                self._record_hook_timing(
                    layer_name=layer_name,
                    phase="P0",
                    hook_name="before_token_dispatch_total",
                    start_ns=hook_start_ns,
                    end_ns=hook_end_ns,
                    scheduled=True,
                    plan_hash=plan.plan_hash,
                    wave_count=int(len(plan.waves)),
                )
                self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=True, plan_hash=plan.plan_hash)
                return
            agreement_start_ns = time.monotonic_ns()
            policy = self._pending_window_adapter() if self.config.execution_mode == "multiphase_pending_window" else self._phase_policy()
            plan = run_phase_plan_agreement(local_context=phase_ctx, policy=policy, group=self.ep_process_group)
            agreement_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="run_phase_plan_agreement",
                start_ns=agreement_start_ns,
                end_ns=agreement_end_ns,
                wave_count=int(len(plan.waves)),
                build_plan_time_us=float(plan.metrics.get("build_plan_time_us", 0.0) or 0.0),
                all_gather_time_us=float(plan.metrics.get("all_gather_time_us", 0.0) or 0.0),
                summary_build_time_us=float(plan.metrics.get("summary_build_time_us", 0.0) or 0.0),
                summary_encode_time_us=float(plan.metrics.get("summary_encode_time_us", 0.0) or 0.0),
                summary_decode_time_us=float(plan.metrics.get("summary_decode_time_us", 0.0) or 0.0),
                broadcast_time_us=float(plan.metrics.get("broadcast_time_us", 0.0) or 0.0),
                abstract_encode_time_us=float(plan.metrics.get("abstract_encode_time_us", 0.0) or 0.0),
                abstract_decode_time_us=float(plan.metrics.get("abstract_decode_time_us", 0.0) or 0.0),
                materialize_local_plan_time_us=float(plan.metrics.get("materialize_local_plan_time_us", 0.0) or 0.0),
                verify_time_us=float(plan.metrics.get("verify_time_us", 0.0) or 0.0),
            )
            if self.observation_recorder is not None:
                self.observation_recorder.record_scheduled_plan(
                    scheduled_plan_artifact(plan=plan, perf_profile=self._is_perf_profile())
                )
            self._record_control_replay_trace(phase_ctx=phase_ctx, plan=plan)
            self._record_pending_window_driver(layer_name=layer_name, phase="P0", plan=plan)
            self._activate_transport(layer_name=layer_name, phase="P0", context=phase_ctx, plan=plan)
            self._timeline(
                "phase_execution_plan_agreed",
                layer_name=layer_name,
                phase_name="P0",
                plan_hash=plan.plan_hash,
                wave_count=len(plan.waves),
                bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
                execution_mode=plan.execution_mode,
                all_gather_time_us=float(plan.metrics.get("all_gather_time_us", 0.0) or 0.0),
                build_plan_time_us=float(plan.metrics.get("build_plan_time_us", 0.0) or 0.0),
                summary_build_time_us=float(plan.metrics.get("summary_build_time_us", 0.0) or 0.0),
                summary_encode_time_us=float(plan.metrics.get("summary_encode_time_us", 0.0) or 0.0),
                summary_decode_time_us=float(plan.metrics.get("summary_decode_time_us", 0.0) or 0.0),
                broadcast_time_us=float(plan.metrics.get("broadcast_time_us", 0.0) or 0.0),
                abstract_encode_time_us=float(plan.metrics.get("abstract_encode_time_us", 0.0) or 0.0),
                abstract_decode_time_us=float(plan.metrics.get("abstract_decode_time_us", 0.0) or 0.0),
                materialize_local_plan_time_us=float(plan.metrics.get("materialize_local_plan_time_us", 0.0) or 0.0),
                verify_time_us=float(plan.metrics.get("verify_time_us", 0.0) or 0.0),
                total_agreement_time_us=float(plan.metrics.get("total_agreement_time_us", 0.0) or 0.0),
            )
            self._timeline("after_phase_plan", layer_name=layer_name, phase_name="P0", plan_hash=plan.plan_hash)
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P0",
                hook_name="before_token_dispatch_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                scheduled=True,
                plan_hash=plan.plan_hash,
                wave_count=int(len(plan.waves)),
            )
            self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=True, plan_hash=plan.plan_hash)
            return
        if self.config.scheduler_mode != "native_passthrough_identity":
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P0",
                hook_name="before_token_dispatch_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                scheduled=False,
                reason="phase_policy_not_selected",
            )
            self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=False)
            return
        context = replace(self._context(layer_name), expert_placement_hash=observation.expert_placement_hash)
        local_observations = (observation,)
        plan, agreement = run_policy_agreement(
            local_observations=local_observations,
            context=context,
            policy=self._phase_policy(),
            device=torch.device(f"cuda:{self.local_rank}"),
            group=self.ep_process_group,
        )
        post_input_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "input_splits", None)))
        post_output_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "output_splits", None)))
        self.assertion_state["native_splits_unchanged"] = pre_input_splits == post_input_splits and pre_output_splits == post_output_splits
        self.assertion_state["native_buffers_unchanged"] = (
            hidden_ptr == (int(packed_hidden_states.data_ptr()) if isinstance(packed_hidden_states, torch.Tensor) else -1)
            and probs_ptr == (int(packed_probs.data_ptr()) if isinstance(packed_probs, torch.Tensor) else -1)
        )
        current_version = self._active_plan_versions.get(layer_name, 0)
        self._active_plan_versions[layer_name] = current_version
        self._active_plan_hashes[layer_name] = plan.plan_hash
        decision = InjectionDecision(
            accepted=True,
            fallback="native",
            plan_hash=plan.plan_hash,
            reason="identity_pre_transport_passthrough",
            policy_name=plan.policy_name,
            control_mode=self.config.control_mode,
        )
        self.completed.append(
            PolicyRuntimeRecord(
                layer_name=layer_name,
                context=context,
                local_observations=local_observations,
                plan=plan,
                agreement=agreement,
                decision=decision,
            )
        )
        self._timeline("root_plan_broadcast_received", layer_name=layer_name, root_wire_hash=agreement.root_wire_hash)
        self._timeline(
            "root_plan_decoded",
            layer_name=layer_name,
            root_semantic_hash=agreement.root_semantic_hash,
            decoded_semantic_hash=agreement.decoded_semantic_hash,
        )
        self._timeline(
            "plan_agreement_verified",
            layer_name=layer_name,
            agreement_status=agreement.agreement_status,
            root_semantic_hash=agreement.root_semantic_hash,
            decoded_semantic_hash=agreement.decoded_semantic_hash,
        )
        self._timeline(
            "identity_plan_agreed",
            layer_name=layer_name,
            root_wire_hash=agreement.root_wire_hash,
            root_semantic_hash=agreement.root_semantic_hash,
            decoded_semantic_hash=agreement.decoded_semantic_hash,
            agreement_status=agreement.agreement_status,
            version=current_version,
        )
        self._record_observer(
            phase="policy_plan",
            layer=layer_name,
            rank=self.rank,
            local_rank=self.local_rank,
            policy_name=plan.policy_name,
            scheduler_mode=self.config.scheduler_mode,
            control_mode=self.config.control_mode,
            plan_hash=plan.plan_hash,
            execution_mode=plan.execution_mode,
            wave_count=len(plan.waves),
            agreement=agreement.to_dict(),
            decision=decision.to_dict(),
        )
        if self.config.control_mode == "default_continue" and self.config.shadow_command_arrival == "before_commit":
            self._active_plan_versions[layer_name] = current_version + 1
            self.control_commands.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "event_seq": len(self.control_commands) + 1,
                    "layer": layer_name,
                    "rank": self.rank,
                    "command_kind": "shadow_replace",
                    "old_version": current_version,
                    "new_version": current_version + 1,
                    "status": "applied",
                    "transport_mutation": False,
                }
            )
            self._timeline(
                "shadow_command_replaced_active",
                layer_name=layer_name,
                old_version=current_version,
                new_version=current_version + 1,
                transport_mutation=False,
            )
        hook_end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase="P0",
            hook_name="before_token_dispatch_total",
            start_ns=hook_start_ns,
            end_ns=hook_end_ns,
            scheduled=False,
            reason="native_passthrough_identity",
        )
        self._timeline("before_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=False)

    def mark_token_dispatch_committed(self, *, layer_name: str) -> None:
        if self.config.scheduler_mode != "native_passthrough_identity" and not bool(self._effective_phase_policy_name()):
            return
        self._timeline(
            "p0_native_dispatch_committed",
            layer_name=layer_name,
            active_version=self._active_plan_versions.get(layer_name, 0),
        )

    def after_token_dispatch(self, *, layer_name: str) -> None:
        hook_start_ns = time.monotonic_ns()
        self._timeline("after_token_dispatch_enter", layer_name=layer_name, phase_name="P0")
        if self.layer_role_for_name(layer_name) == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P0",
                hook_name="after_token_dispatch_total",
                start_ns=hook_start_ns,
            )
            self._timeline("after_token_dispatch_exit", layer_name=layer_name, phase_name="P0", scheduled=False)
            return
        active_transport = self.current_transport()
        if self.layer_role_for_name(layer_name) == "selected" and active_transport is not None and str(active_transport.get("layer_name")) == str(layer_name) and str(active_transport.get("phase")) == "P0":
            clear_start_ns = time.monotonic_ns()
            self.clear_transport(layer_name=layer_name, phase="P0")
            if self._is_joint_window_async_mode() and self.target_plan_store is not None:
                key = self._target_plan_key(layer_name=layer_name)
                self.target_plan_store.close_key_if_unclaimed(
                    key,
                    final_status="EXPIRED",
                    execution_origin="too_late_no_effect",
                )
            if self._is_joint_window_async_mode():
                self._runtime_state.write("after_async_p2p_phase_count", int(self._runtime_state.read("after_async_p2p_phase_count", 0) or 0) + 1)
                self._runtime_state.write("all_layer_async_phase_count", int(self._runtime_state.read("all_layer_async_phase_count", 0) or 0) + 1)
                if self._layer_selected(layer_name):
                    self._runtime_state.write("selected_layer_after_async_p2p_phase_count", int(self._runtime_state.read("selected_layer_after_async_p2p_phase_count", 0) or 0) + 1)
            clear_end_ns = time.monotonic_ns()
            self._runtime_state.write("dispatch_transport_end_ns", int(clear_end_ns))
            self._runtime_state.write("rank_release_ns", int(clear_end_ns))
            self._runtime_state.write("expert_compute_start_ns", int(clear_end_ns))
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P0",
                stage="clear_transport",
                start_ns=clear_start_ns,
                end_ns=clear_end_ns,
            )
            self._record_release_update(layer_name=layer_name, event="p0_dispatch_completed")
            if str(self.config.schedule_phase_selector).lower() == "p0" and self._should_stop_after_layer(layer_name=layer_name, phase="P0"):
                raise SelectedLayerStop(f"Stopped after selected P0 layer {layer_name}")
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P0",
                hook_name="after_token_dispatch_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
            )
            self._timeline("after_token_dispatch_exit", layer_name=layer_name, phase_name="P0")
            return
        if self.config.scheduler_mode != "native_passthrough_identity":
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P0",
                hook_name="after_token_dispatch_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                skipped=True,
            )
            self._timeline("after_token_dispatch_exit", layer_name=layer_name, phase_name="P0", skipped=True)
            return
        if self.config.control_mode == "default_continue" and self.config.shadow_command_arrival == "after_commit":
            current = self._active_plan_versions.get(layer_name, 0)
            self.control_commands.append(
                {
                    "ts_us": int(time.time() * 1e6),
                    "event_seq": len(self.control_commands) + 1,
                    "layer": layer_name,
                    "rank": self.rank,
                    "command_kind": "shadow_replace",
                    "old_version": current,
                    "new_version": current + 1,
                    "status": "expired_late",
                    "transport_mutation": False,
                }
            )
            self._timeline(
                "shadow_command_expired_late",
                layer_name=layer_name,
                old_version=current,
                attempted_version=current + 1,
                transport_mutation=False,
            )
        hook_end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase="P0",
            hook_name="after_token_dispatch_total",
            start_ns=hook_start_ns,
            end_ns=hook_end_ns,
        )
        self._timeline("after_token_dispatch_exit", layer_name=layer_name, phase_name="P0")

    def before_token_combine(self, *, layer_name: str, dispatcher: Any, packed_hidden_states: Any) -> None:
        hook_start_ns = time.monotonic_ns()
        layer_role = self.layer_role_for_name(layer_name)
        if layer_role == "selected":
            self._runtime_state.metrics.selected_p1_hook_count = int(self._runtime_state.metrics.selected_p1_hook_count) + 1
        self._timeline("before_token_combine_enter", layer_name=layer_name, phase_name="P1")
        if layer_role == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P1",
                hook_name="before_token_combine_total",
                start_ns=hook_start_ns,
            )
            self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=False)
            return
        if layer_role != "selected":
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P1",
                hook_name="before_token_combine_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                scheduled=False,
                reason="layer_role_not_selected",
            )
            self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=False)
            return
        self._pump_target_planner_publications()
        self._runtime_state.write("expert_compute_end_ns", int(hook_start_ns))
        observation_start_ns = time.monotonic_ns()
        observation = build_runtime_observation(
            run_id=self.run_id,
            step_id=self.step_id,
            microbatch_id=self.microbatch_id,
            model_revision_hash=self.model_revision_hash,
            request_table_hash=self.request_table_hash,
            hostname=self.hostname,
            layer_name=layer_name,
            rank=self.rank,
            local_rank=self.local_rank,
            ep_group_ranks=self.ep_group_ranks,
            ep_group_hash=compute_ep_group_hash(self.ep_group_ranks),
            dispatcher=dispatcher,
            phase="P1",
            hidden_states=packed_hidden_states,
        )
        observation_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P1",
            stage="build_runtime_observation",
            start_ns=observation_start_ns,
            end_ns=observation_end_ns,
            remote_rows=int(observation.remote_rows),
            remote_bytes=int(sum(int(v) for v in observation.per_peer_bytes)),
        )
        self._pending_p1[layer_name] = observation
        self._record_window_state(layer_name=layer_name, p1_observation=observation)
        self._record_release_update(layer_name=layer_name, event="p1_return_materialized")
        self._record_plan_arrival(layer_name=layer_name, phase="P1")
        p2_hint = self._build_p2_hint(layer_name=layer_name, phase="P1")
        phase_ctx_start_ns = time.monotonic_ns()
        phase_ctx = build_phase_ready_context(
            PhaseContextBuildRequest(
                plan_key=self._plan_key(layer_name, "P1"),
                runtime_identity=RuntimeIdentity(
                    run_id=self.run_id,
                    forward_epoch=int(self._forward_epoch),
                    layer_id=parse_layer_id(layer_name),
                    layer_name=layer_name,
                    global_rank=self.rank,
                    local_rank=self.local_rank,
                    ep_group_ranks=self.ep_group_ranks,
                    ep_group_root_rank=self.ep_group_root_global_rank,
                ),
                topology=observation.topology.to_dict(),
                dispatcher_snapshot=DispatcherSnapshot(
                    dispatcher_class=type(dispatcher).__name__,
                    dispatcher_fingerprint={"dispatcher_class": type(dispatcher).__name__},
                    expert_placement_hash=observation.expert_placement_hash,
                    input_splits=observation.input_splits,
                    output_splits=observation.output_splits,
                ),
                payload_contract=PhasePayloadContract(
                    phase="P1",
                    payload_roles=("hidden_states",),
                    atomic_submit=False,
                ),
                packed_tensors=(packed_hidden_states,) if isinstance(packed_hidden_states, torch.Tensor) else (),
                control_mode=self.config.control_mode,
                release_state="ready",
                demand_known_at="router_ready",
                payload_exists=True,
                p2_hint=p2_hint,
            )
        )
        phase_ctx_end_ns = time.monotonic_ns()
        self._record_planning_timing(
            layer_name=layer_name,
            phase="P1",
            stage="build_phase_ready_context",
            start_ns=phase_ctx_start_ns,
            end_ns=phase_ctx_end_ns,
            remote_rows=int(observation.remote_rows),
            hint_mode=str(p2_hint.hint_mode),
        )
        if self.observation_recorder is not None:
            self.observation_recorder.record_phase_context(
                phase_context_artifact(context=phase_ctx, perf_profile=self._is_perf_profile())
            )
            for bundle in phase_ctx.transport_bundles:
                self.observation_recorder.record_transport_bundle(
                    transport_bundle_artifact(bundle=bundle, perf_profile=self._is_perf_profile())
                )
        self._record_prepared_phase_plan_shadow(
            layer_name=layer_name,
            phase="P1",
            local_context=phase_ctx,
            global_contexts=(
                reconstruct_global_phase_contexts_from_byte_matrix(
                    local_context=phase_ctx,
                    matrix=tuple(
                        tuple(int(value) for value in row)
                        for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("inferred_p1_row_matrix")) or [])
                    ),
                    matrix_unit="rows",
                )
                if self._is_joint_window_async_mode()
                and ((self._runtime_state.read("global_joint_window_plan") or {}).get("inferred_p1_row_matrix"))
                else (phase_ctx,)
            ),
        )
        self._timeline(
            "p1_pre_transport_observation_ready",
            layer_name=layer_name,
            p2_hint_mode=p2_hint.hint_mode,
            p2_hint_digest=p2_hint.hint_digest,
        )
        self._timeline("before_phase_plan", layer_name=layer_name, phase_name="P1")
        if self._should_schedule_phase(layer_name=layer_name, phase="P1"):
            if self._is_joint_window_async_mode():
                binding = self._current_prepared_plan_binding(layer_name=layer_name)
                stored_digest = str(self._runtime_state.read("stored_p1_plan_digest", "") or "")
                stored_logical_digest = str(self._runtime_state.read("stored_p1_logical_plan_digest", "") or "")
                stored_compile_input_digest = str(self._runtime_state.read("stored_p1_compile_input_digest", "") or "")
                if stored_digest:
                    self._runtime_state.write("consumed_p1_plan_digest", stored_digest)
                    self._runtime_state.write("consumed_p1_logical_plan_digest", stored_logical_digest or stored_digest)
                    self._runtime_state.write("consumed_p1_compile_input_digest", stored_compile_input_digest)
                elif binding is not None:
                    self._runtime_state.write("consumed_p1_plan_digest", str(binding.source_logical_plan_hash))
                    self._runtime_state.write("consumed_p1_logical_plan_digest", str(binding.source_logical_plan_hash))
                self._timeline(
                    "prepared_p1_plan_consumed",
                    layer_name=layer_name,
                    stored_p1_plan_digest=str(stored_digest),
                    consumed_p1_plan_digest=str(self._runtime_state.read("consumed_p1_plan_digest", "") or ""),
                    p1_plan_source_window=str(binding.window_key) if binding is not None else "",
                    p1_plan_consumed_once=True,
                )
                inferred_p1 = tuple(
                    tuple(int(value) for value in row)
                    for row in (
                        ((self._runtime_state.read("global_joint_window_plan") or {}).get("inferred_p1_row_matrix"))
                        or self._runtime_state.read("p1_inferred_from_p0")
                        or []
                    )
                )
                expected_send = tuple(int(value) for value in phase_ctx.send_splits)
                expected_recv = tuple(int(value) for value in phase_ctx.recv_splits)
                if inferred_p1:
                    local_index = tuple(int(v) for v in self.ep_group_ranks).index(int(self.rank))
                    inferred_send = tuple(int(inferred_p1[local_index][dst]) for dst in range(len(expected_send)))
                    inferred_recv = tuple(int(inferred_p1[src][local_index]) for src in range(len(expected_recv)))
                    inferred_total = int(sum(inferred_send) + sum(inferred_recv))
                    expected_total = int(sum(expected_send) + sum(expected_recv))
                    if inferred_total <= 0 and expected_total > 0:
                        self._timeline(
                            "p1_invariant_skipped_zero_inferred",
                            layer_name=layer_name,
                            inferred_send=list(inferred_send),
                            inferred_recv=list(inferred_recv),
                            actual_send=list(expected_send),
                            actual_recv=list(expected_recv),
                        )
                    elif inferred_send != expected_send or inferred_recv != expected_recv:
                        actual_p0_full = tuple(
                            tuple(int(value) for value in row)
                            for row in (((self._runtime_state.read("global_joint_window_plan") or {}).get("actual_p0_full_row_matrix")) or [])
                        )
                        raise RuntimeError(
                            f"local P1 invariant mismatch for {layer_name}: "
                            f"inferred_send={inferred_send} actual_send={expected_send} "
                            f"inferred_recv={inferred_recv} actual_recv={expected_recv} "
                            f"local_index={local_index} actual_p0_full_row={actual_p0_full[local_index] if actual_p0_full and local_index < len(actual_p0_full) else ()}"
                        )
                plan = self._compile_async_local_phase_plan(
                    layer_name=layer_name,
                    phase="P1",
                    local_context=phase_ctx,
                )
                if self.target_plan_store is not None and getattr(self, "execution_pipeline", None) is not None:
                    key = self._target_plan_key(layer_name=layer_name)
                    published_execution_plan = self._execution_plan_cache().get(self.target_plan_store._key(key))
                    if published_execution_plan is not None:
                        prepared_execution = self.execution_pipeline.prepare(
                            published_execution_plan,
                            self._actual_phase_context_from_ready_context(phase_ctx=phase_ctx),
                        )
                        self._record_instrumentation_measurement(
                            event_type="materialization",
                            layer_id=str(phase_ctx.layer_id),
                            phase="P1",
                            started_at_ns=int(time.monotonic_ns()),
                            ended_at_ns=int(time.monotonic_ns()),
                            details={"valid": bool(prepared_execution.validation.valid)},
                        )
                        if not prepared_execution.validation.valid:
                            self.target_plan_store.fail(key, execution_origin="materialization_invalid_p1")
                            self._runtime_state.write("execution_origin", "materialization_invalid_p1")
                            return
                        self._prepared_execution_cache()[self.target_plan_store._key(key)] = prepared_execution
                self._runtime_state.write("p1_planning_collective_count", 0)
                if self.observation_recorder is not None:
                    self.observation_recorder.record_scheduled_plan(
                        scheduled_plan_artifact(plan=plan, perf_profile=self._is_perf_profile())
                    )
                self._activate_transport(layer_name=layer_name, phase="P1", context=phase_ctx, plan=plan)
                self._runtime_state.write("before_async_p2p_phase_count", int(self._runtime_state.read("before_async_p2p_phase_count", 0) or 0) + 1)
                self._runtime_state.write("combine_transport_start_ns", int(time.monotonic_ns()))
                self._runtime_state.write("all_layer_async_phase_count", int(self._runtime_state.read("all_layer_async_phase_count", 0) or 0) + 1)
                if self._layer_selected(layer_name):
                    self._runtime_state.write("selected_layer_before_async_p2p_phase_count", int(self._runtime_state.read("selected_layer_before_async_p2p_phase_count", 0) or 0) + 1)
                self._timeline(
                    "phase_execution_plan_agreed",
                    layer_name=layer_name,
                    phase_name="P1",
                    plan_hash=plan.plan_hash,
                    wave_count=len(plan.waves),
                    bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
                    execution_mode=plan.execution_mode,
                    p1_planning_collective_count=0,
                )
                hook_end_ns = time.monotonic_ns()
                self._record_hook_timing(
                    layer_name=layer_name,
                    phase="P1",
                    hook_name="before_token_combine_total",
                    start_ns=hook_start_ns,
                    end_ns=hook_end_ns,
                    scheduled=True,
                    plan_hash=plan.plan_hash,
                    wave_count=int(len(plan.waves)),
                )
                self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=True, plan_hash=plan.plan_hash)
                return
            agreement_start_ns = time.monotonic_ns()
            policy = self._pending_window_adapter() if self.config.execution_mode == "multiphase_pending_window" else self._phase_policy()
            plan = run_phase_plan_agreement(local_context=phase_ctx, policy=policy, group=self.ep_process_group)
            agreement_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P1",
                stage="run_phase_plan_agreement",
                start_ns=agreement_start_ns,
                end_ns=agreement_end_ns,
                wave_count=int(len(plan.waves)),
                build_plan_time_us=float(plan.metrics.get("build_plan_time_us", 0.0) or 0.0),
                all_gather_time_us=float(plan.metrics.get("all_gather_time_us", 0.0) or 0.0),
                summary_build_time_us=float(plan.metrics.get("summary_build_time_us", 0.0) or 0.0),
                summary_encode_time_us=float(plan.metrics.get("summary_encode_time_us", 0.0) or 0.0),
                summary_decode_time_us=float(plan.metrics.get("summary_decode_time_us", 0.0) or 0.0),
                broadcast_time_us=float(plan.metrics.get("broadcast_time_us", 0.0) or 0.0),
                abstract_encode_time_us=float(plan.metrics.get("abstract_encode_time_us", 0.0) or 0.0),
                abstract_decode_time_us=float(plan.metrics.get("abstract_decode_time_us", 0.0) or 0.0),
                materialize_local_plan_time_us=float(plan.metrics.get("materialize_local_plan_time_us", 0.0) or 0.0),
                verify_time_us=float(plan.metrics.get("verify_time_us", 0.0) or 0.0),
            )
            if self.observation_recorder is not None:
                self.observation_recorder.record_scheduled_plan(
                    scheduled_plan_artifact(plan=plan, perf_profile=self._is_perf_profile())
                )
            self._record_control_replay_trace(phase_ctx=phase_ctx, plan=plan)
            self._record_pending_window_driver(layer_name=layer_name, phase="P1", plan=plan)
            self._activate_transport(layer_name=layer_name, phase="P1", context=phase_ctx, plan=plan)
            self._timeline(
                "phase_execution_plan_agreed",
                layer_name=layer_name,
                phase_name="P1",
                plan_hash=plan.plan_hash,
                wave_count=len(plan.waves),
                bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
                execution_mode=plan.execution_mode,
                all_gather_time_us=float(plan.metrics.get("all_gather_time_us", 0.0) or 0.0),
                build_plan_time_us=float(plan.metrics.get("build_plan_time_us", 0.0) or 0.0),
                summary_build_time_us=float(plan.metrics.get("summary_build_time_us", 0.0) or 0.0),
                summary_encode_time_us=float(plan.metrics.get("summary_encode_time_us", 0.0) or 0.0),
                summary_decode_time_us=float(plan.metrics.get("summary_decode_time_us", 0.0) or 0.0),
                broadcast_time_us=float(plan.metrics.get("broadcast_time_us", 0.0) or 0.0),
                abstract_encode_time_us=float(plan.metrics.get("abstract_encode_time_us", 0.0) or 0.0),
                abstract_decode_time_us=float(plan.metrics.get("abstract_decode_time_us", 0.0) or 0.0),
                materialize_local_plan_time_us=float(plan.metrics.get("materialize_local_plan_time_us", 0.0) or 0.0),
                verify_time_us=float(plan.metrics.get("verify_time_us", 0.0) or 0.0),
                total_agreement_time_us=float(plan.metrics.get("total_agreement_time_us", 0.0) or 0.0),
            )
            self._timeline("after_phase_plan", layer_name=layer_name, phase_name="P1", plan_hash=plan.plan_hash)
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P1",
                hook_name="before_token_combine_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                scheduled=True,
                plan_hash=plan.plan_hash,
                wave_count=int(len(plan.waves)),
            )
            self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=True, plan_hash=plan.plan_hash)
            return
        if self.config.scheduler_mode != "native_passthrough_identity":
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P1",
                hook_name="before_token_combine_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                scheduled=False,
                reason="phase_policy_not_selected",
            )
            self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=False)
            return
        hook_end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase="P1",
            hook_name="before_token_combine_total",
            start_ns=hook_start_ns,
            end_ns=hook_end_ns,
            scheduled=False,
            reason="native_passthrough_identity",
        )
        self._timeline("before_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=False)

    def after_token_combine(self, *, layer_name: str) -> None:
        hook_start_ns = time.monotonic_ns()
        self._timeline("after_token_combine_enter", layer_name=layer_name, phase_name="P1")
        layer_id = parse_layer_id(layer_name)
        if str(layer_id).isdigit():
            next_layer_id = str(int(layer_id) + 1)
            self._pump_target_planner_publications()
            self._poll_target_plan_slot(target_layer_id=next_layer_id, safe_point="source_combine_complete")
        if self.layer_role_for_name(layer_name) == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P1",
                hook_name="after_token_combine_total",
                start_ns=hook_start_ns,
            )
            self._timeline("after_token_combine_exit", layer_name=layer_name, phase_name="P1", scheduled=False)
            return
        active_transport = self.current_transport()
        if self.layer_role_for_name(layer_name) == "selected" and active_transport is not None and str(active_transport.get("layer_name")) == str(layer_name) and str(active_transport.get("phase")) == "P1":
            clear_start_ns = time.monotonic_ns()
            self.clear_transport(layer_name=layer_name, phase="P1")
            if self._is_joint_window_async_mode():
                self._runtime_state.write("after_async_p2p_phase_count", int(self._runtime_state.read("after_async_p2p_phase_count", 0) or 0) + 1)
                self._runtime_state.write("combine_transport_end_ns", int(time.monotonic_ns()))
                self._runtime_state.write("all_layer_async_phase_count", int(self._runtime_state.read("all_layer_async_phase_count", 0) or 0) + 1)
                if self._layer_selected(layer_name):
                    self._runtime_state.write("selected_layer_after_async_p2p_phase_count", int(self._runtime_state.read("selected_layer_after_async_p2p_phase_count", 0) or 0) + 1)
            clear_end_ns = time.monotonic_ns()
            self._record_planning_timing(
                layer_name=layer_name,
                phase="P1",
                stage="clear_transport",
                start_ns=clear_start_ns,
                end_ns=clear_end_ns,
            )
            observation_p1 = self._pending_p1.pop(layer_name, None)
            if observation_p1 is not None and self.config.p2_hint_mode == "calibrated_artifact" and not self._is_joint_window_async_mode():
                self._store_prepared_plan(layer_name=layer_name, observation_p1=observation_p1)
            if self._is_joint_window_async_mode():
                self._runtime_state.write("prepared_plan", None)
                self._runtime_state.remove("prepared_priority_cache", None)
            if observation_p1 is not None:
                self._record_window_state(layer_name=layer_name, p1_observation=observation_p1)
            if self._is_joint_window_async_mode() and self.target_plan_store is not None:
                execution_origin = str(self._runtime_state.read("execution_origin", "") or "")
                if execution_origin in {"prepared_exact", "prepared_repaired"}:
                    try:
                        self.target_plan_store.complete(
                            self._target_plan_key(layer_name=layer_name),
                            execution_origin=execution_origin,
                        )
                    except Exception as exc:
                        try:
                            self.target_plan_store.fail(
                                self._target_plan_key(layer_name=layer_name),
                                execution_origin="complete_failed",
                            )
                        except Exception:
                            pass
                        raise RuntimeError(f"prepared target completion failed for {layer_name}") from exc
            self._record_release_update(layer_name=layer_name, event="p1_return_completed")
            if self._should_stop_after_layer(layer_name=layer_name, phase="P1"):
                raise SelectedLayerStop(f"Stopped after selected P1 layer {layer_name}")
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P1",
                hook_name="after_token_combine_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
            )
            self._timeline("after_token_combine_exit", layer_name=layer_name, phase_name="P1")
            return
        if self.config.scheduler_mode == "native_passthrough_identity":
            self._timeline("native_p1_observed", layer_name=layer_name)
        hook_end_ns = time.monotonic_ns()
        self._record_hook_timing(
            layer_name=layer_name,
            phase="P1",
            hook_name="after_token_combine_total",
            start_ns=hook_start_ns,
            end_ns=hook_end_ns,
        )
        self._timeline("after_token_combine_exit", layer_name=layer_name, phase_name="P1")

    # Shadow-only native observation hooks

    def on_dispatch(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        hook_start_ns = time.monotonic_ns()
        layer_role = self.layer_role_for_name(layer_name)
        if layer_role == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P0",
                hook_name="on_dispatch_total",
                start_ns=hook_start_ns,
            )
            return
        try:
            hook_mode = self._hook_execution_mode(layer_name=layer_name)
            if hook_mode in {"DISABLED", "OBSERVATION_ONLY"} or layer_role != "selected":
                return
            if hook_mode == "REAL_EXECUTION_WITH_OBSERVATION":
                self._finalize_dispatch_observation(
                    layer_name=layer_name,
                    dispatcher=dispatcher,
                    hidden_states=hidden_states,
                )
                return
            self._runtime_state.metrics.shadow_dispatch_execution_count = int(
                self._runtime_state.metrics.shadow_dispatch_execution_count
            ) + 1
            observation = build_runtime_observation(
                run_id=self.run_id,
                step_id=self.step_id,
                microbatch_id=self.microbatch_id,
                model_revision_hash=self.model_revision_hash,
                request_table_hash=self.request_table_hash,
                hostname=self.hostname,
                layer_name=layer_name,
                rank=self.rank,
                local_rank=self.local_rank,
                ep_group_ranks=self.ep_group_ranks,
                ep_group_hash=compute_ep_group_hash(self.ep_group_ranks),
                dispatcher=dispatcher,
                phase="P0",
                hidden_states=hidden_states,
            )
            self._pending_p0[layer_name] = observation
        finally:
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P0",
                hook_name="on_dispatch_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                mode=self._hook_execution_mode(layer_name=layer_name),
            )

    def on_combine(self, *, layer_name: str, dispatcher: Any, hidden_states: Any) -> None:
        hook_start_ns = time.monotonic_ns()
        layer_role = self.layer_role_for_name(layer_name)
        if layer_role == "none":
            self._record_none_heavy_hook(
                layer_name=layer_name,
                phase="P1",
                hook_name="on_combine_total",
                start_ns=hook_start_ns,
            )
            return
        try:
            hook_mode = self._hook_execution_mode(layer_name=layer_name)
            if hook_mode in {"DISABLED", "OBSERVATION_ONLY"} or layer_role != "selected":
                return
            if hook_mode == "REAL_EXECUTION_WITH_OBSERVATION":
                self._finalize_combine_observation(
                    layer_name=layer_name,
                    dispatcher=dispatcher,
                    hidden_states=hidden_states,
                )
                return
            if layer_name not in self._pending_p0:
                return
            self._runtime_state.metrics.shadow_combine_execution_count = int(
                self._runtime_state.metrics.shadow_combine_execution_count
            ) + 1
            p0_observation = self._pending_p0.pop(layer_name)
            p1_observation = build_runtime_observation(
                run_id=self.run_id,
                step_id=self.step_id,
                microbatch_id=self.microbatch_id,
                model_revision_hash=self.model_revision_hash,
                request_table_hash=self.request_table_hash,
                hostname=self.hostname,
                layer_name=layer_name,
                rank=self.rank,
                local_rank=self.local_rank,
                ep_group_ranks=self.ep_group_ranks,
                ep_group_hash=compute_ep_group_hash(self.ep_group_ranks),
                dispatcher=dispatcher,
                phase="P1",
                hidden_states=hidden_states,
            )
            context = replace(self._context(layer_name), expert_placement_hash=p0_observation.expert_placement_hash)
            local_observations = (p0_observation, p1_observation)
            policy = self._phase_policy()
            self._runtime_state.metrics.shadow_plan_build_count = int(
                self._runtime_state.metrics.shadow_plan_build_count
            ) + 1
            self._runtime_state.metrics.shadow_policy_agreement_count = int(
                self._runtime_state.metrics.shadow_policy_agreement_count
            ) + 1
            self._runtime_state.metrics.shadow_control_collective_count = int(
                self._runtime_state.metrics.shadow_control_collective_count
            ) + 1
            plan, agreement = run_policy_agreement(
                local_observations=local_observations,
                context=context,
                policy=policy,
                device=torch.device(f"cuda:{self.local_rank}"),
                group=self.ep_process_group,
            )
            decision = InjectionDecision(
                accepted=True,
                fallback="native",
                plan_hash=plan.plan_hash,
                reason="native_order_passthrough" if plan.policy_name == "native_order" else "shadow_only_passthrough",
                policy_name=plan.policy_name,
                control_mode=self.config.control_mode,
            )
            self.completed.append(
                PolicyRuntimeRecord(
                    layer_name=layer_name,
                    context=context,
                    local_observations=local_observations,
                    plan=plan,
                    agreement=agreement,
                    decision=decision,
                )
            )
            self._record_observer(
                phase="policy_plan",
                layer=layer_name,
                rank=self.rank,
                local_rank=self.local_rank,
                policy_name=plan.policy_name,
                scheduler_mode=self.config.scheduler_mode,
                control_mode=self.config.control_mode,
                plan_hash=plan.plan_hash,
                execution_mode=plan.execution_mode,
                wave_count=len(plan.waves),
                ready_wave_count=len(plan.ready_waves),
                blocked_future_wave_count=len(plan.blocked_future_waves),
                agreement=agreement.to_dict(),
                decision=decision.to_dict(),
            )
        finally:
            hook_end_ns = time.monotonic_ns()
            self._record_hook_timing(
                layer_name=layer_name,
                phase="P1",
                hook_name="on_combine_total",
                start_ns=hook_start_ns,
                end_ns=hook_end_ns,
                mode=self._hook_execution_mode(layer_name=layer_name),
            )

    # Export helpers

    def _export_list(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(rows)

    def _export_observation_rows(self, method_name: str) -> list[dict[str, Any]]:
        if self.observation_recorder is None:
            return []
        export_fn = getattr(self.observation_recorder, method_name)
        return list(export_fn())

    def export_records(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.completed]

    def export_control_timeline(self) -> list[dict[str, Any]]:
        return self._export_list(self.control_timeline)

    def export_control_commands(self) -> list[dict[str, Any]]:
        return self._export_list(self.control_commands)

    def export_plan_arrival_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.plan_arrival_records)

    def export_window_state_records(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.window_state_records)

    def export_prepared_plan_bindings(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.prepared_plan_bindings)

    def export_release_events(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.release_events)

    def export_window_schedule_shadows(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.window_schedule_shadows)

    def export_prepared_phase_plan_shadows(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.prepared_phase_plan_shadows)

    def export_pending_window_driver_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.pending_window_driver_records)

    def export_planning_timing_records(self) -> list[dict[str, Any]]:
        return self._export_list(self.planning_timing_records)

    def export_control_replay_traces(self) -> list[dict[str, Any]]:
        if not self._replay_trace_enabled():
            return []
        return self._export_list(self.control_replay_traces)

    def export_prediction_audits(self) -> list[dict[str, Any]]:
        if self._is_perf_profile():
            return []
        return self._export_list(self.prediction_audits)

    def export_expert_route_traces(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_expert_route_traces")

    def export_source_expert_counts(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_source_expert_counts")

    def export_expert_to_traffic_audits(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_expert_to_traffic_audits")

    def export_expert_trace_warnings(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_expert_trace_warnings")

    def export_assertions(self) -> dict[str, Any]:
        return dict(self.assertion_state)

    def export_prepared_plan_summary(self) -> dict[str, Any]:
        return build_prepared_plan_summary(runtime_state=self._runtime_state)

    def export_phase_contexts(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_phase_contexts")

    def export_transport_bundles(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_transport_bundles")

    def export_scheduled_phase_plans(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_scheduled_phase_plans")

    def export_transport_execution_results(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_transport_execution")

    def export_captured_phase_tensors(self) -> list[dict[str, Any]]:
        rows = self._export_observation_rows("export_captured_phase_tensors")
        return [{key: value for key, value in item.items() if key != "tensor"} for item in rows]

    def export_captured_phase_tensors_with_payload(self) -> list[dict[str, Any]]:
        return self._export_observation_rows("export_captured_phase_tensors")
