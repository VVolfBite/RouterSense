"""Megatron EP 正式执行链路的 P0/P1 生命周期主线。

这个文件是在线运行时的核心编排器，主要负责：
- before/after token_dispatch
- before/after token_combine
- phase context 构建、计划协商、transport 激活/清理
- prepared plan、release state、pending-window shadow 的记录
如果想看“运行时一层里到底发生了什么”，优先看这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rs.core.contracts.result import ResultBundle
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig, RuntimeObservation
from rs.runtime.online.megatron_ep.observation import PolicyRuntimeRecord, RouterSenseObserver, RuntimeObservationRecorder
from rs.runtime.online.megatron_ep.public_types import ControlGroupHandle
from rs.runtime.online.megatron_ep.state import PreparedWindowRuntimeState
from rs.runtime.online.megatron_ep.target_planning import TargetLayerPlannerService, TargetPlanStore, reconcile_once
from .lifecycle_parts import (
    ExpectedEvidence, LifecycleConfigurationMixin, LifecycleEvidenceMixin, LifecycleExportMixin,
    LifecycleHooksMixin, LifecyclePlanningMixin, LifecyclePredictionMixin, ReleaseStateLedger,
    RuntimeEvidenceCounters, RuntimePredictionCompatResult,
)


@dataclass
class RouterSenseInjectionRuntime(
    LifecycleConfigurationMixin,
    LifecycleEvidenceMixin,
    LifecyclePredictionMixin,
    LifecyclePlanningMixin,
    LifecycleHooksMixin,
    LifecycleExportMixin,
):
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
    expert_route_context_provider: Any | None = None
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
    _effective_planner_id_cache: str = ""
    _effective_planner_spec_cache: Any | None = None
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
    evidence_counters: RuntimeEvidenceCounters = field(default_factory=RuntimeEvidenceCounters)
    expected_evidence: ExpectedEvidence = field(default_factory=ExpectedEvidence)
    _runtime_failure_reason: str = ""

    # Configuration and policy selection



__all__ = [
    "ExpectedEvidence", "ReleaseStateLedger", "RouterSenseInjectionRuntime",
    "RuntimeEvidenceCounters", "RuntimePredictionCompatResult",
]
