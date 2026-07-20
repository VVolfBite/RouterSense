"""Shared imports for lifecycle mixins.

This module centralizes the runtime dependency surface so each lifecycle
stage module stays focused on orchestration logic.
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
from rs.core.contracts.result import ResultBundle, RunIdentity
from rs.evidence.result_builder import ResultBundleDraft, build_result_bundle
from rs.prediction import PredictionRegistry, resolve_predictor_id
from rs.planning import (
    CommonCorePlanEstimator,
    PlannerPolicyConfig,
    PlannerRegistry,
    PlannerSelectionMode,
    PlannerSelector,
    PlanningCostModel,
)
from rs.planning.request_builder import build_window_planning_request
from rs.scheduling.p012_future._kernel.axes import is_axes_planner_id, parse_planner_axes
from rs.planning.runtime_compat import resolve_phase_policy
from rs.runtime.online.megatron_ep.contracts import (
    HookExecutionMode,
    InjectionDecision,
    ObservationBundle,
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
from rs.runtime.online.megatron_ep.target_planning.contracts import _compat_logical_plan_from_window_plan
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
    DispatcherSynchronizationError,
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
from rs.runtime.online.megatron_ep.target_planning.contracts import PreparationToken
from rs.runtime.online.megatron_ep.target_planning.planner_service import PreparationSubmitStatus
from rs.runtime.online.megatron_ep.target_planning.fate_two_horizon import FateTwoHorizonRuntimePredictor
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



from .state import ExpectedEvidence, ReleaseStateLedger, RuntimeEvidenceCounters, RuntimePredictionCompatResult

__all__ = [name for name in globals() if not name.startswith("__")]
