"""Megatron EP 在线运行时的顶层合同定义。

这个文件主要放：
- OnlineRuntimeConfig / RouterSenseInjectionConfig
- 运行时记录、计划、断言等共享 dataclass
它不负责执行逻辑，只定义 host/lifecycle/runtime 之间共享的数据形状。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from rs.scheduling.observation_contracts import (
    PeerFlow,
    PhaseDemand,
    PolicyContext,
    RankTopologyRecord,
    RouterSensePlan,
    RuntimeObservation,
    PlanWave,
)


DemandKnowledgeState = Literal["router_ready", "predictor_output"]
ReleaseState = Literal["ready", "blocked", "advisory_only"]
ReleaseDependency = Literal["none", "remote_expert_compute_complete"]
AssertionStatus = Literal["passed", "failed", "not_applicable"]
HookExecutionMode = Literal[
    "DISABLED",
    "OBSERVATION_ONLY",
    "REAL_EXECUTION_WITH_OBSERVATION",
    "LEGACY_SHADOW",
]


@dataclass(frozen=True)
class ExecutionSelection:
    layer_selector: str = "all"
    phase_selector: str = "both"
    selected_layer_ids: tuple[str, ...] = ()
    bucket_mode: str = "dynamic_current"
    bucket_rows: int = 0
    max_waves: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnlinePolicyParameters:
    p0_weight: float = 1.0
    p1_reservation_weight: float = 1.0
    p2_hint_weight: float = 0.0
    residual_weight: float = 0.75
    barrier_weight: float = 1.75
    age_weight: float = 0.15
    prediction_weight: float = 0.35
    p2_hint_mode: str = "none"
    p2_hint_artifact: str = ""
    calibrated_p2_enabled: bool = False
    online_p2_predictor: str = "copy_current_dispatch"
    safe_projection_mode: str = "host_select"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineValidationConfig:
    stop_after_selected_layer: bool = False
    executor_heartbeat_path: str = ""
    executor_phase_timeout_sec: int = 0
    preflight_mode: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OnlineRuntimeConfig:
    policy_name: str
    execution_mode: str
    control_mode: str
    execution_selection: ExecutionSelection
    policy_parameters: OnlinePolicyParameters
    observation: dict[str, Any] = field(default_factory=dict)
    validation: OnlineValidationConfig = field(default_factory=OnlineValidationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowInjectionConfig:
    scheduler_mode: str = "disabled"
    future_hint_mode: str = "none"
    shadow_command_arrival: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouterSenseInjectionConfig:
    policy: str = ""
    scheduler_mode: str = "disabled"
    execution_mode: str = "native_passthrough"
    future_hint_mode: str = "none"
    p2_hint_mode: str = "none"
    control_mode: str = "default_continue"
    policy_version: str = "v1"
    shadow_command_arrival: str = "none"
    bucket_mode: str = "dynamic_current"
    bucket_rows: int = 0
    max_waves: int = 256
    p0_weight: float = 1.0
    p1_reservation_weight: float = 1.0
    p2_hint_weight: float = 1.0
    residual_weight: float = 0.75
    barrier_weight: float = 1.75
    age_weight: float = 0.15
    prediction_weight: float = 0.35
    p2_hint_artifact: str = ""
    online_p2_predictor: str = "copy_current_dispatch"
    safe_projection_mode: str = "host_select"
    schedule_layer_selector: str = "all"
    schedule_phase_selector: str = "both"
    selected_layer_ids: tuple[str, ...] = ()
    capture_phase_tensors: bool = False
    capture_expert_trace: bool = False
    observation_profile: str = "minimal"
    invariant_mode: str = "diagnostic"
    legacy_compiler_bridge: bool = False
    capture_layer_selector: str = ""
    capture_phase_selector: str = ""
    heartbeat_enabled: bool = False
    per_wave_timing_enabled: bool = False
    replay_trace_enabled: bool = False
    stop_after_selected_layer: bool = False
    executor_heartbeat_path: str = ""
    executor_phase_timeout_sec: int = 0
    preflight_mode: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)




@dataclass(frozen=True)
class PlanAgreement:
    root_rank: int
    rank_count: int
    root_wire_hash: str
    root_semantic_hash: str
    decoded_semantic_hash: str
    observation_digest: str
    agreement_status: str
    policy_name: str
    policy_version: str
    control_mode: str
    observation_encode_ms: float
    observation_all_gather_ms: float
    planner_ms: float
    plan_broadcast_ms: float
    agreement_ms: float
    rank_hashes: tuple[str, ...]
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InjectionDecision:
    accepted: bool
    fallback: str
    plan_hash: str
    reason: str
    policy_name: str
    control_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeEPSummary:
    pipeline: str = "host_runtime_native_ep"
    host_runtime: str = "megatron_core"
    model_family: str = "olmoe"
    ep_size: int = 0
    dispatcher: str = "alltoall"
    backend: str = "nccl"
    forward_completed: bool = False
    remote_dispatch_exercised: bool = False
    remote_combine_exercised: bool = False
    is_legacy_harness: bool = False
    performance_claim_eligible: bool = False
    status: str = "blocked_environment"
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
