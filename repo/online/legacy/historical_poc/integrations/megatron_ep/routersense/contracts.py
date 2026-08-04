from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


DemandKnowledgeState = Literal["router_ready", "predictor_output"]
ReleaseState = Literal["ready", "blocked", "advisory_only"]
ReleaseDependency = Literal["none", "remote_expert_compute_complete"]
AssertionStatus = Literal["passed", "failed", "not_applicable"]


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
    bucket_rows: int = 0
    p0_weight: float = 1.0
    p1_reservation_weight: float = 1.0
    p2_hint_weight: float = 1.0
    p2_hint_artifact: str = ""
    schedule_layer_selector: str = "all"
    schedule_phase_selector: str = "both"
    capture_phase_tensors: bool = False
    stop_after_selected_layer: bool = False
    executor_heartbeat_path: str = ""
    executor_phase_timeout_sec: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankTopologyRecord:
    global_rank: int
    local_rank: int
    node_index: int
    hostname_digest: str
    device_index: int
    ep_group_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeObservation:
    run_id: str
    step_id: str
    microbatch_id: str
    layer_id: str
    layer_name: str
    global_rank: int
    local_rank: int
    node_id: str
    device: str
    ep_group_ranks: tuple[int, ...]
    ep_group_size: int
    dispatcher_class: str
    expert_placement_hash: str
    model_revision_hash: str
    dispatcher_hash: str
    ep_group_hash: str
    request_table_hash: str
    run_id_digest: str
    step_id_digest: str
    microbatch_id_digest: str
    phase: str
    per_peer_rows: tuple[int, ...]
    per_peer_bytes: tuple[int, ...]
    local_rows: int
    remote_rows: int
    topology: RankTopologyRecord
    tokens_per_expert: tuple[int, ...] = ()
    input_splits: tuple[int, ...] = ()
    output_splits: tuple[int, ...] = ()
    observation_digest: str = ""
    availability: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PeerFlow:
    flow_id: str
    src_rank: int
    dst_rank: int
    phase: str
    rows: int
    bytes: int
    demand_known_at: DemandKnowledgeState
    release_state: ReleaseState
    release_dependency: ReleaseDependency
    payload_exists: bool
    is_cross_rank: bool
    is_cross_node: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseDemand:
    phase: str
    demand_known_at: DemandKnowledgeState
    release_state: ReleaseState
    release_dependency: ReleaseDependency
    payload_exists: bool
    flows: tuple[PeerFlow, ...]
    total_remote_rows: int
    total_remote_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyContext:
    run_id: str
    step_id: str
    microbatch_id: str
    layer_id: str
    run_id_digest: str
    step_id_digest: str
    microbatch_id_digest: str
    request_table_hash: str
    model_revision_hash: str
    expert_placement_hash: str
    ep_group_ranks: tuple[int, ...]
    ep_group_size: int
    ep_group_hash: str
    future_hint_mode: str
    control_mode: str
    full_duplex: bool = True
    max_outgoing_per_rank_per_wave: int = 1
    max_incoming_per_rank_per_wave: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanWave:
    wave_id: int
    release_state: ReleaseState
    flows: tuple[PeerFlow, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouterSensePlan:
    run_id: str
    step_id: str
    microbatch_id: str
    layer_id: str
    ep_group_hash: str
    request_table_hash: str
    model_revision_hash: str
    expert_placement_hash: str
    observation_digest: str
    plan_hash: str
    policy_name: str
    policy_version: str
    execution_mode: str
    transport_mutation: bool
    future_hint_mode: str
    control_mode: str
    is_shadow_only: bool
    can_preempt: bool = False
    phase_demands: tuple[PhaseDemand, ...] = ()
    ready_waves: tuple[PlanWave, ...] = ()
    blocked_future_waves: tuple[PlanWave, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def waves(self) -> tuple[PlanWave, ...]:
        return self.ready_waves + self.blocked_future_waves

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["waves"] = [wave.to_dict() for wave in self.waves]
        return payload


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
