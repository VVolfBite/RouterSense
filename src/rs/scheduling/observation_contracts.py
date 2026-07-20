"""Shared observation and shadow-planning contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


DemandKnowledgeState = Literal["router_ready", "predictor_output"]
ReleaseState = Literal["ready", "blocked", "advisory_only"]
ReleaseDependency = Literal["none", "remote_expert_compute_complete"]


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
class ObservationBundle:
    run_id: str
    forward_generation: int
    microbatch_id: str
    layer_id: str
    ep_group_ranks: tuple[int, ...]
    observations_by_phase: dict[str, RuntimeObservation]

    def validate(self) -> None:
        if not str(self.run_id):
            raise ValueError("run_id must be non-empty")
        if int(self.forward_generation) < 0:
            raise ValueError("forward_generation must be >= 0")
        if not str(self.microbatch_id):
            raise ValueError("microbatch_id must be non-empty")
        if not str(self.layer_id):
            raise ValueError("layer_id must be non-empty")
        if not self.ep_group_ranks:
            raise ValueError("ep_group_ranks must be non-empty")
        if not self.observations_by_phase:
            raise ValueError("observations_by_phase must be non-empty")
        for phase, observation in self.observations_by_phase.items():
            if str(phase) != str(observation.phase):
                raise ValueError("bundle phase key must match observation.phase")
            if str(observation.layer_id) != str(self.layer_id):
                raise ValueError("bundle layer_id mismatch")
            if tuple(int(v) for v in observation.ep_group_ranks) != tuple(int(v) for v in self.ep_group_ranks):
                raise ValueError("bundle ep_group_ranks mismatch")

    def semantic_digest(self) -> str:
        self.validate()
        return hashlib.sha256(
            json.dumps(
                {
                    "run_id": str(self.run_id),
                    "forward_generation": int(self.forward_generation),
                    "microbatch_id": str(self.microbatch_id),
                    "layer_id": str(self.layer_id),
                    "ep_group_ranks": [int(v) for v in self.ep_group_ranks],
                    "observations_by_phase": {
                        str(phase): observation.to_dict()
                        for phase, observation in sorted(self.observations_by_phase.items())
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": str(self.run_id),
            "forward_generation": int(self.forward_generation),
            "microbatch_id": str(self.microbatch_id),
            "layer_id": str(self.layer_id),
            "ep_group_ranks": [int(v) for v in self.ep_group_ranks],
            "observations_by_phase": {
                str(phase): observation.to_dict()
                for phase, observation in sorted(self.observations_by_phase.items())
            },
            "semantic_digest": self.semantic_digest(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObservationBundle":
        bundle = cls(
            run_id=str(payload.get("run_id", "")),
            forward_generation=int(payload.get("forward_generation", 0) or 0),
            microbatch_id=str(payload.get("microbatch_id", "")),
            layer_id=str(payload.get("layer_id", "")),
            ep_group_ranks=tuple(int(v) for v in payload.get("ep_group_ranks", ())),
            observations_by_phase={
                str(phase): RuntimeObservation(**dict(observation_payload))
                for phase, observation_payload in dict(payload.get("observations_by_phase", {}) or {}).items()
            },
        )
        bundle.validate()
        return bundle


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
