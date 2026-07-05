"""Observation profile contracts and runtime observation builders."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from rs.runtime.online.megatron_ep.contracts import (
    InjectionDecision,
    PlanAgreement,
    PolicyContext,
    RankTopologyRecord,
    RouterSensePlan,
    RuntimeObservation,
)
from rs.scheduling.validation import stable_hash


ObservationProfile = Literal["minimal", "execution", "debug"]
ExecutionAuditStatus = Literal["passed", "failed", "not_applicable"]


@dataclass(frozen=True)
class ObservationConfig:
    profile: ObservationProfile
    capture_enabled: bool = False
    capture_layer_selector: str = ""
    capture_phase_selector: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionAudit:
    status: ExecutionAuditStatus
    policy_name: str
    plan_hash: str
    phase: str
    layer_id: str
    planned_wave_count: int
    executed_wave_count: int
    planned_task_ids: tuple[str, ...] = ()
    executed_task_ids: tuple[str, ...] = ()
    missing_tasks: tuple[str, ...] = ()
    unexpected_tasks: tuple[str, ...] = ()
    duplicate_tasks: tuple[str, ...] = ()
    order_mismatches: tuple[str, ...] = ()
    planned_rows: int = 0
    actual_rows: int = 0
    planned_bytes: int = 0
    actual_bytes: int = 0
    native_fallback_events: int = 0
    contract_violation_events: int = 0
    p0_bundle_atomicity_preserved: bool = True
    local_copy_coverage_passed: bool = True
    remote_flow_coverage_passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObservationEmitter:
    """Small helper for profile-gated event collection."""

    def __init__(self, config: ObservationConfig) -> None:
        self.config = config

    def includes_execution(self) -> bool:
        return self.config.profile in {"execution", "debug"}

    def includes_debug(self) -> bool:
        return self.config.profile == "debug"


def parse_layer_id(layer_name: str) -> str:
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return match.group(1)
    return "unknown"


def extract_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        return tuple(int(item) for item in value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        flattened: list[int] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                flattened.extend(int(sub_item) for sub_item in item)
            else:
                flattened.append(int(item))
        return tuple(flattened)
    if hasattr(value, "tolist"):
        try:
            return extract_int_tuple(value.tolist())
        except Exception:
            return ()
    return ()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_runtime_observation(
    *,
    run_id: str,
    step_id: str,
    microbatch_id: str,
    model_revision_hash: str,
    request_table_hash: str,
    hostname: str,
    layer_name: str,
    rank: int,
    local_rank: int,
    ep_group_ranks: tuple[int, ...],
    ep_group_hash: str,
    dispatcher: Any,
    phase: str,
    hidden_states: Any,
) -> RuntimeObservation:
    peer_count = len(ep_group_ranks)
    rank_index = ep_group_ranks.index(rank) if rank in ep_group_ranks else 0
    split_attr = "input_splits" if phase == "P0" else "output_splits"
    splits = list(extract_int_tuple(getattr(dispatcher, split_attr, None))[:peer_count])
    splits.extend([0] * max(0, peer_count - len(splits)))
    per_peer_rows = tuple(int(v) for v in splits)
    elem_size = int(hidden_states.element_size()) if isinstance(hidden_states, torch.Tensor) else 0
    hidden_dim = int(hidden_states.shape[-1]) if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim >= 2 else 0
    per_peer_bytes = tuple(int(rows * hidden_dim * elem_size) for rows in per_peer_rows)
    local_rows = int(per_peer_rows[rank_index]) if rank_index < len(per_peer_rows) else 0
    remote_rows = sum(int(value) for idx, value in enumerate(per_peer_rows) if idx != rank_index)
    tokens_per_expert = extract_int_tuple(getattr(dispatcher, "num_global_tokens_per_local_expert", None))
    input_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "input_splits", None))[:peer_count])
    output_splits = tuple(int(v) for v in extract_int_tuple(getattr(dispatcher, "output_splits", None))[:peer_count])
    run_id_digest = digest_text(run_id)
    step_id_digest = digest_text(step_id)
    microbatch_id_digest = digest_text(microbatch_id)
    dispatcher_hash = digest_text(type(dispatcher).__name__)
    hostname_digest = digest_text(hostname)
    availability = {
        "step_id": "unknown" if step_id == "unknown" else "available",
        "microbatch_id": "unknown" if microbatch_id == "unknown" else "available",
        "layer_id": "unknown" if parse_layer_id(layer_name) == "unknown" else "available",
        "tokens_per_expert": "available" if tokens_per_expert else "unknown",
    }
    expert_placement_hash = digest_text(
        stable_hash(
            {
                "placement_mode": "megatron_native_ep",
                "ep_group_ranks": list(ep_group_ranks),
                "ep_group_size": len(ep_group_ranks),
                "dispatcher_class": type(dispatcher).__name__,
            }
        )
    )
    digest_payload = {
        "run_id_digest": run_id_digest,
        "step_id_digest": step_id_digest,
        "microbatch_id_digest": microbatch_id_digest,
        "layer_id": parse_layer_id(layer_name),
        "rank": rank,
        "phase": phase,
        "per_peer_rows": list(per_peer_rows),
        "per_peer_bytes": list(per_peer_bytes),
        "local_rows": local_rows,
        "remote_rows": remote_rows,
        "expert_placement_hash": expert_placement_hash,
        "model_revision_hash": model_revision_hash,
        "request_table_hash": request_table_hash,
        "hostname_digest": hostname_digest,
    }
    topology = RankTopologyRecord(
        global_rank=rank,
        local_rank=local_rank,
        node_index=-1,
        hostname_digest=hostname_digest,
        device_index=local_rank,
        ep_group_rank=rank_index,
    )
    return RuntimeObservation(
        run_id=run_id,
        step_id=step_id,
        microbatch_id=microbatch_id,
        layer_id=parse_layer_id(layer_name),
        layer_name=layer_name,
        global_rank=rank,
        local_rank=local_rank,
        node_id=hostname,
        device=f"cuda:{local_rank}",
        ep_group_ranks=ep_group_ranks,
        ep_group_size=len(ep_group_ranks),
        dispatcher_class=type(dispatcher).__name__,
        expert_placement_hash=expert_placement_hash,
        model_revision_hash=model_revision_hash,
        dispatcher_hash=dispatcher_hash,
        ep_group_hash=ep_group_hash,
        request_table_hash=request_table_hash,
        run_id_digest=run_id_digest,
        step_id_digest=step_id_digest,
        microbatch_id_digest=microbatch_id_digest,
        phase=phase,
        per_peer_rows=per_peer_rows,
        per_peer_bytes=per_peer_bytes,
        local_rows=local_rows,
        remote_rows=remote_rows,
        topology=topology,
        tokens_per_expert=tokens_per_expert,
        input_splits=input_splits,
        output_splits=output_splits,
        observation_digest=stable_hash(digest_payload),
        availability=availability,
    )


@dataclass
class PolicyRuntimeRecord:
    layer_name: str
    context: PolicyContext
    local_observations: tuple[RuntimeObservation, ...]
    plan: RouterSensePlan
    agreement: PlanAgreement
    decision: InjectionDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "context": self.context.to_dict(),
            "local_observations": [item.to_dict() for item in self.local_observations],
            "plan": self.plan.to_dict(),
            "agreement": self.agreement.to_dict(),
            "decision": self.decision.to_dict(),
        }
