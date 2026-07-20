"""观测面 schema 与 recorder。

主要内容：
- ObservationProfile / RuntimeObservationRecorder
- ExecutionAudit 数据结构
- build_runtime_observation() 运行时观测构建
这里定义“记录什么、保留什么、如何导出”。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from rs.core.contracts.observation import RuntimeObservationConfig
from rs.runtime.online.megatron_ep.contracts import (
    InjectionDecision,
    PlanAgreement,
    PolicyContext,
    RankTopologyRecord,
    RouterSensePlan,
    RuntimeObservation,
)
from rs.scheduling.validation import stable_hash


ObservationProfile = Literal["minimal", "perf", "timeline_light", "attribution_light", "execution", "debug"]
ExecutionAuditStatus = Literal["passed", "failed", "not_applicable"]


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

    def __init__(self, config: RuntimeObservationConfig) -> None:
        self.config = config

    def includes_execution(self) -> bool:
        return self.config.profile in {"perf", "timeline_light", "attribution_light", "execution", "debug"}

    def includes_debug(self) -> bool:
        return self.config.profile == "debug"

    def includes_perf(self) -> bool:
        return self.config.profile in {"perf", "timeline_light", "attribution_light"}


@dataclass(frozen=True)
class RuntimeObservationSnapshot:
    counters: dict[str, Any]
    phase_contexts: tuple[dict[str, Any], ...] = ()
    transport_bundles: tuple[dict[str, Any], ...] = ()
    scheduled_phase_plans: tuple[dict[str, Any], ...] = ()
    transport_execution: tuple[dict[str, Any], ...] = ()
    execution_audits: tuple[dict[str, Any], ...] = ()
    expert_route_traces: tuple[dict[str, Any], ...] = ()
    source_expert_counts: tuple[dict[str, Any], ...] = ()
    expert_to_traffic_audits: tuple[dict[str, Any], ...] = ()
    expert_trace_warnings: tuple[dict[str, Any], ...] = ()
    heartbeats: tuple[dict[str, Any], ...] = ()
    failures: tuple[dict[str, Any], ...] = ()
    captured_phase_tensors: tuple[dict[str, Any], ...] = ()


class RuntimeObservationRecorder:
    """Profile-aware runtime observation retention."""

    def __init__(self, config: RuntimeObservationConfig) -> None:
        self.config = config
        self._emitter = ObservationEmitter(config)
        self._phase_contexts: list[dict[str, Any]] = []
        self._transport_bundles: list[dict[str, Any]] = []
        self._scheduled_phase_plans: list[dict[str, Any]] = []
        self._transport_execution: list[dict[str, Any]] = []
        self._execution_audits: list[dict[str, Any]] = []
        self._expert_route_traces: list[dict[str, Any]] = []
        self._source_expert_counts: list[dict[str, Any]] = []
        self._expert_to_traffic_audits: list[dict[str, Any]] = []
        self._expert_trace_warnings: list[dict[str, Any]] = []
        self._heartbeats: list[dict[str, Any]] = []
        self._failures: list[dict[str, Any]] = []
        self._captured_phase_tensors: list[dict[str, Any]] = []
        self._counters: dict[str, Any] = {
            "phase_context_count": 0,
            "transport_bundle_count": 0,
            "scheduled_plan_count": 0,
            "transport_execution_count": 0,
            "execution_audit_count": 0,
            "expert_route_trace_count": 0,
            "source_expert_count_count": 0,
            "expert_to_traffic_audit_count": 0,
            "expert_trace_warning_count": 0,
            "heartbeat_count": 0,
            "failure_count": 0,
            "fallback_count": 0,
            "contract_violation_count": 0,
            "per_layer_phase_counts": {},
        }

    def record_phase_context(self, payload: dict[str, Any]) -> None:
        layer_id = str(payload.get("layer_id", "unknown"))
        phase = str(payload.get("phase", "unknown"))
        counters = self._counters["per_layer_phase_counts"]
        layer_key = f"{layer_id}:{phase}"
        counters[layer_key] = int(counters.get(layer_key, 0)) + 1
        self._counters["phase_context_count"] = int(self._counters["phase_context_count"]) + 1
        if self._emitter.includes_execution():
            self._phase_contexts.append(dict(payload))

    def record_transport_bundle(self, payload: dict[str, Any]) -> None:
        self._counters["transport_bundle_count"] = int(self._counters["transport_bundle_count"]) + 1
        if self._emitter.includes_execution():
            self._transport_bundles.append(dict(payload))

    def record_scheduled_plan(self, payload: dict[str, Any]) -> None:
        self._counters["scheduled_plan_count"] = int(self._counters["scheduled_plan_count"]) + 1
        if self._emitter.includes_execution():
            self._scheduled_phase_plans.append(dict(payload))

    def record_transport_execution(self, payload: dict[str, Any]) -> None:
        self._counters["transport_execution_count"] = int(self._counters["transport_execution_count"]) + 1
        event_type = str(payload.get("event_type", ""))
        if event_type == "native_fallback":
            self._counters["fallback_count"] = int(self._counters["fallback_count"]) + 1
        if event_type == "contract_violation":
            self._counters["contract_violation_count"] = int(self._counters["contract_violation_count"]) + 1
        if self._emitter.includes_execution():
            self._transport_execution.append(dict(payload))

    def record_execution_audit(self, payload: dict[str, Any]) -> None:
        self._counters["execution_audit_count"] = int(self._counters["execution_audit_count"]) + 1
        if self._emitter.includes_execution():
            self._execution_audits.append(dict(payload))

    def record_expert_route_trace(self, payload: dict[str, Any]) -> None:
        self._counters["expert_route_trace_count"] = int(self._counters["expert_route_trace_count"]) + 1
        if self._emitter.includes_debug() and self.config.capture_expert_trace:
            self._expert_route_traces.append(dict(payload))

    def record_source_expert_counts(self, payload: dict[str, Any]) -> None:
        self._counters["source_expert_count_count"] = int(self._counters["source_expert_count_count"]) + 1
        if self._emitter.includes_debug() and self.config.capture_expert_trace:
            self._source_expert_counts.append(dict(payload))

    def record_expert_to_traffic_audit(self, payload: dict[str, Any]) -> None:
        self._counters["expert_to_traffic_audit_count"] = int(self._counters["expert_to_traffic_audit_count"]) + 1
        if self._emitter.includes_debug() and self.config.capture_expert_trace:
            self._expert_to_traffic_audits.append(dict(payload))

    def record_expert_trace_warning(self, payload: dict[str, Any]) -> None:
        self._counters["expert_trace_warning_count"] = int(self._counters["expert_trace_warning_count"]) + 1
        if self._emitter.includes_debug() and self.config.capture_expert_trace:
            self._expert_trace_warnings.append(dict(payload))

    def record_heartbeat(self, payload: dict[str, Any]) -> None:
        self._counters["heartbeat_count"] = int(self._counters["heartbeat_count"]) + 1
        if self._emitter.includes_debug() and self.config.heartbeat_enabled:
            self._heartbeats.append(dict(payload))

    def record_failure(self, payload: dict[str, Any]) -> None:
        self._counters["failure_count"] = int(self._counters["failure_count"]) + 1
        if self._emitter.includes_debug():
            self._failures.append(dict(payload))

    def should_capture_tensor(self, *, layer_id: str, phase: str) -> bool:
        if not (self._emitter.includes_debug() and self.config.capture_enabled):
            return False
        layers = {value.strip() for value in self.config.capture_layer_selector.split(",") if value.strip()}
        phases = {value.strip().lower() for value in self.config.capture_phase_selector.split(",") if value.strip()}
        return (not layers or layer_id in layers) and (not phases or phase.lower() in phases)

    def record_captured_tensor(self, payload: dict[str, Any]) -> None:
        if self._emitter.includes_debug() and self.config.capture_enabled:
            self._captured_phase_tensors.append(dict(payload))

    def export_phase_contexts(self) -> list[dict[str, Any]]:
        return list(self._phase_contexts)

    def export_transport_bundles(self) -> list[dict[str, Any]]:
        return list(self._transport_bundles)

    def export_scheduled_phase_plans(self) -> list[dict[str, Any]]:
        return list(self._scheduled_phase_plans)

    def export_transport_execution(self) -> list[dict[str, Any]]:
        return list(self._transport_execution)

    def export_execution_audits(self) -> list[dict[str, Any]]:
        return list(self._execution_audits)

    def export_expert_route_traces(self) -> list[dict[str, Any]]:
        return list(self._expert_route_traces)

    def export_source_expert_counts(self) -> list[dict[str, Any]]:
        return list(self._source_expert_counts)

    def export_expert_to_traffic_audits(self) -> list[dict[str, Any]]:
        return list(self._expert_to_traffic_audits)

    def export_expert_trace_warnings(self) -> list[dict[str, Any]]:
        return list(self._expert_trace_warnings)

    def export_heartbeats(self) -> list[dict[str, Any]]:
        return list(self._heartbeats)

    def export_failures(self) -> list[dict[str, Any]]:
        return list(self._failures)

    def export_captured_phase_tensors(self) -> list[dict[str, Any]]:
        return list(self._captured_phase_tensors)

    def summary_counters(self) -> dict[str, Any]:
        return dict(self._counters)

    def snapshot(self) -> RuntimeObservationSnapshot:
        return RuntimeObservationSnapshot(
            counters=self.summary_counters(),
            phase_contexts=tuple(self._phase_contexts),
            transport_bundles=tuple(self._transport_bundles),
            scheduled_phase_plans=tuple(self._scheduled_phase_plans),
            transport_execution=tuple(self._transport_execution),
            execution_audits=tuple(self._execution_audits),
            expert_route_traces=tuple(self._expert_route_traces),
            source_expert_counts=tuple(self._source_expert_counts),
            expert_to_traffic_audits=tuple(self._expert_to_traffic_audits),
            expert_trace_warnings=tuple(self._expert_trace_warnings),
            heartbeats=tuple(self._heartbeats),
            failures=tuple(self._failures),
            captured_phase_tensors=tuple(self._captured_phase_tensors),
        )


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
