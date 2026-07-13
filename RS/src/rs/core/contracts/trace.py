"""Trace contracts used by offline and online execution paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

from .online_ep import OnlineExpertPlacement, OnlineLayerRouteTrace, RankManifest, TransportOperationRecord
from .router_trace import ExpertBucketRecord, LayerRouteTrace
from .validation import ValidationResult


class TraceOrigin:
    SINGLE_GPU_PROXY_ROUTER = "single_gpu_proxy_router"
    OBSERVED_SINGLE_RANK_LOCAL_MOE = "observed_single_rank_local_moe"
    OBSERVED_ONLINE_WS2_ROUTE_PARTITION = "observed_online_ws2_route_partition"
    OBSERVED_ONLINE_WS2_HIDDEN_DISPATCH = "observed_online_ws2_hidden_dispatch"
    OBSERVED_ONLINE_WS2_MOE_LAYER_HARNESS = "observed_online_ws2_moe_layer_harness"
    OBSERVED_ONLINE_NATIVE_EP = "observed_online_native_ep"
    OBSERVED_ONLINE_SCHEDULED_EP = "observed_online_scheduled_ep"
    LEGACY_TRACE_REPLAY = "legacy_trace_replay"


class FutureInformationMode:
    NONE = "none"
    ORACLE_FULL_TRACE = "oracle_full_trace"
    PREDICTED = "predicted"


class AuditEvidenceLevel(str, Enum):
    FULL = "full"
    SUMMARY_ONLY = "summary_only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RankStageTiming:
    rank: int
    stage: str
    wall_ms: float
    cuda_event_ms: float | None = None
    barrier_ms: float | None = None
    idle_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpExecutionTrace:
    trace_origin: str
    future_information_mode: str
    route_traces: list[LayerRouteTrace] = field(default_factory=list)
    stage_timings: list[RankStageTiming] = field(default_factory=list)
    expert_buckets: list[ExpertBucketRecord] = field(default_factory=list)
    online_route_traces: list[OnlineLayerRouteTrace] = field(default_factory=list)
    rank_manifests: list[RankManifest] = field(default_factory=list)
    expert_placements: list[OnlineExpertPlacement] = field(default_factory=list)
    transport_operations: list[TransportOperationRecord] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_origin": self.trace_origin,
            "future_information_mode": self.future_information_mode,
            "route_traces": [trace.to_dict() for trace in self.route_traces],
            "stage_timings": [timing.to_dict() for timing in self.stage_timings],
            "expert_buckets": [bucket.to_dict() for bucket in self.expert_buckets],
            "online_route_traces": [trace.to_dict() for trace in self.online_route_traces],
            "rank_manifests": [manifest.to_dict() for manifest in self.rank_manifests],
            "expert_placements": [placement.to_dict() for placement in self.expert_placements],
            "transport_operations": [record.to_dict() for record in self.transport_operations],
            "validation_results": [result.to_dict() for result in self.validation_results],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceEvent:
    event_type: str
    ts_ns: int
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["details"] = dict(self.details)
        return payload


class TraceSink(Protocol):
    def record(self, event: TraceEvent) -> None:
        ...

    def flush(self) -> tuple[TraceEvent, ...]:
        ...


@dataclass(frozen=True)
class ReferenceTraceBundle:
    run_identity: Mapping[str, object]
    topology: Mapping[str, object]
    traffic_observations: tuple[Mapping[str, object], ...] = ()
    prediction_records: tuple[Mapping[str, object], ...] = ()
    planning_records: tuple[Mapping[str, object], ...] = ()
    published_plan_records: tuple[Mapping[str, object], ...] = ()
    materialized_summaries: tuple[Mapping[str, object], ...] = ()
    execution_outcomes: tuple[Mapping[str, object], ...] = ()
    execution_truth: Mapping[str, object] = field(default_factory=dict)
    check_results: tuple[Mapping[str, object], ...] = ()
    measurement_snapshot: Mapping[str, object] = field(default_factory=dict)
    evidence_level: AuditEvidenceLevel = AuditEvidenceLevel.UNAVAILABLE

    def to_dict(self) -> dict[str, object]:
        return {
            "run_identity": dict(self.run_identity),
            "topology": dict(self.topology),
            "traffic_observations": [dict(item) for item in self.traffic_observations],
            "prediction_records": [dict(item) for item in self.prediction_records],
            "planning_records": [dict(item) for item in self.planning_records],
            "published_plan_records": [dict(item) for item in self.published_plan_records],
            "materialized_summaries": [dict(item) for item in self.materialized_summaries],
            "execution_outcomes": [dict(item) for item in self.execution_outcomes],
            "execution_truth": dict(self.execution_truth),
            "check_results": [dict(item) for item in self.check_results],
            "measurement_snapshot": dict(self.measurement_snapshot),
            "evidence_level": str(self.evidence_level.value),
        }


__all__ = [
    "AuditEvidenceLevel",
    "EpExecutionTrace",
    "FutureInformationMode",
    "RankStageTiming",
    "ReferenceTraceBundle",
    "TraceEvent",
    "TraceSink",
    "TraceOrigin",
]
