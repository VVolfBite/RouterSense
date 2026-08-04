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
class TrafficObservationRecord:
    run_id: str
    layer_id: str
    phase: str
    layout_digest: str
    payload_roles: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrafficObservationRecord":
        return cls(
            run_id=str(payload.get("run_id", "")),
            layer_id=str(payload.get("layer_id", "")),
            phase=str(payload.get("phase", "")),
            layout_digest=str(payload.get("layout_digest", "")),
            payload_roles=tuple(str(item) for item in payload.get("payload_roles", ())),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
        )

    def validate(self) -> None:
        if not str(self.run_id):
            raise ValueError("traffic observation run_id must be non-empty")
        if not str(self.layer_id) or not str(self.phase):
            raise ValueError("traffic observation layer/phase must be non-empty")
        if not str(self.layout_digest):
            raise ValueError("traffic observation layout_digest must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "run_id": str(self.run_id),
            "layer_id": str(self.layer_id),
            "phase": str(self.phase),
            "layout_digest": str(self.layout_digest),
            "payload_roles": list(self.payload_roles),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _SimpleTraceRecord:
    record_type: str
    payload: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, record_type: str, payload: Mapping[str, object]) -> "_SimpleTraceRecord":
        return cls(record_type=str(record_type), payload=dict(payload))

    def validate(self) -> None:
        if not str(self.record_type):
            raise ValueError("trace record_type must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return dict(self.payload)


@dataclass(frozen=True)
class ReferenceTraceBundle:
    run_identity: Mapping[str, object]
    topology: Mapping[str, object]
    traffic_observations: tuple[TrafficObservationRecord, ...] = ()
    prediction_records: tuple[_SimpleTraceRecord, ...] = ()
    planning_records: tuple[_SimpleTraceRecord, ...] = ()
    published_plan_records: tuple[_SimpleTraceRecord, ...] = ()
    materialized_summaries: tuple[_SimpleTraceRecord, ...] = ()
    execution_outcomes: tuple[_SimpleTraceRecord, ...] = ()
    execution_truth: Mapping[str, object] = field(default_factory=dict)
    check_results: tuple[_SimpleTraceRecord, ...] = ()
    measurement_snapshot: Mapping[str, object] = field(default_factory=dict)
    evidence_level: AuditEvidenceLevel = AuditEvidenceLevel.UNAVAILABLE

    def validate(self) -> None:
        run_identity = dict(self.run_identity)
        topology = dict(self.topology)
        if not str(run_identity.get("run_id", "")).strip():
            raise ValueError("reference trace bundle run_identity.run_id must be non-empty")
        if "world_size" not in topology:
            raise ValueError("reference trace bundle topology.world_size is required")
        for item in self.traffic_observations:
            item.validate()
        for collection in (
            self.prediction_records,
            self.planning_records,
            self.published_plan_records,
            self.materialized_summaries,
            self.execution_outcomes,
            self.check_results,
        ):
            for item in collection:
                item.validate()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReferenceTraceBundle":
        bundle = cls(
            run_identity=dict(payload.get("run_identity", {})) if isinstance(payload.get("run_identity"), Mapping) else {},
            topology=dict(payload.get("topology", {})) if isinstance(payload.get("topology"), Mapping) else {},
            traffic_observations=tuple(
                TrafficObservationRecord.from_dict(item)
                for item in payload.get("traffic_observations", ())
                if isinstance(item, Mapping)
            ),
            prediction_records=tuple(
                _SimpleTraceRecord.from_dict("prediction", item)
                for item in payload.get("prediction_records", ())
                if isinstance(item, Mapping)
            ),
            planning_records=tuple(
                _SimpleTraceRecord.from_dict("planning", item)
                for item in payload.get("planning_records", ())
                if isinstance(item, Mapping)
            ),
            published_plan_records=tuple(
                _SimpleTraceRecord.from_dict("published_plan", item)
                for item in payload.get("published_plan_records", ())
                if isinstance(item, Mapping)
            ),
            materialized_summaries=tuple(
                _SimpleTraceRecord.from_dict("materialized_plan", item)
                for item in payload.get("materialized_summaries", ())
                if isinstance(item, Mapping)
            ),
            execution_outcomes=tuple(
                _SimpleTraceRecord.from_dict("execution_outcome", item)
                for item in payload.get("execution_outcomes", ())
                if isinstance(item, Mapping)
            ),
            execution_truth=dict(payload.get("execution_truth", {})) if isinstance(payload.get("execution_truth"), Mapping) else {},
            check_results=tuple(
                _SimpleTraceRecord.from_dict("check", item)
                for item in payload.get("check_results", ())
                if isinstance(item, Mapping)
            ),
            measurement_snapshot=dict(payload.get("measurement_snapshot", {})) if isinstance(payload.get("measurement_snapshot"), Mapping) else {},
            evidence_level=AuditEvidenceLevel(str(payload.get("evidence_level", AuditEvidenceLevel.UNAVAILABLE.value))),
        )
        bundle.validate()
        return bundle

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "run_identity": dict(self.run_identity),
            "topology": dict(self.topology),
            "traffic_observations": [item.to_dict() for item in self.traffic_observations],
            "prediction_records": [item.to_dict() for item in self.prediction_records],
            "planning_records": [item.to_dict() for item in self.planning_records],
            "published_plan_records": [item.to_dict() for item in self.published_plan_records],
            "materialized_summaries": [item.to_dict() for item in self.materialized_summaries],
            "execution_outcomes": [item.to_dict() for item in self.execution_outcomes],
            "execution_truth": dict(self.execution_truth),
            "check_results": [item.to_dict() for item in self.check_results],
            "measurement_snapshot": dict(self.measurement_snapshot),
            "evidence_level": str(self.evidence_level.value),
        }


__all__ = [
    "AuditEvidenceLevel",
    "EpExecutionTrace",
    "FutureInformationMode",
    "RankStageTiming",
    "ReferenceTraceBundle",
    "TrafficObservationRecord",
    "TraceEvent",
    "TraceSink",
    "TraceOrigin",
]
