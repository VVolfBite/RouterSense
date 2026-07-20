"""Typed performance metric contracts shared by offline replay and online runtime.

The contracts deliberately distinguish logical offline time from measured wall
clock time.  A metric record is not publishable unless its strategy identity,
baseline identity, time unit and provenance are explicit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


METRIC_SCHEMA_VERSION = "routersense.performance_metrics.v1"
METRIC_DOMAINS = {"offline_logical", "online_wall_clock"}
TIME_UNITS = {"logical_time", "ns", "us", "ms", "s"}
PREDICTION_FIDELITIES = {
    "none",
    "faithful_fate",
    "fate_causal_remap_proxy",
    "perfect_trace",
    "ridge",
    "bridge",
    "zero_hint",
    "unknown",
}


def _finite_non_negative(value: float | int | None, *, name: str) -> None:
    if value is None:
        return
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _finite(value: float | int | None, *, name: str) -> None:
    if value is None:
        return
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class StrategyIdentity:
    planner_id: str
    timing: str
    horizon: str
    scope: str
    engine: str
    core: str
    predictor: str = "none"
    prediction_fidelity: str = "none"
    safety: str = "raw"

    def validate(self) -> None:
        if not self.planner_id.strip():
            raise ValueError("planner_id must be non-empty")
        if self.timing not in {"current", "future"}:
            raise ValueError("timing must be current or future")
        if self.horizon not in {"p01", "p012", "p0123"}:
            raise ValueError("horizon must be p01, p012 or p0123")
        if self.scope not in {"local", "joint"}:
            raise ValueError("scope must be local or joint")
        if self.engine not in {"event", "global"}:
            raise ValueError("engine must be event or global")
        if not self.core.strip():
            raise ValueError("core must be non-empty")
        if not self.predictor.strip():
            raise ValueError("predictor must be non-empty")
        if self.prediction_fidelity not in PREDICTION_FIDELITIES:
            raise ValueError("unsupported prediction_fidelity")
        if self.safety not in {"raw", "safe"}:
            raise ValueError("safety must be raw or safe")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StrategyIdentity":
        value = cls(
            planner_id=str(payload.get("planner_id", "")),
            timing=str(payload.get("timing", "")),
            horizon=str(payload.get("horizon", "")),
            scope=str(payload.get("scope", "")),
            engine=str(payload.get("engine", "")),
            core=str(payload.get("core", "")),
            predictor=str(payload.get("predictor", "none")),
            prediction_fidelity=str(payload.get("prediction_fidelity", "none")),
            safety=str(payload.get("safety", "raw")),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class MetricBaselineIdentity:
    planner_id: str
    comparison_key: str

    def validate(self) -> None:
        if not self.planner_id.strip():
            raise ValueError("baseline planner_id must be non-empty")
        if not self.comparison_key.strip():
            raise ValueError("baseline comparison_key must be non-empty")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MetricBaselineIdentity":
        value = cls(
            planner_id=str(payload.get("planner_id", "")),
            comparison_key=str(payload.get("comparison_key", "")),
        )
        value.validate()
        return value


@dataclass(frozen=True)
class MetricProvenance:
    metric_domain: str
    time_unit: str
    trace_digest: str
    sample_set_digest: str
    measurement_status: str
    source: str
    ep_size: int
    sample_count: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if self.metric_domain not in METRIC_DOMAINS:
            raise ValueError("unsupported metric_domain")
        if self.time_unit not in TIME_UNITS:
            raise ValueError("unsupported time_unit")
        if not self.trace_digest.strip():
            raise ValueError("trace_digest must be non-empty")
        if not self.sample_set_digest.strip():
            raise ValueError("sample_set_digest must be non-empty")
        if self.measurement_status not in {"complete", "partial", "invalid"}:
            raise ValueError("measurement_status must be complete, partial or invalid")
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if int(self.ep_size) <= 0:
            raise ValueError("ep_size must be positive")
        if int(self.sample_count) <= 0:
            raise ValueError("sample_count must be positive")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MetricProvenance":
        value = cls(
            metric_domain=str(payload.get("metric_domain", "")),
            time_unit=str(payload.get("time_unit", "")),
            trace_digest=str(payload.get("trace_digest", "")),
            sample_set_digest=str(payload.get("sample_set_digest", "")),
            measurement_status=str(payload.get("measurement_status", "")),
            source=str(payload.get("source", "")),
            ep_size=int(payload.get("ep_size", 0) or 0),
            sample_count=int(payload.get("sample_count", 1) or 1),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
        )
        value.validate()
        return value


@dataclass(frozen=True)
class OfflineWindowMetrics:
    communication_makespan: float
    tail_latency_p95: float | None
    tail_latency_p99: float | None
    tail_latency_max: float | None
    first_token_time: float | None
    planning_ms: float
    bind_ms: float
    target_entry_overhead_ms: float
    total_control_ms: float
    wave_count: int
    p1_remote_token_count: int
    current_layer_completion: float | None = None
    first_dispatch_arrival: float | None = None

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name in {"wave_count", "p1_remote_token_count"}:
                if int(value) < 0:
                    raise ValueError(f"{name} must be non-negative")
            else:
                _finite_non_negative(value, name=name)
        if self.tail_latency_p95 is not None and self.tail_latency_p99 is not None:
            if float(self.tail_latency_p95) > float(self.tail_latency_p99):
                raise ValueError("tail_latency_p95 must not exceed tail_latency_p99")
        if self.tail_latency_p99 is not None and self.tail_latency_max is not None:
            if float(self.tail_latency_p99) > float(self.tail_latency_max):
                raise ValueError("tail_latency_p99 must not exceed tail_latency_max")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OfflineWindowMetrics":
        def opt(name: str) -> float | None:
            value = payload.get(name)
            return None if value is None else float(value)
        result = cls(
            communication_makespan=float(payload.get("communication_makespan", 0.0) or 0.0),
            tail_latency_p95=opt("tail_latency_p95"),
            tail_latency_p99=opt("tail_latency_p99"),
            tail_latency_max=opt("tail_latency_max"),
            first_token_time=opt("first_token_time"),
            planning_ms=float(payload.get("planning_ms", 0.0) or 0.0),
            bind_ms=float(payload.get("bind_ms", 0.0) or 0.0),
            target_entry_overhead_ms=float(payload.get("target_entry_overhead_ms", 0.0) or 0.0),
            total_control_ms=float(payload.get("total_control_ms", 0.0) or 0.0),
            wave_count=int(payload.get("wave_count", 0) or 0),
            p1_remote_token_count=int(payload.get("p1_remote_token_count", 0) or 0),
            current_layer_completion=opt("current_layer_completion"),
            first_dispatch_arrival=opt("first_dispatch_arrival"),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class OptimizationMetrics:
    communication_optimization_pct: float | None
    tail_optimization_pct: float | None
    first_token_optimization_pct: float | None
    scope_communication_gain_pct: float | None = None
    scope_tail_gain_pct: float | None = None
    scope_first_token_gain_pct: float | None = None

    def validate(self) -> None:
        for name, value in asdict(self).items():
            _finite(value, name=name)

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "OptimizationMetrics":
        def opt(name: str) -> float | None:
            value = payload.get(name)
            return None if value is None else float(value)
        result = cls(**{name: opt(name) for name in cls.__dataclass_fields__})
        result.validate()
        return result


@dataclass(frozen=True)
class PerformanceMetricRecord:
    strategy: StrategyIdentity
    baseline: MetricBaselineIdentity
    provenance: MetricProvenance
    metrics: OfflineWindowMetrics
    optimization: OptimizationMetrics
    schema_version: str = METRIC_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != METRIC_SCHEMA_VERSION:
            raise ValueError("unsupported performance metric schema")
        self.strategy.validate()
        self.baseline.validate()
        self.provenance.validate()
        self.metrics.validate()
        self.optimization.validate()

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy.to_dict(),
            "baseline": self.baseline.to_dict(),
            "provenance": self.provenance.to_dict(),
            "metrics": self.metrics.to_dict(),
            "optimization": self.optimization.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PerformanceMetricRecord":
        for key in ("strategy", "baseline", "provenance", "metrics", "optimization"):
            if not isinstance(payload.get(key), Mapping):
                raise ValueError(f"performance metric {key} must be a mapping")
        result = cls(
            strategy=StrategyIdentity.from_dict(payload["strategy"]),
            baseline=MetricBaselineIdentity.from_dict(payload["baseline"]),
            provenance=MetricProvenance.from_dict(payload["provenance"]),
            metrics=OfflineWindowMetrics.from_dict(payload["metrics"]),
            optimization=OptimizationMetrics.from_dict(payload["optimization"]),
            schema_version=str(payload.get("schema_version", "")),
        )
        result.validate()
        return result



def validate_performance_metrics_payload(payload: object) -> tuple[PerformanceMetricRecord, ...]:
    """Validate a ResultBundle ``details.performance_metrics`` payload."""
    if payload is None:
        return ()
    rows = payload if isinstance(payload, list) else [payload]
    records: list[PerformanceMetricRecord] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("performance_metrics entries must be mappings")
        records.append(PerformanceMetricRecord.from_dict(row))
    return tuple(records)


__all__ = [
    "METRIC_SCHEMA_VERSION",
    "MetricBaselineIdentity",
    "MetricProvenance",
    "OfflineWindowMetrics",
    "OptimizationMetrics",
    "PerformanceMetricRecord",
    "StrategyIdentity",
    "validate_performance_metrics_payload",
]
