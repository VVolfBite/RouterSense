"""Algorithm-layer scheduling contracts shared by offline and online paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FlowDemand:
    flow_id: str
    phase: str
    src_rank: int
    dst_rank: int
    byte_count: int
    release_state: str
    is_executable: bool
    dependency_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlowWindow:
    ready_flows: tuple[FlowDemand, ...] = ()
    blocked_flows: tuple[FlowDemand, ...] = ()
    forecast_pressure: tuple[FlowDemand, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LogicalWave:
    wave_id: int
    flows: tuple[FlowDemand, ...] = ()
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LogicalSchedulePlan:
    policy_name: str
    waves: tuple[LogicalWave, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseConstraint:
    phase: str
    rank: int
    release_after_phase: str = ""
    release_after_rank: int | None = None
    expert_compute_delay: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LogicalTopology:
    num_gpus: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastPressure:
    source: str
    digest: str
    oracle: bool
    evaluation_eligible: bool
    matrix_shape: tuple[int, int]
    matrix_total_bytes: int
    matrix: tuple[tuple[int, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalReadySetOptions:
    scheduling_mode: str
    information_mode: str = "p0_only"
    prediction_confidence: float = 0.0
    p0_weight: float = 1.0
    p1_reservation_weight: float = 1.0
    p2_hint_weight: float = 1.0
    p3_return_weight: float = 0.0
    max_waves: int = 256
    bucket_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultiPhaseSchedulingProblem:
    flow_window: FlowWindow
    topology: LogicalTopology
    release_model: ReleaseConstraint
    forecast: ForecastPressure | None
    options: GlobalReadySetOptions
    p0_dispatch_matrix: tuple[tuple[int, ...], ...]
    p1_return_matrix: tuple[tuple[int, ...], ...]
    p2_next_dispatch_forecast_matrix: tuple[tuple[int, ...], ...]
    p3_next_return_advisory_matrix: tuple[tuple[int, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedWindowPlan:
    window_key: str
    forecast_digest: str
    logical_plan: LogicalSchedulePlan
    created_at_layer_id: str
    applies_from_layer_id: str
    execution_capability_required: str
    forecast_matrix: tuple[tuple[int, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
