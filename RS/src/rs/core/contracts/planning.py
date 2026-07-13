from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from rs.core.hashing import stable_hash_dict

from .prediction import MatrixRows, PredictionHint


@dataclass(frozen=True)
class PlanningIdentity:
    request_id: str
    run_id: str | None = None
    forward_id: str | None = None
    window_id: str | None = None
    source_layer_id: str | None = None
    target_layer_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningTraffic:
    p0_dispatch_rows: MatrixRows
    p1_return_rows: MatrixRows

    def to_dict(self) -> dict[str, Any]:
        return {
            "p0_dispatch_rows": [list(row) for row in self.p0_dispatch_rows],
            "p1_return_rows": [list(row) for row in self.p1_return_rows],
        }


@dataclass(frozen=True)
class PlanningTopology:
    world_size: int
    full_duplex: bool = True
    max_outgoing_per_rank_per_wave: int = 1
    max_incoming_per_rank_per_wave: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningConstraints:
    bucket_rows: int
    max_waves: int
    expert_compute_delay: float
    phase_release_model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningWeights:
    p0_weight: float = 1.0
    p1_weight: float = 1.0
    p2_weight: float = 1.0
    residual_weight: float = 0.75
    barrier_weight: float = 1.75
    age_weight: float = 0.15
    prediction_weight: float = 0.35
    criticality_weight: float = 0.0
    iteration_budget: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanningRequest:
    identity: PlanningIdentity
    traffic: PlanningTraffic
    prediction_hint: PredictionHint
    topology: PlanningTopology
    constraints: PlanningConstraints
    weights: PlanningWeights
    information_mode: str

    def validate(self) -> None:
        world_size = int(self.topology.world_size)
        for name, matrix in {
            "p0_dispatch_rows": self.traffic.p0_dispatch_rows,
            "p1_return_rows": self.traffic.p1_return_rows,
            "prediction_hint.target_dispatch_rows": self.prediction_hint.target_dispatch_rows,
        }.items():
            if len(matrix) != world_size:
                raise ValueError(f"{name} row count {len(matrix)} does not match world_size {world_size}")
            widths = {len(row) for row in matrix}
            if widths != {world_size}:
                raise ValueError(f"{name} column widths {sorted(widths)} do not match world_size {world_size}")
        if int(self.constraints.bucket_rows) < 0:
            raise ValueError("bucket_rows must be >= 0")
        if int(self.constraints.max_waves) <= 0:
            raise ValueError("max_waves must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "traffic": self.traffic.to_dict(),
            "prediction_hint": self.prediction_hint.to_dict(),
            "topology": self.topology.to_dict(),
            "constraints": self.constraints.to_dict(),
            "weights": self.weights.to_dict(),
            "information_mode": str(self.information_mode),
        }

    def semantic_digest(self) -> str:
        self.validate()
        return stable_hash_dict(self.to_dict())


@dataclass(frozen=True)
class PlannedFlow:
    flow_id: str
    phase: str
    src_rank: int
    dst_rank: int
    row_count: int
    release_state: str
    executable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanWave:
    wave_id: int
    flows: tuple[PlannedFlow, ...]
    estimated_duration: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_id": int(self.wave_id),
            "flows": [flow.to_dict() for flow in self.flows],
            "estimated_duration": float(self.estimated_duration),
        }


@dataclass(frozen=True)
class WindowPlan:
    planner_id: str
    planner_family: str
    request_digest: str
    waves: tuple[PlanWave, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_id": str(self.planner_id),
            "planner_family": str(self.planner_family),
            "request_digest": str(self.request_digest),
            "waves": [wave.to_dict() for wave in self.waves],
            "metadata": dict(self.metadata),
        }

    def semantic_digest(self) -> str:
        return stable_hash_dict(self.to_dict())


@dataclass(frozen=True)
class PlanScore:
    estimated_makespan: float
    estimator_id: str
    cost_model_id: str
    valid: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_makespan": float(self.estimated_makespan),
            "estimator_id": str(self.estimator_id),
            "cost_model_id": str(self.cost_model_id),
            "valid": bool(self.valid),
            "reason": self.reason,
        }


__all__ = [
    "PlanScore",
    "PlanWave",
    "PlannedFlow",
    "PlanningConstraints",
    "PlanningIdentity",
    "PlanningRequest",
    "PlanningTopology",
    "PlanningTraffic",
    "PlanningWeights",
    "WindowPlan",
]
