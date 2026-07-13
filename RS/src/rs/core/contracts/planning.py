from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
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

    def validate(self) -> None:
        if not str(self.request_id):
            raise ValueError("request_id must be non-empty")
        for name, value in {
            "run_id": self.run_id,
            "forward_id": self.forward_id,
            "window_id": self.window_id,
            "source_layer_id": self.source_layer_id,
            "target_layer_id": self.target_layer_id,
        }.items():
            if value is not None and not str(value):
                raise ValueError(f"{name} must not be empty when provided")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PlanningTraffic:
    p0_dispatch_rows: MatrixRows
    p1_return_rows: MatrixRows

    def validate(self, *, world_size: int | None = None) -> None:
        _validate_matrix("p0_dispatch_rows", self.p0_dispatch_rows, world_size=world_size)
        _validate_matrix("p1_return_rows", self.p1_return_rows, world_size=world_size)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
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

    def validate(self) -> None:
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be > 0")
        if int(self.max_outgoing_per_rank_per_wave) <= 0:
            raise ValueError("max_outgoing_per_rank_per_wave must be > 0")
        if int(self.max_incoming_per_rank_per_wave) <= 0:
            raise ValueError("max_incoming_per_rank_per_wave must be > 0")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PlanningConstraints:
    bucket_rows: int
    max_waves: int
    expert_compute_delay: float
    phase_release_model: str

    def validate(self) -> None:
        if int(self.bucket_rows) < 0:
            raise ValueError("bucket_rows must be >= 0")
        if int(self.max_waves) <= 0:
            raise ValueError("max_waves must be > 0")
        if not math.isfinite(float(self.expert_compute_delay)) or float(self.expert_compute_delay) < 0.0:
            raise ValueError("expert_compute_delay must be finite and >= 0")
        if str(self.phase_release_model) not in {"p0_dispatch", "p1_return", "after_p1", "ready"}:
            raise ValueError(f"unsupported phase_release_model {self.phase_release_model!r}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
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

    def validate(self) -> None:
        for name, value in self.to_semantic_dict().items():
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
        if self.iteration_budget is not None and int(self.iteration_budget) <= 0:
            raise ValueError("iteration_budget must be > 0 when provided")

    def to_semantic_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return self.to_semantic_dict()


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
        self.identity.validate()
        self.topology.validate()
        world_size = int(self.topology.world_size)
        self.traffic.validate(world_size=world_size)
        self.prediction_hint.validate(world_size=world_size)
        self.constraints.validate()
        self.weights.validate()
        if str(self.information_mode) not in {"p0_p1_p2", "p0_only", "p0_p1"}:
            raise ValueError(f"unsupported information_mode {self.information_mode!r}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "identity": self.identity.to_dict(),
            "traffic": self.traffic.to_dict(),
            "prediction_hint": self.prediction_hint.to_dict(),
            "topology": self.topology.to_dict(),
            "constraints": self.constraints.to_dict(),
            "weights": self.weights.to_dict(),
            "information_mode": str(self.information_mode),
        }

    def semantic_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "semantic_version": "planning_request_v3",
            "traffic": self.traffic.to_dict(),
            "prediction_hint_target_dispatch_rows": [list(row) for row in self.prediction_hint.target_dispatch_rows],
            "topology": self.topology.to_dict(),
            "constraints": self.constraints.to_dict(),
            "weights": self.weights.to_dict(),
            "information_mode": str(self.information_mode),
        }

    def semantic_digest(self) -> str:
        return stable_hash_dict(self.semantic_payload())

    def identity_digest(self) -> str:
        return stable_hash_dict(
            {
                "identity_version": "planning_identity_v1",
                "identity": self.identity.to_dict(),
            }
        )


@dataclass(frozen=True)
class PlannedFlow:
    flow_id: str
    phase: str
    src_rank: int
    dst_rank: int
    row_count: int
    release_state: str
    executable: bool

    def validate(self) -> None:
        if not str(self.flow_id):
            raise ValueError("flow_id must be non-empty")
        if str(self.phase) not in {"p0_dispatch", "p1_return", "p2_next_dispatch_forecast", "p2_next_dispatch"}:
            raise ValueError(f"unsupported phase {self.phase!r}")
        if int(self.src_rank) < 0 or int(self.dst_rank) < 0:
            raise ValueError("src_rank and dst_rank must be >= 0")
        if int(self.row_count) < 0:
            raise ValueError("row_count must be >= 0")
        if str(self.release_state) not in {"ready", "none", "blocked", "after_p1", "advisory_only"}:
            raise ValueError(f"unsupported release_state {self.release_state!r}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PlanWave:
    wave_id: int
    flows: tuple[PlannedFlow, ...]
    estimated_duration: float

    def validate(self) -> None:
        seen_flow_ids: set[str] = set()
        if int(self.wave_id) < 0:
            raise ValueError("wave_id must be >= 0")
        if not math.isfinite(float(self.estimated_duration)) or float(self.estimated_duration) < 0.0:
            raise ValueError("estimated_duration must be finite and >= 0")
        for flow in self.flows:
            flow.validate()
            if flow.flow_id in seen_flow_ids:
                raise ValueError(f"duplicate flow_id {flow.flow_id!r} within wave")
            seen_flow_ids.add(flow.flow_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
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

    def validate(self) -> None:
        if not str(self.planner_id):
            raise ValueError("planner_id must be non-empty")
        if str(self.planner_family) not in {
            "baseline",
            "local",
            "joint",
            "reference_local",
            "reference_joint",
            "exact_local",
            "exact_joint",
        }:
            raise ValueError(f"unsupported planner_family {self.planner_family!r}")
        if not str(self.request_digest):
            raise ValueError("request_digest must be non-empty")
        seen_flow_ids: set[str] = set()
        for wave in self.waves:
            wave.validate()
            for flow in wave.flows:
                if flow.flow_id in seen_flow_ids:
                    raise ValueError(f"duplicate flow_id across plan: {flow.flow_id!r}")
                seen_flow_ids.add(flow.flow_id)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "planner_id": str(self.planner_id),
            "planner_family": str(self.planner_family),
            "request_digest": str(self.request_digest),
            "waves": [wave.to_dict() for wave in self.waves],
            "metadata": dict(self.metadata),
        }

    def semantic_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "semantic_version": "window_plan_v3",
            "planner_id": str(self.planner_id),
            "planner_family": str(self.planner_family),
            "request_digest": str(self.request_digest),
            "waves": [wave.to_dict() for wave in self.waves],
        }

    def semantic_digest(self) -> str:
        return stable_hash_dict(self.semantic_payload())

    def audit_digest(self) -> str:
        self.validate()
        return stable_hash_dict(
            {
                "audit_version": "window_plan_audit_v1",
                **self.to_dict(),
            }
        )


@dataclass(frozen=True)
class PlanScore:
    estimated_makespan: float
    estimator_id: str
    cost_model_id: str
    valid: bool
    reason: str | None = None

    def validate(self) -> None:
        if not math.isfinite(float(self.estimated_makespan)) and self.valid:
            raise ValueError("valid plan score must have finite estimated_makespan")
        if not str(self.estimator_id):
            raise ValueError("estimator_id must be non-empty")
        if not str(self.cost_model_id):
            raise ValueError("cost_model_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
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


def _validate_matrix(name: str, matrix: MatrixRows, *, world_size: int | None) -> None:
    widths = {len(row) for row in matrix}
    if world_size is None:
        if len(widths) > 1:
            raise ValueError(f"{name} has ragged row widths {sorted(widths)}")
    else:
        if len(matrix) != int(world_size):
            raise ValueError(f"{name} row count {len(matrix)} does not match world_size {world_size}")
        if widths != {int(world_size)}:
            raise ValueError(f"{name} column widths {sorted(widths)} do not match world_size {world_size}")
    for row in matrix:
        for value in row:
            if int(value) < 0:
                raise ValueError(f"{name} values must be non-negative")
