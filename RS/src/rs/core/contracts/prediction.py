from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


MatrixRows = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PredictionIdentity:
    request_id: str
    run_id: str | None = None
    forward_id: str | None = None
    source_layer_id: str | None = None
    target_layer_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpertRouteContext:
    identity: PredictionIdentity
    hidden_features: object
    gate_features: object | None
    top_k: int
    expert_owner_by_id: tuple[int, ...]
    world_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "hidden_features_type": type(self.hidden_features).__name__,
            "gate_features_type": None if self.gate_features is None else type(self.gate_features).__name__,
            "top_k": int(self.top_k),
            "expert_owner_by_id": list(int(item) for item in self.expert_owner_by_id),
            "world_size": int(self.world_size),
        }


@dataclass(frozen=True)
class TrafficHistoryContext:
    identity: PredictionIdentity
    current_dispatch_rows: MatrixRows
    history_dispatch_rows: tuple[MatrixRows, ...]
    world_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "current_dispatch_rows": [list(row) for row in self.current_dispatch_rows],
            "history_dispatch_rows": [[list(cell) for cell in rows] for rows in self.history_dispatch_rows],
            "world_size": int(self.world_size),
        }


PredictionContext = ExpertRouteContext | TrafficHistoryContext


@dataclass(frozen=True)
class PredictionHint:
    predictor_id: str
    hint_type: str
    target_dispatch_rows: MatrixRows
    confidence: float | None
    oracle: bool = False
    source_layer_id: str | None = None
    target_layer_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictor_id": str(self.predictor_id),
            "hint_type": str(self.hint_type),
            "target_dispatch_rows": [list(row) for row in self.target_dispatch_rows],
            "confidence": None if self.confidence is None else float(self.confidence),
            "oracle": bool(self.oracle),
            "source_layer_id": self.source_layer_id,
            "target_layer_id": self.target_layer_id,
        }


@dataclass(frozen=True)
class ExpertRoutePrediction:
    expert_ids: tuple[tuple[int, ...], ...]
    route_weights: tuple[tuple[float, ...], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert_ids": [list(row) for row in self.expert_ids],
            "route_weights": None if self.route_weights is None else [list(row) for row in self.route_weights],
        }


@dataclass(frozen=True)
class RankedExpertRoutes:
    routes_by_source_rank: tuple[ExpertRoutePrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "routes_by_source_rank": [route.to_dict() for route in self.routes_by_source_rank],
        }


@dataclass(frozen=True)
class PredictionResult:
    identity: PredictionIdentity
    hint: PredictionHint
    expert_route: ExpertRoutePrediction | None = None
    auxiliary: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "hint": self.hint.to_dict(),
            "expert_route": None if self.expert_route is None else self.expert_route.to_dict(),
            "auxiliary": dict(self.auxiliary),
        }


__all__ = [
    "ExpertRouteContext",
    "ExpertRoutePrediction",
    "RankedExpertRoutes",
    "MatrixRows",
    "PredictionContext",
    "PredictionHint",
    "PredictionIdentity",
    "PredictionResult",
    "TrafficHistoryContext",
]
