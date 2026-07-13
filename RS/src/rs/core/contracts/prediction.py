from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


MatrixRows = tuple[tuple[int, ...], ...]


def _validate_matrix(name: str, matrix: MatrixRows, *, world_size: int | None = None) -> None:
    if world_size is not None and int(world_size) <= 0:
        raise ValueError("world_size must be > 0")
    if world_size is not None and len(matrix) != int(world_size):
        raise ValueError(f"{name} row count {len(matrix)} does not match world_size {world_size}")
    widths = {len(row) for row in matrix}
    if world_size is not None:
        if widths != {int(world_size)}:
            raise ValueError(f"{name} column widths {sorted(widths)} do not match world_size {world_size}")
    elif len(widths) > 1:
        raise ValueError(f"{name} has ragged row widths {sorted(widths)}")
    for row in matrix:
        for value in row:
            if int(value) < 0:
                raise ValueError(f"{name} values must be non-negative")


@dataclass(frozen=True)
class PredictionIdentity:
    request_id: str
    run_id: str | None = None
    forward_id: str | None = None
    source_layer_id: str | None = None
    target_layer_id: str | None = None

    def validate(self) -> None:
        if not str(self.request_id):
            raise ValueError("request_id must be non-empty")
        for name, value in {
            "run_id": self.run_id,
            "forward_id": self.forward_id,
            "source_layer_id": self.source_layer_id,
            "target_layer_id": self.target_layer_id,
        }.items():
            if value is not None and not str(value):
                raise ValueError(f"{name} must not be empty when provided")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class ExpertRouteContext:
    identity: PredictionIdentity
    hidden_features: object
    gate_features: object | None
    top_k: int
    expert_owner_by_id: tuple[int, ...]
    world_size: int

    def validate(self) -> None:
        self.identity.validate()
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be > 0")
        if int(self.world_size) <= 0:
            raise ValueError("world_size must be > 0")
        if len(self.expert_owner_by_id) == 0:
            raise ValueError("expert_owner_by_id must be non-empty")
        for owner_rank in self.expert_owner_by_id:
            if int(owner_rank) < 0 or int(owner_rank) >= int(self.world_size):
                raise ValueError(f"expert owner rank {owner_rank} outside world_size {self.world_size}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
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
    current_return_rows: MatrixRows
    history_dispatch_rows: tuple[MatrixRows, ...]
    world_size: int

    def validate(self) -> None:
        self.identity.validate()
        _validate_matrix("current_dispatch_rows", self.current_dispatch_rows, world_size=int(self.world_size))
        _validate_matrix("current_return_rows", self.current_return_rows, world_size=int(self.world_size))
        for index, history_rows in enumerate(self.history_dispatch_rows):
            _validate_matrix(f"history_dispatch_rows[{index}]", history_rows, world_size=int(self.world_size))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "identity": self.identity.to_dict(),
            "current_dispatch_rows": [list(row) for row in self.current_dispatch_rows],
            "current_return_rows": [list(row) for row in self.current_return_rows],
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

    def validate(self, *, world_size: int | None = None) -> None:
        if not str(self.predictor_id):
            raise ValueError("predictor_id must be non-empty")
        if not str(self.hint_type):
            raise ValueError("hint_type must be non-empty")
        _validate_matrix("target_dispatch_rows", self.target_dispatch_rows, world_size=world_size)
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
                raise ValueError("confidence must be finite and within [0, 1]")
        for name, value in {
            "source_layer_id": self.source_layer_id,
            "target_layer_id": self.target_layer_id,
        }.items():
            if value is not None and not str(value):
                raise ValueError(f"{name} must not be empty when provided")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
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

    def validate(self, *, top_k: int | None = None, expert_count: int | None = None) -> None:
        expected_width: int | None = int(top_k) if top_k is not None else None
        if expected_width is not None and expected_width <= 0:
            raise ValueError("top_k must be > 0 when provided")
        for token_index, token_route in enumerate(self.expert_ids):
            if expected_width is not None and len(token_route) != expected_width:
                raise ValueError(
                    f"expert_ids[{token_index}] width {len(token_route)} does not match expected top_k {expected_width}"
                )
            for expert_id in token_route:
                if int(expert_id) < 0:
                    raise ValueError("expert IDs must be non-negative")
                if expert_count is not None and int(expert_id) >= int(expert_count):
                    raise ValueError(f"expert ID {expert_id} outside expert_count {expert_count}")
        if self.route_weights is not None:
            if len(self.route_weights) != len(self.expert_ids):
                raise ValueError("route_weights token count must match expert_ids token count")
            for token_index, (weights, token_route) in enumerate(zip(self.route_weights, self.expert_ids, strict=True)):
                if len(weights) != len(token_route):
                    raise ValueError(
                        f"route_weights[{token_index}] width {len(weights)} does not match expert_ids width {len(token_route)}"
                    )
                for value in weights:
                    if not math.isfinite(float(value)):
                        raise ValueError("route_weights values must be finite")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "expert_ids": [list(row) for row in self.expert_ids],
            "route_weights": None if self.route_weights is None else [list(row) for row in self.route_weights],
        }


@dataclass(frozen=True)
class RankedExpertRoutes:
    routes_by_source_rank: tuple[ExpertRoutePrediction, ...]

    def validate(self, *, world_size: int | None = None, expert_count: int | None = None) -> None:
        if world_size is not None and len(self.routes_by_source_rank) != int(world_size):
            raise ValueError(
                f"routes_by_source_rank count {len(self.routes_by_source_rank)} does not match world_size {world_size}"
            )
        for route in self.routes_by_source_rank:
            route.validate(expert_count=expert_count)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "routes_by_source_rank": [route.to_dict() for route in self.routes_by_source_rank],
        }


@dataclass(frozen=True)
class PredictionResult:
    identity: PredictionIdentity
    hint: PredictionHint
    expert_route: ExpertRoutePrediction | None = None
    auxiliary: Mapping[str, object] = field(default_factory=dict)

    def validate(self, *, world_size: int | None = None) -> None:
        self.identity.validate()
        self.hint.validate(world_size=world_size)
        if self.hint.source_layer_id is not None and self.identity.source_layer_id is not None:
            if str(self.hint.source_layer_id) != str(self.identity.source_layer_id):
                raise ValueError("hint.source_layer_id must match identity.source_layer_id")
        if self.hint.target_layer_id is not None and self.identity.target_layer_id is not None:
            if str(self.hint.target_layer_id) != str(self.identity.target_layer_id):
                raise ValueError("hint.target_layer_id must match identity.target_layer_id")
        if self.expert_route is not None:
            self.expert_route.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
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
