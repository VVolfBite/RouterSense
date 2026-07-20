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
    prediction_kind: str | None = None

    @staticmethod
    def _canonical_kind(*, predictor_id: str, hint_type: str, oracle: bool) -> str:
        normalized_kind = str(hint_type or "").strip()
        normalized_predictor = str(predictor_id or "").strip()
        if normalized_kind in {
            "zero_hint",
            "copy_current_dispatch",
            "learned_prediction",
            "perfect_trace_hint",
        }:
            return normalized_kind
        if normalized_kind == "traffic_matrix":
            if bool(oracle):
                return "perfect_trace_hint"
            if normalized_predictor in {"zero", "zero_hint", "none"}:
                return "zero_hint"
            if normalized_predictor in {"copy_current", "copy_current_dispatch"}:
                return "copy_current_dispatch"
            return "learned_prediction"
        if normalized_kind in {
            "different_hint",
            "history_ema",
            "history_linear_trend",
            "ridge_linear_trace_predictor",
            "shuffled_control",
        }:
            return "learned_prediction"
        return normalized_kind

    def validate(self, *, world_size: int | None = None) -> None:
        if not str(self.predictor_id):
            raise ValueError("predictor_id must be non-empty")
        if not str(self.hint_type):
            raise ValueError("hint_type must be non-empty")
        canonical_kind = self._canonical_kind(
            predictor_id=str(self.predictor_id),
            hint_type=str(self.hint_type),
            oracle=bool(self.oracle),
        )
        if canonical_kind not in {
            "zero_hint",
            "copy_current_dispatch",
            "learned_prediction",
            "perfect_trace_hint",
            "expert_route",
        }:
            raise ValueError(f"unsupported prediction kind {canonical_kind!r}")
        _validate_matrix("target_dispatch_rows", self.target_dispatch_rows, world_size=world_size)
        if self.confidence is not None:
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
                raise ValueError("confidence must be finite and within [0, 1]")
        if canonical_kind == "perfect_trace_hint" and not bool(self.oracle):
            raise ValueError("perfect trace must set oracle=True")
        if bool(self.oracle) and canonical_kind != "perfect_trace_hint":
            raise ValueError("oracle=True is only legal for perfect_trace_hint")
        if canonical_kind == "zero_hint":
            if any(int(value) != 0 for row in self.target_dispatch_rows for value in row):
                raise ValueError("zero_hint matrix must be all zero")
            if self.confidence is not None and float(self.confidence) != 0.0:
                raise ValueError("zero_hint confidence must equal 0.0")
        if canonical_kind == "perfect_trace_hint" and self.confidence is not None and float(self.confidence) != 1.0:
            raise ValueError("perfect_trace_hint confidence must equal 1.0")
        for name, value in {
            "source_layer_id": self.source_layer_id,
            "target_layer_id": self.target_layer_id,
        }.items():
            if value is not None and not str(value):
                raise ValueError(f"{name} must not be empty when provided")

    def semantic_payload(self) -> dict[str, Any]:
        self.validate()
        canonical_kind = self._canonical_kind(
            predictor_id=str(self.predictor_id),
            hint_type=str(self.hint_type),
            oracle=bool(self.oracle),
        )
        return {
            "prediction_semantic_version": "prediction_hint_v2",
            "predictor_id": str(self.predictor_id),
            "prediction_kind": canonical_kind,
            "target_dispatch_rows": [list(row) for row in self.target_dispatch_rows],
            "confidence": None if self.confidence is None else float(self.confidence),
            "oracle": bool(self.oracle),
            "source_layer_id": self.source_layer_id,
            "target_layer_id": self.target_layer_id,
        }

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
            "prediction_kind": self._canonical_kind(
                predictor_id=str(self.predictor_id),
                hint_type=str(self.hint_type),
                oracle=bool(self.oracle),
            ),
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
class ExpertScoreDistribution:
    """Full routed-expert scores grouped by logical source rank.

    The contract is deliberately model agnostic: each source rank owns a
    sequence of tokens, every token has one score per routed expert, and
    ``top_k`` fixes the total assignment mass. Shared/local experts must be
    excluded before constructing this contract.
    """

    scores_by_source_rank: tuple[tuple[tuple[float, ...], ...], ...]
    top_k: int
    score_domain: str

    def validate(
        self,
        *,
        world_size: int | None = None,
        expert_count: int | None = None,
    ) -> None:
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be > 0")
        if self.score_domain not in {"logits", "probabilities", "nonnegative_scores"}:
            raise ValueError(f"unsupported score_domain {self.score_domain!r}")
        if world_size is not None and len(self.scores_by_source_rank) != int(world_size):
            raise ValueError("scores_by_source_rank count does not match world_size")
        inferred_expert_count = expert_count
        for source_rank, token_rows in enumerate(self.scores_by_source_rank):
            for token_index, row in enumerate(token_rows):
                if inferred_expert_count is None:
                    inferred_expert_count = len(row)
                if len(row) != int(inferred_expert_count):
                    raise ValueError(
                        f"score width mismatch at source {source_rank}, token {token_index}"
                    )
                for value in row:
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        raise ValueError("expert scores must be finite")
                    if self.score_domain != "logits" and numeric < 0.0:
                        raise ValueError("non-logit expert scores must be non-negative")
        if inferred_expert_count is None or int(inferred_expert_count) <= 0:
            raise ValueError("expert score distribution must contain expert-width metadata")
        if int(self.top_k) > int(inferred_expert_count):
            raise ValueError("top_k cannot exceed routed expert count")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "scores_by_source_rank": [
                [list(float(value) for value in row) for row in token_rows]
                for token_rows in self.scores_by_source_rank
            ],
            "top_k": int(self.top_k),
            "score_domain": str(self.score_domain),
        }


@dataclass(frozen=True)
class RankPressureForecast:
    """Per-source future remote-traffic pressure with calibrated bounds."""

    mean: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    confidence: tuple[float, ...]

    def validate(self, *, world_size: int | None = None) -> None:
        widths = {len(self.mean), len(self.lower), len(self.upper), len(self.confidence)}
        if len(widths) != 1:
            raise ValueError("rank-pressure vectors must have equal length")
        if world_size is not None and len(self.mean) != int(world_size):
            raise ValueError("rank-pressure width does not match world_size")
        for index, (mean, lower, upper, confidence) in enumerate(
            zip(self.mean, self.lower, self.upper, self.confidence, strict=True)
        ):
            values = (float(mean), float(lower), float(upper), float(confidence))
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"rank-pressure values at {index} must be finite")
            if lower < 0.0 or mean < 0.0 or upper < 0.0:
                raise ValueError("rank-pressure values must be non-negative")
            if float(lower) > float(mean) + 1e-9 or float(mean) > float(upper) + 1e-9:
                raise ValueError("rank-pressure bounds must satisfy lower <= mean <= upper")
            if float(confidence) < 0.0 or float(confidence) > 1.0:
                raise ValueError("rank-pressure confidence must be within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "mean": list(float(value) for value in self.mean),
            "lower": list(float(value) for value in self.lower),
            "upper": list(float(value) for value in self.upper),
            "confidence": list(float(value) for value in self.confidence),
        }


@dataclass(frozen=True)
class StableEdgePrecedence:
    """A prediction-supported partial order; it is never executable truth."""

    before_src: int
    before_dst: int
    after_src: int
    after_dst: int
    margin: float
    confidence: float

    def validate(self, *, world_size: int | None = None) -> None:
        endpoints = (self.before_src, self.before_dst, self.after_src, self.after_dst)
        if world_size is not None:
            for endpoint in endpoints:
                if int(endpoint) < 0 or int(endpoint) >= int(world_size):
                    raise ValueError("precedence endpoint outside world_size")
        if int(self.before_src) == int(self.before_dst) or int(self.after_src) == int(self.after_dst):
            raise ValueError("precedence edges must be remote")
        if (int(self.before_src), int(self.before_dst)) == (int(self.after_src), int(self.after_dst)):
            raise ValueError("precedence edges must be distinct")
        if not math.isfinite(float(self.margin)) or float(self.margin) < 0.0:
            raise ValueError("precedence margin must be finite and non-negative")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("precedence confidence must be within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class TrafficForecastEnvelope:
    """Model-agnostic forecast contract consumed by planning and evaluation.

    ``mean_rows`` remains available for exact/predict-then-optimize controls.
    Online heuristics should primarily consume ``rank_pressure`` and only use
    ``stable_precedence`` as a low-weight tie-break. Predicted bytes never become
    executable bytes without a later actual-traffic reveal.
    """

    predictor_id: str
    mean_rows: MatrixRows
    lower_rows: MatrixRows
    upper_rows: MatrixRows
    rank_pressure: RankPressureForecast
    stable_precedence: tuple[StableEdgePrecedence, ...] = ()
    calibration_id: str = ""
    source_layer_id: str | None = None
    target_layer_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self, *, world_size: int | None = None) -> None:
        if not str(self.predictor_id):
            raise ValueError("predictor_id must be non-empty")
        _validate_matrix("mean_rows", self.mean_rows, world_size=world_size)
        _validate_matrix("lower_rows", self.lower_rows, world_size=world_size)
        _validate_matrix("upper_rows", self.upper_rows, world_size=world_size)
        if not (len(self.mean_rows) == len(self.lower_rows) == len(self.upper_rows)):
            raise ValueError("forecast matrices must have equal shapes")
        for src in range(len(self.mean_rows)):
            if not (
                len(self.mean_rows[src])
                == len(self.lower_rows[src])
                == len(self.upper_rows[src])
            ):
                raise ValueError("forecast matrices must have equal shapes")
            for mean, lower, upper in zip(
                self.mean_rows[src], self.lower_rows[src], self.upper_rows[src], strict=True
            ):
                if int(lower) > int(mean) or int(mean) > int(upper):
                    raise ValueError("forecast bounds must satisfy lower <= mean <= upper")
        inferred_world_size = len(self.mean_rows) if world_size is None else int(world_size)
        self.rank_pressure.validate(world_size=inferred_world_size)
        for precedence in self.stable_precedence:
            precedence.validate(world_size=inferred_world_size)
        for name, value in {
            "source_layer_id": self.source_layer_id,
            "target_layer_id": self.target_layer_id,
        }.items():
            if value is not None and not str(value):
                raise ValueError(f"{name} must not be empty when provided")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "prediction_semantic_version": "traffic_forecast_envelope_v1",
            "predictor_id": str(self.predictor_id),
            "mean_rows": [list(row) for row in self.mean_rows],
            "lower_rows": [list(row) for row in self.lower_rows],
            "upper_rows": [list(row) for row in self.upper_rows],
            "rank_pressure": self.rank_pressure.to_dict(),
            "stable_precedence": [item.to_dict() for item in self.stable_precedence],
            "calibration_id": str(self.calibration_id),
            "source_layer_id": self.source_layer_id,
            "target_layer_id": self.target_layer_id,
            "metadata": dict(self.metadata),
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
    "ExpertScoreDistribution",
    "RankedExpertRoutes",
    "RankPressureForecast",
    "StableEdgePrecedence",
    "TrafficForecastEnvelope",
    "MatrixRows",
    "PredictionContext",
    "PredictionHint",
    "PredictionIdentity",
    "PredictionResult",
    "TrafficHistoryContext",
]
