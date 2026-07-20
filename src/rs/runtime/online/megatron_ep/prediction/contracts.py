"""Contracts for lightweight next-dispatch prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PredictionInput:
    run_id_digest: str
    layer_id: str
    next_layer_id: str
    rank: int
    world_size: int
    current_dispatch_matrix_digest: str
    current_dispatch_total_bytes: int
    current_dispatch_nonzero_edges: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictedTrafficMatrix:
    predictor_name: str
    predictor_version: str
    source_layer_id: str
    predicted_layer_id: str
    matrix: Matrix
    matrix_digest: str
    total_bytes: int
    nonzero_edge_count: int
    confidence: float
    is_oracle: bool
    evaluation_eligible: bool
    created_at_phase: str
    valid: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "matrix": [list(row) for row in self.matrix],
        }


@dataclass(frozen=True)
class PredictionAuditRecord:
    predictor_name: str
    source_layer_id: str
    predicted_layer_id: str
    predicted_digest: str
    actual_digest: str
    predicted_total_bytes: int
    actual_total_bytes: int
    predicted_remote_bytes: int
    actual_remote_bytes: int
    predicted_self_bytes: int
    actual_self_bytes: int
    relative_l1_error: float
    absolute_l1_error: float
    cosine_similarity: float
    topk_edge_overlap: float
    nonzero_edge_precision: float
    nonzero_edge_recall: float
    evaluation_eligible: bool
    self_bytes_ignored_for_scheduling: bool = True
    valid: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActiveNextDispatchPrediction:
    source_layer_id: str
    target_layer_id: str
    forecast_matrix: Matrix
    matrix_digest: str
    predictor_name: str
    predictor_version: str
    confidence: float
    evaluation_eligible: bool
    is_oracle: bool
    created_at_phase: str
    created_at_stage: str
    prediction_time_us: float = 0.0
    valid: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "forecast_matrix": [list(row) for row in self.forecast_matrix],
        }
