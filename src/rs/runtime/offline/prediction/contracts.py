"""Lightweight offline traffic predictor contracts used by replay fixtures and evidence suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PredictorSample:
    layer_id: str
    next_layer_id: str
    current_dispatch_matrix: Matrix
    current_return_matrix: Matrix
    previous_dispatch_matrix: Matrix
    target_next_dispatch_matrix: Matrix

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictorArtifact:
    predictor_name: str
    predictor_version: str
    feature_spec: str
    world_size: int
    metadata: dict[str, Any]
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictorEvaluationRecord:
    predictor_name: str
    predictor_version: str
    layer_id: str
    next_layer_id: str
    predicted_matrix: Matrix
    actual_matrix: Matrix
    confidence: float
    relative_l1_error: float
    absolute_l1_error: float
    cosine_similarity: float
    topk_edge_overlap: float
    nonzero_precision: float
    nonzero_recall: float
    row_sum_error: float
    col_sum_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
