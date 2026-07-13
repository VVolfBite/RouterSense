from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from rs.core.contracts import MatrixRows, PredictionContext, PredictionResult


class Predictor(Protocol):
    @property
    def predictor_id(self) -> str:
        ...

    def predict(self, context: PredictionContext) -> PredictionResult:
        ...


@dataclass(frozen=True)
class PredictorSpec:
    predictor_id: str
    category: str
    deployable: bool
    offline_only: bool
    test_only: bool = False
    historical_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrafficPredictionTrainingSample:
    current_dispatch_rows: MatrixRows
    history_dispatch_rows: tuple[MatrixRows, ...]
    target_next_dispatch_rows: MatrixRows
    current_return_rows: MatrixRows | None = None
    layer_id: str | None = None
    next_layer_id: str | None = None


class TrainableTrafficPredictor(Protocol):
    def fit(self, samples: Sequence[TrafficPredictionTrainingSample]) -> "TrainableTrafficPredictor":
        ...


__all__ = ["Predictor", "PredictorSpec", "TrafficPredictionTrainingSample", "TrainableTrafficPredictor"]
