from __future__ import annotations

from dataclasses import dataclass
import math
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
    current_return_rows: MatrixRows
    history_dispatch_rows: tuple[MatrixRows, ...]
    target_next_dispatch_rows: MatrixRows
    layer_id: str | None = None
    next_layer_id: str | None = None

    def validate(self) -> None:
        _validate_matrix("current_dispatch_rows", self.current_dispatch_rows)
        _validate_matrix("current_return_rows", self.current_return_rows, world_size=len(self.current_dispatch_rows))
        _validate_matrix("target_next_dispatch_rows", self.target_next_dispatch_rows, world_size=len(self.current_dispatch_rows))
        for index, history in enumerate(self.history_dispatch_rows):
            _validate_matrix(f"history_dispatch_rows[{index}]", history, world_size=len(self.current_dispatch_rows))
        for name, value in {"layer_id": self.layer_id, "next_layer_id": self.next_layer_id}.items():
            if value is not None and not str(value):
                raise ValueError(f"{name} must not be empty when provided")


class TrainableTrafficPredictor(Protocol):
    def fit(self, samples: Sequence[TrafficPredictionTrainingSample]) -> "TrainableTrafficPredictor":
        ...


__all__ = ["Predictor", "PredictorSpec", "TrafficPredictionTrainingSample", "TrainableTrafficPredictor"]


def _validate_matrix(name: str, matrix: MatrixRows, *, world_size: int | None = None) -> None:
    if world_size is not None and int(world_size) <= 0:
        raise ValueError("world_size must be > 0")
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
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} values must be finite and non-negative")
