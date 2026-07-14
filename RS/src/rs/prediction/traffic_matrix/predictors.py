from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rs.core.contracts import PredictionHint, PredictionIdentity, PredictionResult, TrafficHistoryContext
from rs.scheduling.traffic_matrix import matrix_col_sums_remote, matrix_row_sums_remote
import torch

from ..api import Predictor, TrafficPredictionTrainingSample, TrainableTrafficPredictor


def _zero_matrix(rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in row) for row in rows)


def _blend(current: tuple[tuple[int, ...], ...], previous: tuple[tuple[int, ...], ...], alpha: float) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(round(float(alpha) * int(cur) + (1.0 - float(alpha)) * int(prev))) for cur, prev in zip(current_row, previous_row, strict=True))
        for current_row, previous_row in zip(current, previous, strict=True)
    )


def _flatten(matrix: tuple[tuple[int, ...], ...]) -> list[float]:
    return [float(value) for row in matrix for value in row]


def _feature_vector(
    *,
    previous_dispatch: tuple[tuple[int, ...], ...],
    current_dispatch: tuple[tuple[int, ...], ...],
    current_return: tuple[tuple[int, ...], ...],
    layer_id: str | None,
) -> list[float]:
    current = _flatten(current_dispatch)
    returns = _flatten(current_return)
    previous = _flatten(previous_dispatch)
    return [
        *current,
        *returns,
        *previous,
        *[float(value) for value in matrix_row_sums_remote(current_dispatch)],
        *[float(value) for value in matrix_col_sums_remote(current_dispatch)],
        *[float(value) for value in matrix_row_sums_remote(current_return)],
        *[float(value) for value in matrix_col_sums_remote(current_return)],
        float(sum(current)),
        float(sum(returns)),
        float(int(layer_id) if layer_id is not None and str(layer_id).isdigit() else 0),
    ]


def _reshape(values: list[float], *, rows: int, cols: int) -> tuple[tuple[int, ...], ...]:
    clipped = [max(0, int(round(value))) for value in values]
    return tuple(tuple(clipped[row * cols + col] for col in range(cols)) for row in range(rows))


@dataclass
class ZeroTrafficPredictor(Predictor):
    @property
    def predictor_id(self) -> str:
        return "zero"

    def predict(self, context: TrafficHistoryContext) -> PredictionResult:
        context.validate()
        hint = PredictionHint(
            predictor_id=self.predictor_id,
            hint_type="traffic_matrix",
            target_dispatch_rows=_zero_matrix(context.current_dispatch_rows),
            confidence=0.0,
            oracle=False,
            source_layer_id=context.identity.source_layer_id,
            target_layer_id=context.identity.target_layer_id,
        )
        return PredictionResult(identity=context.identity, hint=hint)


@dataclass
class CopyCurrentTrafficPredictor(Predictor):
    @property
    def predictor_id(self) -> str:
        return "copy_current"

    def predict(self, context: TrafficHistoryContext) -> PredictionResult:
        context.validate()
        hint = PredictionHint(
            predictor_id=self.predictor_id,
            hint_type="traffic_matrix",
            target_dispatch_rows=context.current_dispatch_rows,
            confidence=1.0,
            oracle=False,
            source_layer_id=context.identity.source_layer_id,
            target_layer_id=context.identity.target_layer_id,
        )
        return PredictionResult(identity=context.identity, hint=hint)


@dataclass
class HistoryTrafficPredictor(Predictor):
    alpha: float = 0.5

    @property
    def predictor_id(self) -> str:
        return "history"

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("HistoryTrafficPredictor alpha must be within [0, 1]")

    def predict(self, context: TrafficHistoryContext) -> PredictionResult:
        context.validate()
        previous = context.history_dispatch_rows[-1] if context.history_dispatch_rows else context.current_dispatch_rows
        predicted = _blend(context.current_dispatch_rows, previous, self.alpha)
        hint = PredictionHint(
            predictor_id=self.predictor_id,
            hint_type="traffic_matrix",
            target_dispatch_rows=predicted,
            confidence=0.75,
            oracle=False,
            source_layer_id=context.identity.source_layer_id,
            target_layer_id=context.identity.target_layer_id,
        )
        return PredictionResult(identity=context.identity, hint=hint)


@dataclass
class LinearTrafficPredictor(Predictor, TrainableTrafficPredictor):
    ridge_lambda: float = 1e-3
    predictor_name: str = "linear"
    _weight: torch.Tensor | None = None
    _bias: torch.Tensor | None = None
    _shape: tuple[int, int] | None = None

    @property
    def predictor_id(self) -> str:
        return self.predictor_name

    def __post_init__(self) -> None:
        ridge = float(self.ridge_lambda)
        if not torch.isfinite(torch.tensor(ridge, dtype=torch.float64)) or ridge < 0.0:
            raise ValueError("ridge_lambda must be finite and >= 0")

    def fit(self, samples: Sequence[TrafficPredictionTrainingSample]) -> "LinearTrafficPredictor":
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("linear predictor requires at least one training sample")
        for sample in sample_list:
            sample.validate()
        world_size = len(sample_list[0].current_dispatch_rows)
        shape = (
            len(sample_list[0].target_next_dispatch_rows),
            len(sample_list[0].target_next_dispatch_rows[0]) if sample_list[0].target_next_dispatch_rows else 0,
        )
        for sample in sample_list[1:]:
            if len(sample.current_dispatch_rows) != world_size:
                raise ValueError("all training samples must share world_size")
            sample_shape = (
                len(sample.target_next_dispatch_rows),
                len(sample.target_next_dispatch_rows[0]) if sample.target_next_dispatch_rows else 0,
            )
            if sample_shape != shape:
                raise ValueError("all training targets must share shape")
        features = torch.tensor(
            [
                _feature_vector(
                    previous_dispatch=(sample.history_dispatch_rows[-1] if sample.history_dispatch_rows else _zero_matrix(sample.current_dispatch_rows)),
                    current_dispatch=sample.current_dispatch_rows,
                    current_return=sample.current_return_rows,
                    layer_id=sample.layer_id,
                )
                for sample in sample_list
            ],
            dtype=torch.float64,
        )
        targets = torch.tensor([_flatten(sample.target_next_dispatch_rows) for sample in sample_list], dtype=torch.float64)
        ones = torch.ones((features.shape[0], 1), dtype=torch.float64)
        design = torch.cat([features, ones], dim=1)
        eye = torch.eye(design.shape[1], dtype=torch.float64)
        eye[-1, -1] = 0.0
        normal_matrix = design.T @ design + float(self.ridge_lambda) * eye
        rhs = design.T @ targets
        try:
            solution = torch.linalg.solve(normal_matrix, rhs)
        except RuntimeError:
            solution = torch.linalg.pinv(normal_matrix) @ rhs
        if not torch.isfinite(solution).all():
            raise ValueError("linear predictor fit produced non-finite parameters")
        self._weight = solution[:-1, :]
        self._bias = solution[-1, :]
        self._shape = shape
        return self

    def predict(self, context: TrafficHistoryContext) -> PredictionResult:
        context.validate()
        if self._weight is None or self._bias is None or self._shape is None:
            raise ValueError("linear predictor must be fit before predict; it remains offline_only in M0")
        previous_dispatch = context.history_dispatch_rows[-1] if context.history_dispatch_rows else _zero_matrix(context.current_dispatch_rows)
        infer_feature = torch.tensor(
            _feature_vector(
                previous_dispatch=previous_dispatch,
                current_dispatch=context.current_dispatch_rows,
                current_return=context.current_return_rows,
                layer_id=context.identity.source_layer_id,
            ),
            dtype=torch.float64,
        )
        output = infer_feature @ self._weight + self._bias
        rows, cols = self._shape
        predicted = _reshape(output.tolist(), rows=rows, cols=cols)
        confidence = 0.8
        hint = PredictionHint(
            predictor_id=self.predictor_id,
            hint_type="traffic_matrix",
            target_dispatch_rows=predicted,
            confidence=confidence,
            oracle=False,
            source_layer_id=context.identity.source_layer_id,
            target_layer_id=context.identity.target_layer_id,
        )
        return PredictionResult(identity=context.identity, hint=hint)


__all__ = [
    "CopyCurrentTrafficPredictor",
    "HistoryTrafficPredictor",
    "LinearTrafficPredictor",
    "ZeroTrafficPredictor",
]
