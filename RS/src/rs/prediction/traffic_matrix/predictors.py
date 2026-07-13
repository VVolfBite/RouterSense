from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import PredictionHint, PredictionIdentity, PredictionResult, TrafficHistoryContext
from rs.scheduling.traffic_matrix import matrix_col_sums_remote, matrix_row_sums_remote
import torch

from ..api import Predictor


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
    layer_id: str | None,
) -> list[float]:
    current = _flatten(current_dispatch)
    returns = [float(current_dispatch[col][row]) for row in range(len(current_dispatch)) for col in range(len(current_dispatch))]
    previous = _flatten(previous_dispatch)
    return [
        *current,
        *returns,
        *previous,
        *[float(value) for value in matrix_row_sums_remote(current_dispatch)],
        *[float(value) for value in matrix_col_sums_remote(current_dispatch)],
        *[float(value) for value in matrix_row_sums_remote(tuple(tuple(int(current_dispatch[col][row]) for col in range(len(current_dispatch))) for row in range(len(current_dispatch))))],
        *[float(value) for value in matrix_col_sums_remote(tuple(tuple(int(current_dispatch[col][row]) for col in range(len(current_dispatch))) for row in range(len(current_dispatch))))],
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

    def predict(self, context: TrafficHistoryContext) -> PredictionResult:
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
class LinearTrafficPredictor(Predictor):
    ridge_lambda: float = 1e-3
    predictor_name: str = "linear"

    @property
    def predictor_id(self) -> str:
        return self.predictor_name

    def predict(self, context: TrafficHistoryContext) -> PredictionResult:
        if len(context.history_dispatch_rows) < 2:
            predicted = context.current_dispatch_rows
            confidence = 0.2
        else:
            history = list(context.history_dispatch_rows)
            features = []
            targets = []
            previous = history[0]
            for current in history[1:]:
                features.append(_feature_vector(previous_dispatch=previous, current_dispatch=current, layer_id=context.identity.source_layer_id))
                targets.append(_flatten(current))
                previous = current
            feature_tensor = torch.tensor(features, dtype=torch.float64)
            target_tensor = torch.tensor(targets, dtype=torch.float64)
            ones = torch.ones((feature_tensor.shape[0], 1), dtype=torch.float64)
            design = torch.cat([feature_tensor, ones], dim=1)
            eye = torch.eye(design.shape[1], dtype=torch.float64)
            eye[-1, -1] = 0.0
            normal_matrix = design.T @ design + float(self.ridge_lambda) * eye
            rhs = design.T @ target_tensor
            try:
                solution = torch.linalg.solve(normal_matrix, rhs)
            except RuntimeError:
                solution = torch.linalg.pinv(normal_matrix) @ rhs
            weight = solution[:-1, :]
            bias = solution[-1, :]
            infer_feature = torch.tensor(
                _feature_vector(previous_dispatch=history[-1], current_dispatch=context.current_dispatch_rows, layer_id=context.identity.source_layer_id),
                dtype=torch.float64,
            )
            output = infer_feature @ weight + bias
            rows = len(context.current_dispatch_rows)
            cols = len(context.current_dispatch_rows[0]) if context.current_dispatch_rows else 0
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
