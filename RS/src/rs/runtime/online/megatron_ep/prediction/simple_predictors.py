"""Simple online predictors used for debug/replay and cheap runtime baselines."""

from __future__ import annotations

from typing import Protocol

from rs.scheduling.validation import stable_hash

from .contracts import Matrix, PredictionInput, PredictedTrafficMatrix


class TrafficPredictor(Protocol):
    predictor_name: str
    predictor_version: str

    def predict(self, *, prediction_input: PredictionInput, current_dispatch_matrix: Matrix) -> PredictedTrafficMatrix:
        ...


class ZeroHintPredictor:
    predictor_name = "zero_hint"
    predictor_version = "v1"

    def predict(self, *, prediction_input: PredictionInput, current_dispatch_matrix: Matrix) -> PredictedTrafficMatrix:
        matrix = tuple(tuple(0 for _ in row) for row in current_dispatch_matrix)
        return PredictedTrafficMatrix(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            source_layer_id=prediction_input.layer_id,
            predicted_layer_id=prediction_input.next_layer_id,
            matrix=matrix,
            matrix_digest=stable_hash({"predictor": self.predictor_name, "matrix": matrix}),
            total_bytes=0,
            nonzero_edge_count=0,
            confidence=0.0,
            is_oracle=False,
            evaluation_eligible=True,
            created_at_phase="P0",
        )


class CopyCurrentDispatchPredictor:
    predictor_name = "copy_current_dispatch"
    predictor_version = "v1"

    def predict(self, *, prediction_input: PredictionInput, current_dispatch_matrix: Matrix) -> PredictedTrafficMatrix:
        matrix = tuple(tuple(int(value) for value in row) for row in current_dispatch_matrix)
        total_bytes = int(sum(sum(row) for row in matrix))
        nonzero_edge_count = int(
            sum(1 for src, row in enumerate(matrix) for dst, value in enumerate(row) if src != dst and int(value) > 0)
        )
        return PredictedTrafficMatrix(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            source_layer_id=prediction_input.layer_id,
            predicted_layer_id=prediction_input.next_layer_id,
            matrix=matrix,
            matrix_digest=stable_hash({"predictor": self.predictor_name, "matrix": matrix}),
            total_bytes=total_bytes,
            nonzero_edge_count=nonzero_edge_count,
            confidence=1.0,
            is_oracle=False,
            evaluation_eligible=True,
            created_at_phase="P0",
        )
