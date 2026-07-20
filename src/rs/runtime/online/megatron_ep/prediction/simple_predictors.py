"""Simple online predictors used for debug/replay and cheap runtime baselines."""

from __future__ import annotations

from typing import Protocol

from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_digest_remote, matrix_nonzero_remote_edge_count, matrix_remote_bytes

from .contracts import Matrix, PredictionInput, PredictedTrafficMatrix


class TrafficPredictor(Protocol):
    predictor_name: str
    predictor_version: str

    def predict(self, *, prediction_input: PredictionInput, current_dispatch_matrix: Matrix) -> PredictedTrafficMatrix:
        ...


def _validated_matrix(
    *,
    matrix: Matrix,
    world_size: int,
) -> tuple[Matrix, bool, str]:
    canonical = canonicalize_remote_matrix(matrix)
    if len(canonical) != int(world_size):
        return canonical, False, f"shape_mismatch:rows:{len(canonical)}!=world_size:{int(world_size)}"
    widths = {len(row) for row in canonical}
    if widths != {int(world_size)}:
        return canonical, False, f"shape_mismatch:cols:{sorted(widths)}!=world_size:{int(world_size)}"
    return canonical, True, ""


class ZeroHintPredictor:
    predictor_name = "zero_hint"
    predictor_version = "v1"

    def predict(self, *, prediction_input: PredictionInput, current_dispatch_matrix: Matrix) -> PredictedTrafficMatrix:
        matrix, valid, error = _validated_matrix(
            matrix=tuple(tuple(0 for _ in row) for row in current_dispatch_matrix),
            world_size=prediction_input.world_size,
        )
        return PredictedTrafficMatrix(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            source_layer_id=prediction_input.layer_id,
            predicted_layer_id=prediction_input.next_layer_id,
            matrix=matrix,
            matrix_digest=matrix_digest_remote(matrix),
            total_bytes=0,
            nonzero_edge_count=0,
            confidence=0.0,
            is_oracle=False,
            evaluation_eligible=True,
            created_at_phase="P0",
            valid=valid,
            error=error,
        )


class CopyCurrentDispatchPredictor:
    predictor_name = "copy_current_dispatch"
    predictor_version = "v1"

    def predict(self, *, prediction_input: PredictionInput, current_dispatch_matrix: Matrix) -> PredictedTrafficMatrix:
        matrix, valid, error = _validated_matrix(matrix=current_dispatch_matrix, world_size=prediction_input.world_size)
        total_bytes = matrix_remote_bytes(matrix)
        nonzero_edge_count = matrix_nonzero_remote_edge_count(matrix)
        return PredictedTrafficMatrix(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            source_layer_id=prediction_input.layer_id,
            predicted_layer_id=prediction_input.next_layer_id,
            matrix=matrix,
            matrix_digest=matrix_digest_remote(matrix),
            total_bytes=total_bytes,
            nonzero_edge_count=nonzero_edge_count,
            confidence=1.0,
            is_oracle=False,
            evaluation_eligible=True,
            created_at_phase="P0",
            valid=valid,
            error=error,
        )


class HistoryEMATrafficPredictor:
    predictor_name = "history_ema"
    predictor_version = "v1"

    def __init__(self, *, alpha: float = 0.5) -> None:
        self.alpha = float(alpha)

    def predict(self, *, prediction_input: PredictionInput, current_dispatch_matrix: Matrix) -> PredictedTrafficMatrix:
        current, current_valid, current_error = _validated_matrix(
            matrix=current_dispatch_matrix,
            world_size=prediction_input.world_size,
        )
        previous = prediction_input.metadata.get("previous_dispatch_matrix")
        if previous is None:
            blended = current
        else:
            previous_matrix, previous_valid, previous_error = _validated_matrix(
                matrix=tuple(tuple(int(value) for value in row) for row in previous),
                world_size=prediction_input.world_size,
            )
            if not previous_valid:
                return PredictedTrafficMatrix(
                    predictor_name=self.predictor_name,
                    predictor_version=self.predictor_version,
                    source_layer_id=prediction_input.layer_id,
                    predicted_layer_id=prediction_input.next_layer_id,
                    matrix=current,
                    matrix_digest=matrix_digest_remote(current),
                    total_bytes=matrix_remote_bytes(current),
                    nonzero_edge_count=matrix_nonzero_remote_edge_count(current),
                    confidence=0.0,
                    is_oracle=False,
                    evaluation_eligible=True,
                    created_at_phase="P0",
                    valid=False,
                    error=previous_error,
                )
            blended_rows = []
            for current_row, previous_row in zip(current, previous_matrix, strict=True):
                blended_rows.append(
                    tuple(
                        int(round(self.alpha * int(cur) + (1.0 - self.alpha) * int(prev)))
                        for cur, prev in zip(current_row, previous_row, strict=True)
                    )
                )
            blended = canonicalize_remote_matrix(tuple(blended_rows))
        total_bytes = matrix_remote_bytes(blended)
        nonzero_edge_count = matrix_nonzero_remote_edge_count(blended)
        return PredictedTrafficMatrix(
            predictor_name=self.predictor_name,
            predictor_version=self.predictor_version,
            source_layer_id=prediction_input.layer_id,
            predicted_layer_id=prediction_input.next_layer_id,
            matrix=blended,
            matrix_digest=matrix_digest_remote(blended),
            total_bytes=total_bytes,
            nonzero_edge_count=nonzero_edge_count,
            confidence=0.75,
            is_oracle=False,
            evaluation_eligible=True,
            created_at_phase="P0",
            valid=current_valid,
            error=current_error,
        )
