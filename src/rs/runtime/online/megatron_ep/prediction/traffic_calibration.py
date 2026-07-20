"""Lightweight traffic-matrix calibration helpers for predicted dispatch matrices."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_col_sums_remote, matrix_remote_bytes, matrix_row_sums_remote


Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TrafficCalibrationAudit:
    before_relative_l1: float
    after_relative_l1: float
    before_cosine: float
    after_cosine: float
    before_row_sum_error: float
    after_row_sum_error: float
    before_col_sum_error: float
    after_col_sum_error: float
    calibration_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_traffic_matrix(
    predicted_matrix: Matrix,
    *,
    actual_matrix: Matrix | None = None,
    current_dispatch_matrix: Matrix | None = None,
    historical_reference_matrix: Matrix | None = None,
    mode: str = "none",
    historical_scale: float | None = None,
) -> tuple[Matrix, TrafficCalibrationAudit]:
    predicted = canonicalize_remote_matrix(predicted_matrix)
    before = _compare(predicted, actual_matrix or predicted)
    calibrated = predicted
    if mode in {"total", "oracle_total"} and actual_matrix is not None:
        calibrated = _scale_total(predicted, target_total=matrix_remote_bytes(actual_matrix))
    elif mode == "current_total" and current_dispatch_matrix is not None:
        calibrated = _scale_total(predicted, target_total=matrix_remote_bytes(current_dispatch_matrix))
    elif mode in {"layer_scale", "history_layer_scale"} and historical_scale is not None:
        calibrated = _scale_total(predicted, scale=float(historical_scale))
    elif mode in {"row_col", "row_col_current"} and current_dispatch_matrix is not None:
        calibrated = _row_col_rescale(predicted, current_dispatch_matrix)
    elif mode == "row_col_history" and historical_reference_matrix is not None:
        calibrated = _row_col_rescale(predicted, historical_reference_matrix)
    after = _compare(calibrated, actual_matrix or calibrated)
    return calibrated, TrafficCalibrationAudit(
        before_relative_l1=before["relative_l1_error"],
        after_relative_l1=after["relative_l1_error"],
        before_cosine=before["cosine_similarity"],
        after_cosine=after["cosine_similarity"],
        before_row_sum_error=before["row_sum_error"],
        after_row_sum_error=after["row_sum_error"],
        before_col_sum_error=before["col_sum_error"],
        after_col_sum_error=after["col_sum_error"],
        calibration_mode=mode,
    )


def _scale_total(matrix: Matrix, *, target_total: int | None = None, scale: float | None = None) -> Matrix:
    current_total = max(1, matrix_remote_bytes(matrix))
    if scale is None:
        scale = 1.0 if target_total is None else float(target_total) / float(current_total)
    return canonicalize_remote_matrix(tuple(tuple(int(round(value * scale)) for value in row) for row in matrix))


def _row_col_rescale(matrix: Matrix, reference_matrix: Matrix) -> Matrix:
    predicted = canonicalize_remote_matrix(matrix)
    reference = canonicalize_remote_matrix(reference_matrix)
    pred_row = matrix_row_sums_remote(predicted)
    ref_row = matrix_row_sums_remote(reference)
    pred_col = matrix_col_sums_remote(predicted)
    ref_col = matrix_col_sums_remote(reference)
    scaled: list[list[int]] = []
    for src, row in enumerate(predicted):
        row_scale = 1.0 if pred_row[src] <= 0 else float(ref_row[src]) / float(pred_row[src])
        new_row: list[int] = []
        for dst, value in enumerate(row):
            col_scale = 1.0 if pred_col[dst] <= 0 else float(ref_col[dst]) / float(pred_col[dst])
            new_row.append(int(round(float(value) * math.sqrt(max(0.0, row_scale * col_scale)))))
        scaled.append(new_row)
    return canonicalize_remote_matrix(tuple(tuple(row) for row in scaled))


def _compare(predicted: Matrix, actual: Matrix) -> dict[str, float]:
    pred = canonicalize_remote_matrix(predicted)
    act = canonicalize_remote_matrix(actual)
    pred_values = [float(v) for row in pred for v in row]
    act_values = [float(v) for row in act for v in row]
    abs_l1 = float(sum(abs(a - b) for a, b in zip(pred_values, act_values, strict=False)))
    actual_total = float(sum(act_values))
    relative_l1 = 0.0 if actual_total <= 0.0 else abs_l1 / actual_total
    dot = float(sum(a * b for a, b in zip(pred_values, act_values, strict=False)))
    pred_norm = math.sqrt(sum(v * v for v in pred_values))
    act_norm = math.sqrt(sum(v * v for v in act_values))
    cosine = 0.0 if pred_norm == 0.0 or act_norm == 0.0 else dot / (pred_norm * act_norm)
    row_err = float(sum(abs(a - b) for a, b in zip(matrix_row_sums_remote(pred), matrix_row_sums_remote(act), strict=False)))
    col_err = float(sum(abs(a - b) for a, b in zip(matrix_col_sums_remote(pred), matrix_col_sums_remote(act), strict=False)))
    return {
        "relative_l1_error": relative_l1,
        "cosine_similarity": cosine,
        "row_sum_error": row_err,
        "col_sum_error": col_err,
    }


__all__ = ["TrafficCalibrationAudit", "calibrate_traffic_matrix"]
