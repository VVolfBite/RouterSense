from __future__ import annotations

import math
import statistics
from typing import Any

from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_col_sums_remote, matrix_row_sums_remote

from .contracts import Matrix, PredictorEvaluationRecord, PredictorSample
from .feature_builder import load_fixture_samples
from .history_predictor import FATEStyleHistoryPredictor
from .linear_predictor import FATEStyleLinearTrafficPredictor


def _flatten(matrix: Matrix) -> list[float]:
    return [float(value) for row in matrix for value in row]


def _row_sums(matrix: Matrix) -> list[float]:
    return [float(value) for value in matrix_row_sums_remote(matrix)]


def _col_sums(matrix: Matrix) -> list[float]:
    return [float(value) for value in matrix_col_sums_remote(matrix)]


def compare_prediction(*, predictor_name: str, predictor_version: str, sample: PredictorSample, predicted: Matrix, confidence: float) -> PredictorEvaluationRecord:
    predicted = canonicalize_remote_matrix(predicted)
    predicted_flat = _flatten(predicted)
    actual_flat = _flatten(sample.target_next_dispatch_matrix)
    abs_l1 = float(sum(abs(left - right) for left, right in zip(predicted_flat, actual_flat, strict=False)))
    actual_total = float(sum(actual_flat))
    relative_l1 = 0.0 if actual_total <= 0.0 else abs_l1 / actual_total
    dot = float(sum(left * right for left, right in zip(predicted_flat, actual_flat, strict=False)))
    norm_pred = math.sqrt(sum(value * value for value in predicted_flat))
    norm_actual = math.sqrt(sum(value * value for value in actual_flat))
    cosine = 0.0 if norm_pred == 0.0 or norm_actual == 0.0 else dot / (norm_pred * norm_actual)
    topk = min(16, len(predicted_flat))
    pred_pairs = sorted(enumerate(predicted_flat), key=lambda item: item[1], reverse=True)[:topk]
    actual_pairs = sorted(enumerate(actual_flat), key=lambda item: item[1], reverse=True)[:topk]
    pred_idx = {idx for idx, value in pred_pairs if value > 0}
    actual_idx = {idx for idx, value in actual_pairs if value > 0}
    overlap = float(len(pred_idx & actual_idx) / max(len(actual_idx), 1))
    pred_nonzero = {idx for idx, value in enumerate(predicted_flat) if value > 0}
    actual_nonzero = {idx for idx, value in enumerate(actual_flat) if value > 0}
    precision = float(len(pred_nonzero & actual_nonzero) / max(len(pred_nonzero), 1))
    recall = float(len(pred_nonzero & actual_nonzero) / max(len(actual_nonzero), 1))
    row_sum_error = float(sum(abs(left - right) for left, right in zip(_row_sums(predicted), _row_sums(sample.target_next_dispatch_matrix), strict=False)))
    col_sum_error = float(sum(abs(left - right) for left, right in zip(_col_sums(predicted), _col_sums(sample.target_next_dispatch_matrix), strict=False)))
    return PredictorEvaluationRecord(
        predictor_name=predictor_name,
        predictor_version=predictor_version,
        layer_id=sample.layer_id,
        next_layer_id=sample.next_layer_id,
        predicted_matrix=predicted,
        actual_matrix=sample.target_next_dispatch_matrix,
        confidence=float(confidence),
        relative_l1_error=relative_l1,
        absolute_l1_error=abs_l1,
        cosine_similarity=cosine,
        topk_edge_overlap=overlap,
        nonzero_precision=precision,
        nonzero_recall=recall,
        row_sum_error=row_sum_error,
        col_sum_error=col_sum_error,
    )


def rolling_predictor_records(*, fixture_dir, predictor_name: str) -> list[PredictorEvaluationRecord]:
    samples = load_fixture_samples(fixture_dir)
    records: list[PredictorEvaluationRecord] = []
    history: list[PredictorSample] = []
    for sample in samples:
        if predictor_name == "zero_hint":
            predicted = tuple(tuple(0 for _ in row) for row in sample.target_next_dispatch_matrix)
            records.append(compare_prediction(predictor_name="zero_hint", predictor_version="v1", sample=sample, predicted=predicted, confidence=0.0))
        elif predictor_name == "copy_current_dispatch":
            predicted = sample.current_dispatch_matrix
            records.append(compare_prediction(predictor_name="copy_current_dispatch", predictor_version="v1", sample=sample, predicted=predicted, confidence=1.0))
        elif predictor_name in {"history_ema", "fate_style_history"}:
            if not history:
                predicted = sample.current_dispatch_matrix
                records.append(compare_prediction(predictor_name="history_ema", predictor_version="v1", sample=sample, predicted=predicted, confidence=0.25))
            else:
                previous_matrix = history[-1].current_dispatch_matrix
                alpha = 0.5
                predicted = canonicalize_remote_matrix(
                    tuple(
                        tuple(
                            int(round(alpha * int(cur) + (1.0 - alpha) * int(prev)))
                            for cur, prev in zip(current_row, previous_row, strict=True)
                        )
                        for current_row, previous_row in zip(sample.current_dispatch_matrix, previous_matrix, strict=True)
                    )
                )
                predictor = FATEStyleHistoryPredictor(alpha=alpha)
                records.append(compare_prediction(predictor_name=predictor.predictor_name, predictor_version=predictor.predictor_version, sample=sample, predicted=predicted, confidence=0.75))
        elif predictor_name in {"ridge_linear_trace_predictor", "fate_style_linear", "history_linear_trend"}:
            if len(history) < 2:
                predicted = sample.current_dispatch_matrix
                records.append(compare_prediction(predictor_name="ridge_linear_trace_predictor", predictor_version="v1", sample=sample, predicted=predicted, confidence=0.2))
            else:
                predictor = FATEStyleLinearTrafficPredictor().fit(history)
                predicted = predictor.predict_matrix(sample)
                records.append(compare_prediction(predictor_name=predictor.predictor_name, predictor_version=predictor.predictor_version, sample=sample, predicted=predicted, confidence=0.8))
        else:
            raise ValueError(f"unsupported predictor {predictor_name!r}")
        history.append(sample)
    return records


def summarize_prediction_records(records: list[PredictorEvaluationRecord]) -> dict[str, Any]:
    def _mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    payload = {
        "record_count": len(records),
        "mean_relative_l1_error": _mean([record.relative_l1_error for record in records]),
        "median_relative_l1_error": statistics.median([record.relative_l1_error for record in records]) if records else 0.0,
        "mean_cosine_similarity": _mean([record.cosine_similarity for record in records]),
        "mean_topk_edge_overlap": _mean([record.topk_edge_overlap for record in records]),
        "mean_nonzero_precision": _mean([record.nonzero_precision for record in records]),
        "mean_nonzero_recall": _mean([record.nonzero_recall for record in records]),
        "mean_row_sum_error": _mean([record.row_sum_error for record in records]),
        "mean_col_sum_error": _mean([record.col_sum_error for record in records]),
        "per_layer": {},
    }
    by_layer: dict[str, list[PredictorEvaluationRecord]] = {}
    for record in records:
        by_layer.setdefault(record.layer_id, []).append(record)
    for layer_id, layer_records in by_layer.items():
        payload["per_layer"][layer_id] = {
            "record_count": len(layer_records),
            "mean_relative_l1_error": _mean([record.relative_l1_error for record in layer_records]),
            "mean_cosine_similarity": _mean([record.cosine_similarity for record in layer_records]),
            "mean_topk_edge_overlap": _mean([record.topk_edge_overlap for record in layer_records]),
        }
    return payload
