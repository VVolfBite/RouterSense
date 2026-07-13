from __future__ import annotations

import math
import statistics
from typing import Any

from rs.core.contracts import PredictionIdentity, TrafficHistoryContext
from rs.prediction import PredictionRegistry, TrafficPredictionTrainingSample, resolve_predictor_id
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix, matrix_col_sums_remote, matrix_row_sums_remote

from .contracts import Matrix, PredictorEvaluationRecord, PredictorSample
from .feature_builder import load_fixture_samples


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
    resolved_name = resolve_predictor_id(str(predictor_name))
    for sample in samples:
        predictor = PredictionRegistry.create(str(predictor_name), {"alpha": 0.5})
        cold_start_prediction: Matrix | None = None
        if resolved_name == "linear":
            training_samples = tuple(
                TrafficPredictionTrainingSample(
                    current_dispatch_rows=item.current_dispatch_matrix,
                    history_dispatch_rows=(item.previous_dispatch_matrix,),
                    target_next_dispatch_rows=item.target_next_dispatch_matrix,
                    current_return_rows=item.current_return_matrix,
                    layer_id=item.layer_id,
                    next_layer_id=item.next_layer_id,
                )
                for item in history
            )
            if training_samples and hasattr(predictor, "fit"):
                predictor.fit(training_samples)
            else:
                cold_start_prediction = canonicalize_remote_matrix(sample.current_dispatch_matrix)
        context = TrafficHistoryContext(
            identity=PredictionIdentity(
                request_id=f"{sample.layer_id}:{sample.next_layer_id}:{predictor_name}",
                source_layer_id=str(sample.layer_id),
                target_layer_id=str(sample.next_layer_id),
            ),
            current_dispatch_rows=sample.current_dispatch_matrix,
            history_dispatch_rows=tuple(item.current_dispatch_matrix for item in history),
            world_size=len(sample.current_dispatch_matrix),
        )
        if cold_start_prediction is None:
            result = predictor.predict(context)
            predicted_matrix = canonicalize_remote_matrix(result.hint.target_dispatch_rows)
            confidence = float(result.hint.confidence or 0.0)
            result_name = str(result.hint.predictor_id)
        else:
            predicted_matrix = cold_start_prediction
            confidence = 0.0
            result_name = str(resolve_predictor_id(str(predictor_name)))
        records.append(
            compare_prediction(
                predictor_name=result_name,
                predictor_version="v1",
                sample=sample,
                predicted=predicted_matrix,
                confidence=confidence,
            )
        )
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
