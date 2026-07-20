from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from rs.core.contracts import ExpertRoutePrediction, MatrixRows, PredictionResult


@dataclass(frozen=True)
class PredictionTruth:
    actual_dispatch_rows: MatrixRows
    actual_expert_route: ExpertRoutePrediction | None = None
    expert_owner_by_id: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PredictionEvaluation:
    predictor_id: str
    hint_type: str
    valid: bool
    reason: str | None = None
    predicted_shape: tuple[int, ...] = ()
    truth_shape: tuple[int, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)


class PredictionEvaluator:
    def evaluate(
        self,
        prediction: PredictionResult,
        truth: PredictionTruth,
    ) -> PredictionEvaluation:
        prediction.validate()
        if prediction.expert_route is not None and truth.actual_expert_route is not None:
            valid, reason, predicted_shape, truth_shape, metrics = self._evaluate_expert_route(
                prediction.expert_route,
                truth.actual_expert_route,
                expert_owner_by_id=truth.expert_owner_by_id,
            )
        else:
            valid, reason, predicted_shape, truth_shape, metrics = self._evaluate_traffic(
                prediction.hint.target_dispatch_rows,
                truth.actual_dispatch_rows,
            )
        return PredictionEvaluation(
            predictor_id=str(prediction.hint.predictor_id),
            hint_type=str(prediction.hint.hint_type),
            valid=valid,
            reason=reason,
            predicted_shape=predicted_shape,
            truth_shape=truth_shape,
            metrics=metrics,
        )

    def _evaluate_traffic(
        self,
        predicted: MatrixRows,
        actual: MatrixRows,
    ) -> tuple[bool, str | None, tuple[int, ...], tuple[int, ...], dict[str, float]]:
        predicted_shape = (len(predicted), len(predicted[0]) if predicted else 0)
        actual_shape = (len(actual), len(actual[0]) if actual else 0)
        if predicted_shape != actual_shape:
            return False, "traffic_shape_mismatch", predicted_shape, actual_shape, {}
        if len({len(row) for row in predicted}) > 1 or len({len(row) for row in actual}) > 1:
            return False, "traffic_ragged_matrix", predicted_shape, actual_shape, {}
        for name, matrix in {"predicted": predicted, "actual": actual}.items():
            for row in matrix:
                for value in row:
                    if float(value) < 0.0:
                        return False, f"{name}_contains_negative_value", predicted_shape, actual_shape, {}
        pred_flat = [float(v) for row in predicted for v in row]
        actual_flat = [float(v) for row in actual for v in row]
        abs_l1 = float(sum(abs(a - b) for a, b in zip(pred_flat, actual_flat, strict=True)))
        actual_total = float(sum(actual_flat))
        relative_l1 = 0.0 if actual_total <= 0.0 else abs_l1 / actual_total
        dot = float(sum(a * b for a, b in zip(pred_flat, actual_flat, strict=True)))
        pred_norm = math.sqrt(sum(v * v for v in pred_flat))
        act_norm = math.sqrt(sum(v * v for v in actual_flat))
        cosine = 0.0 if pred_norm == 0.0 or act_norm == 0.0 else dot / (pred_norm * act_norm)
        row_error = float(
            sum(abs(sum(pred_row) - sum(act_row)) for pred_row, act_row in zip(predicted, actual, strict=True))
        )
        return True, None, predicted_shape, actual_shape, {
            "relative_l1": relative_l1,
            "cosine_similarity": cosine,
            "row_matrix_error": row_error,
        }

    def _evaluate_expert_route(
        self,
        predicted: ExpertRoutePrediction,
        actual: ExpertRoutePrediction,
        *,
        expert_owner_by_id: tuple[int, ...] | None,
    ) -> tuple[bool, str | None, tuple[int, ...], tuple[int, ...], dict[str, float]]:
        predicted_shape = (len(predicted.expert_ids), len(predicted.expert_ids[0]) if predicted.expert_ids else 0)
        actual_shape = (len(actual.expert_ids), len(actual.expert_ids[0]) if actual.expert_ids else 0)
        if len(predicted.expert_ids) != len(actual.expert_ids):
            return False, "expert_route_token_count_mismatch", predicted_shape, actual_shape, {}
        predicted_widths = {len(row) for row in predicted.expert_ids}
        actual_widths = {len(row) for row in actual.expert_ids}
        if len(predicted_widths) != 1 or len(actual_widths) != 1:
            return False, "expert_route_ragged_topk", predicted_shape, actual_shape, {}
        if predicted_widths != actual_widths:
            return False, "expert_route_topk_width_mismatch", predicted_shape, actual_shape, {}
        if predicted.route_weights is not None and len(predicted.route_weights) != len(predicted.expert_ids):
            return False, "predicted_route_weight_token_mismatch", predicted_shape, actual_shape, {}
        if actual.route_weights is not None and len(actual.route_weights) != len(actual.expert_ids):
            return False, "truth_route_weight_token_mismatch", predicted_shape, actual_shape, {}
        expert_count = len(expert_owner_by_id) if expert_owner_by_id is not None else None
        try:
            predicted.validate(top_k=next(iter(predicted_widths), None), expert_count=expert_count)
            actual.validate(top_k=next(iter(actual_widths), None), expert_count=expert_count)
        except ValueError as exc:
            return False, str(exc), predicted_shape, actual_shape, {}
        pair_count = len(predicted.expert_ids)
        overlap_total = 0.0
        weight_similarity_total = 0.0
        owner_accuracy_total = 0.0
        exact_accuracy_total = 0.0
        for index in range(pair_count):
            pred_row = tuple(int(v) for v in predicted.expert_ids[index])
            act_row = tuple(int(v) for v in actual.expert_ids[index])
            pred_set = set(pred_row)
            act_set = set(act_row)
            overlap_total += float(len(pred_set & act_set) / max(len(act_set), 1))
            width = min(len(pred_row), len(act_row))
            if width > 0:
                exact_accuracy_total += float(sum(1 for i in range(width) if pred_row[i] == act_row[i]) / width)
                if expert_owner_by_id is not None:
                    owner_accuracy_total += float(
                        sum(
                            1
                            for i in range(width)
                            if int(expert_owner_by_id[pred_row[i]]) == int(expert_owner_by_id[act_row[i]])
                        )
                        / width
                    )
            if predicted.route_weights is not None and actual.route_weights is not None:
                pred_weights = predicted.route_weights[index]
                act_weights = actual.route_weights[index]
                if len(pred_weights) != len(act_weights):
                    return False, "route_weight_width_mismatch", predicted_shape, actual_shape, {}
                weight_width = len(pred_weights)
                weight_similarity_total += float(
                    sum(1.0 - abs(float(pred_weights[i]) - float(act_weights[i])) for i in range(weight_width)) / max(weight_width, 1)
                )
        denom = max(pair_count, 1)
        return True, None, predicted_shape, actual_shape, {
            "topk_overlap": overlap_total / denom,
            "exact_expert_accuracy": exact_accuracy_total / denom,
            "expert_owner_gpu_accuracy": owner_accuracy_total / denom if expert_owner_by_id is not None else 0.0,
            "route_weight_similarity": weight_similarity_total / denom,
        }


__all__ = ["PredictionEvaluation", "PredictionEvaluator", "PredictionTruth"]
