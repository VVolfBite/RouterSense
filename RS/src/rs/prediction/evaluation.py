from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from rs.core.contracts import ExpertRoutePrediction, MatrixRows, PredictionResult


@dataclass(frozen=True)
class PredictionTruth:
    actual_dispatch_rows: MatrixRows
    actual_expert_route: ExpertRoutePrediction | None = None


@dataclass(frozen=True)
class PredictionEvaluation:
    predictor_id: str
    hint_type: str
    metrics: dict[str, float] = field(default_factory=dict)


class PredictionEvaluator:
    def evaluate(
        self,
        prediction: PredictionResult,
        truth: PredictionTruth,
    ) -> PredictionEvaluation:
        if prediction.expert_route is not None and truth.actual_expert_route is not None:
            metrics = self._evaluate_expert_route(prediction.expert_route, truth.actual_expert_route)
        else:
            metrics = self._evaluate_traffic(prediction.hint.target_dispatch_rows, truth.actual_dispatch_rows)
        return PredictionEvaluation(
            predictor_id=str(prediction.hint.predictor_id),
            hint_type=str(prediction.hint.hint_type),
            metrics=metrics,
        )

    def _evaluate_traffic(self, predicted: MatrixRows, actual: MatrixRows) -> dict[str, float]:
        pred_flat = [float(v) for row in predicted for v in row]
        actual_flat = [float(v) for row in actual for v in row]
        abs_l1 = float(sum(abs(a - b) for a, b in zip(pred_flat, actual_flat, strict=False)))
        actual_total = float(sum(actual_flat))
        relative_l1 = 0.0 if actual_total <= 0.0 else abs_l1 / actual_total
        dot = float(sum(a * b for a, b in zip(pred_flat, actual_flat, strict=False)))
        pred_norm = math.sqrt(sum(v * v for v in pred_flat))
        act_norm = math.sqrt(sum(v * v for v in actual_flat))
        cosine = 0.0 if pred_norm == 0.0 or act_norm == 0.0 else dot / (pred_norm * act_norm)
        row_error = float(
            sum(abs(sum(pred_row) - sum(act_row)) for pred_row, act_row in zip(predicted, actual, strict=False))
        )
        return {
            "relative_l1": relative_l1,
            "cosine_similarity": cosine,
            "row_matrix_error": row_error,
        }

    def _evaluate_expert_route(self, predicted: ExpertRoutePrediction, actual: ExpertRoutePrediction) -> dict[str, float]:
        pair_count = min(len(predicted.expert_ids), len(actual.expert_ids))
        overlap_total = 0.0
        weight_similarity_total = 0.0
        gpu_accuracy_total = 0.0
        for index in range(pair_count):
            pred_row = tuple(int(v) for v in predicted.expert_ids[index])
            act_row = tuple(int(v) for v in actual.expert_ids[index])
            pred_set = set(pred_row)
            act_set = set(act_row)
            overlap_total += float(len(pred_set & act_set) / max(len(act_set), 1))
            gpu_accuracy_total += float(1.0 if pred_row[:1] == act_row[:1] else 0.0)
            if predicted.route_weights is not None and actual.route_weights is not None:
                pred_weights = predicted.route_weights[index]
                act_weights = actual.route_weights[index]
                width = min(len(pred_weights), len(act_weights))
                weight_similarity_total += float(
                    sum(1.0 - abs(float(pred_weights[i]) - float(act_weights[i])) for i in range(width)) / max(width, 1)
                )
        denom = max(pair_count, 1)
        return {
            "topk_overlap": overlap_total / denom,
            "expert_owner_gpu_accuracy": gpu_accuracy_total / denom,
            "route_weight_similarity": weight_similarity_total / denom,
        }


__all__ = ["PredictionEvaluation", "PredictionEvaluator", "PredictionTruth"]
