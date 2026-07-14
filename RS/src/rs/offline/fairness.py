from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rs.core.contracts import EvaluationSpec, ExecutionTruth, PlanningRequest, PredictionResult
from rs.offline.builder import prediction_digest


@dataclass(frozen=True)
class PlannerComparisonEligibility:
    eligible: bool
    reason: str | None = None


def evaluate_comparison_eligibility(
    *,
    left_spec: EvaluationSpec,
    right_spec: EvaluationSpec,
    left_truth: ExecutionTruth,
    right_truth: ExecutionTruth,
    left_request: PlanningRequest,
    right_request: PlanningRequest,
    left_prediction: PredictionResult,
    right_prediction: PredictionResult,
    metadata: Mapping[str, object] | None = None,
) -> PlannerComparisonEligibility:
    del metadata
    if left_spec.semantic_digest() != right_spec.semantic_digest():
        return PlannerComparisonEligibility(False, "evaluation_spec_mismatch")
    if left_truth.task_set.task_set_digest != right_truth.task_set.task_set_digest:
        return PlannerComparisonEligibility(False, "task_set_mismatch")
    if left_truth.truth_digest != right_truth.truth_digest:
        return PlannerComparisonEligibility(False, "execution_truth_mismatch")
    if left_request.semantic_digest() != right_request.semantic_digest():
        return PlannerComparisonEligibility(False, "planning_request_mismatch")
    if prediction_digest(left_prediction) != prediction_digest(right_prediction):
        return PlannerComparisonEligibility(False, "prediction_result_mismatch")
    return PlannerComparisonEligibility(True, None)


__all__ = ["PlannerComparisonEligibility", "evaluate_comparison_eligibility"]
