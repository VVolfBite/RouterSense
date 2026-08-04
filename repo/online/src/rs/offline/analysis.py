from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Mapping, Sequence

from rs.core.contracts import (
    EvaluationSpec,
    OfflineEvaluationBundle,
    OfflineEvaluationRecord,
    OfflineWindow,
    PlanEvaluation,
    PlanningRequest,
    PredictionResult,
    WindowPlan,
)
from rs.offline.builder import prediction_digest
from rs.prediction.evaluation import PredictionEvaluation


def build_offline_record(
    *,
    window: OfflineWindow,
    spec: EvaluationSpec,
    task_set_digest: str,
    request: PlanningRequest,
    prediction: PredictionResult,
    plan: WindowPlan,
    execution_truth_digest: str,
    evaluation: PlanEvaluation,
    planner_reported_makespan: float | None,
    audit_status: str,
    coverage_status: str,
    fallback_status: str = "none",
    oracle_status: str = "not_run",
    eligibility: Mapping[str, object] | None = None,
    metrics: Mapping[str, object] | None = None,
) -> OfflineEvaluationRecord:
    return OfflineEvaluationRecord(
        window_identity=str(window.window_identity),
        evaluation_spec_digest=spec.semantic_digest(),
        task_set_digest=str(task_set_digest),
        planning_request_digest=request.semantic_digest(),
        prediction_digest=prediction_digest(prediction),
        logical_plan_digest=plan.semantic_digest(),
        execution_truth_digest=str(execution_truth_digest),
        planner_id=str(plan.planner_id),
        planner_family=str(plan.planner_family),
        predictor_id=str(prediction.hint.predictor_id),
        track=str(spec.track),
        realized_makespan=evaluation.realized_makespan,
        planner_reported_makespan=planner_reported_makespan,
        audit_status=str(audit_status),
        coverage_status=str(coverage_status),
        fallback_status=str(fallback_status),
        oracle_status=str(oracle_status),
        eligibility=dict(eligibility or {}),
        metrics=dict(metrics or {}) | dict(evaluation.metrics),
    )


def validate_comparable_records(left: OfflineEvaluationRecord, right: OfflineEvaluationRecord) -> None:
    comparable_fields = (
        "evaluation_spec_digest",
        "task_set_digest",
        "execution_truth_digest",
        "track",
    )
    for field in comparable_fields:
        if getattr(left, field) != getattr(right, field):
            raise ValueError(f"records_not_comparable:{field}")


def _eligible_for_aggregate(record: OfflineEvaluationRecord) -> bool:
    eligibility = dict(record.eligibility)
    return (
        str(record.audit_status) == "valid"
        and str(record.coverage_status) == "complete"
        and str(record.fallback_status) == "none"
        and record.realized_makespan is not None
        and not bool(eligibility.get("timeout", False))
        and bool(eligibility.get("performance_eligible", eligibility.get("offline_replay_eligible", False)))
    )


def paired_aggregate(
    records: Sequence[OfflineEvaluationRecord],
    *,
    baseline_predictor_id: str,
    candidate_predictor_id: str | None = None,
    planner_id: str | None = None,
    track: str | None = None,
) -> dict[str, object]:
    grouped: dict[tuple[str, str, str, str, str, str], list[OfflineEvaluationRecord]] = {}
    for record in records:
        if planner_id is not None and str(record.planner_id) != str(planner_id):
            continue
        if track is not None and str(record.track) != str(track):
            continue
        key = (
            str(record.window_identity),
            str(record.evaluation_spec_digest),
            str(record.task_set_digest),
            str(record.execution_truth_digest),
            str(record.planner_id),
            str(record.track),
            str(record.metrics.get("repeat", "")),
        )
        grouped.setdefault(key, []).append(record)
    gains: list[float] = []
    invalid_count = 0
    fallback_count = 0
    for bucket in grouped.values():
        baseline = next((item for item in bucket if item.predictor_id == baseline_predictor_id), None)
        if baseline is None or not _eligible_for_aggregate(baseline):
            invalid_count += 1
            continue
        for item in bucket:
            if item.predictor_id == baseline_predictor_id:
                continue
            if candidate_predictor_id is not None and str(item.predictor_id) != str(candidate_predictor_id):
                continue
            if not _eligible_for_aggregate(item):
                invalid_count += 1
                if str(item.fallback_status) != "none":
                    fallback_count += 1
                continue
            gains.append(float(baseline.realized_makespan) - float(item.realized_makespan))
            if str(item.fallback_status) != "none":
                fallback_count += 1
    if not gains:
        return {
            "baseline_predictor_id": str(baseline_predictor_id),
            "candidate_predictor_id": candidate_predictor_id,
            "planner_id": planner_id,
            "track": track,
            "sample_count": 0,
            "mean_paired_gain": None,
            "median_paired_gain": None,
            "p25": None,
            "p75": None,
            "win_rate": None,
            "tie_rate": None,
            "loss_rate": None,
            "worst_case": None,
            "best_case": None,
            "invalid_count": invalid_count,
            "fallback_count": fallback_count,
        }
    ordered = sorted(gains)
    wins = sum(1 for gain in gains if gain > 0.0)
    ties = sum(1 for gain in gains if gain == 0.0)
    losses = sum(1 for gain in gains if gain < 0.0)
    return {
        "baseline_predictor_id": str(baseline_predictor_id),
        "candidate_predictor_id": candidate_predictor_id,
        "planner_id": planner_id,
        "track": track,
        "sample_count": len(gains),
        "mean_paired_gain": sum(gains) / len(gains),
        "median_paired_gain": median(gains),
        "p25": ordered[int((len(ordered) - 1) * 0.25)],
        "p75": ordered[int((len(ordered) - 1) * 0.75)],
        "win_rate": wins / len(gains),
        "tie_rate": ties / len(gains),
        "loss_rate": losses / len(gains),
        "worst_case": min(gains),
        "best_case": max(gains),
        "invalid_count": invalid_count,
        "fallback_count": fallback_count,
    }


def build_evaluation_bundle(
    *,
    spec: EvaluationSpec,
    records: Sequence[OfflineEvaluationRecord],
    oracle_records: Sequence[Mapping[str, object]] = (),
    prediction_records: Sequence[Mapping[str, object]] = (),
    parity_records: Sequence[Mapping[str, object]] = (),
    paired_aggregates: Sequence[Mapping[str, object]] = (),
    eligibility_summary: Mapping[str, object] | None = None,
) -> OfflineEvaluationBundle:
    bundle = OfflineEvaluationBundle(
        schema_version="offline_bundle_v1",
        evaluation_spec=spec,
        records=tuple(records),
        oracle_records=tuple(dict(item) for item in oracle_records),
        prediction_records=tuple(dict(item) for item in prediction_records),
        parity_records=tuple(dict(item) for item in parity_records),
        paired_aggregates=tuple(dict(item) for item in paired_aggregates),
        eligibility_summary=dict(eligibility_summary or {}),
    )
    bundle.validate()
    return bundle


def schedule_quality_metrics(
    *,
    predicted_record: OfflineEvaluationRecord,
    zero_record: OfflineEvaluationRecord,
    perfect_record: OfflineEvaluationRecord,
) -> dict[str, object]:
    validate_comparable_records(predicted_record, zero_record)
    validate_comparable_records(predicted_record, perfect_record)
    if predicted_record.realized_makespan is None or zero_record.realized_makespan is None or perfect_record.realized_makespan is None:
        return {"valid": False, "reason": "missing_realized_makespan"}
    return {
        "valid": True,
        "gain_vs_zero": float(zero_record.realized_makespan) - float(predicted_record.realized_makespan),
        "regret_vs_perfect": float(predicted_record.realized_makespan) - float(perfect_record.realized_makespan),
        "plan_digest_difference_vs_zero": predicted_record.logical_plan_digest != zero_record.logical_plan_digest,
        "plan_digest_difference_vs_perfect": predicted_record.logical_plan_digest != perfect_record.logical_plan_digest,
    }


__all__ = [
    "build_evaluation_bundle",
    "build_offline_record",
    "paired_aggregate",
    "schedule_quality_metrics",
    "validate_comparable_records",
]
