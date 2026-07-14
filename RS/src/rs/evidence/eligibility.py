from __future__ import annotations

from rs.core.contracts.result import EligibilityResult, ResultBundle


def evaluate_result_bundle_eligibility(bundle: ResultBundle) -> EligibilityResult:
    correctness_reasons: list[str] = []
    summary = dict(bundle.summary)
    details = dict(bundle.details)
    run_kind = str(details.get("run_kind", "") or summary.get("run_kind", "")).strip().upper()
    if str(bundle.status) != "success":
        correctness_reasons.append("non_success_status")
    if str(bundle.correctness_status) != "valid":
        correctness_reasons.append("correctness_invalid")
    if not summary:
        correctness_reasons.append("empty_summary")
    if not str(bundle.commit_sha).strip():
        correctness_reasons.append("missing_commit_sha")
    if bundle.git_clean is not True:
        correctness_reasons.append("dirty_git")
    if str(bundle.instrumentation_mode) == "debug":
        correctness_reasons.append("debug_mode")
    elif str(bundle.instrumentation_mode) not in {"off", "contract", "perf_light"}:
        correctness_reasons.append("instrumentation_mode")
    if "all_work_completed" not in summary:
        correctness_reasons.append("missing_all_work_completed")
    elif not bool(summary.get("all_work_completed")):
        correctness_reasons.append("all_work_incomplete")
    if "fallback_count" not in summary:
        correctness_reasons.append("missing_fallback_count")
    elif int(summary.get("fallback_count", 0) or 0) > 0:
        correctness_reasons.append("fallback_present")
    if "timeout_count" not in summary:
        correctness_reasons.append("missing_timeout_count")
    elif int(summary.get("timeout_count", 0) or 0) > 0:
        correctness_reasons.append("timeout_present")
    if "check_failure_count" not in summary:
        correctness_reasons.append("missing_check_failure_count")
    elif int(summary.get("check_failure_count", 0) or 0) > 0:
        correctness_reasons.append("check_failures_present")
    if bundle.measurement_complete is not True:
        correctness_reasons.append("measurement_incomplete")
    if str(bundle.audit_evidence_level) == "unavailable":
        correctness_reasons.append("audit_unavailable")
    correctness_eligible = not correctness_reasons
    performance_allowed_kinds = {"GPU_PERFORMANCE", "MULTINODE_PERFORMANCE", "OFFLINE_EVALUATION_FORMAL"}
    performance_reasons = list(correctness_reasons)
    if run_kind not in performance_allowed_kinds:
        performance_reasons.append("run_kind_not_performance_claimable")
    if not bool(summary.get("performance_measurement_complete", False)):
        performance_reasons.append("performance_measurement_incomplete")
    if int(summary.get("measured_repeat_count", 0) or 0) <= 0:
        performance_reasons.append("missing_measured_repeats")
    if summary.get("warmup_excluded") is not True:
        performance_reasons.append("warmup_not_excluded")
    performance_eligible = not performance_reasons
    prediction_reasons: list[str] = []
    if correctness_reasons:
        prediction_reasons.append("base_ineligible")
    if summary.get("prediction_evaluation_complete") is not True:
        prediction_reasons.append("prediction_evaluation_incomplete")
    if not str(summary.get("prediction_truth_digest", "")).strip():
        prediction_reasons.append("missing_prediction_truth_digest")
    if int(summary.get("prediction_record_count", 0) or 0) <= 0:
        prediction_reasons.append("missing_prediction_records")
    if int(summary.get("prediction_metric_count", 0) or 0) <= 0:
        prediction_reasons.append("missing_prediction_metrics")
    if str(summary.get("prediction_audit_status", "")).lower() != "valid":
        prediction_reasons.append("prediction_audit_invalid")
    if summary.get("truth_leakage_check") is not True:
        prediction_reasons.append("truth_leakage_check_failed")
    offline_reasons: list[str] = []
    if correctness_reasons:
        offline_reasons.append("base_ineligible")
    if summary.get("offline_replay_complete") is not True:
        offline_reasons.append("offline_replay_incomplete")
    if not str(summary.get("evaluation_spec_digest", "")).strip():
        offline_reasons.append("missing_evaluation_spec_digest")
    if not str(summary.get("task_set_digest", "")).strip():
        offline_reasons.append("missing_task_set_digest")
    if not str(summary.get("execution_truth_digest", "")).strip():
        offline_reasons.append("missing_execution_truth_digest")
    if int(summary.get("offline_record_count", 0) or 0) <= 0:
        offline_reasons.append("missing_offline_records")
    if str(summary.get("offline_audit_status", "")).lower() != "valid":
        offline_reasons.append("offline_audit_invalid")
    if str(summary.get("coverage_status", "")).lower() != "complete":
        offline_reasons.append("coverage_incomplete")
    reasons = list(correctness_reasons)
    if correctness_eligible and str(bundle.correctness_status).lower() != "valid":
        reasons.append("correctness_status_missing")
    if str(bundle.performance_status).lower() == "eligible" and not performance_eligible:
        reasons.append("performance_status_inconsistent")
    if performance_eligible and str(bundle.performance_status).lower() not in {"eligible", ""}:
        reasons.append("performance_status_missing")
    return EligibilityResult(
        correctness_eligible=bool(correctness_eligible),
        performance_eligible=bool(performance_eligible),
        prediction_evaluation_eligible=not prediction_reasons,
        offline_replay_eligible=not offline_reasons,
        reasons=tuple(
            reasons
            + [f"performance:{item}" for item in performance_reasons]
            + [f"prediction:{item}" for item in prediction_reasons]
            + [f"offline:{item}" for item in offline_reasons]
        ),
    )
