from __future__ import annotations

from rs.core.contracts.result import EligibilityResult, ResultBundle


def _run_kind(bundle: ResultBundle) -> str:
    summary = dict(bundle.summary)
    details = dict(bundle.details)
    return str(details.get("run_kind", "") or summary.get("run_kind", "")).strip().upper()


def evaluate_result_bundle_eligibility(bundle: ResultBundle) -> EligibilityResult:
    summary = dict(bundle.summary)
    run_kind = _run_kind(bundle)
    correctness_reasons: list[str] = []
    if str(bundle.status) != "success":
        correctness_reasons.append("non_success_status")
    if str(bundle.correctness_status) != "valid":
        correctness_reasons.append("correctness_invalid")
    if not str(bundle.commit_sha).strip() or str(bundle.commit_sha).strip().lower() == "unknown":
        correctness_reasons.append("missing_commit_identity")
    if bundle.git_clean is not True:
        correctness_reasons.append("dirty_git")
    if str(bundle.instrumentation_mode) == "debug":
        correctness_reasons.append("debug_mode")
    if bundle.measurement_complete is not True:
        correctness_reasons.append("measurement_incomplete")
    if str(bundle.audit_evidence_level) == "unavailable":
        correctness_reasons.append("audit_unavailable")
    if summary.get("all_work_completed") is not True:
        correctness_reasons.append("all_work_incomplete")
    if int(summary.get("timeout_count", 0) or 0) > 0:
        correctness_reasons.append("timeout_present")
    if int(summary.get("check_failure_count", 0) or 0) > 0:
        correctness_reasons.append("check_failures_present")
    if int(summary.get("cleanup_failure_count", 0) or 0) > 0:
        correctness_reasons.append("cleanup_failures_present")
    if int(summary.get("missing_execution_outcome_count", 0) or 0) > 0:
        correctness_reasons.append("missing_execution_outcome")
    if int(summary.get("missing_expected_payload_role_count", 0) or 0) > 0:
        correctness_reasons.append("missing_expected_payload_roles")
    if int(summary.get("missing_selected_layer_count", 0) or 0) > 0:
        correctness_reasons.append("missing_selected_layers")
    if bool(summary.get("formal_execution_expected")) and int(summary.get("execution_outcome_count", 0) or 0) <= 0:
        correctness_reasons.append("expected_execution_without_outcome")
    if int(summary.get("execution_outcome_count", 0) or 0) <= 0 and str(bundle.run_identity.claim_scope) == "formal":
        correctness_reasons.append("formal_execution_missing")
    if str(bundle.correctness_status).lower() == "valid" and correctness_reasons:
        correctness_reasons.append("correctness_status_inconsistent")

    performance_reasons = list(correctness_reasons)
    performance_allowed_kinds = {"GPU_PERFORMANCE", "MULTINODE_PERFORMANCE", "OFFLINE_EVALUATION_FORMAL"}
    if run_kind not in performance_allowed_kinds:
        performance_reasons.append("run_kind_not_performance_claimable")
    if str(bundle.instrumentation_mode) != "perf_light":
        performance_reasons.append("instrumentation_mode_not_perf_light")
    if summary.get("performance_measurement_complete") is not True:
        performance_reasons.append("performance_measurement_incomplete")
    if int(summary.get("measured_repeat_count", 0) or 0) <= 0:
        performance_reasons.append("missing_measured_repeats")
    if summary.get("warmup_excluded") is not True:
        performance_reasons.append("warmup_not_excluded")
    if int(summary.get("fallback_count", 0) or 0) > 0:
        performance_reasons.append("fallback_present")
    if str(bundle.performance_status).lower() == "eligible" and performance_reasons:
        performance_reasons.append("performance_status_inconsistent")

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
    if run_kind != "OFFLINE_EVALUATION_FORMAL":
        offline_reasons.append("run_kind_not_offline_formal")
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
    if int(summary.get("fallback_count", 0) or 0) > 0:
        offline_reasons.append("fallback_present")

    preparation_reasons: list[str] = []
    if correctness_reasons:
        preparation_reasons.append("base_ineligible")
    if int(summary.get("preparation_miss_count", 0) or 0) > 0:
        preparation_reasons.append("preparation_miss_present")

    return EligibilityResult(
        correctness_eligible=not correctness_reasons,
        performance_eligible=not performance_reasons,
        prediction_evaluation_eligible=not prediction_reasons,
        offline_replay_eligible=not offline_reasons,
        preparation_claim_eligible=not preparation_reasons,
        correctness_reasons=tuple(correctness_reasons),
        performance_reasons=tuple(performance_reasons),
        prediction_reasons=tuple(prediction_reasons),
        offline_replay_reasons=tuple(offline_reasons),
        preparation_claim_reasons=tuple(preparation_reasons),
    )
