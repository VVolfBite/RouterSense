from __future__ import annotations

from rs.core.contracts.result import EligibilityResult, ResultBundle


def evaluate_result_bundle_eligibility(bundle: ResultBundle) -> EligibilityResult:
    reasons: list[str] = []
    summary = dict(bundle.summary)
    if str(bundle.status) != "success":
        reasons.append("non_success_status")
    if str(bundle.correctness_status) != "valid":
        reasons.append("correctness_invalid")
    if not summary:
        reasons.append("empty_summary")
    if not str(bundle.commit_sha).strip():
        reasons.append("missing_commit_sha")
    if bundle.git_clean is not True:
        reasons.append("dirty_git")
    if str(bundle.instrumentation_mode) == "debug":
        reasons.append("debug_mode")
    elif str(bundle.instrumentation_mode) not in {"off", "contract", "perf_light"}:
        reasons.append("instrumentation_mode")
    if "all_work_completed" not in summary:
        reasons.append("missing_all_work_completed")
    elif not bool(summary.get("all_work_completed")):
        reasons.append("all_work_incomplete")
    if "fallback_count" not in summary:
        reasons.append("missing_fallback_count")
    elif int(summary.get("fallback_count", 0) or 0) > 0:
        reasons.append("fallback_present")
    if "timeout_count" not in summary:
        reasons.append("missing_timeout_count")
    elif int(summary.get("timeout_count", 0) or 0) > 0:
        reasons.append("timeout_present")
    if "check_failure_count" not in summary:
        reasons.append("missing_check_failure_count")
    elif int(summary.get("check_failure_count", 0) or 0) > 0:
        reasons.append("check_failures_present")
    if bundle.measurement_complete is not True:
        reasons.append("measurement_incomplete")
    if str(bundle.audit_evidence_level) == "unavailable":
        reasons.append("audit_unavailable")
    performance_eligible = not reasons
    return EligibilityResult(
        correctness_eligible=bool(bundle.status == "success" and bundle.correctness_status == "valid"),
        performance_eligible=bool(performance_eligible),
        prediction_evaluation_eligible=bool(bundle.status == "success" and bundle.correctness_status == "valid"),
        offline_replay_eligible=bool(bundle.status == "success" and bundle.correctness_status == "valid"),
        reasons=tuple(reasons),
    )
