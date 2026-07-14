from __future__ import annotations

from rs.core.contracts.result import EligibilityResult, ResultBundle


def evaluate_result_bundle_eligibility(bundle: ResultBundle) -> EligibilityResult:
    reasons: list[str] = []
    summary = dict(bundle.summary)
    details = dict(bundle.details)
    if not summary:
        reasons.append("empty_summary")
    if "all_work_completed" not in summary:
        reasons.append("missing_all_work_completed")
    elif not bool(summary.get("all_work_completed")):
        reasons.append("all_work_incomplete")
    if int(summary.get("fallback_count", 0) or 0) > 0:
        reasons.append("fallback_count")
    if int(summary.get("timeout_count", 0) or 0) > 0:
        reasons.append("timeout_count")
    if str(details.get("instrumentation_mode", "")) == "debug":
        reasons.append("debug_mode")
    performance_eligible = not reasons
    return EligibilityResult(
        correctness_eligible=bool(bundle.status == "success"),
        performance_eligible=bool(performance_eligible),
        prediction_evaluation_eligible=bool(bundle.status == "success"),
        offline_replay_eligible=bool(bundle.status == "success"),
        reasons=tuple(reasons),
    )
