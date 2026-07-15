from __future__ import annotations

from pathlib import Path

from .schema import ReportBundle, load_manifest, load_result_bundle


def build_performance_report(run_dir: Path, *, report_type: str) -> ReportBundle:
    manifest = load_manifest(run_dir)
    bundle = load_result_bundle(run_dir)
    summary = dict(bundle.summary)
    details = dict(bundle.details)
    payload = {
        "run_id": manifest["run_id"],
        "status": bundle.status,
        "report_type": report_type,
        "performance_eligible": bool(bundle.eligibility.performance_eligible),
        "correctness_eligible": bool(bundle.eligibility.correctness_eligible),
        "run_kind": str(details.get("run_kind", summary.get("run_kind", ""))),
        "measurement_complete": bool(bundle.measurement_complete),
        "execution_outcome_count": int(summary.get("execution_outcome_count", 0) or 0),
        "fallback_count": int(summary.get("fallback_count", 0) or 0),
        "timeout_count": int(summary.get("timeout_count", 0) or 0),
        "check_failure_count": int(summary.get("check_failure_count", 0) or 0),
    }
    markdown = "\n".join(
        [
            f"# {report_type.upper()} Report",
            "",
            f"- run_id: `{manifest['run_id']}`",
            f"- status: `{bundle.status}`",
            f"- run_kind: `{payload['run_kind']}`",
            f"- performance_eligible: `{payload['performance_eligible']}`",
            f"- execution_outcomes: `{payload['execution_outcome_count']}`",
        ]
    )
    return ReportBundle(report_type=report_type, title=f"{report_type.upper()} Report", summary=payload, markdown=markdown)

