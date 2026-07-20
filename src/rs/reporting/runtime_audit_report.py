from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import ReportBundle, load_manifest, load_result_bundle


def build_runtime_audit_report(run_dir: Path) -> ReportBundle:
    manifest = load_manifest(run_dir)
    bundle = load_result_bundle(run_dir)
    summary = dict(bundle.summary)
    details = dict(bundle.details)
    payload: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "status": bundle.status,
        "fallback_count": int(summary.get("fallback_count", 0) or 0),
        "timeout_count": int(summary.get("timeout_count", 0) or 0),
        "check_failure_count": int(summary.get("check_failure_count", 0) or 0),
        "execution_outcome_count": int(summary.get("execution_outcome_count", 0) or 0),
        "stored_consumed_match": bool(details.get("stored_consumed_match", False)),
    }
    markdown = "\n".join(
        [
            "# Runtime Audit Report",
            "",
            f"- run_id: `{manifest['run_id']}`",
            f"- status: `{bundle.status}`",
            f"- execution_outcome_count: `{payload['execution_outcome_count']}`",
            f"- fallback_count: `{payload['fallback_count']}`",
            f"- stored_consumed_match: `{payload['stored_consumed_match']}`",
        ]
    )
    return ReportBundle(report_type="runtime_audit", title="Runtime Audit Report", summary=payload, markdown=markdown)

