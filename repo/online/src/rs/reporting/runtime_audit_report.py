from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import ReportBundle, load_manifest, read_json


def build_runtime_audit_report(run_dir: Path) -> ReportBundle:
    manifest = load_manifest(run_dir)
    summary_path = run_dir / "metrics" / "summary.json"
    if not summary_path.exists():
        summary_path = run_dir / "summary.json"
    summary = read_json(summary_path)
    payload: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "status": summary.get("status", manifest.get("status", "")),
        "fallback_count": summary.get("phase_sync_fallback_count", summary.get("fallback_count", 0)),
        "batch_isend_irecv_call_count": summary.get("batch_isend_irecv_call_count", 0),
        "stored_consumed_match": summary.get("stored_consumed_p1_digest_match", summary.get("stored_equals_consumed", False)),
    }
    markdown = "\n".join(
        [
            "# Runtime Audit Report",
            "",
            f"- run_id: `{manifest['run_id']}`",
            f"- status: `{payload['status']}`",
            f"- batch_isend_irecv_call_count: `{payload['batch_isend_irecv_call_count']}`",
            f"- fallback_count: `{payload['fallback_count']}`",
            f"- stored_consumed_match: `{payload['stored_consumed_match']}`",
        ]
    )
    return ReportBundle(report_type="runtime_audit", title="Runtime Audit Report", summary=payload, markdown=markdown)

