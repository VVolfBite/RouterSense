from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import ReportBundle, load_manifest, load_result_bundle


def build_offline_report(run_dir: Path) -> ReportBundle:
    manifest = load_manifest(run_dir)
    bundle = load_result_bundle(run_dir)
    details = dict(bundle.details)
    summary = dict(bundle.summary)
    row_count = int(details.get("row_count", summary.get("offline_record_count", 0)) or 0)
    invariant_count = int(details.get("invariant_count", 0) or 0)
    policies = sorted(str(item) for item in details.get("policy_names", ()) if str(item).strip())
    hint_types = sorted(str(item) for item in details.get("hint_names", ()) if str(item).strip())
    payload: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "status": bundle.status,
        "row_count": row_count,
        "invariant_count": invariant_count,
        "policies": policies,
        "hint_types": hint_types,
        "offline_replay_complete": bool(summary.get("offline_replay_complete", False)),
        "offline_replay_eligible": bool(bundle.eligibility.offline_replay_eligible),
    }
    markdown = "\n".join(
        [
            "# Offline Replay Report",
            "",
            f"- run_id: `{manifest['run_id']}`",
            f"- status: `{bundle.status}`",
            f"- rows: `{row_count}`",
            f"- invariants: `{invariant_count}`",
            f"- policies: `{', '.join(payload['policies'])}`",
            f"- hints: `{', '.join(payload['hint_types'])}`",
        ]
    )
    return ReportBundle(report_type="offline", title="Offline Replay Report", summary=payload, markdown=markdown)

