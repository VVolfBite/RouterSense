from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import ReportBundle, load_manifest, read_json


def build_offline_report(run_dir: Path) -> ReportBundle:
    manifest = load_manifest(run_dir)
    summary_path = run_dir / "metrics" / "summary.json"
    if not summary_path.exists():
        summary_path = run_dir / "summary.json"
    summary = read_json(summary_path)
    rows = list(summary.get("rows", []))
    invariants = list(summary.get("invariants", []))
    payload: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "row_count": len(rows),
        "invariant_count": len(invariants),
        "policies": sorted({str(row.get("canonical_policy_name", "")) for row in rows}),
        "hint_types": sorted({str(row.get("hint_type", "")) for row in rows}),
    }
    markdown = "\n".join(
        [
            "# Offline Replay Report",
            "",
            f"- run_id: `{manifest['run_id']}`",
            f"- rows: `{len(rows)}`",
            f"- invariants: `{len(invariants)}`",
            f"- policies: `{', '.join(payload['policies'])}`",
            f"- hints: `{', '.join(payload['hint_types'])}`",
        ]
    )
    return ReportBundle(report_type="offline", title="Offline Replay Report", summary=payload, markdown=markdown)

