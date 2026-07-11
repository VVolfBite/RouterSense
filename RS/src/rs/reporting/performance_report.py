from __future__ import annotations

from pathlib import Path

from .schema import ReportBundle, load_manifest, read_json


def build_performance_report(run_dir: Path, *, report_type: str) -> ReportBundle:
    manifest = load_manifest(run_dir)
    if report_type == "comparison":
        summary_path = run_dir / "comparison_report.json"
    elif report_type == "a2":
        summary_path = run_dir / "a2_runner_summary.json"
    else:
        summary_path = run_dir / "c2_runner_summary.json"
    if not summary_path.exists():
        summary_path = run_dir / "metrics" / "summary.json"
    summary = read_json(summary_path)
    markdown = "\n".join(
        [
            f"# {report_type.upper()} Report",
            "",
            f"- run_id: `{manifest['run_id']}`",
            f"- status: `{summary.get('status', manifest.get('status', 'unknown'))}`",
            f"- source: `{summary_path.name}`",
        ]
    )
    return ReportBundle(report_type=report_type, title=f"{report_type.upper()} Report", summary=summary, markdown=markdown)

