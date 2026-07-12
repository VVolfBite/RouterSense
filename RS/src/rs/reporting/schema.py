from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReportBundle:
    report_type: str
    title: str
    summary: dict[str, Any]
    markdown: str


@dataclass(frozen=True)
class ReportEligibility:
    eligible: bool
    failures: tuple[str, ...] = field(default_factory=tuple)
    valid_for_performance_comparison: bool = False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "manifest.json")


def validate_report_eligibility(
    run_dir: Path,
    *,
    report_type: str,
    allow_invalid_diagnostic: bool = False,
) -> ReportEligibility:
    manifest = load_manifest(run_dir)
    status_path = run_dir / "status.json"
    status = read_json(status_path) if status_path.exists() else {"status": manifest.get("status", "")}
    failures: list[str] = []
    if str(manifest.get("status", "")) not in {"completed", "success"}:
        failures.append("manifest_status_not_completed")
    if str(status.get("status", "")) not in {"completed", "success"}:
        failures.append("status_not_completed")
    if bool(manifest.get("git_dirty", False)):
        failures.append("git_dirty")
    if str(manifest.get("commit_sha", "")) != str(manifest.get("runtime_commit_sha", manifest.get("commit_sha", ""))):
        failures.append("commit_sha_mismatch")
    if not bool(manifest.get("valid_for_evaluation", True)):
        failures.append("valid_for_evaluation_false")
    summary_candidates = [
        run_dir / "metrics" / "summary.json",
        run_dir / "summary.json",
        run_dir / "comparison_report.json",
        run_dir / "c2_runner_summary.json",
        run_dir / "a2_runner_summary.json",
        run_dir / "b2_runner_summary.json",
    ]
    summary = {}
    for path in summary_candidates:
        if path.exists():
            summary = read_json(path)
            break
    if int(summary.get("fallback_count", summary.get("phase_sync_fallback_count", 0)) or 0) > 0:
        failures.append("fallback_count_nonzero")
    if int(summary.get("timeout_count", 0) or 0) > 0:
        failures.append("timeout_count_nonzero")
    if int(summary.get("selected_layer_match_count", 1) or 0) <= 0:
        failures.append("selected_layer_match_count_zero")
    if int(summary.get("selected_transport_execution_count", 1) or 0) <= 0:
        failures.append("selected_transport_execution_count_zero")
    if not bool(summary.get("all_work_completed", True)):
        failures.append("all_work_completed_false")
    if int(summary.get("audit_invalid_count", 0) or 0) > 0:
        failures.append("audit_invalid_count_nonzero")
    if int(summary.get("legacy_secondary_policy_call_count", summary.get("legacy_secondary_policy_invocation_count", 0)) or 0) > 0:
        failures.append("legacy_secondary_policy_call_count_nonzero")
    if int(summary.get("compiler_shadow_compare_count", 0) or 0) > 0:
        failures.append("compiler_shadow_compare_count_nonzero")
    if "result_eligible_for_performance_comparison" in summary and not bool(summary.get("result_eligible_for_performance_comparison", False)):
        failures.append("performance_eligibility_false")
    if report_type == "a2" and not bool(summary.get("valid_for_a2", summary.get("c2_pass", False))):
        failures.append("a2_missing_c2_eligibility")
    eligible = not failures
    if failures and not allow_invalid_diagnostic:
        return ReportEligibility(eligible=False, failures=tuple(failures), valid_for_performance_comparison=False)
    return ReportEligibility(
        eligible=eligible,
        failures=tuple(failures),
        valid_for_performance_comparison=eligible,
    )
