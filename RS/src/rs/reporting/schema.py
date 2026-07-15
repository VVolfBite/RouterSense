from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rs.core.contracts.result import ResultBundle
from rs.evidence.eligibility import evaluate_result_bundle_eligibility


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
    diagnostic_only: bool = False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return read_json(run_dir / "manifest.json")


def load_result_bundle(run_dir: Path) -> ResultBundle:
    return ResultBundle.from_dict(read_json(run_dir / "result_bundle.json"))


def validate_report_eligibility(
    run_dir: Path,
    *,
    report_type: str,
    allow_invalid_diagnostic: bool = False,
) -> ReportEligibility:
    failures: list[str] = []
    manifest = load_manifest(run_dir)
    bundle_path = run_dir / "result_bundle.json"
    if not bundle_path.exists():
        failures.append("missing_result_bundle")
        return ReportEligibility(
            eligible=False,
            failures=tuple(failures),
            valid_for_performance_comparison=False,
            diagnostic_only=bool(allow_invalid_diagnostic),
        )
    try:
        bundle = load_result_bundle(run_dir)
    except Exception as exc:
        failures.append(f"invalid_result_bundle:{type(exc).__name__}")
        return ReportEligibility(
            eligible=False,
            failures=tuple(failures),
            valid_for_performance_comparison=False,
            diagnostic_only=bool(allow_invalid_diagnostic),
        )
    recomputed = evaluate_result_bundle_eligibility(bundle)
    if bundle.eligibility.to_dict() != recomputed.to_dict():
        failures.append("eligibility_recompute_mismatch")
    manifest_commit = str(manifest.get("commit_sha", "")).strip()
    if manifest_commit and manifest_commit != str(bundle.commit_sha).strip():
        failures.append("commit_sha_mismatch")
    if bool(manifest.get("git_dirty", False)) != bool(bundle.git_clean is False):
        # manifest.git_dirty should be the inverse of bundle.git_clean when both are present
        if "git_dirty" in manifest:
            expected_clean = not bool(manifest.get("git_dirty", False))
            if bool(bundle.git_clean) != expected_clean:
                failures.append("git_state_mismatch")
    if bundle.measurement_complete is not True:
        failures.append("measurement_incomplete")
    if "execution_outcome_count" not in bundle.summary:
        failures.append("missing_execution_outcome_count")
    if bundle.summary.get("all_work_completed") is not True:
        failures.append("all_work_incomplete")
    if int(bundle.summary.get("fallback_count", 0) or 0) > 0:
        failures.append("fallback_count_nonzero")
    if int(bundle.summary.get("timeout_count", 0) or 0) > 0:
        failures.append("timeout_count_nonzero")
    if int(bundle.summary.get("check_failure_count", 0) or 0) > 0:
        failures.append("check_failure_count_nonzero")
    if str(bundle.instrumentation_mode) == "debug":
        failures.append("debug_mode")
    if report_type in {"a2", "c2", "comparison"} and not bool(bundle.eligibility.performance_eligible):
        failures.append("performance_eligibility_false")
    eligible = not failures and bool(bundle.eligibility.correctness_eligible)
    if report_type in {"a2", "c2", "comparison"}:
        eligible = eligible and bool(bundle.eligibility.performance_eligible)
    if failures and not allow_invalid_diagnostic:
        return ReportEligibility(
            eligible=False,
            failures=tuple(failures),
            valid_for_performance_comparison=False,
            diagnostic_only=False,
        )
    return ReportEligibility(
        eligible=eligible,
        failures=tuple(failures),
        valid_for_performance_comparison=eligible and bool(bundle.eligibility.performance_eligible),
        diagnostic_only=bool(failures),
    )
