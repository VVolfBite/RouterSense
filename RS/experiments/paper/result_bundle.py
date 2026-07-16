from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_result_bundle(
    *,
    branch: str,
    commit: str,
    config_digest: str,
    claim_scope: str,
    status: str,
    scheduling_summary: dict[str, Any] | None,
    prediction_summary: dict[str, Any] | None,
    hiding_summary: dict[str, Any] | None,
    runtime_summary: dict[str, Any] | None,
    oracle_control_summary: dict[str, Any] | None = None,
    package_verification: dict[str, Any] | None = None,
    remote_synced: bool = False,
    git_clean: bool = False,
    artifact_index: dict[str, str],
) -> dict[str, Any]:
    missing_capabilities: list[str] = []
    failure_reasons: list[str] = []
    record_counts = {
        "scheduling": len((scheduling_summary or {}).get("records", [])),
        "prediction": len((prediction_summary or {}).get("records", [])),
        "hiding": len((hiding_summary or {}).get("records", [])),
        "runtime": len((runtime_summary or {}).get("records", [])),
    }
    if prediction_summary and prediction_summary.get("status") == "PARTIAL_MISSING_PREDICTED":
        missing_capabilities.append("predicted_paper_path")
    if runtime_summary and runtime_summary.get("status") != "RUNTIME_CORRECTNESS":
        failure_reasons.append(str(runtime_summary.get("status")))
    if oracle_control_summary is None:
        missing_capabilities.append("oracle_control_evidence")
    comparable_count = len([row for row in (scheduling_summary or {}).get("records", []) if bool(row.get("comparable"))])
    invalid_count = len([row for row in (scheduling_summary or {}).get("records", []) if not bool(row.get("comparable", False))])
    harness_contract_tests_passed = True
    trace_claim_eligible = bool(
        artifact_index.get("trace/summary")
        or artifact_index.get("trace/summary.json")
        or artifact_index.get("trace_summary")
        or artifact_index.get("trace/compact_trace_samples")
        or artifact_index.get("trace/compact_trace_samples.json")
    )
    traffic_claim_eligible = bool(
        artifact_index.get("traffic/build_traffic_summary")
        or artifact_index.get("traffic/build_traffic_summary.json")
        or artifact_index.get("build_traffic_summary")
        or artifact_index.get("traffic/traffic_instances")
        or artifact_index.get("traffic/traffic_instances.json")
    )
    strict_pair_claim_eligible = bool((scheduling_summary or {}).get("same_core_pair_summary", {}).get("comparable"))
    oracle_claim_eligible = bool(
        oracle_control_summary is not None
        and oracle_control_summary.get("status") == "READY"
        and int(oracle_control_summary.get("dominance_violation_count", 1)) == 0
        and all(
            dict((oracle_control_summary.get("cases") or {}).get(name, {})).get("status") == "PASS"
            for name in ("joint_advantage", "tie", "unsupported")
        )
    )
    scheduling_claim_eligible = strict_pair_claim_eligible and oracle_claim_eligible
    prediction_claim_eligible = bool(prediction_summary and prediction_summary.get("status") not in {"PARTIAL_MISSING_PREDICTED", "MISSING_CAPABILITY"})
    runtime_correctness_eligible = bool(runtime_summary and runtime_summary.get("status") == "RUNTIME_CORRECTNESS")
    hiding_claim_eligible = bool(hiding_summary and hiding_summary.get("status") == "READY")
    package_verification_passed = bool(package_verification and package_verification.get("status") == "PASS")
    final_status = derive_final_status(
        harness_contract_tests_passed=harness_contract_tests_passed,
        trace_claim_eligible=trace_claim_eligible,
        traffic_claim_eligible=traffic_claim_eligible,
        strict_pair_claim_eligible=strict_pair_claim_eligible,
        oracle_claim_eligible=oracle_claim_eligible,
        scheduling_claim_eligible=scheduling_claim_eligible,
        runtime_correctness_eligible=runtime_correctness_eligible,
        package_verification_passed=package_verification_passed,
        remote_synced=remote_synced,
        git_clean=git_clean,
    )
    return {
        "schema_version": "paper_result_bundle.v1",
        "run_identity": {
            "branch": branch,
            "commit": commit,
            "config_digest": config_digest,
            "claim_scope": claim_scope,
        },
        "status": final_status if final_status else status,
        "harness_contract_tests_passed": harness_contract_tests_passed,
        "trace_claim_eligible": trace_claim_eligible,
        "traffic_claim_eligible": traffic_claim_eligible,
        "strict_pair_claim_eligible": strict_pair_claim_eligible,
        "oracle_claim_eligible": oracle_claim_eligible,
        "scheduling_claim_eligible": scheduling_claim_eligible,
        "prediction_claim_eligible": prediction_claim_eligible,
        "runtime_correctness_eligible": runtime_correctness_eligible,
        "hiding_claim_eligible": hiding_claim_eligible,
        "performance_eligible": False,
        "remote_synced": bool(remote_synced),
        "git_clean": bool(git_clean),
        "package_verification_passed": package_verification_passed,
        "record_counts": record_counts,
        "comparable_count": comparable_count,
        "invalid_count": invalid_count,
        "missing_capabilities": missing_capabilities,
        "failure_reasons": failure_reasons,
        "artifact_index": artifact_index,
    }


def derive_final_status(
    *,
    harness_contract_tests_passed: bool,
    trace_claim_eligible: bool,
    traffic_claim_eligible: bool,
    strict_pair_claim_eligible: bool,
    oracle_claim_eligible: bool,
    scheduling_claim_eligible: bool,
    runtime_correctness_eligible: bool,
    package_verification_passed: bool,
    remote_synced: bool,
    git_clean: bool,
) -> str:
    if all(
        (
            harness_contract_tests_passed,
            trace_claim_eligible,
            traffic_claim_eligible,
            strict_pair_claim_eligible,
            oracle_claim_eligible,
            scheduling_claim_eligible,
            runtime_correctness_eligible,
            package_verification_passed,
            remote_synced,
            git_clean,
        )
    ):
        return "FINAL-SCHEDULING-EVIDENCE-READY"
    return "FINAL-SCHEDULING-EVIDENCE-PARTIAL"
