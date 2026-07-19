from __future__ import annotations

from experiments.paper.result_bundle import build_result_bundle, sha256_file


def test_result_bundle_sha256(tmp_path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("abc", encoding="utf-8")
    assert len(sha256_file(path)) == 64


def test_result_bundle_marks_runtime_ineligible_without_real_execution() -> None:
    bundle = build_result_bundle(
        branch="b",
        commit="c",
        config_digest="d",
        claim_scope="runtime_correctness",
        status="PAPER-EVAL-HARNESS-PARTIAL",
        scheduling_summary={"records": []},
        prediction_summary={"records": [], "status": "PARTIAL_MISSING_PREDICTED"},
        hiding_summary={"records": []},
        runtime_summary={"status": "MATERIALIZATION_CONTRACT_SMOKE", "records": []},
        oracle_control_summary=None,
        package_verification=None,
        remote_synced=False,
        git_clean=False,
        artifact_index={"runtime_summary": "runtime_summary.json"},
    )
    assert bundle["harness_contract_tests_passed"] is True
    assert bundle["trace_claim_eligible"] is False
    assert bundle["traffic_claim_eligible"] is False
    assert bundle["strict_pair_claim_eligible"] is False
    assert bundle["oracle_claim_eligible"] is False
    assert bundle["scheduling_claim_eligible"] is False
    assert bundle["prediction_claim_eligible"] is False
    assert bundle["runtime_correctness_eligible"] is False
    assert bundle["hiding_claim_eligible"] is False
    assert bundle["performance_eligible"] is False
    assert "predicted_paper_path" in bundle["missing_capabilities"]
    assert bundle["status"] == "FINAL-SCHEDULING-EVIDENCE-PARTIAL"


def test_result_bundle_fails_closed_when_any_configured_scheduling_policy_is_invalid() -> None:
    bundle = build_result_bundle(
        branch="b",
        commit="c",
        config_digest="d",
        claim_scope="scheduling",
        status="IGNORED",
        scheduling_summary={
            "status": "PARTIAL_INVALID_POLICY",
            "same_core_pair_summary": {"comparable": True},
            "records": [
                {"policy_id": "B", "comparable": True, "coverage_valid": True},
                {"policy_id": "islip_bucket", "comparable": False, "coverage_valid": False},
            ],
        },
        prediction_summary={"records": [], "status": "PARTIAL_MISSING_PREDICTED"},
        hiding_summary={"records": []},
        runtime_summary={"status": "RUNTIME_CORRECTNESS", "records": []},
        oracle_control_summary={
            "status": "READY",
            "dominance_violation_count": 0,
            "cases": {name: {"status": "PASS"} for name in ("joint_advantage", "tie", "unsupported")},
        },
        package_verification={"status": "PASS"},
        remote_synced=True,
        git_clean=True,
        artifact_index={
            "trace/summary.json": "trace/summary.json",
            "traffic/traffic_instances.json": "traffic/traffic_instances.json",
        },
    )
    assert bundle["strict_pair_claim_eligible"] is True
    assert bundle["scheduling_claim_eligible"] is False
    assert bundle["status"] == "FINAL-SCHEDULING-EVIDENCE-PARTIAL"
    assert any("PARTIAL_INVALID_POLICY" in item for item in bundle["failure_reasons"])
