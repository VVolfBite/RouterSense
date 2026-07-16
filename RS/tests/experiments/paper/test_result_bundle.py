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
        artifact_index={"runtime_summary": "runtime_summary.json"},
    )
    assert bundle["harness_contract_tests_passed"] is True
    assert bundle["trace_claim_eligible"] is True
    assert bundle["traffic_claim_eligible"] is True
    assert bundle["scheduling_claim_eligible"] is False
    assert bundle["prediction_claim_eligible"] is False
    assert bundle["runtime_correctness_eligible"] is False
    assert bundle["performance_eligible"] is False
    assert "predicted_paper_path" in bundle["missing_capabilities"]
