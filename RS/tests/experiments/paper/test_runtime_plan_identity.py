from __future__ import annotations

from experiments.paper.contracts import RecordMetadata
from experiments.paper.runtime_evaluation import evaluate_materialization_contract_smoke


def test_materialization_smoke_does_not_claim_real_execution() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    result = evaluate_materialization_contract_smoke(metadata=metadata)
    row = result["records"][0]
    assert result["status"] == "MATERIALIZATION_CONTRACT_SMOKE"
    assert row["runtime_status"] == "MATERIALIZATION_CONTRACT_SMOKE"
    assert row["executed_plan_digest"] is None
    assert row["execution_backend_id"] is None
    assert row["completed_tasks"] is None
    assert row["reference_output_digest"] is None
    assert row["executed_output_digest"] is None
    assert row["parity_status"] == "NOT_EXECUTED"
