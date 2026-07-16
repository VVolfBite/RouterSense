from __future__ import annotations

from experiments.paper.contracts import RecordMetadata
from experiments.paper.runtime_evaluation import evaluate_runtime_correctness


def test_runtime_record_keeps_plan_identity() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    result = evaluate_runtime_correctness(metadata=metadata)
    row = result["records"][0]
    assert row["published_plan_digest"]
    assert row["materialized_plan_digest"] == row["executed_plan_digest"]
