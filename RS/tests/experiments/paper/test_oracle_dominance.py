from __future__ import annotations

from pathlib import Path

from experiments.paper.contracts import RecordMetadata
from experiments.paper.scheduling_evaluation import evaluate_scheduling


def test_unsupported_exact_result_is_not_comparable() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    result = evaluate_scheduling(
        fixture_dir=Path(__file__).resolve().parents[2] / "fixtures" / "offline_replay_smoke",
        metadata=metadata,
        model_id="m",
        model_revision="rev",
        policy_ids=("exact_small_instance_reference",),
    )
    row = result["records"][0]
    assert result["o_local_status"] == "SEMANTICALLY_INVALID"
    assert row["is_exact"] is True
    assert row["comparable"] is False
    assert row["objective"] is None
    assert row["best_bound"] is None
    assert row["optimality_gap"] is None
