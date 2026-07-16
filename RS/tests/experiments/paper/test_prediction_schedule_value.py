from __future__ import annotations

from pathlib import Path

from experiments.paper.contracts import RecordMetadata
from experiments.paper.prediction_evaluation import evaluate_prediction


def test_prediction_eval_keeps_predicted_missing_explicit() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    result = evaluate_prediction(
        fixture_dir=Path(__file__).resolve().parents[2] / "fixtures" / "offline_replay_smoke",
        metadata=metadata,
        model_id="m",
        model_revision="rev",
        joint_policy_id="U_barrier_criticality_global_matching",
    )
    assert result["status"] == "PARTIAL_MISSING_PREDICTED"
    assert result["records"][0]["predicted_plan_metrics"]["status"] == "MISSING_CAPABILITY"
