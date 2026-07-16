from __future__ import annotations

from pathlib import Path

from experiments.paper.contracts import RecordMetadata
from experiments.paper.prediction_evaluation import evaluate_prediction


def test_missing_predictor_keeps_leakage_and_regret_null() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    result = evaluate_prediction(
        fixture_dir=Path(__file__).resolve().parents[2] / "fixtures" / "offline_replay_smoke",
        metadata=metadata,
        model_id="m",
        model_revision="rev",
        joint_policy_id="U_barrier_criticality_global_matching",
    )
    row = result["records"][0]
    assert row["predictor_id"] is None
    assert row["no_future_leakage"] is None
    assert row["prediction_regret"] is None
    assert row["gain_over_zero"] is None
    assert row["predicted_plan_metrics"]["status"] == "MISSING_CAPABILITY"
