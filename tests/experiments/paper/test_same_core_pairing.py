from __future__ import annotations

from pathlib import Path

from experiments.paper.contracts import RecordMetadata
from experiments.paper.scheduling_evaluation import evaluate_scheduling


def test_same_core_pairing_summary_uses_strict_pair_not_family_pair() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    result = evaluate_scheduling(
        fixture_dir=Path(__file__).resolve().parents[2] / "fixtures" / "offline_replay_smoke",
        metadata=metadata,
        model_id="m",
        model_revision="rev",
        policy_ids=("B_barrier_criticality_core_independent", "U_barrier_criticality_global_matching"),
    )
    summary = result["same_core_pair_summary"]
    assert summary["status"] == "READY"
    assert summary["family_pair"]["b_policy_id"] == "B_barrier_criticality_matching"
    assert summary["strict_same_core_pair"]["b_policy_id"] == "B_barrier_criticality_core_independent"
    assert summary["strict_same_core_pair"]["u_policy_id"] == "U_barrier_criticality_global_matching"
