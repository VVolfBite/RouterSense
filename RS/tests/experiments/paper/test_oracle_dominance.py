from __future__ import annotations

from experiments.paper.capability_audit import run_capability_audit
from experiments.paper.contracts import RecordMetadata
from experiments.paper.scheduling_evaluation import evaluate_scheduling


def test_exact_oracle_dominance_holds_on_smoke_fixture() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    result = evaluate_scheduling(
        fixture_dir=__import__("pathlib").Path(__file__).resolve().parents[2] / "fixtures" / "offline_replay_smoke",
        metadata=metadata,
        model_id="m",
        model_revision="rev",
        policy_ids=("birkhoff_von_neumann_fluid", "exact_small_instance_reference"),
    )
    assert result["oracle_dominance_ok"] is True
