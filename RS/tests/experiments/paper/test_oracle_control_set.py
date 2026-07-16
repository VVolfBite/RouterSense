from __future__ import annotations

from pathlib import Path

from experiments.paper.contracts import RecordMetadata
from experiments.paper.oracle_evaluation import evaluate_oracle_control_set, load_oracle_fixture


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "paper_oracle_controls"


def test_oracle_control_fixtures_all_load() -> None:
    names = {path.name for path in FIXTURE_DIR.glob("*.json")}
    assert names == {"joint_advantage.json", "tie.json", "unsupported.json"}
    fixture = load_oracle_fixture(FIXTURE_DIR / "joint_advantage.json")
    assert fixture.rank_count == 3
    assert fixture.expected_relation == "joint_strictly_better"


def test_oracle_control_set_expected_relations() -> None:
    metadata = RecordMetadata("b", "c", "d", "run", 0, "m", "rev")
    result = evaluate_oracle_control_set(FIXTURE_DIR, metadata, "formal_replay_makespan")
    assert result["cases"]["joint_advantage"]["observed_relation"] == "joint_strictly_better"
    assert result["cases"]["tie"]["observed_relation"] == "tie"
    assert result["cases"]["unsupported"]["observed_relation"] == "not_comparable"
    assert result["dominance_violation_count"] == 0


def test_oracle_control_set_joint_advantage_and_tie_are_replay_valid() -> None:
    metadata = RecordMetadata("b", "c", "d", "run", 0, "m", "rev")
    result = evaluate_oracle_control_set(FIXTURE_DIR, metadata, "formal_replay_makespan")
    by_case = {case["case_id"]: case for case in result["case_results"]}
    joint = by_case["oracle_joint_advantage_v1"]
    tie = by_case["oracle_tie_v1"]
    assert joint["o_local"]["coverage_valid"] is True
    assert joint["o_joint"]["coverage_valid"] is True
    assert float(joint["o_joint"]["objective"]) < float(joint["o_local"]["objective"])
    assert tie["o_local"]["coverage_valid"] is True
    assert tie["o_joint"]["coverage_valid"] is True
    assert float(tie["o_local"]["objective"]) == float(tie["o_joint"]["objective"])


def test_oracle_control_set_unsupported_fails_closed() -> None:
    metadata = RecordMetadata("b", "c", "d", "run", 0, "m", "rev")
    result = evaluate_oracle_control_set(FIXTURE_DIR, metadata, "formal_replay_makespan")
    unsupported = next(case for case in result["case_results"] if case["case_id"] == "oracle_unsupported_scale_v1")
    assert unsupported["o_local"]["comparable"] is False
    assert unsupported["o_joint"]["comparable"] is False
    assert unsupported["o_local"]["objective"] is None
    assert unsupported["o_joint"]["objective"] is None
    assert unsupported["o_local"]["solver_status"] == "unsupported_scale"
    assert unsupported["o_joint"]["solver_status"] == "unsupported_scale"
