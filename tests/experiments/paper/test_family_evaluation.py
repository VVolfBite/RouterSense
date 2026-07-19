from __future__ import annotations

import json
from pathlib import Path

from experiments.paper.adapters.scheduling_adapter import replay_window_from_matrices
from experiments.paper.family_evaluation import evaluate_family_pairs


def test_family_evaluation_reports_effect_and_overhead() -> None:
    fixture = json.loads(Path("tests/fixtures/offline_replay_smoke/replay_layer_1.json").read_text(encoding="utf-8"))
    matrix = lambda key: tuple(tuple(int(value) for value in row) for row in fixture[key])
    window = replay_window_from_matrices(
        fixture_id="family-eval",
        layer_id=1,
        p0_matrix=matrix("p0_dispatch_matrix"),
        p1_matrix=matrix("p1_return_matrix"),
        p2_matrix=matrix("p2_next_dispatch_matrix"),
    )
    result = evaluate_family_pairs(
        replay_window=window,
        family_ids=("greedy_control", "gmwd", "rsbc"),
        repeats=2,
        warmups=0,
    )
    assert result["status"] == "READY"
    assert len(result["records"]) == 3
    for row in result["records"]:
        assert row["contract_equal"] is True
        assert row["contract_mismatches"] == []
        assert row["local"]["valid"] is True
        assert row["joint"]["valid"] is True
        assert row["local"]["deterministic_plan"] is True
        assert row["joint"]["deterministic_plan"] is True
        assert row["effect"]["outcome"] in {"win", "tie", "loss"}
        assert "joint_improvement_pct" in row["effect"]
        assert "joint_minus_local_runtime_ms" in row["overhead"]
        assert row["local"]["planning_runtime_ms"]["median"] >= 0.0
        assert row["joint"]["planning_runtime_ms"]["median"] >= 0.0


def test_main_scheduling_evaluator_emits_family_pair_summaries(tmp_path: Path) -> None:
    from experiments.paper.contracts import RecordMetadata
    from experiments.paper.scheduling_evaluation import evaluate_scheduling

    fixture_src = Path("tests/fixtures/tier1/unlock_hotspot_4rank.json")
    fixture_dst = tmp_path / "replay_layer_1.json"
    fixture_dst.write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")
    metadata = RecordMetadata(
        branch="test",
        commit="deadbeef",
        config_digest="cfg",
        run_id="family-main-eval",
        seed=0,
        model_id="fixture",
        model_revision="v1",
    )
    result = evaluate_scheduling(
        fixture_dir=tmp_path,
        metadata=metadata,
        model_id="fixture",
        model_revision="v1",
        policy_ids=(
            "greedy_control_local",
            "greedy_control_joint",
            "gmwd_local",
            "gmwd_joint",
            "rsbc_local",
            "rsbc_joint",
        ),
    )
    assert result["status"] == "OK"
    summaries = {row["family_id"]: row for row in result["family_pair_summaries"]}
    assert summaries["greedy_control"]["status"] == "READY"
    assert summaries["gmwd"]["status"] == "READY"
    assert summaries["rsbc"]["status"] == "READY"
    assert summaries["greedy_control"]["pair_count"] == 1
    assert summaries["greedy_control"]["win_count"] + summaries["greedy_control"]["tie_count"] + summaries["greedy_control"]["loss_count"] == 1
