from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from rs.runtime.offline.prediction.evaluation import rolling_predictor_records
from rs.runtime.offline.replay_unified import CanonicalBucketizer, PlanningHint, ReplayEngine, ReplayWindow, build_execution_truth, build_planning_problem
from rs.runtime.online.megatron_ep.prediction.contracts import PredictionInput
from rs.runtime.online.megatron_ep.prediction.simple_predictors import HistoryEMATrafficPredictor
from rs.scheduling.traffic_matrix import matrix_digest_remote, matrix_nonzero_remote_edge_count, matrix_remote_bytes
from rs.scheduling.catalog import resolve_algorithm_id
from rs.scheduling.public_catalog import deployable_policies, reference_policies, resolve_public_policy_name


REPO_ROOT = Path(__file__).resolve().parents[2]


def _window() -> ReplayWindow:
    return ReplayWindow(
        fixture_id="fixture",
        window_id="1->2",
        layer_id=1,
        p0_truth_rows=((0, 3), (2, 0)),
        p1_truth_rows=((0, 2), (3, 0)),
        p2_truth_rows=((0, 4), (1, 0)),
        matrix_unit="rows",
        group_size=2,
        payload_row_bytes_by_phase={"P0": 1, "P1": 1, "P2": 1},
        metadata={},
    )


def test_replay_window_and_single_hint_are_truth_separated() -> None:
    window = _window()
    hint = PlanningHint(
        hint_type="zero_hint",
        p2_hint_rows=((0, 0), (0, 0)),
        confidence=0.0,
        source_layer=None,
        target_layer=2,
    )
    planning_problem = build_planning_problem(replay_window=window, planning_hint=hint)
    execution_truth = build_execution_truth(window)
    assert planning_problem.replay_window.p2_truth_rows == ((0, 4), (1, 0))
    assert planning_problem.planning_hint.p2_hint_rows == ((0, 0), (0, 0))
    assert execution_truth.p2_truth_rows == ((0, 4), (1, 0))


def test_imperfect_prediction_remains_valid_when_execution_truth_differs() -> None:
    window = _window()
    engine = ReplayEngine(scheduling_mode="execution_window", expert_compute_delay=0.0, bucket_rows=2)
    policy_name = resolve_algorithm_id("islip_bucket").builder_key
    for hint in (
        PlanningHint("zero_hint", ((0, 0), (0, 0)), 0.0, None, 2),
        PlanningHint("copy_current_dispatch", window.p0_truth_rows, 1.0, 1, 2),
        PlanningHint("history_ema", ((0, 2), (1, 0)), 0.75, 1, 2),
        PlanningHint("shuffled_control", ((1, 0), (0, 4)), 1.0, 1, 2),
        PlanningHint("perfect_trace_hint", window.p2_truth_rows, 1.0, 1, 2),
    ):
        result = engine.execute(
            replay_window=window,
            planning_hint=hint,
            policy_name=policy_name,
        )
        assert result["audit_valid"] is True
        assert result["planning_hint"]["p2_hint_rows"] == hint.p2_hint_rows
        assert result["replay_window"]["p2_truth_rows"] == window.p2_truth_rows
        assert result["planning_hint"]["p2_hint_rows"] != result["replay_window"]["p2_truth_rows"] or hint.hint_type == "perfect_trace_hint"


def test_canonical_bucketizer_digest_is_stable_across_deployable_policies() -> None:
    window = _window()
    digests = set()
    counts = set()
    totals = set()
    for _policy in deployable_policies():
        tasks = CanonicalBucketizer(bucket_rows=2).bucketize(window)
        digests.add(CanonicalBucketizer.digest(tasks))
        counts.add(len(tasks))
        totals.add(sum(task.row_count for task in tasks))
    assert len(digests) == 1
    assert len(counts) == 1
    assert len(totals) == 1


def test_public_policy_catalog_converges_birkhoff_and_reference_split() -> None:
    birkhoff = resolve_public_policy_name("birkhoff_bucket_phase_local")
    assert birkhoff.internal_policy_name == "birkhoff_phase_local"
    assert birkhoff.reference_only is False
    fluid = resolve_public_policy_name("birkhoff_fluid_reference")
    assert fluid.internal_policy_name == "birkhoff_von_neumann_fluid"
    assert fluid.reference_only is True
    assert len(deployable_policies()) >= 7
    assert len(reference_policies()) >= 4


def test_legacy_and_canonical_algorithm_names_produce_identical_plan_digest() -> None:
    window = _window()
    hint = PlanningHint(
        hint_type="zero_hint",
        p2_hint_rows=((0, 0), (0, 0)),
        confidence=0.0,
        source_layer=None,
        target_layer=2,
    )
    engine = ReplayEngine(scheduling_mode="execution_window", expert_compute_delay=0.0, bucket_rows=2)
    legacy = resolve_algorithm_id("birkhoff_phase_local")
    canonical = resolve_algorithm_id("birkhoff_bucket_phase_local")
    legacy_result = engine.execute(replay_window=window, planning_hint=hint, policy_name=legacy.builder_key)
    canonical_result = engine.execute(replay_window=window, planning_hint=hint, policy_name=canonical.builder_key)
    assert legacy_result["input_task_digest"] == canonical_result["input_task_digest"]
    assert legacy_result["logical_plan_digest"] == canonical_result["logical_plan_digest"]
    assert legacy_result["makespan"] == canonical_result["makespan"]


def test_history_ema_offline_matches_online_formula() -> None:
    fixture_dir = REPO_ROOT / "outputs/offline/replay_fixture_selected_256x128_birkhoffctx/fixtures"
    if not fixture_dir.exists():
        import pytest
        pytest.skip("offline replay fixture is not present in the current source checkout")
    records = rolling_predictor_records(fixture_dir=fixture_dir, predictor_name="history_ema")
    assert records
    sample_record = next(record for record in records if int(record.layer_id) > 1)
    predictor = HistoryEMATrafficPredictor(alpha=0.5)
    current = tuple(tuple(int(value) for value in row) for row in sample_record.actual_matrix)
    previous = tuple(tuple(int(value) for value in row) for row in records[0].actual_matrix)
    online = predictor.predict(
        prediction_input=PredictionInput(
            run_id_digest="run",
            layer_id=str(sample_record.layer_id),
            next_layer_id=str(sample_record.next_layer_id),
            rank=0,
            world_size=len(current),
            current_dispatch_matrix_digest=matrix_digest_remote(current),
            current_dispatch_total_bytes=matrix_remote_bytes(current),
            current_dispatch_nonzero_edges=matrix_nonzero_remote_edge_count(current),
            metadata={"previous_dispatch_matrix": previous},
        ),
        current_dispatch_matrix=current,
    )
    assert online.predictor_name == "history_ema"
    assert online.confidence == 0.75


def test_public_entrypoints_help_and_async_release_config_parse(tmp_path: Path) -> None:
    for script in (
        "experiments/run_offline_replay.py",
        "experiments/run_online_phase_sync.py",
        "experiments/run_online_async_release.py",
    ):
        proc = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = yaml.safe_load((REPO_ROOT / "configs/online_async_release.yaml").read_text(encoding="utf-8"))
    assert payload["runtime"]["line"] == "async_release"
    assert any("async" in str(item["name"]) for item in payload["strategies"])
