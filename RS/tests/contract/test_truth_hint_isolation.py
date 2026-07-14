from __future__ import annotations

import pytest

from rs.core.contracts import (
    PlanEvaluation,
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
    TrafficProvenance,
)
from rs.planning import CommonCorePlanEstimator, PlannerRegistry, PlanningCostModel
from rs.runtime.offline.replay_unified import (
    PlanningHint as ReplayPlanningHint,
    ReplayEngine,
    ReplayWindow,
    build_multiphase_problem,
    build_planning_problem,
    build_execution_truth,
    execution_truth_digest,
)


def _request(*, hint_rows: tuple[tuple[int, ...], ...], request_id: str = "req") -> PlanningRequest:
    return PlanningRequest(
        identity=PlanningIdentity(request_id=request_id, run_id="run-x", forward_id="fwd-x", window_id="win-x", source_layer_id="1", target_layer_id="2"),
        traffic=PlanningTraffic(
            p0_dispatch_rows=((0, 2), (3, 0)),
            p1_return_rows=((0, 3), (2, 0)),
        ),
        prediction_hint=PredictionHint(
            predictor_id="copy_current",
            hint_type="traffic_matrix",
            target_dispatch_rows=hint_rows,
            confidence=1.0,
            source_layer_id="1",
            target_layer_id="2",
        ),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=2, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )


def _window(*, p2_truth_rows: tuple[tuple[int, ...], ...]) -> ReplayWindow:
    return ReplayWindow(
        fixture_id="fixture",
        window_id="1->2",
        layer_id=1,
        p0_truth_rows=((0, 2), (3, 0)),
        p1_truth_rows=((0, 3), (2, 0)),
        p2_truth_rows=p2_truth_rows,
        matrix_unit="rows",
        group_size=2,
        payload_row_bytes_by_phase={"P0": 1, "P1": 1, "P2": 1},
        metadata={},
        traffic_provenance=TrafficProvenance.ROUTE_RECONSTRUCTED,
        return_model="transpose_dispatch",
        raw_token_count=5,
        used_token_count=5,
    )


def test_truth_change_keeps_planner_semantics_but_changes_execution_truth_digest() -> None:
    hint_rows = ((0, 1), (4, 0))
    request = _request(hint_rows=hint_rows)
    planner = PlannerRegistry.create("fifo_bucket", None)
    estimator = CommonCorePlanEstimator()
    plan = planner.plan(request)
    score = estimator.estimate(plan, request, PlanningCostModel())
    engine = ReplayEngine(scheduling_mode="execution_window", expert_compute_delay=0.0, bucket_rows=2)
    hint = ReplayPlanningHint("copy_current_dispatch", hint_rows, 1.0, 1, 2)
    result_a = engine.execute(replay_window=_window(p2_truth_rows=((0, 7), (1, 0))), planning_hint=hint, policy_name="fifo_bucket")
    result_b = engine.execute(replay_window=_window(p2_truth_rows=((0, 5), (3, 0))), planning_hint=hint, policy_name="fifo_bucket")
    assert request.semantic_digest() == _request(hint_rows=hint_rows, request_id="other").semantic_digest()
    assert result_a["planning_task_digest"] == result_b["planning_task_digest"]
    assert result_a["logical_plan_digest"] == result_b["logical_plan_digest"]
    assert result_a["execution_truth_digest"] != result_b["execution_truth_digest"]
    assert result_a["planning_request_digest"] == result_b["planning_request_digest"]
    assert result_a["logical_plan_digest"] == result_b["logical_plan_digest"]
    assert score.to_dict() == estimator.estimate(planner.plan(request), request, PlanningCostModel()).to_dict()


def test_hint_change_changes_planning_semantics_even_if_local_plan_matches() -> None:
    request_a = _request(hint_rows=((0, 1), (4, 0)))
    request_b = _request(hint_rows=((0, 4), (1, 0)))
    engine = ReplayEngine(scheduling_mode="execution_window", expert_compute_delay=0.0, bucket_rows=2)
    truth_window = _window(p2_truth_rows=((0, 9), (2, 0)))
    result_a = engine.execute(replay_window=truth_window, planning_hint=ReplayPlanningHint("history_ema", ((0, 1), (4, 0)), 0.75, 1, 2), policy_name="fifo_bucket")
    result_b = engine.execute(replay_window=truth_window, planning_hint=ReplayPlanningHint("history_ema", ((0, 4), (1, 0)), 0.75, 1, 2), policy_name="fifo_bucket")
    assert request_a.semantic_digest() != request_b.semantic_digest()
    assert result_a["planning_task_digest"] != result_b["planning_task_digest"]


def test_perfect_trace_hint_is_oracle_but_remains_separate_from_truth_contract() -> None:
    truth_window = _window(p2_truth_rows=((0, 5), (2, 0)))
    truth = build_execution_truth(truth_window)
    perfect_hint = ReplayPlanningHint("perfect_trace_hint", truth_window.p2_truth_rows, 1.0, 1, 2)
    request = PlanningRequest(
        identity=PlanningIdentity(request_id="oracle"),
        traffic=PlanningTraffic(p0_dispatch_rows=truth_window.p0_truth_rows, p1_return_rows=truth_window.p1_truth_rows),
        prediction_hint=PredictionHint(
            predictor_id="perfect_trace_hint",
            hint_type="traffic_matrix",
            target_dispatch_rows=perfect_hint.p2_hint_rows,
            confidence=1.0,
            oracle=True,
        ),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=2, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    assert request.prediction_hint.target_dispatch_rows == truth.p2_truth_rows
    assert request.prediction_hint.oracle is True
    assert request.prediction_hint is not truth


def test_four_hints_share_one_truth_digest_but_change_request_digest() -> None:
    truth_window = _window(p2_truth_rows=((0, 6), (2, 0)))
    truth = build_execution_truth(truth_window)
    hints = (
        PredictionHint("zero", "traffic_matrix", ((0, 0), (0, 0)), 0.0, oracle=False),
        PredictionHint("copy_current", "traffic_matrix", truth_window.p0_truth_rows, 1.0, oracle=False),
        PredictionHint("history", "traffic_matrix", ((0, 1), (4, 0)), 0.75, oracle=False),
        PredictionHint("perfect_trace_hint", "traffic_matrix", truth_window.p2_truth_rows, 1.0, oracle=True),
    )
    digests = {
        PlanningRequest(
            identity=PlanningIdentity(request_id=f"hint-{idx}"),
            traffic=PlanningTraffic(p0_dispatch_rows=truth_window.p0_truth_rows, p1_return_rows=truth_window.p1_truth_rows),
            prediction_hint=hint,
            topology=PlanningTopology(world_size=2),
            constraints=PlanningConstraints(bucket_rows=2, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
            weights=PlanningWeights(),
            information_mode="p0_p1_p2",
        ).semantic_digest()
        for idx, hint in enumerate(hints)
    }
    assert len(digests) == 4
    assert execution_truth_digest(truth) == execution_truth_digest(build_execution_truth(truth_window))


def test_replay_planning_metadata_does_not_expose_execution_truth_digest() -> None:
    truth_window = _window(p2_truth_rows=((0, 6), (2, 0)))
    hint = ReplayPlanningHint("history_ema", ((0, 1), (4, 0)), 0.75, 1, 2)
    planning_problem = build_planning_problem(replay_window=truth_window, planning_hint=hint)
    problem = build_multiphase_problem(
        planning_problem=planning_problem,
        execution_truth=build_execution_truth(truth_window),
        scheduling_mode="execution_window",
        expert_compute_delay=0.0,
    )
    metadata = problem.forecast.metadata
    assert "execution_truth_digest" not in metadata


def test_invalid_replay_fails_closed_with_no_formal_makespan(monkeypatch: pytest.MonkeyPatch) -> None:
    from rs.runtime.offline import replay_unified

    truth_window = _window(p2_truth_rows=((0, 6), (2, 0)))
    engine = ReplayEngine(scheduling_mode="execution_window", expert_compute_delay=0.0, bucket_rows=2)
    hint = ReplayPlanningHint("history_ema", ((0, 1), (4, 0)), 0.75, 1, 2)

    class _InvalidEvaluator:
        def evaluate(self, *args, **kwargs) -> PlanEvaluation:
            return PlanEvaluation(
                valid=False,
                reason="forced_invalid",
                realized_makespan=None,
                coverage_valid=False,
                port_valid=False,
                metrics={},
            )

    monkeypatch.setattr(replay_unified, "OfflineEvaluator", lambda: _InvalidEvaluator())
    result = engine.execute(replay_window=truth_window, planning_hint=hint, policy_name="fifo_bucket")
    assert result["audit_valid"] is False
    assert result["makespan"] is None
    assert result["legacy_diagnostic_makespan"] is not None
