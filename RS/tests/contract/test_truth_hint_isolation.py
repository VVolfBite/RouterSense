from __future__ import annotations

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
)
from rs.planning import CommonCorePlanEstimator, PlannerRegistry, PlanningCostModel


def _request(*, hint_rows: tuple[tuple[int, ...], ...]) -> PlanningRequest:
    return PlanningRequest(
        identity=PlanningIdentity(request_id="req", source_layer_id="1", target_layer_id="2"),
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
        constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )


def test_truth_changes_do_not_change_request_or_plan_when_hint_stays_fixed() -> None:
    request = _request(hint_rows=((0, 1), (4, 0)))
    planner = PlannerRegistry.create("fifo_bucket", None)
    plan_a = planner.plan(request)
    plan_b = planner.plan(request)
    estimator = CommonCorePlanEstimator()
    score_a = estimator.estimate(plan_a, request, PlanningCostModel())
    score_b = estimator.estimate(plan_b, request, PlanningCostModel())
    assert request.semantic_digest() == request.semantic_digest()
    assert plan_a.semantic_digest() == plan_b.semantic_digest()
    assert score_a.to_dict() == score_b.to_dict()


def test_hint_change_changes_request_digest() -> None:
    request_a = _request(hint_rows=((0, 1), (4, 0)))
    request_b = _request(hint_rows=((0, 4), (1, 0)))
    assert request_a.semantic_digest() != request_b.semantic_digest()


def test_perfect_trace_hint_is_separate_contract_from_truth() -> None:
    truth_rows = ((0, 5), (2, 0))
    request = _request(hint_rows=truth_rows)
    assert request.prediction_hint.target_dispatch_rows == truth_rows
    assert request.prediction_hint.oracle is False
