from __future__ import annotations

from rs.core.contracts import ExpertRouteContext, PredictionIdentity, TrafficHistoryContext
from rs.prediction import PredictionEvaluator, PredictionRegistry, PredictionTruth, RouteToTrafficMapper


def test_traffic_predictor_registry_uses_formal_api() -> None:
    predictor = PredictionRegistry.create("copy_current")
    context = TrafficHistoryContext(
        identity=PredictionIdentity(request_id="req", source_layer_id="1", target_layer_id="2"),
        current_dispatch_rows=((0, 3), (2, 0)),
        history_dispatch_rows=(),
        world_size=2,
    )
    result = predictor.predict(context)
    assert result.hint.predictor_id == "copy_current"
    assert result.hint.target_dispatch_rows == ((0, 3), (2, 0))


def test_expert_route_predictor_uses_route_to_traffic_mapper() -> None:
    predictor = PredictionRegistry.create("mock_gate_replay")
    context = ExpertRouteContext(
        identity=PredictionIdentity(request_id="req", source_layer_id="3", target_layer_id="4"),
        hidden_features={"expert_ids": ((1, 2), (2, 3))},
        gate_features=None,
        top_k=2,
        expert_owner_by_id=(0, 0, 1, 1),
        world_size=2,
    )
    result = predictor.predict(context)
    expected = RouteToTrafficMapper().map(result.expert_route, source_rank=0, expert_owner_by_id=(0, 0, 1, 1), world_size=2)
    assert result.hint.target_dispatch_rows == expected


def test_prediction_evaluator_separates_traffic_and_expert_route_metrics() -> None:
    traffic_prediction = PredictionRegistry.create("zero").predict(
        TrafficHistoryContext(
            identity=PredictionIdentity(request_id="traffic", source_layer_id="1", target_layer_id="2"),
            current_dispatch_rows=((0, 2), (1, 0)),
            history_dispatch_rows=(),
            world_size=2,
        )
    )
    evaluator = PredictionEvaluator()
    traffic_eval = evaluator.evaluate(
        traffic_prediction,
        PredictionTruth(actual_dispatch_rows=((0, 1), (2, 0))),
    )
    assert "relative_l1" in traffic_eval.metrics
    expert_prediction = PredictionRegistry.create("mock_gate_replay").predict(
        ExpertRouteContext(
            identity=PredictionIdentity(request_id="expert", source_layer_id="1", target_layer_id="2"),
            hidden_features={"expert_ids": ((0, 1),)},
            gate_features=None,
            top_k=2,
            expert_owner_by_id=(0, 1),
            world_size=2,
        )
    )
    expert_eval = evaluator.evaluate(
        expert_prediction,
        PredictionTruth(
            actual_dispatch_rows=((0, 1), (0, 0)),
            actual_expert_route=expert_prediction.expert_route,
        ),
    )
    assert "topk_overlap" in expert_eval.metrics
