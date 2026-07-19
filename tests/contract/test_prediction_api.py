from __future__ import annotations

import pytest

from rs.core.contracts import ExpertRouteContext, PredictionIdentity, RankedExpertRoutes, TrafficHistoryContext
from rs.prediction import (
    LinearTrafficPredictor,
    PredictionEvaluator,
    PredictionRegistry,
    PredictionTruth,
    RouteToTrafficMapper,
    TrafficPredictionTrainingSample,
)
from rs.runtime.offline.prediction.contracts import PredictorSample
from rs.runtime.offline.prediction.linear_predictor import FATEStyleLinearTrafficPredictor


def test_traffic_predictor_registry_uses_formal_api() -> None:
    predictor = PredictionRegistry.create("copy_current", usage="test")
    context = TrafficHistoryContext(
        identity=PredictionIdentity(request_id="req", source_layer_id="1", target_layer_id="2"),
        current_dispatch_rows=((0, 3), (2, 0)),
        current_return_rows=((0, 2), (3, 0)),
        history_dispatch_rows=(),
        world_size=2,
    )
    result = predictor.predict(context)
    assert result.hint.predictor_id == "copy_current"
    assert result.hint.target_dispatch_rows == ((0, 3), (2, 0))


def test_expert_route_predictor_uses_route_to_traffic_mapper() -> None:
    predictor = PredictionRegistry.create("mock_gate_replay", usage="test")
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
    specs = {spec.predictor_id: spec for spec in PredictionRegistry.specs()}
    assert specs["mock_gate_replay"].test_only is True


def test_test_only_predictor_is_rejected_outside_test_usage() -> None:
    with pytest.raises(ValueError, match="test_only"):
        PredictionRegistry.create("mock_gate_replay", usage="runtime")
    with pytest.raises(ValueError, match="test_only"):
        PredictionRegistry.create("mock_gate_replay", usage="offline")


def test_prediction_evaluator_separates_traffic_and_expert_route_metrics() -> None:
    traffic_prediction = PredictionRegistry.create("zero", usage="test").predict(
        TrafficHistoryContext(
            identity=PredictionIdentity(request_id="traffic", source_layer_id="1", target_layer_id="2"),
            current_dispatch_rows=((0, 2), (1, 0)),
            current_return_rows=((0, 1), (2, 0)),
            history_dispatch_rows=(),
            world_size=2,
        )
    )
    evaluator = PredictionEvaluator()
    traffic_eval = evaluator.evaluate(
        traffic_prediction,
        PredictionTruth(actual_dispatch_rows=((0, 1), (2, 0))),
    )
    assert traffic_eval.valid is True
    assert "relative_l1" in traffic_eval.metrics
    expert_prediction = PredictionRegistry.create("mock_gate_replay", usage="test").predict(
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
            expert_owner_by_id=(0, 1),
        ),
    )
    assert expert_eval.valid is True
    assert "topk_overlap" in expert_eval.metrics
    assert "exact_expert_accuracy" in expert_eval.metrics


def test_prediction_evaluator_marks_shape_mismatch_invalid() -> None:
    evaluator = PredictionEvaluator()
    prediction = PredictionRegistry.create("copy_current", usage="test").predict(
        TrafficHistoryContext(
            identity=PredictionIdentity(request_id="shape", source_layer_id="1", target_layer_id="2"),
            current_dispatch_rows=((0, 2), (1, 0)),
            current_return_rows=((0, 1), (2, 0)),
            history_dispatch_rows=(),
            world_size=2,
        )
    )
    evaluation = evaluator.evaluate(
        prediction,
        PredictionTruth(actual_dispatch_rows=((0, 2, 0), (1, 0, 0), (0, 0, 0))),
    )
    assert evaluation.valid is False
    assert evaluation.reason == "traffic_shape_mismatch"


def test_prediction_evaluator_marks_incomplete_expert_route_invalid() -> None:
    evaluator = PredictionEvaluator()
    prediction = PredictionRegistry.create("mock_gate_replay", usage="test").predict(
        ExpertRouteContext(
            identity=PredictionIdentity(request_id="expert-shape", source_layer_id="1", target_layer_id="2"),
            hidden_features={"expert_ids": ((0, 1),)},
            gate_features=None,
            top_k=2,
            expert_owner_by_id=(0, 1),
            world_size=2,
        )
    )
    evaluation = evaluator.evaluate(
        prediction,
        PredictionTruth(
            actual_dispatch_rows=((0, 1), (0, 0)),
            actual_expert_route=type(prediction.expert_route)(expert_ids=((0,),)),
            expert_owner_by_id=(0, 1),
        ),
    )
    assert evaluation.valid is False
    assert evaluation.reason == "expert_route_topk_width_mismatch"


def test_route_to_traffic_mapper_supports_ranked_routes_and_skips_diagonal() -> None:
    ranked = RankedExpertRoutes(
        routes_by_source_rank=(
            expert_route := PredictionRegistry.create("mock_gate_replay", usage="test").predict(
                ExpertRouteContext(
                    identity=PredictionIdentity(request_id="src0"),
                    hidden_features={"expert_ids": ((0, 1),)},
                    gate_features=None,
                    top_k=2,
                    expert_owner_by_id=(0, 1),
                    world_size=2,
                )
            ).expert_route,
            expert_route,
        )
    )
    matrix = RouteToTrafficMapper().map_ranked(ranked, expert_owner_by_id=(0, 1), world_size=2)
    assert matrix[0][0] == 0
    assert matrix[1][1] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"source_rank": -1, "expert_owner_by_id": (0, 1), "world_size": 2}, "source_rank"),
        ({"source_rank": 2, "expert_owner_by_id": (0, 1), "world_size": 2}, "source_rank"),
        ({"source_rank": 0, "expert_owner_by_id": (-1, 1), "world_size": 2}, "owner_rank"),
        ({"source_rank": 0, "expert_owner_by_id": (0, 2), "world_size": 2}, "owner_rank"),
    ),
)
def test_route_to_traffic_mapper_rejects_invalid_ranks(kwargs: dict[str, object], message: str) -> None:
    route = PredictionRegistry.create("mock_gate_replay", usage="test").predict(
        ExpertRouteContext(
            identity=PredictionIdentity(request_id="invalid-rank"),
            hidden_features={"expert_ids": ((0, 1),)},
            gate_features=None,
            top_k=2,
            expert_owner_by_id=(0, 1),
            world_size=2,
        )
    ).expert_route
    with pytest.raises(ValueError, match=message):
        RouteToTrafficMapper().map(route, **kwargs)


def test_route_to_traffic_mapper_rejects_world_size_mismatch_and_invalid_expert() -> None:
    mapper = RouteToTrafficMapper()
    ranked = RankedExpertRoutes(routes_by_source_rank=())
    with pytest.raises(ValueError, match="world_size"):
        mapper.map_ranked(ranked, expert_owner_by_id=(0, 1), world_size=2)
    bad_route = type(
        PredictionRegistry.create("mock_gate_replay", usage="test").predict(
            ExpertRouteContext(
                identity=PredictionIdentity(request_id="bad-expert"),
                hidden_features={"expert_ids": ((0, 1),)},
                gate_features=None,
                top_k=2,
                expert_owner_by_id=(0, 1),
                world_size=2,
            )
        ).expert_route
    )(expert_ids=((99, 1),))
    with pytest.raises(ValueError, match="expert ID|expert_id"):
        mapper.map(bad_route, source_rank=0, expert_owner_by_id=(0, 1), world_size=2)


def test_linear_predictor_uses_next_traffic_as_training_target() -> None:
    predictor = LinearTrafficPredictor()
    predictor.fit(
        (
            TrafficPredictionTrainingSample(
                current_dispatch_rows=((0, 1), (2, 0)),
                current_return_rows=((0, 2), (1, 0)),
                history_dispatch_rows=(((0, 0), (1, 0)),),
                target_next_dispatch_rows=((0, 5), (7, 0)),
                layer_id="1",
            ),
            TrafficPredictionTrainingSample(
                current_dispatch_rows=((0, 2), (3, 0)),
                current_return_rows=((0, 3), (2, 0)),
                history_dispatch_rows=(((0, 1), (2, 0)),),
                target_next_dispatch_rows=((0, 6), (8, 0)),
                layer_id="2",
            ),
        )
    )
    result = predictor.predict(
        TrafficHistoryContext(
            identity=PredictionIdentity(request_id="linear", source_layer_id="3", target_layer_id="4"),
            current_dispatch_rows=((0, 3), (4, 0)),
            current_return_rows=((0, 4), (3, 0)),
            history_dispatch_rows=(((0, 2), (3, 0)),),
            world_size=2,
        )
    )
    assert result.hint.target_dispatch_rows != ((0, 3), (4, 0))


def test_linear_predictor_matches_legacy_fate_style_features() -> None:
    samples = [
        TrafficPredictionTrainingSample(
            current_dispatch_rows=((0, 8), (4, 0)),
            current_return_rows=((0, 4), (8, 0)),
            history_dispatch_rows=(((0, 0), (0, 0)),),
            target_next_dispatch_rows=((0, 10), (5, 0)),
            layer_id="0",
            next_layer_id="1",
        ),
        TrafficPredictionTrainingSample(
            current_dispatch_rows=((0, 10), (5, 0)),
            current_return_rows=((0, 5), (10, 0)),
            history_dispatch_rows=(((0, 8), (4, 0)),),
            target_next_dispatch_rows=((0, 12), (6, 0)),
            layer_id="1",
            next_layer_id="2",
        ),
    ]
    formal = LinearTrafficPredictor().fit(samples)
    legacy = FATEStyleLinearTrafficPredictor().fit(
        [
            PredictorSample(
                layer_id=str(sample.layer_id),
                next_layer_id=str(sample.next_layer_id),
                current_dispatch_matrix=sample.current_dispatch_rows,
                current_return_matrix=sample.current_return_rows,
                previous_dispatch_matrix=sample.history_dispatch_rows[-1],
                target_next_dispatch_matrix=sample.target_next_dispatch_rows,
            )
            for sample in samples
        ]
    )
    context = TrafficHistoryContext(
        identity=PredictionIdentity(request_id="parity", source_layer_id="2", target_layer_id="3"),
        current_dispatch_rows=((0, 12), (6, 0)),
        current_return_rows=((0, 6), (12, 0)),
        history_dispatch_rows=(((0, 10), (5, 0)),),
        world_size=2,
    )
    formal_prediction = formal.predict(context).hint.target_dispatch_rows
    legacy_prediction = legacy.predict_matrix(
        PredictorSample(
            layer_id="2",
            next_layer_id="3",
            current_dispatch_matrix=context.current_dispatch_rows,
            current_return_matrix=context.current_return_rows,
            previous_dispatch_matrix=context.history_dispatch_rows[-1],
            target_next_dispatch_matrix=((0, 14), (7, 0)),
        )
    )
    assert formal_prediction == legacy_prediction
