from __future__ import annotations

from dataclasses import dataclass

from rs.core.contracts import ExpertRouteContext, ExpertRoutePrediction, PredictionHint, PredictionResult

from ..api import Predictor
from ..route_to_traffic import RouteToTrafficMapper


@dataclass
class MockGateReplayExpertRoutePredictor(Predictor):
    mapper: RouteToTrafficMapper = RouteToTrafficMapper()

    @property
    def predictor_id(self) -> str:
        return "mock_gate_replay"

    def predict(self, context: ExpertRouteContext) -> PredictionResult:
        context.validate()
        hidden = context.hidden_features
        expert_ids = getattr(hidden, "expert_ids", None)
        route_weights = getattr(hidden, "route_weights", None)
        if isinstance(hidden, dict):
            expert_ids = hidden.get("expert_ids", expert_ids)
            route_weights = hidden.get("route_weights", route_weights)
        if expert_ids is None:
            raise ValueError("mock_gate_replay requires hidden_features.expert_ids or hidden_features['expert_ids']")
        expert_route = ExpertRoutePrediction(
            expert_ids=tuple(tuple(int(value) for value in row) for row in expert_ids),
            route_weights=None if route_weights is None else tuple(tuple(float(value) for value in row) for row in route_weights),
        )
        target_rows = self.mapper.map(
            expert_route,
            source_rank=0,
            expert_owner_by_id=context.expert_owner_by_id,
            world_size=context.world_size,
        )
        hint = PredictionHint(
            predictor_id=self.predictor_id,
            hint_type="expert_route",
            target_dispatch_rows=target_rows,
            confidence=1.0,
            oracle=False,
            source_layer_id=context.identity.source_layer_id,
            target_layer_id=context.identity.target_layer_id,
        )
        return PredictionResult(identity=context.identity, hint=hint, expert_route=expert_route)


__all__ = ["MockGateReplayExpertRoutePredictor"]
