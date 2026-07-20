from __future__ import annotations

"""Canonical Predictor wrapper for the migrated faithful FATE implementation."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rs.core.contracts import (
    ExpertRouteContext,
    ExpertRoutePrediction,
    PredictionContext,
    PredictionHint,
    PredictionResult,
)
from rs.prediction.api import PredictorSpec
from rs.scheduling.p012_future._kernel.fate import (
    FateObservation,
    FatePredictorService,
)


@dataclass(frozen=True)
class FateFormalConfig:
    percentile: float = 75.0
    min_candidates: int | None = None
    confidence: float = 0.75
    gate_output_domain: str = "logits"


class FateFormalPredictor:
    """Formal online/offline FATE predictor.

    ``ExpertRouteContext.hidden_features`` carries the current gate input.
    ``gate_features`` must expose ``next_layer_gate`` and ``source_ranks``; it
    may be either a mapping or an object with those attributes. This keeps the
    model-specific hook in the runtime adapter while the predictor remains
    independent of Megatron/CUDA.
    """

    def __init__(self, config: FateFormalConfig | None = None) -> None:
        self.config = config or FateFormalConfig()
        from rs.scheduling.p012_future._kernel.fate import FateCrossLayerGatePredictor

        kernel = FateCrossLayerGatePredictor(
            percentile=float(self.config.percentile),
            min_candidates=self.config.min_candidates,
        )
        self._service = FatePredictorService(
            predictor=kernel,
            default_confidence=float(self.config.confidence),
        )

    @property
    def predictor_id(self) -> str:
        return "fate_cross_layer_gate"

    @staticmethod
    def _field(value: object, name: str, default: Any = None) -> Any:
        if isinstance(value, Mapping):
            return value.get(name, default)
        return getattr(value, name, default)

    def predict(self, context: PredictionContext) -> PredictionResult:
        if not isinstance(context, ExpertRouteContext):
            raise TypeError("faithful FATE requires ExpertRouteContext")
        context.validate()
        gate_features = context.gate_features
        if gate_features is None:
            raise ValueError("FATE gate_features must provide next_layer_gate and source_ranks")
        next_layer_gate = self._field(gate_features, "next_layer_gate")
        source_ranks = self._field(gate_features, "source_ranks")
        gate_output_domain = str(
            self._field(gate_features, "gate_output_domain", self.config.gate_output_domain)
        )
        if next_layer_gate is None or source_ranks is None:
            raise ValueError("FATE gate_features missing next_layer_gate/source_ranks")

        observation = FateObservation(
            current_gate_input=np.asarray(context.hidden_features),
            source_ranks=np.asarray(source_ranks),
            next_layer_gate=next_layer_gate,
            expert_to_rank=np.asarray(context.expert_owner_by_id),
            world_size=int(context.world_size),
            top_k=int(context.top_k),
            gate_output_domain=gate_output_domain,
            layer_id=None,
            request_id=str(context.identity.request_id),
        )
        bundle = self._service.predict(observation)
        expert = bundle.expert_prediction
        # Formal expert-route output records the top-k route, while the full
        # expected assignment mass remains in auxiliary for evaluation/audit.
        top_indices = np.argsort(-expert.expected_assignment_mass, axis=1, kind="stable")[:, : int(expert.top_k)]
        top_weights = np.take_along_axis(expert.expected_assignment_mass, top_indices, axis=1)
        expert_route = ExpertRoutePrediction(
            expert_ids=tuple(tuple(int(item) for item in row) for row in top_indices),
            route_weights=tuple(tuple(float(item) for item in row) for row in top_weights),
        )
        hint = PredictionHint(
            predictor_id=self.predictor_id,
            hint_type="expert_route",
            target_dispatch_rows=tuple(
                tuple(int(item) for item in row)
                for row in bundle.traffic_hint.target_dispatch_rows
            ),
            confidence=float(bundle.traffic_hint.confidence),
            oracle=False,
            source_layer_id=context.identity.source_layer_id,
            target_layer_id=context.identity.target_layer_id,
        )
        result = PredictionResult(
            identity=context.identity,
            hint=hint,
            expert_route=expert_route,
            auxiliary={
                "faithful_fate": True,
                "percentile": float(self.config.percentile),
                "min_candidates": self.config.min_candidates,
                "candidate_counts": tuple(int(value) for value in expert.prefetch_mask.sum(axis=1)),
                "expected_assignment_mass": tuple(
                    tuple(float(item) for item in row)
                    for row in expert.expected_assignment_mass
                ),
            },
        )
        result.validate(world_size=int(context.world_size))
        return result


FATE_PREDICTOR_SPEC = PredictorSpec(
    predictor_id="fate_cross_layer_gate",
    category="expert_route",
    deployable=True,
    offline_only=False,
    test_only=False,
    historical_aliases=("fate", "faithful_fate", "fate_cross_layer_gate_v1"),
)


def create_fate_predictor(config: Mapping[str, object] | None = None) -> FateFormalPredictor:
    values = dict(config or {})
    return FateFormalPredictor(FateFormalConfig(**values))


__all__ = [
    "FATE_PREDICTOR_SPEC",
    "FateFormalConfig",
    "FateFormalPredictor",
    "create_fate_predictor",
]
