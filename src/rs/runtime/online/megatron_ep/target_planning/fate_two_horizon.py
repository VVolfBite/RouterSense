from __future__ import annotations

"""Stateless runtime wrapper that makes faithful FATE usable by the existing
``TargetLayerPlannerService`` two-horizon injection point.

H1 is produced from an ``ExpertRouteContext`` supplied by the model hook.  H2
is produced by an ordinary formal traffic predictor from H1.  The wrapper owns
no queue, store, token, deadline, or thread; those remain the responsibility of
the existing target-planning service and ``TargetPlanStore``.
"""

from dataclasses import dataclass
import time
from typing import Callable, Protocol

from rs.core.contracts import (
    ExpertRouteContext,
    PredictionIdentity,
    PredictionResult,
    TrafficHistoryContext,
)
from rs.scheduling.validation import stable_hash
from rs.prediction.fate_future import FateFormalPredictor
from .contracts import MatrixRows, TwoHorizonPrediction


class ExpertRouteContextProvider(Protocol):
    def __call__(self, *, source_layer_id: str, target_layer_id: str) -> ExpertRouteContext: ...


class TrafficPredictorLike(Protocol):
    @property
    def predictor_id(self) -> str: ...
    def predict(self, context: TrafficHistoryContext) -> PredictionResult: ...


def _transpose(matrix: MatrixRows) -> MatrixRows:
    if not matrix:
        return ()
    return tuple(
        tuple(int(matrix[col][row]) for col in range(len(matrix)))
        for row in range(len(matrix))
    )


def _next_layer_id(source_layer_id: str, offset: int) -> str:
    value = str(source_layer_id)
    return str(int(value) + int(offset)) if value.isdigit() else f"{value}+{int(offset)}"


def _prediction_record(
    *,
    horizon: int,
    source_layer_id: str,
    target_layer_id: str,
    prediction: PredictionResult,
    created_at_ns: int,
    elapsed_us: float,
) -> TwoHorizonPrediction:
    rows = tuple(tuple(int(value) for value in row) for row in prediction.hint.target_dispatch_rows)
    return TwoHorizonPrediction(
        forecast_horizon=int(horizon),
        source_layer_id=str(source_layer_id),
        target_layer_id=str(target_layer_id),
        matrix_unit="rows",
        matrix_rows=rows,
        matrix_digest=str(stable_hash({"matrix_unit": "rows", "matrix_rows": [list(row) for row in rows]})),
        predictor=str(prediction.hint.predictor_id),
        confidence=float(prediction.hint.confidence or 0.0),
        created_at_ns=int(created_at_ns),
        prediction_us=float(elapsed_us),
        terminal=False,
    )


@dataclass
class FateTwoHorizonRuntimePredictor:
    """Duck-typed replacement for ``SharedTwoHorizonPredictor``.

    The runtime constructs this object through the already-supported
    ``two_horizon_predictor_factory`` hook.  It does not alter the planner
    service lifecycle or publication ownership.
    """

    context_provider: ExpertRouteContextProvider | None
    second_hop_predictor: TrafficPredictorLike
    fixed_context: ExpertRouteContext | None = None
    fate_predictor: FateFormalPredictor = FateFormalPredictor()

    def predict_two_horizon(
        self,
        *,
        source_layer_id: str,
        current_dispatch_matrix: MatrixRows,
        previous_dispatch_matrix: MatrixRows | None = None,
        history_matrices: tuple[MatrixRows, ...] = (),
    ):
        source = str(source_layer_id)
        target_h1 = _next_layer_id(source, 1)
        target_h2 = _next_layer_id(source, 2)

        h1_created = time.perf_counter_ns()
        if self.fixed_context is not None:
            expert_context = self.fixed_context
            if str(expert_context.identity.source_layer_id) != source:
                raise ValueError("fixed FATE context source layer does not match request")
            if str(expert_context.identity.target_layer_id) != target_h1:
                raise ValueError("fixed FATE context target layer does not match request")
        else:
            if self.context_provider is None:
                raise RuntimeError("FATE runtime predictor requires a context provider or fixed context")
            expert_context = self.context_provider(
                source_layer_id=source,
                target_layer_id=target_h1,
            )
        h1_started = time.perf_counter_ns()
        h1_result = self.fate_predictor.predict(expert_context)
        h1_ended = time.perf_counter_ns()
        h1 = _prediction_record(
            horizon=1,
            source_layer_id=source,
            target_layer_id=target_h1,
            prediction=h1_result,
            created_at_ns=h1_created,
            elapsed_us=(h1_ended - h1_started) / 1000.0,
        )

        first_history = history_matrices or (
            () if previous_dispatch_matrix is None else (previous_dispatch_matrix,)
        )
        h2_context = TrafficHistoryContext(
            identity=PredictionIdentity(
                request_id=f"fate_two_horizon:{source}:{target_h2}:2",
                run_id="fate_two_horizon",
                source_layer_id=target_h1,
                target_layer_id=target_h2,
            ),
            current_dispatch_rows=h1.matrix_rows,
            current_return_rows=_transpose(h1.matrix_rows),
            history_dispatch_rows=first_history
            + (tuple(tuple(int(v) for v in row) for row in current_dispatch_matrix),),
            world_size=len(h1.matrix_rows),
        )
        h2_created = time.perf_counter_ns()
        h2_started = time.perf_counter_ns()
        h2_result = self.second_hop_predictor.predict(h2_context)
        h2_ended = time.perf_counter_ns()
        h2 = _prediction_record(
            horizon=2,
            source_layer_id=target_h1,
            target_layer_id=target_h2,
            prediction=h2_result,
            created_at_ns=h2_created,
            elapsed_us=(h2_ended - h2_started) / 1000.0,
        )

        # Importing the existing bundle type lazily avoids creating a parallel
        # contract and remains compatible with the service's duck-typed factory.
        from .predictor import TwoHorizonPredictionBundle

        return TwoHorizonPredictionBundle(h1=h1, h2=h2)


def make_fate_two_horizon_factory(
    *,
    context_provider: ExpertRouteContextProvider,
    second_hop_predictor_id: str = "copy_current",
    second_hop_config: object | None = None,
) -> Callable[[str], FateTwoHorizonRuntimePredictor]:
    """Build the factory already accepted by ``TargetLayerPlannerService``."""

    def factory(_predictor_name: str) -> FateTwoHorizonRuntimePredictor:
        from rs.prediction.registry import PredictionRegistry

        second = PredictionRegistry.create(
            str(second_hop_predictor_id),
            second_hop_config,
            usage="runtime",
        )
        return FateTwoHorizonRuntimePredictor(
            context_provider=context_provider,
            second_hop_predictor=second,
        )

    return factory


__all__ = [
    "ExpertRouteContextProvider",
    "FateTwoHorizonRuntimePredictor",
    "make_fate_two_horizon_factory",
]
