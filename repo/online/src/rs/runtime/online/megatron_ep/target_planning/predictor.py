from __future__ import annotations

import time
from dataclasses import dataclass

from rs.core.contracts import PredictionIdentity, TrafficHistoryContext
from rs.core.hashing import stable_hash_dict
from rs.prediction import PredictionRegistry, resolve_predictor_id

from .contracts import TwoHorizonPrediction


Matrix = tuple[tuple[int, ...], ...]


def _return_from_dispatch(matrix: Matrix) -> Matrix:
    return tuple(
        tuple(int(matrix[col][row]) for col in range(len(matrix)))
        for row in range(len(matrix))
    )


@dataclass(frozen=True)
class TwoHorizonPredictionBundle:
    h1: TwoHorizonPrediction
    h2: TwoHorizonPrediction

    def to_dict(self) -> dict[str, object]:
        return {"h1": self.h1.to_dict(), "h2": self.h2.to_dict()}


class SharedTwoHorizonPredictor:
    def __init__(self, *, predictor_name: str, history_ema_alpha: float = 0.5) -> None:
        self.predictor_name = str(resolve_predictor_id(predictor_name))
        self.history_ema_alpha = float(history_ema_alpha)
        self._predictor = PredictionRegistry.create(
            self.predictor_name,
            {"alpha": self.history_ema_alpha},
            usage="runtime",
        )

    def _predict_once(
        self,
        *,
        horizon: int,
        source_layer_id: str,
        target_layer_id: str,
        current_dispatch_matrix: Matrix,
        history_matrices: tuple[Matrix, ...] = (),
    ) -> TwoHorizonPrediction:
        created_at_ns = time.perf_counter_ns()
        context = TrafficHistoryContext(
            identity=PredictionIdentity(
                request_id=f"two_horizon:{source_layer_id}:{target_layer_id}:{horizon}",
                run_id="two_horizon",
                source_layer_id=str(source_layer_id),
                target_layer_id=str(target_layer_id),
            ),
            current_dispatch_rows=current_dispatch_matrix,
            current_return_rows=_return_from_dispatch(current_dispatch_matrix),
            history_dispatch_rows=history_matrices,
            world_size=len(current_dispatch_matrix),
        )
        started = time.perf_counter_ns()
        prediction = self._predictor.predict(context)
        ended = time.perf_counter_ns()
        return TwoHorizonPrediction(
            forecast_horizon=int(horizon),
            source_layer_id=str(source_layer_id),
            target_layer_id=str(target_layer_id),
            matrix_unit="rows",
            matrix_rows=prediction.hint.target_dispatch_rows,
            matrix_digest=stable_hash_dict(
                {
                    "matrix_unit": "rows",
                    "matrix_rows": [list(row) for row in prediction.hint.target_dispatch_rows],
                }
            ),
            predictor=str(prediction.hint.predictor_id),
            confidence=float(prediction.hint.confidence or 0.0),
            created_at_ns=int(created_at_ns),
            prediction_us=(ended - started) / 1000.0,
            terminal=False,
        )

    def predict_two_horizon(
        self,
        *,
        source_layer_id: str,
        current_dispatch_matrix: Matrix,
        previous_dispatch_matrix: Matrix | None = None,
        history_matrices: tuple[Matrix, ...] = (),
    ) -> TwoHorizonPredictionBundle:
        current = tuple(tuple(int(v) for v in row) for row in current_dispatch_matrix)
        first_history = history_matrices or (() if previous_dispatch_matrix is None else (previous_dispatch_matrix,))
        h1 = self._predict_once(
            horizon=1,
            source_layer_id=source_layer_id,
            target_layer_id=str(int(source_layer_id) + 1) if str(source_layer_id).isdigit() else f"{source_layer_id}+1",
            current_dispatch_matrix=current,
            history_matrices=first_history,
        )
        h2 = self._predict_once(
            horizon=2,
            source_layer_id=h1.target_layer_id,
            target_layer_id=str(int(source_layer_id) + 2) if str(source_layer_id).isdigit() else f"{source_layer_id}+2",
            current_dispatch_matrix=h1.matrix_rows,
            history_matrices=first_history + (current,),
        )
        return TwoHorizonPredictionBundle(h1=h1, h2=h2)
