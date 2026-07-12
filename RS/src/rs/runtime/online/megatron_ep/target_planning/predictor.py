from __future__ import annotations

import time
from dataclasses import dataclass

from rs.runtime.offline.prediction.linear_predictor import FATEStyleLinearTrafficPredictor
from rs.runtime.online.megatron_ep.prediction.contracts import Matrix, PredictionInput
from rs.runtime.online.megatron_ep.prediction.simple_predictors import (
    CopyCurrentDispatchPredictor,
    HistoryEMATrafficPredictor,
    ZeroHintPredictor,
)
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix

from .contracts import TwoHorizonPrediction


def _matrix_rows(value: Matrix) -> tuple[tuple[int, ...], ...]:
    return canonicalize_remote_matrix(value)


@dataclass(frozen=True)
class TwoHorizonPredictionBundle:
    h1: TwoHorizonPrediction
    h2: TwoHorizonPrediction

    def to_dict(self) -> dict[str, object]:
        return {"h1": self.h1.to_dict(), "h2": self.h2.to_dict()}


class SharedTwoHorizonPredictor:
    def __init__(self, *, predictor_name: str, history_ema_alpha: float = 0.5) -> None:
        self.predictor_name = str(predictor_name)
        self.history_ema_alpha = float(history_ema_alpha)
        if self.predictor_name in {"none", "zero_hint"}:
            self._predictor = ZeroHintPredictor()
        elif self.predictor_name == "copy_current_dispatch":
            self._predictor = CopyCurrentDispatchPredictor()
        elif self.predictor_name == "history_ema":
            self._predictor = HistoryEMATrafficPredictor(alpha=self.history_ema_alpha)
        elif self.predictor_name == "history_linear_trend":
            self._predictor = FATEStyleLinearTrafficPredictor()
        else:
            raise ValueError(f"unsupported two-horizon predictor {self.predictor_name!r}")

    def _predict_once(
        self,
        *,
        horizon: int,
        source_layer_id: str,
        target_layer_id: str,
        current_dispatch_matrix: Matrix,
        previous_dispatch_matrix: Matrix | None,
        history_matrices: tuple[Matrix, ...] = (),
    ) -> TwoHorizonPrediction:
        created_at_ns = time.perf_counter_ns()
        prediction_input = PredictionInput(
            run_id_digest="two_horizon",
            layer_id=str(source_layer_id),
            next_layer_id=str(target_layer_id),
            rank=0,
            world_size=len(current_dispatch_matrix),
            current_dispatch_matrix_digest="",
            current_dispatch_total_bytes=0,
            current_dispatch_nonzero_edges=0,
            metadata={
                "previous_dispatch_matrix": previous_dispatch_matrix,
                "history_matrices": history_matrices,
            },
        )
        started = time.perf_counter_ns()
        if self.predictor_name == "history_linear_trend":
            history = [tuple(tuple(int(v) for v in row) for row in matrix) for matrix in history_matrices if matrix is not None]
            if len(history) < 1:
                matrix = _matrix_rows(current_dispatch_matrix)
            else:
                from rs.runtime.offline.prediction.contracts import PredictorSample

                def _sample(prev_dispatch: Matrix, current_dispatch: Matrix, target_dispatch: Matrix, *, layer: int) -> PredictorSample:
                    return PredictorSample(
                        fixture_id="online-history-linear",
                        sample_id=f"layer-{layer}",
                        layer_id=layer,
                        previous_dispatch_matrix=_matrix_rows(prev_dispatch),
                        current_dispatch_matrix=_matrix_rows(current_dispatch),
                        current_return_matrix=tuple(tuple(int(current_dispatch[col][row]) for col in range(len(current_dispatch))) for row in range(len(current_dispatch))),
                        target_next_dispatch_matrix=_matrix_rows(target_dispatch),
                    )

                fit_samples = []
                previous = history[0]
                for idx, current in enumerate(history[1:], start=1):
                    fit_samples.append(_sample(previous, current, current, layer=idx))
                    previous = current
                if not fit_samples:
                    matrix = _matrix_rows(current_dispatch_matrix)
                else:
                    model = FATEStyleLinearTrafficPredictor().fit(fit_samples)
                    infer_sample = _sample(
                        previous_dispatch_matrix if previous_dispatch_matrix is not None else current_dispatch_matrix,
                        current_dispatch_matrix,
                        current_dispatch_matrix,
                        layer=len(history),
                    )
                    matrix = _matrix_rows(model.predict_matrix(infer_sample))
            confidence = 0.6
            digest = str(hash(tuple(tuple(int(v) for v in row) for row in matrix)))
        else:
            predicted = self._predictor.predict(
                prediction_input=prediction_input,
                current_dispatch_matrix=current_dispatch_matrix,
            )
            matrix = _matrix_rows(predicted.matrix)
            confidence = float(predicted.confidence)
            digest = str(predicted.matrix_digest)
        ended = time.perf_counter_ns()
        return TwoHorizonPrediction(
            forecast_horizon=int(horizon),
            source_layer_id=str(source_layer_id),
            target_layer_id=str(target_layer_id),
            matrix_unit="rows",
            matrix_rows=matrix,
            matrix_digest=str(digest),
            predictor=str(self.predictor_name),
            confidence=float(confidence),
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
        current = _matrix_rows(current_dispatch_matrix)
        h1 = self._predict_once(
            horizon=1,
            source_layer_id=source_layer_id,
            target_layer_id=str(int(source_layer_id) + 1) if str(source_layer_id).isdigit() else f"{source_layer_id}+1",
            current_dispatch_matrix=current,
            previous_dispatch_matrix=previous_dispatch_matrix,
            history_matrices=history_matrices,
        )
        h1_as_current = h1.matrix_rows
        h2 = self._predict_once(
            horizon=2,
            source_layer_id=source_layer_id,
            target_layer_id=str(int(source_layer_id) + 2) if str(source_layer_id).isdigit() else f"{source_layer_id}+2",
            current_dispatch_matrix=h1_as_current,
            previous_dispatch_matrix=current,
            history_matrices=history_matrices + (current,),
        )
        return TwoHorizonPredictionBundle(h1=h1, h2=h2)
