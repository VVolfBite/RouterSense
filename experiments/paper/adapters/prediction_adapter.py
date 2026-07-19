from __future__ import annotations

from typing import Any

from rs.core.contracts import PredictionIdentity, TrafficHistoryContext
from rs.prediction import PredictionEvaluator, PredictionRegistry, PredictionTruth
from rs.scheduling.traffic_matrix import matrix_digest_remote


def create_predictor(predictor_id: str, *, usage: str = "offline"):
    return PredictionRegistry.create(str(predictor_id), usage=usage)


def evaluate_traffic_prediction(
    *,
    predictor_id: str,
    current_dispatch_rows: tuple[tuple[int, ...], ...],
    current_return_rows: tuple[tuple[int, ...], ...],
    target_next_dispatch_rows: tuple[tuple[int, ...], ...],
    history_dispatch_rows: tuple[tuple[tuple[int, ...], ...], ...] = (),
    source_layer_id: str = "0",
    target_layer_id: str = "1",
) -> dict[str, Any]:
    predictor = create_predictor(predictor_id, usage="offline")
    context = TrafficHistoryContext(
        identity=PredictionIdentity(
            request_id=matrix_digest_remote(current_dispatch_rows),
            source_layer_id=str(source_layer_id),
            target_layer_id=str(target_layer_id),
        ),
        current_dispatch_rows=current_dispatch_rows,
        current_return_rows=current_return_rows,
        history_dispatch_rows=history_dispatch_rows,
        world_size=len(current_dispatch_rows),
    )
    prediction = predictor.predict(context)
    evaluation = PredictionEvaluator().evaluate(
        prediction,
        PredictionTruth(actual_dispatch_rows=target_next_dispatch_rows),
    )
    return {
        "predictor_id": predictor_id,
        "input_digest": matrix_digest_remote(current_dispatch_rows),
        "prediction_digest": matrix_digest_remote(prediction.hint.target_dispatch_rows),
        "truth_digest": matrix_digest_remote(target_next_dispatch_rows),
        "prediction_rows": prediction.hint.target_dispatch_rows,
        "metrics": dict(evaluation.metrics),
        "valid": bool(evaluation.valid),
        "reason": evaluation.reason,
    }
