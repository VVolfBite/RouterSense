from __future__ import annotations

import pytest

from rs.core.contracts import PredictionIdentity, TrafficHistoryContext
from rs.prediction import PredictionRegistry, TrafficPredictionTrainingSample


def test_training_sample_validate_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="world_size"):
        TrafficPredictionTrainingSample(
            current_dispatch_rows=((0, 1), (1, 0)),
            current_return_rows=((0, 1, 0), (1, 0, 0)),
            history_dispatch_rows=(),
            target_next_dispatch_rows=((0, 1), (1, 0)),
        ).validate()


def test_history_predictor_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be within"):
        PredictionRegistry.create("history", {"alpha": 2.0}, usage="offline")


def test_linear_predictor_rejects_negative_ridge_lambda() -> None:
    with pytest.raises(ValueError, match="ridge_lambda"):
        PredictionRegistry.create("linear", {"ridge_lambda": -1.0}, usage="offline")


def test_linear_predictor_rejects_predict_before_fit() -> None:
    predictor = PredictionRegistry.create("linear", {"ridge_lambda": 0.0}, usage="offline")
    with pytest.raises(ValueError, match="fit before predict"):
        predictor.predict(  # type: ignore[attr-defined]
            TrafficHistoryContext(
                identity=PredictionIdentity(request_id="req"),
                current_dispatch_rows=((0, 1), (1, 0)),
                current_return_rows=((0, 1), (1, 0)),
                history_dispatch_rows=(),
                world_size=2,
            )
        )
