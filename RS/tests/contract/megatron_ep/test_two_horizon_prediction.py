from __future__ import annotations

from rs.runtime.online.megatron_ep.target_planning import SharedTwoHorizonPredictor


def test_copy_current_two_horizon_prediction_contract() -> None:
    predictor = SharedTwoHorizonPredictor(predictor_name="copy_current_dispatch")
    bundle = predictor.predict_two_horizon(
        source_layer_id="0",
        current_dispatch_matrix=((0, 2), (3, 0)),
        previous_dispatch_matrix=((0, 1), (1, 0)),
    )
    assert bundle.h1.forecast_horizon == 1
    assert bundle.h2.forecast_horizon == 2
    assert bundle.h1.matrix_unit == "rows"
    assert bundle.h1.matrix_rows == ((0, 2), (3, 0))


def test_history_ema_two_horizon_prediction_contract() -> None:
    predictor = SharedTwoHorizonPredictor(predictor_name="history_ema")
    bundle = predictor.predict_two_horizon(
        source_layer_id="3",
        current_dispatch_matrix=((0, 4), (2, 0)),
        previous_dispatch_matrix=((0, 0), (6, 0)),
    )
    assert bundle.h1.target_layer_id == "4"
    assert bundle.h2.target_layer_id == "5"
    assert len(bundle.h1.matrix_rows) == 2

