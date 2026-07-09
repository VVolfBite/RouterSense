from __future__ import annotations

from rs.runtime.online.megatron_ep.prediction import (
    CopyCurrentDispatchPredictor,
    PredictionInput,
    ZeroHintPredictor,
    compare_predicted_to_actual,
)


def _prediction_input() -> PredictionInput:
    return PredictionInput(
        run_id_digest="run",
        layer_id="0",
        next_layer_id="1",
        rank=0,
        world_size=2,
        current_dispatch_matrix_digest="digest",
        current_dispatch_total_bytes=12,
        current_dispatch_nonzero_edges=2,
    )


def test_zero_hint_predictor_outputs_zero_matrix() -> None:
    predicted = ZeroHintPredictor().predict(
        prediction_input=_prediction_input(),
        current_dispatch_matrix=((0, 8), (4, 0)),
    )
    assert predicted.predictor_name == "zero_hint"
    assert predicted.matrix == ((0, 0), (0, 0))
    assert predicted.total_bytes == 0


def test_copy_current_dispatch_predictor_outputs_copy() -> None:
    predicted = CopyCurrentDispatchPredictor().predict(
        prediction_input=_prediction_input(),
        current_dispatch_matrix=((0, 8), (4, 0)),
    )
    assert predicted.predictor_name == "copy_current_dispatch"
    assert predicted.matrix == ((0, 8), (4, 0))
    assert predicted.total_bytes == 12
    assert predicted.matrix_digest == CopyCurrentDispatchPredictor().predict(
        prediction_input=_prediction_input(),
        current_dispatch_matrix=((0, 8), (4, 0)),
    ).matrix_digest


def test_prediction_audit_metrics_are_computed() -> None:
    predicted = CopyCurrentDispatchPredictor().predict(
        prediction_input=_prediction_input(),
        current_dispatch_matrix=((0, 8), (4, 0)),
    )
    audit = compare_predicted_to_actual(predicted, ((0, 10), (2, 0)), topk=2)
    assert audit.valid is True
    assert audit.absolute_l1_error == 4.0
    assert audit.relative_l1_error > 0.0
    assert 0.0 <= audit.cosine_similarity <= 1.0
    assert 0.0 <= audit.topk_edge_overlap <= 1.0
    assert 0.0 <= audit.nonzero_edge_precision <= 1.0
    assert 0.0 <= audit.nonzero_edge_recall <= 1.0

