from rs.core.contracts import RankPressureForecast, StableEdgePrecedence, TrafficForecastEnvelope
from rs.prediction.traffic_envelope import build_traffic_forecast_envelope, envelope_to_prediction_hint


def test_forecast_envelope_builds_rank_pressure_and_partial_order():
    envelope = build_traffic_forecast_envelope(
        predictor_id="test",
        mean_rows=((0, 10, 2), (1, 0, 1), (0, 3, 0)),
        lower_rows=((0, 9, 1), (0, 0, 0), (0, 2, 0)),
        upper_rows=((0, 11, 3), (2, 0, 2), (1, 4, 0)),
        precedence_margin=1.0,
        precedence_confidence_floor=0.0,
    )
    envelope.validate(world_size=3)
    assert envelope.rank_pressure.mean == (12.0, 2.0, 3.0)
    assert any(
        item.before_src == 0 and item.before_dst == 1
        for item in envelope.stable_precedence
    )
    hint = envelope_to_prediction_hint(envelope)
    assert hint.target_dispatch_rows == envelope.mean_rows
    assert hint.oracle is False


def test_contracts_reject_invalid_bounds():
    pressure = RankPressureForecast(mean=(1.0,), lower=(0.0,), upper=(2.0,), confidence=(0.5,))
    envelope = TrafficForecastEnvelope(
        predictor_id="x",
        mean_rows=((1,),),
        lower_rows=((2,),),
        upper_rows=((3,),),
        rank_pressure=pressure,
    )
    try:
        envelope.validate(world_size=1)
    except ValueError as exc:
        assert "lower <= mean <= upper" in str(exc)
    else:
        raise AssertionError("invalid envelope accepted")


def test_forecast_metrics_separate_remote_and_rank_pressure_error() -> None:
    from rs.prediction import evaluate_traffic_forecast

    metrics = evaluate_traffic_forecast(
        ((5, 3), (1, 4)),
        ((4, 2), (2, 3)),
    )
    assert metrics["all_relative_l1"] == 4 / 11
    assert metrics["remote_relative_l1"] == 2 / 4
    assert metrics["rank_pressure_relative_l1"] == 2 / 4
    assert -1.0 <= metrics["remote_total_bias"] <= 1.0
