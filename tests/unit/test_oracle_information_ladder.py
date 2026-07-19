from rs.runtime.offline.oracle_information_ladder import (
    ExactReduction,
    evaluate_exact_information_ladder,
)


def test_exact_information_ladder_true_prediction_matches_perfect():
    p0 = ((0, 2, 0), (0, 0, 1), (1, 0, 0))
    p1 = ((0, 0, 1), (2, 0, 0), (0, 1, 0))
    p2 = ((0, 0, 2), (1, 0, 0), (0, 1, 0))
    reduction = ExactReduction(
        source_instance_id="tiny",
        selected_original_ranks=(0, 1, 2),
        p0=p0,
        p1=p1,
        p2=p2,
    )
    ladder = evaluate_exact_information_ladder(reduction, p2_forecast=p2, time_limit_ms=5000)
    assert ladder.reactive.valid
    assert ladder.predicted is not None and ladder.predicted.valid
    assert ladder.perfect.valid
    assert ladder.reactive.objective >= ladder.perfect.objective
    assert ladder.predicted.objective == ladder.perfect.objective
    assert ladder.predicted.audit["predicted_bytes_executed"] is False
    assert ladder.metrics["prediction_regret_to_perfect"] == 0.0
