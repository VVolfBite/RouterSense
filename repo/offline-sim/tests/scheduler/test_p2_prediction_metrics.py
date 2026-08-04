from rs_sim.scheduler.prediction.metrics import evaluate_p2_prediction


def test_perfect_prediction_has_zero_error_and_full_scores():
    matrix = ((0, 4), (2, 1))
    result = evaluate_p2_prediction(predicted_matrix=matrix, actual_matrix=matrix)
    assert result.absolute_error_bytes == 0
    assert result.relative_absolute_error_ppm == 0
    assert result.matrix_overlap_ppm == 1_000_000
    assert result.exact_edge_accuracy_ppm == 1_000_000
    assert result.top_destination_accuracy_ppm == 1_000_000


def test_zero_prediction_reports_causal_degradation_without_float():
    result = evaluate_p2_prediction(
        predicted_matrix=((0, 0), (0, 0)),
        actual_matrix=((0, 8), (4, 0)),
    )
    assert result.absolute_error_bytes == 12
    assert result.relative_absolute_error_ppm == 1_000_000
    assert result.matrix_overlap_ppm == 0
    assert result.nonzero_edge_recall_ppm == 0
    assert isinstance(result.quality_digest, str)


def test_top_destination_tie_break_is_lowest_rank():
    result = evaluate_p2_prediction(
        predicted_matrix=((5, 5), (1, 3)),
        actual_matrix=((5, 5), (0, 9)),
    )
    assert result.top_destination_accuracy_ppm == 1_000_000
