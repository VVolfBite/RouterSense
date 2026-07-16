from __future__ import annotations

from experiments.paper.prediction_evaluation import _shuffle_rows, _zero_matrix_like


def test_zero_and_shuffled_controls_are_non_future_paths() -> None:
    matrix = ((0, 1), (2, 0))
    assert _zero_matrix_like(matrix) == ((0, 0), (0, 0))
    assert _shuffle_rows(matrix) == ((2, 0), (0, 1))
