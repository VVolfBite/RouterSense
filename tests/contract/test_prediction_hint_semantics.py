from __future__ import annotations

import pytest

from rs.core.contracts import PredictionHint


def test_prediction_hint_semantic_payload_changes_with_confidence() -> None:
    low = PredictionHint(
        predictor_id="copy_current_dispatch",
        hint_type="copy_current_dispatch",
        target_dispatch_rows=((0, 2), (1, 0)),
        confidence=0.1,
        oracle=False,
    )
    high = PredictionHint(
        predictor_id="copy_current_dispatch",
        hint_type="copy_current_dispatch",
        target_dispatch_rows=((0, 2), (1, 0)),
        confidence=0.9,
        oracle=False,
    )
    assert low.semantic_payload() != high.semantic_payload()


def test_zero_hint_requires_zero_matrix() -> None:
    with pytest.raises(ValueError, match="zero_hint matrix must be all zero"):
        PredictionHint(
            predictor_id="zero_hint",
            hint_type="zero_hint",
            target_dispatch_rows=((0, 1), (0, 0)),
            confidence=0.0,
            oracle=False,
        ).validate(world_size=2)


def test_oracle_only_allowed_for_perfect_trace() -> None:
    with pytest.raises(ValueError, match="oracle=True"):
        PredictionHint(
            predictor_id="copy_current_dispatch",
            hint_type="copy_current_dispatch",
            target_dispatch_rows=((0, 1), (1, 0)),
            confidence=1.0,
            oracle=True,
        ).validate(world_size=2)
