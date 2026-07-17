from __future__ import annotations

import pytest

from rs.core.contracts import ExpertScoreDistribution
from rs.prediction import RouteToTrafficMapper


def test_score_distribution_preserves_fixed_topk_row_totals() -> None:
    distribution = ExpertScoreDistribution(
        scores_by_source_rank=(
            (
                (8.0, 7.0, 1.0, 0.0),
                (0.0, 1.0, 7.0, 8.0),
            ),
            (
                (2.0, 2.0, 2.0, 2.0),
            ),
        ),
        top_k=2,
        score_domain="logits",
    )
    mapper = RouteToTrafficMapper()
    matrix = mapper.map_score_distribution(
        distribution,
        expert_owner_by_id=(0, 0, 1, 1),
        world_size=2,
    )
    assert sum(matrix[0]) == 4
    assert sum(matrix[1]) == 2
    assert matrix[0][0] > 0
    assert matrix[0][1] > 0


def test_score_distribution_envelope_is_model_agnostic_and_bounded() -> None:
    distribution = ExpertScoreDistribution(
        scores_by_source_rank=(
            ((0.7, 0.2, 0.1),),
            ((0.1, 0.2, 0.7),),
        ),
        top_k=2,
        score_domain="probabilities",
    )
    envelope = RouteToTrafficMapper().map_score_distribution_to_envelope(
        distribution,
        predictor_id="generic-router-score",
        expert_owner_by_id=(0, 1, 1),
        world_size=2,
        relative_error_bound=(0.1, 0.2),
        calibration_id="validation-v1",
    )
    envelope.validate(world_size=2)
    assert envelope.metadata["adapter"] == "fixed_topk_capped_inclusion_v1"
    assert envelope.metadata["routed_expert_count"] == 3
    assert sum(envelope.mean_rows[0]) == 2
    assert sum(envelope.mean_rows[1]) == 2
    for src in range(2):
        for dst in range(2):
            assert envelope.lower_rows[src][dst] <= envelope.mean_rows[src][dst]
            assert envelope.mean_rows[src][dst] <= envelope.upper_rows[src][dst]


def test_score_distribution_rejects_shared_or_missing_expert_width() -> None:
    distribution = ExpertScoreDistribution(
        scores_by_source_rank=(((0.5, 0.5),),),
        top_k=1,
        score_domain="probabilities",
    )
    with pytest.raises(ValueError, match="score width|world_size"):
        RouteToTrafficMapper().map_score_distribution(
            distribution,
            expert_owner_by_id=(0, 0, 0),
            world_size=1,
        )
