from __future__ import annotations

from rs.runtime.online.megatron_ep.prediction.expert_evaluation import evaluate_expert_prediction
from rs.runtime.online.megatron_ep.prediction.expert_trace import SourceExpertCountMatrix


def test_expert_prediction_metrics_split_expert_and_traffic_error() -> None:
    predicted = SourceExpertCountMatrix(
        layer_id=1,
        world_size=2,
        num_experts=4,
        counts=((3, 1, 0, 0), (0, 0, 2, 1)),
    )
    actual = SourceExpertCountMatrix(
        layer_id=1,
        world_size=2,
        num_experts=4,
        counts=((2, 2, 0, 0), (0, 0, 3, 0)),
    )
    metrics = evaluate_expert_prediction(
        predicted,
        actual,
        expert_to_rank={0: 0, 1: 1, 2: 0, 3: 1},
        bytes_per_token=10,
    )
    assert metrics.expert_count_relative_l1_error > 0.0
    assert 0.0 <= metrics.expert_count_cosine_similarity <= 1.0
    assert 0.0 <= metrics.traffic_cosine_similarity <= 1.0
    assert isinstance(metrics.bottleneck_expert_match, bool)
    assert isinstance(metrics.bottleneck_rank_match, bool)
