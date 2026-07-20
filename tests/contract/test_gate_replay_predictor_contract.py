from __future__ import annotations

from rs.runtime.online.megatron_ep.prediction.expert_trace import SourceExpertCountMatrix
from rs.runtime.online.megatron_ep.prediction.gate_replay_predictor import MockGateReplayPredictor


def test_gate_replay_predictor_contract_is_expert_first() -> None:
    predictor = MockGateReplayPredictor()
    actual = SourceExpertCountMatrix(
        layer_id=2,
        world_size=2,
        num_experts=4,
        counts=((2, 1, 0, 0), (0, 0, 1, 2)),
    )
    result = predictor.predict_next_layer_expert_counts(
        current_layer_id=1,
        current_router_input={
            "world_size": 2,
            "num_experts": 4,
            "source_expert_counts": ((2, 1, 0, 0), (0, 0, 1, 2)),
        },
        next_layer_router={"kind": "mock"},
        expert_to_rank={0: 0, 1: 1, 2: 0, 3: 1},
        top_k=2,
        bytes_per_token=8,
        actual_next_expert_counts=actual,
    )
    assert result.predictor_name == "MockGateReplayPredictor"
    assert result.predictor_family == "fate_style_gate_replay"
    assert result.predictor_version == "mock_v1"
    assert result.faithful_fate_style is False
    assert result.requires_next_layer_router is True
    assert result.requires_router_input is True
    assert result.requires_gpu_trace is True
    assert result.gpu_collection_required is True
    assert result.expert_metrics is not None
    assert result.predicted_source_expert_counts.counts == actual.counts
