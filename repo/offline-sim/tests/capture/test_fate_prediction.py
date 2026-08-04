from __future__ import annotations

import pytest

from rs_sim.trace.collection.fate_online import (
    SampledGateInput,
    deterministic_even_indices,
    predict_rank_row,
)
from rs_sim.trace.collection.fate_reference import predict_fate_routing_rows


def test_deterministic_even_indices_are_bounded_and_stable():
    assert deterministic_even_indices(8, 4) == (1, 3, 5, 7)
    assert deterministic_even_indices(4, 8) == (0, 1, 2, 3)


def test_faithful_percentile_fate_preserves_topk_assignment_mass():
    np = pytest.importorskip("numpy")
    logits = np.asarray(
        [
            [8.0, 7.0, 0.0, -1.0],
            [7.0, 6.0, 1.0, 0.0],
            [0.0, 1.0, 9.0, 8.0],
            [0.0, 1.0, 8.0, 7.0],
        ],
        dtype=np.float64,
    )
    result = predict_fate_routing_rows(
        logits,
        np.asarray([0, 0, 1, 1]),
        expert_to_rank=np.asarray([0, 0, 1, 1]),
        world_size=2,
        top_k=2,
        percentile=75.0,
    )
    assert result.routing_rows == ((4, 0), (0, 4))
    assert sum(sum(row) for row in result.routing_rows) == 8
    assert result.predictor_id == "fate_cross_layer_gate_percentile"


def test_sampled_online_fate_uses_next_router_and_preserves_mass():
    torch = pytest.importorskip("torch")

    class Config:
        moe_router_topk = 2

    class Router:
        config = Config()
        weight = torch.tensor(
            [[4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 3.0]],
            dtype=torch.float32,
        )
        bias = None

    class Target:
        router = Router()
        config = Config()

    sample = SampledGateInput(
        layer_id=0,
        decode_step=0,
        original_token_count=4,
        sampled_hidden_cpu=torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
        ),
        sample_indices=(0, 1, 2, 3),
    )
    row, evidence = predict_rank_row(
        sample,
        target_module=Target(),
        expert_to_rank=(0, 0, 1, 1),
        world_size=2,
    )
    assert row == (4, 4)
    assert sum(row) == sample.original_token_count * 2
    assert evidence["predictor_id"] == "fate_cross_layer_gate_sampled_v1"
