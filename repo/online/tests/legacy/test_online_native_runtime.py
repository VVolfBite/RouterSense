from __future__ import annotations

import pytest
import torch

from rs.online.olmoe_ep import (
    build_input_partition,
    build_route_partition_for_layer,
    execute_world_size_one_local_layer,
    feature_probe_online_olmoe_runtime,
)
from rs.online.scheduler_bridge.plan_protocol import (
    assert_plan_hash_agreement,
    build_plan_protocol_stub,
    compute_plan_hash,
)
from rs.runtime.distributed_ep.adapter.expert_store import LocalExpertWeights


def _fake_local_weights() -> LocalExpertWeights:
    gate_up_proj = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            [
                [0.5, 0.0],
                [0.0, 0.5],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    down_proj = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    return LocalExpertWeights(
        local_expert_ids=[0, 1],
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
        hidden_dim=2,
        intermediate_dim=2,
        activation_name="silu",
    )


def test_route_partition_preserves_local_and_remote_routes() -> None:
    hidden_states = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    router_logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 4.0, 2.0]], dtype=torch.float32)
    partition = build_route_partition_for_layer(
        run_id="run-0",
        request_id="req-0",
        microbatch_id="mb-0",
        layer_id=5,
        source_rank=1,
        hidden_states=hidden_states,
        router_logits=router_logits,
        owner_by_expert={0: 1, 1: 0, 2: 1},
        top_k=2,
    )
    assert len(partition.layer_trace.route_records) == 4
    assert len(partition.local_route_items) == 2
    assert len(partition.remote_route_records) == 2
    assert {record.identity.source_rank for record in partition.layer_trace.route_records} == {1}
    assert {record.identity.token_index_local for record in partition.remote_route_records} == {0, 1}


def test_world_size_one_execution_combines_all_topk_contributions() -> None:
    hidden_states = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    router_logits = torch.tensor([[3.0, 2.0], [1.0, 4.0]], dtype=torch.float32)
    partition = build_input_partition(
        run_id="run-1",
        request_id="req-1",
        microbatch_id="mb-1",
        source_rank=0,
        prompt_text="hello",
        token_count=2,
    )
    result = execute_world_size_one_local_layer(
        hidden_states=hidden_states,
        router_logits=router_logits,
        local_expert_weights=_fake_local_weights(),
        layer_id=0,
        partition=partition,
        top_k=2,
    )
    assert result.output.shape == hidden_states.shape
    assert len(result.route_partition.layer_trace.route_records) == 4
    assert not result.route_partition.remote_route_records
    # Both tokens should receive non-zero aggregated output from two top-k routes.
    assert torch.count_nonzero(result.output[0]).item() > 0
    assert torch.count_nonzero(result.output[1]).item() > 0


def test_plan_hash_mismatch_fails() -> None:
    payload_a = build_plan_protocol_stub(
        plan_id="plan-a",
        plan_hash="placeholder",
        layer_id=0,
        microbatch_id="mb-0",
        information_mode="none",
    )
    payload_b = build_plan_protocol_stub(
        plan_id="plan-b",
        plan_hash="placeholder",
        layer_id=1,
        microbatch_id="mb-0",
        information_mode="none",
    )
    hash_a = compute_plan_hash(payload_a)
    hash_b = compute_plan_hash(payload_b)
    with pytest.raises(RuntimeError, match="plan hash mismatch"):
        assert_plan_hash_agreement([hash_a, hash_b])


def test_feature_probe_detects_required_olmoe_shapes() -> None:
    class _Experts:
        gate_up_proj = torch.zeros((2, 4, 2))
        down_proj = torch.zeros((2, 2, 2))

    class _MLP:
        gate = object()
        experts = _Experts()

    class _Layer:
        mlp = _MLP()

    class _Config:
        num_experts = 2
        num_experts_per_tok = 2
        hidden_size = 2
        intermediate_size = 2

    class _InnerModel:
        layers = [_Layer()]

    class _Model:
        config = _Config()
        model = _InnerModel()

    report = feature_probe_online_olmoe_runtime(_Model())
    assert report["supported"] is True
