from __future__ import annotations

import torch

from rs.online.olmoe_ep import build_online_expert_placement, build_online_route_partition


def test_online_cross_rank_is_distinct_from_cross_node() -> None:
    hidden_states = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    router_logits = torch.tensor([[1.0, 5.0]], dtype=torch.float32)

    same_node_placement = build_online_expert_placement(world_size=2, expert_count=2, rank_to_node_id=[0, 0])
    same_node_partition = build_online_route_partition(
        run_id="run-0",
        request_id="req-0",
        microbatch_id="mb-0",
        request_numeric_id=0,
        microbatch_numeric_id=0,
        layer_id=0,
        source_rank=0,
        source_node_id=0,
        hidden_states=hidden_states,
        router_logits=router_logits,
        placement=same_node_placement,
        top_k=1,
    )
    same_node_route = same_node_partition.remote_send_routes[0]
    assert same_node_route.is_cross_rank is True
    assert same_node_route.is_cross_node is False

    cross_node_placement = build_online_expert_placement(world_size=2, expert_count=2, rank_to_node_id=[0, 1])
    cross_node_partition = build_online_route_partition(
        run_id="run-0",
        request_id="req-0",
        microbatch_id="mb-0",
        request_numeric_id=0,
        microbatch_numeric_id=0,
        layer_id=0,
        source_rank=0,
        source_node_id=0,
        hidden_states=hidden_states,
        router_logits=router_logits,
        placement=cross_node_placement,
        top_k=1,
    )
    cross_node_route = cross_node_partition.remote_send_routes[0]
    assert cross_node_route.is_cross_rank is True
    assert cross_node_route.is_cross_node is True
