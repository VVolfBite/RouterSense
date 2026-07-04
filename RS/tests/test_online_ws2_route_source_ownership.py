from __future__ import annotations

import torch

from rs.online.olmoe_ep import build_online_expert_placement, build_online_route_partition


def test_online_ws2_route_source_ownership_comes_from_rank() -> None:
    placement = build_online_expert_placement(world_size=2, expert_count=4, rank_to_node_id=[0, 0])
    hidden_states = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    router_logits = torch.tensor([[4.0, 3.0, 1.0, 0.0], [0.0, 1.0, 5.0, 4.0]], dtype=torch.float32)
    partition = build_online_route_partition(
        run_id="run-0",
        request_id="req-1",
        microbatch_id="mb-0",
        request_numeric_id=1,
        microbatch_numeric_id=0,
        layer_id=0,
        source_rank=1,
        source_node_id=0,
        hidden_states=hidden_states,
        router_logits=router_logits,
        placement=placement,
        top_k=2,
    )
    assert {record.identity.source_rank for record in partition.all_routes} == {1}
    assert {record.identity.source_node_id for record in partition.all_routes} == {0}
