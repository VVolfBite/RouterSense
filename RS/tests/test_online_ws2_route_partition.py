from __future__ import annotations

import torch

from rs.online.olmoe_ep import build_online_expert_placement, build_online_route_partition


def test_online_ws2_route_partition_preserves_local_and_remote_sets() -> None:
    placement = build_online_expert_placement(world_size=2, expert_count=4, rank_to_node_id=[0, 0])
    hidden_states = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    router_logits = torch.tensor([[8.0, 7.0, 1.0, 0.0], [0.0, 1.0, 6.0, 5.0]], dtype=torch.float32)
    partition = build_online_route_partition(
        run_id="run-0",
        request_id="req-0",
        microbatch_id="mb-0",
        layer_id=0,
        source_rank=0,
        source_node_id=0,
        hidden_states=hidden_states,
        router_logits=router_logits,
        placement=placement,
        top_k=2,
    )
    assert len(partition.all_routes) == 4
    assert len(partition.local_routes) == 2
    assert len(partition.remote_send_routes) == 2
    assert sum(partition.per_peer_send_rows.values()) == 2
    assert partition.per_peer_send_rows[0] == 0
    assert partition.per_peer_send_rows[1] == 2
    assert partition.per_expert_local_bucket_rows == {0: 1, 2: 1}
