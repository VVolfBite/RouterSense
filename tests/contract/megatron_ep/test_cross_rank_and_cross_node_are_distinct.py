from __future__ import annotations

from rs.scheduling.validation import build_remote_flows

from .helpers import make_observation


def test_cross_rank_and_cross_node_are_distinct() -> None:
    observations = (
        make_observation(rank=0, phase="P0", rows=(0, 5), hostname_digest="host-a", node_index=0),
        make_observation(rank=1, phase="P0", rows=(4, 0), hostname_digest="host-b", node_index=1),
    )
    flows = build_remote_flows(observations)
    assert len(flows) == 2
    assert all(flow.is_cross_rank for flow in flows)
    assert all(flow.is_cross_node for flow in flows)
