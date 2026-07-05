from __future__ import annotations

from rs.scheduling.validation import build_remote_flows

from .helpers import make_observation


def test_single_node_two_rank_not_cross_node() -> None:
    observations = (
        make_observation(rank=0, phase="P0", rows=(0, 5), hostname_digest="same-host", node_index=0),
        make_observation(rank=1, phase="P0", rows=(4, 0), hostname_digest="same-host", node_index=0),
    )
    flows = build_remote_flows(observations)
    assert all(flow.is_cross_rank for flow in flows)
    assert all(not flow.is_cross_node for flow in flows)
