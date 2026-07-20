from __future__ import annotations

from rs.scheduling.validation import build_remote_flows

from .helpers import make_observation


def test_p1_demand_known_but_payload_not_ready() -> None:
    flows = build_remote_flows((make_observation(rank=0, phase="P1", rows=(0, 7)),))
    assert len(flows) == 1
    assert flows[0].demand_known_at == "router_ready"
    assert flows[0].release_state == "blocked"
    assert flows[0].payload_exists is False
