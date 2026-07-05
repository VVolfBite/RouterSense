from __future__ import annotations

from integrations.megatron_ep.routersense.policy.validation import build_phase_demands, build_remote_flows

from .helpers import make_observation


def test_p0_ready_p1_blocked_release_semantics() -> None:
    observations = (
        make_observation(rank=0, phase="P0", rows=(0, 5)),
        make_observation(rank=1, phase="P0", rows=(4, 0)),
        make_observation(rank=0, phase="P1", rows=(0, 3)),
        make_observation(rank=1, phase="P1", rows=(2, 0)),
    )
    demands = {d.phase: d for d in build_phase_demands(build_remote_flows(observations))}
    assert demands["P0"].release_state == "ready"
    assert demands["P0"].release_dependency == "none"
    assert demands["P0"].payload_exists is True
    assert demands["P1"].release_state == "blocked"
    assert demands["P1"].release_dependency == "remote_expert_compute_complete"
    assert demands["P1"].payload_exists is False
