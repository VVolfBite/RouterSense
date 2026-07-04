from __future__ import annotations

from integrations.megatron_ep.routersense.policy.validation import build_remote_flows

from integrations.megatron_ep.tests.helpers import make_observation


def test_runtime_observation_builds_remote_flows_with_real_ep_ranks() -> None:
    observation = make_observation(rank=3, local_rank=1, phase="P0", rows=(0, 12), ep_group_ranks=(3, 7))
    flows = build_remote_flows((observation,))
    assert len(flows) == 1
    assert flows[0].src_rank == 3
    assert flows[0].dst_rank == 7
    assert flows[0].phase == "P0"
    assert flows[0].rows == 12
    assert flows[0].bytes == 192
