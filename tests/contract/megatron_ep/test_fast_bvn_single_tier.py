from __future__ import annotations

import pytest

from rs.scheduling.registry import resolve_phase_policy
from .helpers import make_contexts_from_matrix, make_phase_context_generic


def test_fast_bvn_single_tier_builds_matching_waves() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 32, 0, 16), (8, 0, 40, 0), (0, 24, 0, 0), (16, 0, 0, 0)),
    )
    plan = resolve_phase_policy(policy_name="fast_bvn_single_tier", bucket_rows=16).build_plan(
        local_context=contexts[0],
        global_contexts=contexts,
    )
    diagnostics = plan.metrics["policy_diagnostics"]
    assert diagnostics["policy_name"] == "fast_bvn_single_tier"
    assert diagnostics["per_wave_matching_weight"]
    for wave in plan.waves:
        outgoing = set()
        incoming = set()
        for task in wave.bucket_tasks:
            assert task.src_rank not in outgoing
            assert task.dst_rank not in incoming
            outgoing.add(task.src_rank)
            incoming.add(task.dst_rank)


def test_fast_bvn_single_tier_rejects_scale_above_eight() -> None:
    ep_group_ranks = tuple(range(9))
    contexts = tuple(
        make_phase_context_generic(
            rank=rank,
            phase="P0",
            input_splits=tuple(0 for _ in ep_group_ranks),
            output_splits=tuple(0 for _ in ep_group_ranks),
            ep_group_ranks=ep_group_ranks,
        )
        for rank in ep_group_ranks
    )
    with pytest.raises(ValueError, match="unsupported_policy_scale"):
        resolve_phase_policy(policy_name="fast_bvn_single_tier", bucket_rows=16).build_plan(
            local_context=contexts[0],
            global_contexts=contexts,
        )
