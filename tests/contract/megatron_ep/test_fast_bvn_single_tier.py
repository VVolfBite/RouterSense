from __future__ import annotations

from rs.scheduling.registry import resolve_policy
from .helpers import make_contexts_from_matrix


def test_fast_bvn_single_tier_builds_matching_waves() -> None:
    contexts = make_contexts_from_matrix(
        phase="P0",
        matrix=((0, 32, 0, 16), (8, 0, 40, 0), (0, 24, 0, 0), (16, 0, 0, 0)),
    )
    plan = resolve_policy(policy_name="fast_stage_reference", bucket_rows=16).build_plan(
        local_context=contexts[0],
        global_contexts=contexts,
    )
    diagnostics = plan.metrics["policy_diagnostics"]
    assert diagnostics["policy_name"] == "fast_stage_reference"
    assert diagnostics["per_wave_matching_weight"]
    for wave in plan.waves:
        outgoing = set()
        incoming = set()
        for task in wave.bucket_tasks:
            assert task.src_rank not in outgoing
            assert task.dst_rank not in incoming
            outgoing.add(task.src_rank)
            incoming.add(task.dst_rank)


def test_fast_two_tier_style_supports_scale_above_eight() -> None:
    rank_count = 12
    matrix = tuple(
        tuple(0 if src == dst else (16 if dst == (src + 4) % rank_count else 0) for dst in range(rank_count))
        for src in range(rank_count)
    )
    contexts = make_contexts_from_matrix(phase="P0", matrix=matrix)
    plan = resolve_policy(policy_name="fast_stage_reference", bucket_rows=16).build_plan(
        local_context=contexts[0],
        global_contexts=contexts,
    )
    diagnostics = plan.metrics["policy_diagnostics"]
    assert diagnostics["gpus_per_server"] == 4
    assert plan.waves
    for wave in plan.waves:
        outgoing = set()
        incoming = set()
        for task in wave.bucket_tasks:
            assert task.src_rank not in outgoing
            assert task.dst_rank not in incoming
            outgoing.add(task.src_rank)
            incoming.add(task.dst_rank)
