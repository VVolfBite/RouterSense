from __future__ import annotations

from rs_sim.scheduler.core.matching import maximum_weight_bipartite_matching
from rs_sim.scheduler.core.rscf_core import RSCFTask, order_rscf


def test_matching_scales_deterministically_to_ep32():
    weights = {(i, (i + 1) % 32): 10.0 for i in range(32)}
    first = maximum_weight_bipartite_matching(
        sources=tuple(range(32)),
        destinations=tuple(range(32)),
        edge_weight=lambda src, dst: weights.get((src, dst), 0.0),
    )
    second = maximum_weight_bipartite_matching(
        sources=tuple(range(32)),
        destinations=tuple(range(32)),
        edge_weight=lambda src, dst: weights.get((src, dst), 0.0),
    )
    assert first == second
    assert len(first) == 32


def test_rscf_runs_on_ep16():
    tasks = tuple(
        RSCFTask(
            task_id=f"t{i}",
            phase=0,
            src_rank=i,
            dst_rank=(i + 1) % 16,
            payload_units=100 + i,
            chunk_index=0,
            byte_offset=0,
            ready_at=0.0,
        )
        for i in range(16)
    )
    plan = order_rscf(tasks, rank_count=16)
    assert len(plan.ordered_task_ids) == 16
    assert set(plan.ordered_task_ids) == {task.task_id for task in tasks}
