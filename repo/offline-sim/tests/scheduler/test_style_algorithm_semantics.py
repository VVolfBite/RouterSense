from __future__ import annotations

from rs_sim.scheduler.core.literature_cores import (
    LiteratureTask,
    order_aurora,
    order_fast,
    order_islip,
)


def _diagnostics(plan):
    return dict(plan.diagnostics)


def test_aurora_style_uses_receiver_orientation_for_column_hotspot():
    tasks = (
        LiteratureTask("a", 0, 0, 3, 10),
        LiteratureTask("b", 0, 1, 3, 10),
        LiteratureTask("c", 0, 2, 0, 1),
    )
    plan = order_aurora(tasks, rank_count=4)
    diagnostics = _diagnostics(plan)
    assert diagnostics["paper_label"] == "AURORA_STYLE"
    assert diagnostics["orientation_rule"] == "HEAVIER_OF_MAX_ROW_OR_MAX_COLUMN"
    assert diagnostics["column_oriented_phases"] == 1
    assert diagnostics["row_oriented_phases"] == 0
    assert set(plan.ordered_task_ids) == {"a", "b", "c"}
    for wave in plan.waves:
        members = [next(task for task in tasks if task.task_id == task_id) for task_id in wave.task_ids]
        assert len({task.src_rank for task in members}) == len(members)
        assert len({task.dst_rank for task in members}) == len(members)


def test_islip_style_retains_pointers_within_the_supplied_problem():
    tasks = tuple(
        LiteratureTask(f"{src}-{dst}", 0, src, dst, 1)
        for src in range(4)
        for dst in range(4)
        if src != dst
    )
    plan = order_islip(tasks, rank_count=4)
    diagnostics = _diagnostics(plan)
    assert diagnostics["paper_label"] == "ISLIP_STYLE_4"
    assert diagnostics["rounds"] == 4
    assert diagnostics["pointer_state_scope"] == "WITHIN_SUPPLIED_PROBLEM"
    assert diagnostics["pointer_update_first_iteration_only"] is True



def test_fast_style_is_explicitly_fixed_endpoint_two_tier():
    tasks = (
        LiteratureTask("x", 0, 0, 2, 8),
        LiteratureTask("y", 0, 1, 3, 8),
        LiteratureTask("z", 0, 2, 0, 8),
        LiteratureTask("w", 0, 3, 1, 8),
    )
    plan = order_fast(tasks, rank_count=4, gpus_per_server=2)
    diagnostics = _diagnostics(plan)
    assert diagnostics["paper_label"] == "FAST_STYLE"
    assert diagnostics["fixed_endpoints"] is True
    assert diagnostics["literature_mapping"] == "TWO_TIER_FIXED_ENDPOINT_STYLE"
    assert set(plan.ordered_task_ids) == {"x", "y", "z", "w"}


def test_fast_style_refuses_ep_based_topology_guessing():
    tasks = (LiteratureTask("x", 0, 0, 1, 8),)
    try:
        order_fast(tasks, rank_count=2)
    except ValueError as exc:
        assert "explicit rank_to_node" in str(exc)
    else:
        raise AssertionError("FAST-style must fail closed without topology")
