from __future__ import annotations

from rs.runtime.online.megatron_ep.execution.release_frontier import ReleaseBatchFrontier, ReleaseBatchTask


def _tasks(version: int = 0) -> list[ReleaseBatchTask]:
    return [
        ReleaseBatchTask(task_id="t0", phase="P0", src_rank=0, dst_rank=1, row_count=2, plan_version=version),
        ReleaseBatchTask(task_id="t1", phase="P0", src_rank=1, dst_rank=0, row_count=1, plan_version=version),
        ReleaseBatchTask(task_id="t2", phase="P0", src_rank=0, dst_rank=1, row_count=3, plan_version=version),
    ]


def test_release_frontier_committed_prefix_immutable() -> None:
    frontier = ReleaseBatchFrontier(tasks=_tasks())
    batch = frontier.commit_batch(limit=1)
    frontier.mark_in_flight([batch[0].task_id])
    lineage = frontier.apply_late_suffix(
        new_plan_version=1,
        suffix_tasks=_tasks(version=1)[1:],
        plan_origin="late_spliced",
        parent_plan_version=0,
    )
    assert frontier.immutable_prefix_ids() == ("t0",)
    assert frontier.tasks[0].state == "in_flight"
    assert lineage.new_version == 1


def test_release_frontier_too_late_no_effect_shape() -> None:
    frontier = ReleaseBatchFrontier(tasks=_tasks())
    batch = frontier.commit_batch(limit=3)
    frontier.mark_in_flight([task.task_id for task in batch])
    frontier.mark_completed([task.task_id for task in batch])
    lineage = frontier.apply_late_suffix(
        new_plan_version=2,
        suffix_tasks=[],
        plan_origin="late_spliced",
        parent_plan_version=0,
    )
    assert frontier.replaceable_suffix_ids() == ()
    assert lineage.replacement_suffix_digest

