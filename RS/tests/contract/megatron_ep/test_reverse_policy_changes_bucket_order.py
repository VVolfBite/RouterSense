from __future__ import annotations

from integrations.megatron_ep.routersense.policy.registry import resolve_phase_policy
from .helpers import make_phase_context


def _bucket_ids(plan) -> list[str]:
    return [task.task_id for wave in plan.waves for task in wave.bucket_tasks]


def test_bucketed_fifo_and_reverse_share_same_transfer_layout() -> None:
    ctx0 = make_phase_context(rank=0, phase="P0", input_splits=(0, 48), output_splits=(0, 48), rows=48)
    ctx1 = make_phase_context(rank=1, phase="P0", input_splits=(48, 0), output_splits=(48, 0), rows=48)
    global_contexts = (ctx0, ctx1)
    fifo = resolve_phase_policy(policy_name="bucketed_fifo", bucket_rows=16).build_plan(
        local_context=ctx0,
        global_contexts=global_contexts,
    )
    reverse = resolve_phase_policy(policy_name="trivial_reverse_bucket", bucket_rows=16).build_plan(
        local_context=ctx0,
        global_contexts=global_contexts,
    )
    assert fifo.metrics["transfer_layouts"] == reverse.metrics["transfer_layouts"]


def test_reverse_policy_changes_bucket_order() -> None:
    ctx0 = make_phase_context(rank=0, phase="P0", input_splits=(0, 48), output_splits=(0, 48), rows=48)
    ctx1 = make_phase_context(rank=1, phase="P0", input_splits=(48, 0), output_splits=(48, 0), rows=48)
    global_contexts = (ctx0, ctx1)
    fifo = resolve_phase_policy(policy_name="bucketed_fifo", bucket_rows=16).build_plan(
        local_context=ctx0,
        global_contexts=global_contexts,
    )
    reverse = resolve_phase_policy(policy_name="trivial_reverse_bucket", bucket_rows=16).build_plan(
        local_context=ctx0,
        global_contexts=global_contexts,
    )
    assert _bucket_ids(fifo) != _bucket_ids(reverse)


def test_reverse_policy_preserves_bucket_coverage() -> None:
    ctx0 = make_phase_context(rank=0, phase="P0", input_splits=(0, 48), output_splits=(0, 48), rows=48)
    ctx1 = make_phase_context(rank=1, phase="P0", input_splits=(48, 0), output_splits=(48, 0), rows=48)
    global_contexts = (ctx0, ctx1)
    fifo = resolve_phase_policy(policy_name="bucketed_fifo", bucket_rows=16).build_plan(
        local_context=ctx0,
        global_contexts=global_contexts,
    )
    reverse = resolve_phase_policy(policy_name="trivial_reverse_bucket", bucket_rows=16).build_plan(
        local_context=ctx0,
        global_contexts=global_contexts,
    )
    fifo_spans = sorted(
        (task.src_rank, task.dst_rank, task.sender_offset_rows, task.receiver_offset_rows, task.row_count)
        for wave in fifo.waves
        for task in wave.bucket_tasks
    )
    reverse_spans = sorted(
        (task.src_rank, task.dst_rank, task.sender_offset_rows, task.receiver_offset_rows, task.row_count)
        for wave in reverse.waves
        for task in wave.bucket_tasks
    )
    assert fifo_spans == reverse_spans
