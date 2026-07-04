from __future__ import annotations

from collections import defaultdict

from integrations.megatron_ep.routersense.phase import BucketTask, PhaseExecutionPlan, PhaseReadyContext


def validate_phase_execution_plan(context: PhaseReadyContext, plan: PhaseExecutionPlan) -> None:
    if plan.phase != context.phase:
        raise ValueError(f"phase mismatch: context={context.phase} plan={plan.phase}")
    if plan.plan_key != context.plan_key:
        raise ValueError("plan_key mismatch between context and execution plan")
    local_ids = {segment.segment_id for segment in context.outgoing_segments if segment.is_local}
    coverage: dict[tuple[int, int, int], int] = defaultdict(int)
    for wave in plan.waves:
        outgoing = defaultdict(int)
        incoming = defaultdict(int)
        for task in wave.bucket_tasks:
            if task.phase != context.phase:
                raise ValueError("wave task phase mismatch")
            outgoing[int(task.src_rank)] += 1
            incoming[int(task.dst_rank)] += 1
            if outgoing[int(task.src_rank)] > 1:
                raise ValueError(f"wave {wave.wave_id} has >1 outgoing for rank {task.src_rank}")
            if incoming[int(task.dst_rank)] > 1:
                raise ValueError(f"wave {wave.wave_id} has >1 incoming for rank {task.dst_rank}")
            key = (int(task.src_rank), int(task.dst_rank), int(task.segment_ordinal))
            coverage[key] += int(task.row_count)
            if f"{task.phase}:{task.src_rank}->{task.dst_rank}:{task.segment_ordinal}" in local_ids:
                raise ValueError("local flow leaked into network execution plan")

    expected = {
        (segment.src_rank, segment.dst_rank, segment.segment_ordinal): int(segment.row_count)
        for segment in context.outgoing_segments
        if not segment.is_local and int(segment.row_count) > 0 and int(segment.src_rank) == int(context.global_rank)
    }
    local_coverage = {
        key: value
        for key, value in coverage.items()
        if int(key[0]) == int(context.global_rank)
    }
    if set(expected) != set(local_coverage):
        missing = sorted(set(expected) - set(local_coverage))
        extra = sorted(set(local_coverage) - set(expected))
        raise ValueError(f"plan coverage mismatch: missing={missing} extra={extra}")
    for key, expected_rows in expected.items():
        if int(local_coverage[key]) != int(expected_rows):
            raise ValueError(f"plan rows mismatch for {key}: expected={expected_rows} actual={local_coverage[key]}")


def row_digest(tasks: tuple[BucketTask, ...]) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple(
        (
            int(task.src_rank),
            int(task.dst_rank),
            int(task.segment_ordinal),
            int(task.sender_offset_rows),
            int(task.row_count),
        )
        for task in tasks
    )
