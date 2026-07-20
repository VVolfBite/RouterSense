from __future__ import annotations

import pytest

from rs.runtime.online.megatron_ep.control.contracts import BucketDescriptor, PendingCommTask, PlanKey
from rs.runtime.online.megatron_ep.control.state_machine import can_transition, transition_task


def _plan_key() -> PlanKey:
    return PlanKey(
        run_id_digest="run",
        forward_epoch=1,
        step_id="step-1",
        microbatch_id="mb-1",
        layer_id="0",
        phase="P0",
        ep_group_hash="ep",
        ep_group_epoch=1,
        model_revision_hash="model",
        expert_placement_hash="placement",
        request_table_hash="request",
    )


def _task() -> PendingCommTask:
    return PendingCommTask(
        task_id="task-0",
        bucket=BucketDescriptor(
            bucket_id="P0:0:1:0",
            plan_key=_plan_key(),
            phase="P0",
            wave_id=0,
            src_rank=0,
            dst_rank=1,
            source_peer_index=0,
            destination_peer_index=1,
            source_offset_rows=0,
            receive_offset_rows=0,
            row_count=8,
            byte_count=128,
            dtype="float16",
            hidden_shape_suffix=(2048,),
            packed_layout_id="layout-0",
            segment_ordinal=0,
        ),
        release_state="ready",
    )


def test_pending_task_state_machine_happy_path() -> None:
    task = _task()
    task = transition_task(task, "planned")
    task = transition_task(task, "committed")
    task = transition_task(task, "in_flight")
    task = transition_task(task, "completed")
    assert task.commit_state == "completed"


def test_committed_task_cannot_be_preempted() -> None:
    task = transition_task(transition_task(_task(), "planned"), "committed")
    assert can_transition(task.commit_state, "in_flight")
    assert not can_transition(task.commit_state, "planned")
    with pytest.raises(ValueError):
        transition_task(task, "planned")
