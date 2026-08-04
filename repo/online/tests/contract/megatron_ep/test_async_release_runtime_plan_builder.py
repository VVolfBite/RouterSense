from __future__ import annotations

from rs.runtime.online.megatron_ep.async_release import AsyncReleaseRuntimePlanBuilder
from rs.scheduling.phase_execution import BucketTask, PayloadSlice, PhaseExecutionPlan, PhaseReadyContext, PlanWave


def _context() -> PhaseReadyContext:
    return PhaseReadyContext(
        plan_key={"layer_id": "0"},
        phase="P0",
        control_mode="sync_before_phase",
        forward_epoch=0,
        layer_id="0",
        layer_name="layer_0",
        global_rank=0,
        local_rank=0,
        ep_group_ranks=(0, 1),
        ep_group_root_rank=0,
        topology={},
        dispatcher_class="alltoall",
        dispatcher_fingerprint={},
        expert_placement_hash="ep",
        input_splits=(0, 1),
        output_splits=(0, 1),
        send_splits=(0, 1),
        recv_splits=(0, 1),
        per_peer_rows=(0, 1),
        per_peer_bytes=(0, 8),
        packed_send_layout_id="send_layout",
        canonical_receive_layout_id="recv_layout",
        payload_specs=(),
        atomic_submit=True,
        outgoing_segments=(),
        incoming_slots=(),
        transport_bundles=(),
        release_state="ready",
        demand_known_at="router_ready",
        payload_exists=True,
    )


def _task(task_id: str, phase: str, src: int, dst: int) -> BucketTask:
    return BucketTask(
        task_id=task_id,
        bundle_id=f"bundle_{task_id}",
        phase=phase,
        src_rank=src,
        dst_rank=dst,
        source_peer_index=src,
        destination_peer_index=dst,
        segment_ordinal=0,
        bucket_ordinal=0,
        sender_offset_rows=0,
        receiver_offset_rows=0,
        row_count=1,
        byte_count=8,
        packed_send_layout_id=f"send_{task_id}",
        canonical_receive_layout_id=f"recv_{task_id}",
        payload_slices=(
            PayloadSlice(
                bundle_id=f"bundle_{task_id}",
                tensor_role="hidden_states",
                src_rank=src,
                dst_rank=dst,
                segment_ordinal=0,
                sender_offset_rows=0,
                receiver_offset_rows=0,
                row_count=1,
                dtype="torch.float16",
                shape_suffix=(4,),
                element_size_bytes=2,
                payload_byte_count=8,
                packed_layout_id=f"layout_{task_id}",
            ),
        ),
    )


def test_async_release_runtime_plan_builder_uses_unique_task_ids_and_event_table() -> None:
    execution_plan = PhaseExecutionPlan(
        plan_key={"layer_id": "0"},
        phase="P0",
        policy_name="routersense_joint_priority_phase_sync",
        policy_version="v1",
        control_mode="sync_before_phase",
        execution_mode="phase_sync_wave",
        transport_mutation=True,
        is_shadow_only=False,
        future_hint_mode="calibrated_artifact",
        root_rank=0,
        observation_digest="obs",
        plan_hash="plan",
        waves=(
            PlanWave(wave_id=0, phase="P0", bucket_tasks=(_task("t0", "P0", 0, 1),)),
            PlanWave(wave_id=1, phase="P1", bucket_tasks=(_task("t1", "P1", 1, 0),)),
        ),
        metrics={},
    )
    plan = AsyncReleaseRuntimePlanBuilder(executor_available=False).build(
        local_context=_context(),
        execution_plan=execution_plan,
    )
    task_ids = [task["task_id"] for task in plan.phase_tasks]
    assert len(task_ids) == len(set(task_ids))
    assert "P0_PLAN_READY" in plan.event_table
    assert any(task["dependency_event_ids"] for task in plan.phase_tasks if task["phase"] == "P1")
