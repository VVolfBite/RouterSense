from __future__ import annotations

from rs.core.contracts.execution import ExecutionBatch, MaterializedPlan, TransferSlice, ValidationResult
from rs.runtime.online.megatron_ep.phase import PhaseReadyContext


def validate_materialized_layout(plan: MaterializedPlan, context: PhaseReadyContext) -> ValidationResult:
    try:
        plan.validate()
    except Exception as exc:
        return ValidationResult(valid=False, stage="materialized_plan", reason=str(exc))
    if str(plan.layout_digest) != str(context.canonical_receive_layout_id):
        return ValidationResult(valid=False, stage="layout_digest", reason="layout_digest_mismatch")
    world_size = len(context.ep_group_ranks)
    send_coverage = [0] * int(sum(context.send_splits))
    recv_coverage = [0] * int(sum(context.recv_splits))
    seen_task_rows: set[tuple[str, int, int, int, int, int]] = set()
    seen_flow_ids: set[str] = set()
    for batch in plan.batches:
        batch.validate()
        for item in batch.slices:
            item.validate()
            if tuple(str(role) for role in plan.expected_payload_roles) and str(item.payload_role) not in {
                str(role) for role in plan.expected_payload_roles
            }:
                return ValidationResult(valid=False, stage="payload_role", reason="payload_role_mismatch")
            if item.flow_id in seen_flow_ids:
                return ValidationResult(valid=False, stage="flow_id", reason="duplicate_flow_id")
            seen_flow_ids.add(str(item.flow_id))
            if int(item.src_rank) < 0 or int(item.src_rank) >= world_size:
                return ValidationResult(valid=False, stage="rank", reason="src_rank_out_of_range")
            if int(item.dst_rank) < 0 or int(item.dst_rank) >= world_size:
                return ValidationResult(valid=False, stage="rank", reason="dst_rank_out_of_range")
            dedupe_key = (
                str(item.task_id),
                int(item.src_rank),
                int(item.dst_rank),
                int(item.send_offset_rows),
                int(item.recv_offset_rows),
                int(item.row_count),
            )
            if dedupe_key in seen_task_rows:
                continue
            seen_task_rows.add(dedupe_key)
            if int(item.src_rank) == int(context.global_rank):
                for index in range(int(item.send_offset_rows), int(item.send_offset_rows) + int(item.row_count)):
                    if index < 0 or index >= len(send_coverage):
                        return ValidationResult(valid=False, stage="send_offsets", reason="send_offset_out_of_bounds")
                    send_coverage[index] += 1
            if int(item.dst_rank) == int(context.global_rank):
                for index in range(int(item.recv_offset_rows), int(item.recv_offset_rows) + int(item.row_count)):
                    if index < 0 or index >= len(recv_coverage):
                        return ValidationResult(valid=False, stage="recv_offsets", reason="recv_offset_out_of_bounds")
                    recv_coverage[index] += 1
    expected_send_rows = sum(int(segment.row_count) for segment in context.outgoing_segments if not bool(segment.is_local))
    expected_recv_rows = sum(int(slot.row_count) for slot in context.incoming_slots if not bool(slot.is_local))
    if len(send_coverage) != int(sum(context.send_splits)) or len(recv_coverage) != int(sum(context.recv_splits)):
        return ValidationResult(valid=False, stage="coverage", reason="coverage_vector_size_mismatch")
    if expected_send_rows > 0 and any(value != 1 for value in send_coverage if value != 0):
        return ValidationResult(valid=False, stage="send_offsets", reason="send_offset_overlap")
    if expected_recv_rows > 0 and any(value != 1 for value in recv_coverage if value != 0):
        return ValidationResult(valid=False, stage="recv_offsets", reason="recv_offset_overlap")
    return ValidationResult(
        valid=True,
        stage="layout",
        details={
            "send_rows": int(expected_send_rows),
            "recv_rows": int(expected_recv_rows),
            "batch_count": int(len(plan.batches)),
            "slice_count": int(sum(len(batch.slices) for batch in plan.batches)),
        },
    )
