from __future__ import annotations

from collections import defaultdict

from rs.core.contracts.execution import MaterializedPlan, ValidationResult
from rs.runtime.online.megatron_ep.phase import PhaseReadyContext


def validate_materialized_layout(plan: MaterializedPlan, context: PhaseReadyContext) -> ValidationResult:
    try:
        plan.validate()
    except Exception as exc:
        return ValidationResult(valid=False, stage="materialized_plan", reason=str(exc))
    if str(plan.layout_digest) != str(context.canonical_receive_layout_id):
        return ValidationResult(valid=False, stage="layout_digest", reason="layout_digest_mismatch")
    role_specs = {str(spec.payload_role): spec for spec in plan.payload_specs}
    expected_outgoing = {
        str(role): tuple(int(value) for value in rows)
        for role, rows in plan.expected_outgoing_rows.items()
    }
    expected_incoming = {
        str(role): tuple(int(value) for value in rows)
        for role, rows in plan.expected_incoming_rows.items()
    }
    send_coverage = {
        str(role): [0 for value in rows for _ in range(int(value))]
        for role, rows in expected_outgoing.items()
    }
    recv_coverage = {
        str(role): [0 for value in rows for _ in range(int(value))]
        for role, rows in expected_incoming.items()
    }
    seen_task_ids: set[str] = set()
    seen_rows: set[tuple[str, str, int, int, int, int, int]] = set()
    world_size = len(tuple(int(rank) for rank in context.ep_group_ranks))
    for batch in plan.batches:
        batch.validate()
        local_outgoing_by_peer: set[tuple[int, str]] = set()
        local_incoming_by_peer: set[tuple[int, str]] = set()
        for item in batch.slices:
            if item.task_id in seen_task_ids:
                return ValidationResult(valid=False, stage="task_id", reason="duplicate_task_id")
            seen_task_ids.add(str(item.task_id))
            if str(item.payload_role) not in role_specs:
                return ValidationResult(valid=False, stage="payload_role", reason="payload_role_mismatch")
                if int(item.src_rank) < 0 or int(item.src_rank) >= int(world_size):
                    return ValidationResult(valid=False, stage="rank", reason="src_rank_out_of_range")
                if int(item.dst_rank) < 0 or int(item.dst_rank) >= int(world_size):
                    return ValidationResult(valid=False, stage="rank", reason="dst_rank_out_of_range")
            row_key = (
                str(item.task_id),
                str(item.payload_role),
                int(item.src_rank),
                int(item.dst_rank),
                int(item.send_offset_rows),
                int(item.recv_offset_rows),
                int(item.row_count),
            )
            if row_key in seen_rows:
                return ValidationResult(valid=False, stage="coverage", reason="duplicate_slice")
            seen_rows.add(row_key)
            if int(item.src_rank) == int(plan.local_global_rank):
                outgoing_key = (int(item.dst_rank), str(item.payload_role))
                if outgoing_key in local_outgoing_by_peer:
                    return ValidationResult(valid=False, stage="batch_port", reason="multiple_outgoing_same_wave")
                local_outgoing_by_peer.add(outgoing_key)
                coverage = send_coverage[str(item.payload_role)]
                for index in range(int(item.send_offset_rows), int(item.send_offset_rows) + int(item.row_count)):
                    if index < 0 or index >= len(coverage):
                        return ValidationResult(valid=False, stage="send_offsets", reason="send_offset_out_of_bounds")
                    coverage[index] += 1
            if int(item.dst_rank) == int(plan.local_global_rank):
                incoming_key = (int(item.src_rank), str(item.payload_role))
                if incoming_key in local_incoming_by_peer:
                    return ValidationResult(valid=False, stage="batch_port", reason="multiple_incoming_same_wave")
                local_incoming_by_peer.add(incoming_key)
                coverage = recv_coverage[str(item.payload_role)]
                for index in range(int(item.recv_offset_rows), int(item.recv_offset_rows) + int(item.row_count)):
                    if index < 0 or index >= len(coverage):
                        return ValidationResult(valid=False, stage="recv_offsets", reason="recv_offset_out_of_bounds")
                    coverage[index] += 1
    for role, coverage in send_coverage.items():
        if any(value != 1 for value in coverage):
            reason = "send_offset_gap" if any(value == 0 for value in coverage) else "send_offset_overlap"
            return ValidationResult(valid=False, stage="send_offsets", reason=reason)
    for role, coverage in recv_coverage.items():
        if any(value != 1 for value in coverage):
            reason = "recv_offset_gap" if any(value == 0 for value in coverage) else "recv_offset_overlap"
            return ValidationResult(valid=False, stage="recv_offsets", reason=reason)
    return ValidationResult(
        valid=True,
        stage="layout",
        details={
            "payload_roles": tuple(sorted(role_specs)),
            "batch_count": int(len(plan.batches)),
            "slice_count": int(sum(len(batch.slices) for batch in plan.batches)),
        },
    )
