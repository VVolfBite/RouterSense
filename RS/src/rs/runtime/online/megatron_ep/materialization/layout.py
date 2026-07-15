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
        str(role): {int(peer): int(value) for peer, value in rows.items()}
        for role, rows in plan.expected_outgoing_rows.items()
    }
    expected_incoming = {
        str(role): {int(peer): int(value) for peer, value in rows.items()}
        for role, rows in plan.expected_incoming_rows.items()
    }
    send_coverage = {
        str(role): {int(peer): [0 for _ in range(int(value))] for peer, value in rows.items()}
        for role, rows in expected_outgoing.items()
    }
    recv_coverage = {
        str(role): {int(peer): [0 for _ in range(int(value))] for peer, value in rows.items()}
        for role, rows in expected_incoming.items()
    }
    outgoing_segments = {int(segment.dst_rank): segment for segment in context.outgoing_segments if int(segment.row_count) > 0}
    incoming_slots = {int(slot.src_rank): slot for slot in context.incoming_slots if int(slot.row_count) > 0}
    physical_send_coverage = {
        str(role): {int(peer): [0 for _ in range(int(segment.row_count))] for peer, segment in outgoing_segments.items()}
        for role in role_specs
    }
    physical_recv_coverage = {
        str(role): {int(peer): [0 for _ in range(int(slot.row_count))] for peer, slot in incoming_slots.items()}
        for role in role_specs
    }
    seen_task_ids: set[str] = set()
    seen_rows: set[tuple[str, str, int, int, int, int, int]] = set()
    world_size = int(plan.rank_map.world_size)
    for batch in plan.batches:
        batch.validate()
        local_outgoing_flow_peers: set[tuple[int, str]] = set()
        local_incoming_flow_peers: set[tuple[int, str]] = set()
        for item in batch.slices:
            if item.task_id in seen_task_ids:
                return ValidationResult(valid=False, stage="task_id", reason="duplicate_task_id")
            seen_task_ids.add(str(item.task_id))
            if str(item.payload_role) not in role_specs:
                return ValidationResult(valid=False, stage="payload_role", reason="payload_role_mismatch")
            if int(item.src_group_rank) < 0 or int(item.src_group_rank) >= int(world_size):
                return ValidationResult(valid=False, stage="rank", reason="src_group_rank_out_of_range")
            if int(item.dst_group_rank) < 0 or int(item.dst_group_rank) >= int(world_size):
                return ValidationResult(valid=False, stage="rank", reason="dst_group_rank_out_of_range")
            try:
                if int(item.src_global_rank) != int(plan.rank_map.group_rank_to_global_rank(int(item.src_group_rank))):
                    return ValidationResult(valid=False, stage="rank", reason="src_global_rank_mismatch")
                if int(item.dst_global_rank) != int(plan.rank_map.group_rank_to_global_rank(int(item.dst_group_rank))):
                    return ValidationResult(valid=False, stage="rank", reason="dst_global_rank_mismatch")
            except ValueError:
                return ValidationResult(valid=False, stage="rank", reason="rank_out_of_range")
            row_key = (
                str(item.task_id),
                str(item.payload_role),
                int(item.src_group_rank),
                int(item.dst_group_rank),
                int(item.send_offset_rows),
                int(item.recv_offset_rows),
                int(item.row_count),
            )
            if row_key in seen_rows:
                return ValidationResult(valid=False, stage="coverage", reason="duplicate_slice")
            seen_rows.add(row_key)
            if int(item.src_global_rank) == int(plan.local_global_rank) and int(item.dst_global_rank) != int(plan.local_global_rank):
                local_outgoing_flow_peers.add((int(item.dst_group_rank), str(item.flow_id)))
            if int(item.dst_global_rank) == int(plan.local_global_rank) and int(item.src_global_rank) != int(plan.local_global_rank):
                local_incoming_flow_peers.add((int(item.src_group_rank), str(item.flow_id)))
            if int(item.src_global_rank) == int(plan.local_global_rank):
                coverage = send_coverage[str(item.payload_role)][int(item.dst_group_rank)]
                end = int(item.peer_send_offset_rows) + int(item.row_count)
                if int(item.peer_send_offset_rows) < 0 or end > len(coverage):
                    return ValidationResult(valid=False, stage="send_offsets", reason="send_offset_out_of_bounds")
                for index in range(int(item.peer_send_offset_rows), end):
                    coverage[index] += 1
                segment = outgoing_segments.get(int(item.dst_global_rank))
                if segment is None:
                    return ValidationResult(valid=False, stage="physical_send_offsets", reason="missing_outgoing_segment")
                expected_physical = int(segment.send_offset_rows) + int(item.peer_send_offset_rows)
                if int(item.physical_send_offset_rows) != expected_physical:
                    return ValidationResult(valid=False, stage="physical_send_offsets", reason="physical_send_offset_mismatch")
                physical_end = int(item.physical_send_offset_rows) + int(item.row_count)
                if physical_end > int(segment.send_offset_rows) + int(segment.row_count):
                    return ValidationResult(valid=False, stage="physical_send_offsets", reason="physical_send_offset_out_of_bounds")
                segment_coverage = physical_send_coverage[str(item.payload_role)][int(item.dst_global_rank)]
                for index in range(
                    int(item.physical_send_offset_rows) - int(segment.send_offset_rows),
                    physical_end - int(segment.send_offset_rows),
                ):
                    segment_coverage[index] += 1
            if int(item.dst_global_rank) == int(plan.local_global_rank):
                coverage = recv_coverage[str(item.payload_role)][int(item.src_group_rank)]
                end = int(item.peer_recv_offset_rows) + int(item.row_count)
                if int(item.peer_recv_offset_rows) < 0 or end > len(coverage):
                    return ValidationResult(valid=False, stage="recv_offsets", reason="recv_offset_out_of_bounds")
                for index in range(int(item.peer_recv_offset_rows), end):
                    coverage[index] += 1
                slot = incoming_slots.get(int(item.src_global_rank))
                if slot is None:
                    return ValidationResult(valid=False, stage="physical_recv_offsets", reason="missing_incoming_slot")
                expected_physical = int(slot.receive_offset_rows) + int(item.peer_recv_offset_rows)
                if int(item.physical_recv_offset_rows) != expected_physical:
                    return ValidationResult(valid=False, stage="physical_recv_offsets", reason="physical_recv_offset_mismatch")
                physical_end = int(item.physical_recv_offset_rows) + int(item.row_count)
                if physical_end > int(slot.receive_offset_rows) + int(slot.row_count):
                    return ValidationResult(valid=False, stage="physical_recv_offsets", reason="physical_recv_offset_out_of_bounds")
                slot_coverage = physical_recv_coverage[str(item.payload_role)][int(item.src_global_rank)]
                for index in range(
                    int(item.physical_recv_offset_rows) - int(slot.receive_offset_rows),
                    physical_end - int(slot.receive_offset_rows),
                ):
                    slot_coverage[index] += 1
        if len({peer for peer, _flow_id in local_outgoing_flow_peers}) > 1:
            return ValidationResult(valid=False, stage="batch_port", reason="multiple_outgoing_same_wave")
        if len({peer for peer, _flow_id in local_incoming_flow_peers}) > 1:
            return ValidationResult(valid=False, stage="batch_port", reason="multiple_incoming_same_wave")
    for role, peer_map in send_coverage.items():
        for peer, coverage in peer_map.items():
            if any(value != 1 for value in coverage):
                reason = "send_offset_gap" if any(value == 0 for value in coverage) else "send_offset_overlap"
                return ValidationResult(valid=False, stage="send_offsets", reason=reason, details={"role": role, "peer_group_rank": int(peer)})
    for role, peer_map in recv_coverage.items():
        for peer, coverage in peer_map.items():
            if any(value != 1 for value in coverage):
                reason = "recv_offset_gap" if any(value == 0 for value in coverage) else "recv_offset_overlap"
                return ValidationResult(valid=False, stage="recv_offsets", reason=reason, details={"role": role, "peer_group_rank": int(peer)})
    for role, peer_map in physical_send_coverage.items():
        for peer, coverage in peer_map.items():
            if any(value != 1 for value in coverage):
                reason = "physical_send_offset_gap" if any(value == 0 for value in coverage) else "physical_send_offset_overlap"
                return ValidationResult(valid=False, stage="physical_send_offsets", reason=reason, details={"role": role, "peer_global_rank": int(peer)})
    for role, peer_map in physical_recv_coverage.items():
        for peer, coverage in peer_map.items():
            if any(value != 1 for value in coverage):
                reason = "physical_recv_offset_gap" if any(value == 0 for value in coverage) else "physical_recv_offset_overlap"
                return ValidationResult(valid=False, stage="physical_recv_offsets", reason=reason, details={"role": role, "peer_global_rank": int(peer)})
    return ValidationResult(
        valid=True,
        stage="layout",
        details={
            "payload_roles": tuple(sorted(role_specs)),
            "batch_count": int(len(plan.batches)),
            "slice_count": int(sum(len(batch.slices) for batch in plan.batches)),
        },
    )
