"""Rank-level async P2P phase executor.

This executor materializes the local task slices from an already agreed phase
plan and executes remote transfers with ``batch_isend_irecv``. It does not
claim per-bucket compute overlap; completion is rank-local and phase-scoped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.phase import BucketTask, PhaseExecutionPlan, PhaseReadyContext

from .sync_wave_executor import PhaseExecutionResult, _copy_segment, _empty_like_rows


@dataclass(frozen=True)
class AsyncP2PExecutionResult:
    output: torch.Tensor
    summary: PhaseExecutionResult
    execution_entries: list[dict[str, Any]]


@dataclass(frozen=True)
class AsyncPhasePreflightResult:
    ok: bool
    all_ranks_ok: bool
    reason: str
    local_send_count: int
    local_recv_count: int
    send_digest: int
    recv_digest: int


def _sequence_key(
    *,
    context: PhaseReadyContext,
    phase: str,
    tensor_role: str,
    wave_id: int,
    task: BucketTask,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
    plan_key = dict(getattr(context, "plan_key", {}) or {})
    run_id_digest = str(plan_key.get("run_id_digest", "0"))[:16]
    try:
        run_id_value = int(run_id_digest, 16)
    except Exception:
        run_id_value = 0
    microbatch_digest = str(plan_key.get("microbatch_id", "0"))
    microbatch_value = sum(ord(ch) for ch in microbatch_digest) & 0x7FFFFFFF
    layer_id_raw = str(context.layer_id)
    try:
        layer_id_value = int(layer_id_raw)
    except Exception:
        layer_id_value = sum(ord(ch) for ch in layer_id_raw) & 0x7FFFFFFF
    return (
        int(run_id_value & 0x7FFFFFFF),
        int(context.forward_epoch),
        int(microbatch_value),
        int(layer_id_value),
        0 if str(phase) == "P0" else 1,
        0 if str(tensor_role) == "hidden_states" else 1,
        int(wave_id),
        int(task.bucket_ordinal),
        int(task.segment_ordinal),
        int(task.src_rank),
        int(task.dst_rank),
    )


def _digest_sequence_items(items: list[tuple[int, ...]]) -> int:
    acc = 1469598103934665603
    for item in items:
        for value in item:
            acc ^= int(value) & 0xFFFFFFFFFFFFFFFF
            acc = (acc * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(acc & 0x7FFFFFFFFFFFFFFF)


def validate_async_phase_preflight(
    *,
    context: PhaseReadyContext,
    plan: PhaseExecutionPlan,
    tensor_role: str,
    process_group: dist.ProcessGroup | None,
    rank_context: dict[str, int],
) -> AsyncPhasePreflightResult:
    world_group = process_group if process_group is not None else dist.group.WORLD
    rank = int(rank_context["global_rank"])
    world_size = int(len(context.ep_group_ranks) or 1)
    local_reason = ""

    expected_rows = int(sum(context.recv_splits))
    coverage = [0] * max(expected_rows, 0)
    local_send_items: list[tuple[int, ...]] = []
    local_recv_items: list[tuple[int, ...]] = []
    for wave in plan.waves:
        for task in wave.bucket_tasks:
            payload = next((item for item in task.payload_slices if item.tensor_role == tensor_role), None)
            if payload is None or int(payload.row_count) <= 0:
                continue
            if int(task.src_rank) != int(task.dst_rank) and int(task.dst_rank) == rank:
                start = int(payload.receiver_offset_rows)
                end = start + int(payload.row_count)
                if start < 0 or end > len(coverage):
                    local_reason = "recv_offset_out_of_bounds"
                    break
                for idx in range(start, end):
                    coverage[idx] += 1
            seq = _sequence_key(
                context=context,
                phase=str(context.phase),
                tensor_role=str(tensor_role),
                wave_id=int(wave.wave_id),
                task=task,
            )
            if int(task.src_rank) == rank and int(task.src_rank) != int(task.dst_rank):
                local_send_items.append(tuple(int(v) for v in seq if not isinstance(v, str)))
            if int(task.dst_rank) == rank and int(task.src_rank) != int(task.dst_rank):
                local_recv_items.append(tuple(int(v) for v in seq if not isinstance(v, str)))
        if local_reason:
            break
    if not local_reason and coverage and any(value != 1 for value in coverage):
        local_reason = "recv_coverage_invalid"

    device = (
        torch.device("cuda", rank_context["local_rank"])
        if torch.cuda.is_available() and int(rank_context["local_rank"]) < int(torch.cuda.device_count())
        else torch.device("cpu")
    )
    send_count = torch.zeros(world_size * world_size, dtype=torch.long, device=device)
    recv_count = torch.zeros(world_size * world_size, dtype=torch.long, device=device)
    send_rows = torch.zeros(world_size * world_size, dtype=torch.long, device=device)
    recv_rows = torch.zeros(world_size * world_size, dtype=torch.long, device=device)
    send_digest = torch.zeros(world_size * world_size, dtype=torch.long, device=device)
    recv_digest = torch.zeros(world_size * world_size, dtype=torch.long, device=device)
    for wave in plan.waves:
        for task in wave.bucket_tasks:
            payload = next((item for item in task.payload_slices if item.tensor_role == tensor_role), None)
            if payload is None or int(payload.row_count) <= 0 or int(task.src_rank) == int(task.dst_rank):
                continue
            index = int(task.src_rank) * world_size + int(task.dst_rank)
            seq = _sequence_key(
                context=context,
                phase=str(context.phase),
                tensor_role=str(tensor_role),
                wave_id=int(wave.wave_id),
                task=task,
            )
            seq_digest = _digest_sequence_items([tuple(int(v) for v in seq if not isinstance(v, str))])
            if int(task.src_rank) == rank:
                send_count[index] += 1
                send_rows[index] += int(payload.row_count)
                send_digest[index] ^= int(seq_digest)
            if int(task.dst_rank) == rank:
                recv_count[index] += 1
                recv_rows[index] += int(payload.row_count)
                recv_digest[index] ^= int(seq_digest)

    local_ok = 1 if not local_reason else 0
    local_flags = torch.tensor([local_ok], dtype=torch.long, device=device)
    gathered_flags = [torch.empty_like(local_flags) for _ in range(world_size)]
    dist.all_gather(gathered_flags, local_flags, group=world_group)
    all_ranks_ok = all(int(item.item()) == 1 for item in gathered_flags)

    if all_ranks_ok:
        send_count_g = [torch.empty_like(send_count) for _ in range(world_size)]
        recv_count_g = [torch.empty_like(recv_count) for _ in range(world_size)]
        send_rows_g = [torch.empty_like(send_rows) for _ in range(world_size)]
        recv_rows_g = [torch.empty_like(recv_rows) for _ in range(world_size)]
        send_digest_g = [torch.empty_like(send_digest) for _ in range(world_size)]
        recv_digest_g = [torch.empty_like(recv_digest) for _ in range(world_size)]
        dist.all_gather(send_count_g, send_count, group=world_group)
        dist.all_gather(recv_count_g, recv_count, group=world_group)
        dist.all_gather(send_rows_g, send_rows, group=world_group)
        dist.all_gather(recv_rows_g, recv_rows, group=world_group)
        dist.all_gather(send_digest_g, send_digest, group=world_group)
        dist.all_gather(recv_digest_g, recv_digest, group=world_group)
        send_count_sum = torch.stack(send_count_g).sum(dim=0)
        recv_count_sum = torch.stack(recv_count_g).sum(dim=0)
        send_rows_sum = torch.stack(send_rows_g).sum(dim=0)
        recv_rows_sum = torch.stack(recv_rows_g).sum(dim=0)
        send_digest_sum = torch.stack(send_digest_g).sum(dim=0)
        recv_digest_sum = torch.stack(recv_digest_g).sum(dim=0)
        if not torch.equal(send_count_sum, recv_count_sum):
            local_reason = "send_recv_count_mismatch"
            all_ranks_ok = False
        elif not torch.equal(send_rows_sum, recv_rows_sum):
            local_reason = "send_recv_row_mismatch"
            all_ranks_ok = False
        elif not torch.equal(send_digest_sum, recv_digest_sum):
            local_reason = "send_recv_sequence_mismatch"
            all_ranks_ok = False

    return AsyncPhasePreflightResult(
        ok=bool(local_ok),
        all_ranks_ok=bool(all_ranks_ok),
        reason=str(local_reason or ("ok" if all_ranks_ok else "remote_preflight_failed")),
        local_send_count=len(local_send_items),
        local_recv_count=len(local_recv_items),
        send_digest=_digest_sequence_items(local_send_items),
        recv_digest=_digest_sequence_items(local_recv_items),
    )


def execute_async_phase_tensor(
    *,
    context: PhaseReadyContext,
    plan: PhaseExecutionPlan,
    tensor_role: str,
    input_tensor: torch.Tensor,
    process_group: dist.ProcessGroup | None,
    rank_context: dict[str, int],
    timeline_hook: Any | None = None,
) -> AsyncP2PExecutionResult:
    world_group = process_group if process_group is not None else dist.group.WORLD
    rank = int(rank_context["global_rank"])
    total_recv_rows = int(sum(context.recv_splits))
    output = _empty_like_rows(input_tensor, total_recv_rows)

    incoming_by_src = {int(slot.src_rank): slot for slot in context.incoming_slots}
    local_copy_rows = 0
    for bundle in context.transport_bundles:
        segment = bundle.outgoing_segment
        if not segment.is_local or int(segment.row_count) <= 0:
            continue
        incoming_slot = incoming_by_src.get(int(segment.src_rank))
        if incoming_slot is None:
            raise ValueError(f"missing local incoming slot for phase={context.phase} rank={rank}")
        _copy_segment(
            output,
            input_tensor,
            recv_offset=int(incoming_slot.receive_offset_rows),
            send_offset=int(segment.send_offset_rows),
            rows=int(segment.row_count),
        )
        local_copy_rows += int(segment.row_count)

    recv_specs: list[tuple[tuple[int, str, int, int, int, int], dict[str, Any]]] = []
    send_specs: list[tuple[tuple[int, str, int, int, int, int], dict[str, Any]]] = []
    execution_entries: list[dict[str, Any]] = []
    active_wave_ids: set[int] = set()
    remote_copy_rows = 0

    for wave in plan.waves:
        for task in wave.bucket_tasks:
            payload = next((item for item in task.payload_slices if item.tensor_role == tensor_role), None)
            if payload is None or int(payload.row_count) <= 0:
                continue
            seq = _sequence_key(
                context=context,
                phase=str(context.phase),
                tensor_role=str(tensor_role),
                wave_id=int(wave.wave_id),
                task=task,
            )
            base_entry = {
                "phase": str(context.phase),
                "wave_id": int(wave.wave_id),
                "task_id": str(task.task_id),
                "src_rank": int(task.src_rank),
                "dst_rank": int(task.dst_rank),
                "tensor_role": str(tensor_role),
                "sender_offset_rows": int(payload.sender_offset_rows),
                "receiver_offset_rows": int(payload.receiver_offset_rows),
                "row_count": int(payload.row_count),
                "byte_count": int(payload.payload_byte_count),
                "sequence_key": list(seq),
                "record_type": "task",
                "execution_mode": "joint_window_async_p2p",
            }
            execution_entries.append(dict(base_entry))
            if int(task.src_rank) == int(task.dst_rank):
                continue
            active_wave_ids.add(int(wave.wave_id))
            if int(task.dst_rank) == rank:
                recv_specs.append((seq, dict(base_entry)))
                remote_copy_rows += int(payload.row_count)
            if int(task.src_rank) == rank:
                send_specs.append((seq, dict(base_entry)))

    recv_specs.sort(key=lambda item: item[0])
    send_specs.sort(key=lambda item: item[0])
    ordered_entries: list[dict[str, Any]] = []
    work_handles: list[Any] = []
    retained_tensors: list[torch.Tensor] = []

    enqueue_start_ns = time.monotonic_ns()
    if callable(timeline_hook):
        timeline_hook(
            "before_async_p2p_phase",
            phase=context.phase,
            tensor_role=tensor_role,
            plan_hash=plan.plan_hash,
            recv_op_count=len(recv_specs),
            send_op_count=len(send_specs),
        )

    ops: list[Any] = []
    for seq, entry in recv_specs:
        recv_tensor = output.narrow(0, int(entry["receiver_offset_rows"]), int(entry["row_count"]))
        ordered_entries.append({**entry, "op_kind": "recv"})
        retained_tensors.append(recv_tensor)
        ops.append(
            dist.P2POp(
                dist.irecv,
                recv_tensor,
                int(entry["src_rank"]),
                group=world_group,
            )
        )
    for seq, entry in send_specs:
        send_tensor = input_tensor.narrow(0, int(entry["sender_offset_rows"]), int(entry["row_count"]))
        ordered_entries.append({**entry, "op_kind": "send"})
        retained_tensors.append(send_tensor)
        ops.append(
            dist.P2POp(
                dist.isend,
                send_tensor,
                int(entry["dst_rank"]),
                group=world_group,
            )
        )
    enqueue_end_ns = time.monotonic_ns()

    if ops:
        work_handles = list(dist.batch_isend_irecv(ops))

    wait_start_ns = time.monotonic_ns()
    for work in work_handles:
        work.wait()
    wait_end_ns = time.monotonic_ns()

    if callable(timeline_hook):
        timeline_hook(
            "after_async_p2p_phase",
            phase=context.phase,
            tensor_role=tensor_role,
            plan_hash=plan.plan_hash,
            recv_op_count=len(recv_specs),
            send_op_count=len(send_specs),
        )

    execution_entries.append(
        {
            "record_type": "async_phase_summary",
            "phase": str(context.phase),
            "tensor_role": str(tensor_role),
            "execution_mode": "joint_window_async_p2p",
            "recv_op_count": len(recv_specs),
            "send_op_count": len(send_specs),
            "enqueue_start_ns": int(enqueue_start_ns),
            "enqueue_end_ns": int(enqueue_end_ns),
            "wait_start_ns": int(wait_start_ns),
            "wait_end_ns": int(wait_end_ns),
            "retained_tensor_count": len(retained_tensors),
            "work_handle_count": len(work_handles),
            "coalescing_enabled": False,
            "coalesced_task_count": 0,
        }
    )

    summary = PhaseExecutionResult(
        phase=context.phase,
        tensor_role=tensor_role,
        wave_count=len(plan.waves),
        bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
        active_wave_count=len(active_wave_ids),
        local_copy_rows=local_copy_rows,
        remote_copy_rows=remote_copy_rows,
        output_shape=tuple(int(dim) for dim in output.shape),
    )
    return AsyncP2PExecutionResult(output=output, summary=summary, execution_entries=execution_entries + ordered_entries)


__all__ = ["AsyncP2PExecutionResult", "execute_async_phase_tensor"]
