"""Rank-level async P2P phase executor.

This executor materializes the local task slices from an already agreed phase
plan and executes remote transfers with ``batch_isend_irecv``. It does not
claim per-bucket compute overlap; completion is rank-local and phase-scoped.
"""

from __future__ import annotations

import time
import hashlib
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
    collective_count: int
    preflight_mode: str


def _dtype_code(dtype: str) -> int:
    mapping = {
        "torch.float16": 1,
        "torch.bfloat16": 2,
        "torch.float32": 3,
        "torch.int64": 4,
        "torch.int32": 5,
    }
    return int(mapping.get(str(dtype), 255))


def _shape_suffix_digest(shape_suffix: tuple[int, ...]) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for value in shape_suffix:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return int.from_bytes(digest.digest(), "little", signed=False) & 0x7FFFFFFFFFFFFFFF


def _global_group_maps(ep_group_ranks: tuple[int, ...]) -> tuple[dict[int, int], dict[int, int]]:
    global_to_group = {int(rank): idx for idx, rank in enumerate(ep_group_ranks)}
    group_to_global = {idx: int(rank) for idx, rank in enumerate(ep_group_ranks)}
    return global_to_group, group_to_global


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
    digest = hashlib.blake2b(digest_size=16)
    for item in items:
        for value in item:
            digest.update(int(value).to_bytes(8, "little", signed=True))
    return int.from_bytes(digest.digest()[:8], "little", signed=False) & 0x7FFFFFFFFFFFFFFF


def _sequence_entry(
    *,
    context: PhaseReadyContext,
    phase: str,
    tensor_role: str,
    wave_id: int,
    task: BucketTask,
    row_count: int,
    dtype: str,
    shape_suffix: tuple[int, ...],
) -> tuple[int, ...]:
    base = _sequence_key(
        context=context,
        phase=phase,
        tensor_role=tensor_role,
        wave_id=wave_id,
        task=task,
    )
    return (
        *base,
        int(row_count),
        int(_dtype_code(dtype)),
        int(_shape_suffix_digest(shape_suffix)),
    )


def _pair_index(*, src_rank: int, dst_rank: int, ep_group_ranks: tuple[int, ...]) -> int:
    global_to_group, _ = _global_group_maps(ep_group_ranks)
    if int(src_rank) not in global_to_group or int(dst_rank) not in global_to_group:
        raise ValueError(
            f"pair ranks must be members of ep_group_ranks: src={src_rank} dst={dst_rank} group={ep_group_ranks}"
        )
    src_group = int(global_to_group[int(src_rank)])
    dst_group = int(global_to_group[int(dst_rank)])
    group_size = int(len(ep_group_ranks))
    return int(src_group) * group_size + int(dst_group)


def _mark_coverage(coverage: list[int], *, start: int, row_count: int) -> str | None:
    end = int(start) + int(row_count)
    if int(start) < 0 or end > len(coverage):
        return "recv_offset_out_of_bounds"
    for idx in range(int(start), end):
        coverage[idx] += 1
    return None


def _mark_local_copy_coverage(
    coverage: list[int],
    *,
    context: PhaseReadyContext,
    tensor_role: str,
    rank: int,
) -> str | None:
    incoming_by_src = {int(slot.src_rank): slot for slot in context.incoming_slots}
    for bundle in context.transport_bundles:
        segment = bundle.outgoing_segment
        if not bool(segment.is_local) or int(segment.row_count) <= 0 or int(segment.dst_rank) != int(rank):
            continue
        if not any(str(payload.tensor_role) == str(tensor_role) for payload in bundle.payloads):
            continue
        incoming_slot = incoming_by_src.get(int(segment.src_rank))
        if incoming_slot is None:
            return "missing_local_incoming_slot"
        reason = _mark_coverage(
            coverage,
            start=int(incoming_slot.receive_offset_rows),
            row_count=int(segment.row_count),
        )
        if reason:
            return reason
    return None


def validate_async_phase_preflight(
    *,
    context: PhaseReadyContext,
    plan: PhaseExecutionPlan,
    tensor_role: str,
    process_group: dist.ProcessGroup | None,
    rank_context: dict[str, int],
    mode: str = "full",
) -> AsyncPhasePreflightResult:
    world_group = process_group if process_group is not None else dist.group.WORLD
    rank = int(rank_context["global_rank"])
    world_size = int(len(context.ep_group_ranks) or 1)
    local_reason = ""
    collective_count = 1

    expected_rows = int(sum(context.recv_splits))
    coverage = [0] * max(expected_rows, 0)
    local_send_items: list[tuple[int, ...]] = []
    local_recv_items: list[tuple[int, ...]] = []
    local_reason = _mark_local_copy_coverage(
        coverage,
        context=context,
        tensor_role=tensor_role,
        rank=rank,
    )
    for wave in plan.waves:
        for task in wave.bucket_tasks:
            payload = next((item for item in task.payload_slices if item.tensor_role == tensor_role), None)
            if payload is None or int(payload.row_count) <= 0:
                continue
            if int(task.dst_rank) == rank and int(task.src_rank) != int(task.dst_rank):
                reason = _mark_coverage(
                    coverage,
                    start=int(payload.receiver_offset_rows),
                    row_count=int(payload.row_count),
                )
                if reason:
                    local_reason = reason
                    break
            seq = _sequence_entry(
                context=context,
                phase=str(context.phase),
                tensor_role=str(tensor_role),
                wave_id=int(wave.wave_id),
                task=task,
                row_count=int(payload.row_count),
                dtype=str(payload.dtype),
                shape_suffix=tuple(int(v) for v in payload.shape_suffix),
            )
            if int(task.src_rank) == rank and int(task.src_rank) != int(task.dst_rank):
                local_send_items.append(tuple(int(v) for v in seq))
            if int(task.dst_rank) == rank and int(task.src_rank) != int(task.dst_rank):
                local_recv_items.append(tuple(int(v) for v in seq))
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
    role_digest_send = torch.zeros(world_size * world_size, dtype=torch.long, device=device)
    role_digest_recv = torch.zeros(world_size * world_size, dtype=torch.long, device=device)
    per_pair_send_sequences: dict[int, list[tuple[int, ...]]] = {}
    per_pair_recv_sequences: dict[int, list[tuple[int, ...]]] = {}
    per_pair_send_roles: dict[int, list[tuple[int, ...]]] = {}
    per_pair_recv_roles: dict[int, list[tuple[int, ...]]] = {}
    for wave in plan.waves:
        for task in wave.bucket_tasks:
            payload = next((item for item in task.payload_slices if item.tensor_role == tensor_role), None)
            if payload is None or int(payload.row_count) <= 0 or int(task.src_rank) == int(task.dst_rank):
                continue
            index = _pair_index(
                src_rank=int(task.src_rank),
                dst_rank=int(task.dst_rank),
                ep_group_ranks=tuple(int(v) for v in context.ep_group_ranks),
            )
            seq = _sequence_entry(
                context=context,
                phase=str(context.phase),
                tensor_role=str(tensor_role),
                wave_id=int(wave.wave_id),
                task=task,
                row_count=int(payload.row_count),
                dtype=str(payload.dtype),
                shape_suffix=tuple(int(v) for v in payload.shape_suffix),
            )
            role_tuple = (
                int(payload.row_count),
                int(_dtype_code(str(payload.dtype))),
                int(_shape_suffix_digest(tuple(int(v) for v in payload.shape_suffix))),
            )
            if int(task.src_rank) == rank:
                send_count[index] += 1
                send_rows[index] += int(payload.row_count)
                per_pair_send_sequences.setdefault(index, []).append(tuple(int(v) for v in seq))
                per_pair_send_roles.setdefault(index, []).append(role_tuple)
            if int(task.dst_rank) == rank:
                recv_count[index] += 1
                recv_rows[index] += int(payload.row_count)
                per_pair_recv_sequences.setdefault(index, []).append(tuple(int(v) for v in seq))
                per_pair_recv_roles.setdefault(index, []).append(role_tuple)

    for index, items in per_pair_send_sequences.items():
        send_digest[index] = int(_digest_sequence_items(items))
    for index, items in per_pair_recv_sequences.items():
        recv_digest[index] = int(_digest_sequence_items(items))
    for index, items in per_pair_send_roles.items():
        role_digest_send[index] = int(_digest_sequence_items(items))
    for index, items in per_pair_recv_roles.items():
        role_digest_recv[index] = int(_digest_sequence_items(items))

    local_ok = 1 if not local_reason else 0
    local_flags = torch.tensor([local_ok], dtype=torch.long, device=device)
    gathered_flags = [torch.empty_like(local_flags) for _ in range(world_size)]
    dist.all_gather(gathered_flags, local_flags, group=world_group)
    all_ranks_ok = all(int(item.item()) == 1 for item in gathered_flags)

    if all_ranks_ok:
        if str(mode) == "compact":
            packed = torch.stack(
                [
                    send_count,
                    recv_count,
                    send_rows,
                    recv_rows,
                    send_digest,
                    recv_digest,
                    role_digest_send,
                    role_digest_recv,
                ],
                dim=0,
            )
            gathered = [torch.empty_like(packed) for _ in range(world_size)]
            dist.all_gather(gathered, packed, group=world_group)
            collective_count += 1
            stacked = torch.stack(gathered, dim=0)
            send_count_sum = stacked[:, 0, :].sum(dim=0)
            recv_count_sum = stacked[:, 1, :].sum(dim=0)
            send_rows_sum = stacked[:, 2, :].sum(dim=0)
            recv_rows_sum = stacked[:, 3, :].sum(dim=0)
            send_digest_sum = stacked[:, 4, :].sum(dim=0)
            recv_digest_sum = stacked[:, 5, :].sum(dim=0)
            role_send_sum = stacked[:, 6, :].sum(dim=0)
            role_recv_sum = stacked[:, 7, :].sum(dim=0)
        else:
            send_count_g = [torch.empty_like(send_count) for _ in range(world_size)]
            recv_count_g = [torch.empty_like(recv_count) for _ in range(world_size)]
            send_rows_g = [torch.empty_like(send_rows) for _ in range(world_size)]
            recv_rows_g = [torch.empty_like(recv_rows) for _ in range(world_size)]
            send_digest_g = [torch.empty_like(send_digest) for _ in range(world_size)]
            recv_digest_g = [torch.empty_like(recv_digest) for _ in range(world_size)]
            role_digest_send_g = [torch.empty_like(role_digest_send) for _ in range(world_size)]
            role_digest_recv_g = [torch.empty_like(role_digest_recv) for _ in range(world_size)]
            dist.all_gather(send_count_g, send_count, group=world_group)
            dist.all_gather(recv_count_g, recv_count, group=world_group)
            dist.all_gather(send_rows_g, send_rows, group=world_group)
            dist.all_gather(recv_rows_g, recv_rows, group=world_group)
            dist.all_gather(send_digest_g, send_digest, group=world_group)
            dist.all_gather(recv_digest_g, recv_digest, group=world_group)
            dist.all_gather(role_digest_send_g, role_digest_send, group=world_group)
            dist.all_gather(role_digest_recv_g, role_digest_recv, group=world_group)
            collective_count += 8
            send_count_sum = torch.stack(send_count_g).sum(dim=0)
            recv_count_sum = torch.stack(recv_count_g).sum(dim=0)
            send_rows_sum = torch.stack(send_rows_g).sum(dim=0)
            recv_rows_sum = torch.stack(recv_rows_g).sum(dim=0)
            send_digest_sum = torch.stack(send_digest_g).sum(dim=0)
            recv_digest_sum = torch.stack(recv_digest_g).sum(dim=0)
            role_send_sum = torch.stack(role_digest_send_g).sum(dim=0)
            role_recv_sum = torch.stack(role_digest_recv_g).sum(dim=0)
        if not torch.equal(send_count_sum, recv_count_sum):
            local_reason = "send_recv_count_mismatch"
            all_ranks_ok = False
        elif not torch.equal(send_rows_sum, recv_rows_sum):
            local_reason = "send_recv_row_mismatch"
            all_ranks_ok = False
        elif not torch.equal(send_digest_sum, recv_digest_sum):
            local_reason = "send_recv_sequence_mismatch"
            all_ranks_ok = False
        elif not torch.equal(role_send_sum, role_recv_sum):
            local_reason = "send_recv_role_shape_mismatch"
            all_ranks_ok = False

    return AsyncPhasePreflightResult(
        ok=bool(local_ok),
        all_ranks_ok=bool(all_ranks_ok),
        reason=str(local_reason or ("ok" if all_ranks_ok else "remote_preflight_failed")),
        local_send_count=len(local_send_items),
        local_recv_count=len(local_recv_items),
        send_digest=_digest_sequence_items(local_send_items),
        recv_digest=_digest_sequence_items(local_recv_items),
        collective_count=int(collective_count),
        preflight_mode=str(mode),
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
    emit_detailed_artifacts = bool((plan.metrics or {}).get("emit_detailed_task_artifacts", True))

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
            seq = _sequence_entry(
                context=context,
                phase=str(context.phase),
                tensor_role=str(tensor_role),
                wave_id=int(wave.wave_id),
                task=task,
                row_count=int(payload.row_count),
                dtype=str(payload.dtype),
                shape_suffix=tuple(int(v) for v in payload.shape_suffix),
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
            if emit_detailed_artifacts:
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
        if emit_detailed_artifacts:
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
        if emit_detailed_artifacts:
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
