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
from rs.runtime.online.megatron_ep.execution.release_frontier import ReleaseBatchFrontier, ReleaseBatchTask

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
    expected_collective_count: int = 0
    collective_types: dict[str, int] | None = None
    payload_bytes: int = 0
    timing_us: dict[str, float] | None = None


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
    preflight_enter_ns = time.perf_counter_ns()
    local_validation_start_ns = preflight_enter_ns
    if str(mode) == "local_only":
        rank = int(rank_context["global_rank"])
        expected_rows = int(sum(context.recv_splits))
        coverage = [0] * max(expected_rows, 0)
        local_reason = _mark_local_copy_coverage(
            coverage,
            context=context,
            tensor_role=tensor_role,
            rank=rank,
        )
        local_send_items: list[tuple[int, ...]] = []
        local_recv_items: list[tuple[int, ...]] = []
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
        local_validation_end_ns = time.perf_counter_ns()
        result_check_start_ns = local_validation_end_ns
        result_check_end_ns = time.perf_counter_ns()
        preflight_exit_ns = result_check_end_ns
        return AsyncPhasePreflightResult(
            ok=bool(not local_reason),
            all_ranks_ok=bool(not local_reason),
            reason=str(local_reason or "ok"),
            local_send_count=len(local_send_items),
            local_recv_count=len(local_recv_items),
            send_digest=_digest_sequence_items(local_send_items),
            recv_digest=_digest_sequence_items(local_recv_items),
            collective_count=0,
            preflight_mode="local_only",
            expected_collective_count=0,
            collective_types={"all_gather": 0, "all_reduce": 0, "broadcast": 0, "barrier": 0},
            payload_bytes=0,
            timing_us={
                "preflight_total_us": float((preflight_exit_ns - preflight_enter_ns) / 1000.0),
                "preflight_local_validation_us": float((local_validation_end_ns - local_validation_start_ns) / 1000.0),
                "preflight_signature_build_us": 0.0,
                "preflight_collective_submit_us": 0.0,
                "preflight_collective_wait_us": 0.0,
                "preflight_result_check_us": float((result_check_end_ns - result_check_start_ns) / 1000.0),
                "preflight_other_us": 0.0,
            },
        )
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
    local_validation_end_ns = time.perf_counter_ns()

    signature_build_start_ns = time.perf_counter_ns()
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
    signature_build_end_ns = time.perf_counter_ns()

    local_ok = 1 if not local_reason else 0
    local_flags = torch.tensor([local_ok], dtype=torch.long, device=device)
    gathered_flags = [torch.empty_like(local_flags) for _ in range(world_size)]
    collective_submit_us = 0.0
    collective_wait_us = 0.0
    collective_types = {"all_gather": 0, "all_reduce": 0, "broadcast": 0, "barrier": 0}
    submit_start_ns = time.perf_counter_ns()
    dist.all_gather(gathered_flags, local_flags, group=world_group)
    submit_end_ns = time.perf_counter_ns()
    collective_submit_us += float((submit_end_ns - submit_start_ns) / 1000.0)
    collective_types["all_gather"] += 1
    all_ranks_ok = all(int(item.item()) == 1 for item in gathered_flags)
    payload_bytes = int(local_flags.numel() * local_flags.element_size())

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
            payload_bytes += int(packed.numel() * packed.element_size())
            submit_start_ns = time.perf_counter_ns()
            dist.all_gather(gathered, packed, group=world_group)
            submit_end_ns = time.perf_counter_ns()
            collective_submit_us += float((submit_end_ns - submit_start_ns) / 1000.0)
            collective_types["all_gather"] += 1
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
            for gathered, tensor in (
                (send_count_g, send_count),
                (recv_count_g, recv_count),
                (send_rows_g, send_rows),
                (recv_rows_g, recv_rows),
                (send_digest_g, send_digest),
                (recv_digest_g, recv_digest),
                (role_digest_send_g, role_digest_send),
                (role_digest_recv_g, role_digest_recv),
            ):
                payload_bytes += int(tensor.numel() * tensor.element_size())
                submit_start_ns = time.perf_counter_ns()
                dist.all_gather(gathered, tensor, group=world_group)
                submit_end_ns = time.perf_counter_ns()
                collective_submit_us += float((submit_end_ns - submit_start_ns) / 1000.0)
                collective_types["all_gather"] += 1
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
    result_check_end_ns = time.perf_counter_ns()
    preflight_exit_ns = result_check_end_ns
    total_us = float((preflight_exit_ns - preflight_enter_ns) / 1000.0)
    local_us = float((local_validation_end_ns - local_validation_start_ns) / 1000.0)
    signature_us = float((signature_build_end_ns - signature_build_start_ns) / 1000.0)
    result_us = max(0.0, float((result_check_end_ns - signature_build_end_ns) / 1000.0) - collective_submit_us)
    known_us = local_us + signature_us + collective_submit_us + collective_wait_us + result_us
    expected_collective_count = 2 if str(mode) == "compact" else 9

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
        expected_collective_count=int(expected_collective_count),
        collective_types=collective_types,
        payload_bytes=int(payload_bytes),
        timing_us={
            "preflight_total_us": total_us,
            "preflight_local_validation_us": local_us,
            "preflight_signature_build_us": signature_us,
            "preflight_collective_submit_us": float(collective_submit_us),
            "preflight_collective_wait_us": float(collective_wait_us),
            "preflight_result_check_us": float(result_us),
            "preflight_other_us": max(0.0, total_us - known_us),
        },
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
    total_start_ns = time.perf_counter_ns()
    world_group = process_group if process_group is not None else dist.group.WORLD
    rank = int(rank_context["global_rank"])
    total_recv_rows = int(sum(context.recv_splits))
    output = _empty_like_rows(input_tensor, total_recv_rows)
    emit_detailed_artifacts = bool((plan.metrics or {}).get("emit_detailed_task_artifacts", True))

    incoming_by_src = {int(slot.src_rank): slot for slot in context.incoming_slots}
    local_copy_rows = 0
    local_copy_task_count = 0
    local_copy_start_ns = time.perf_counter_ns()
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
        local_copy_task_count += 1
    local_copy_end_ns = time.perf_counter_ns()

    execution_entries: list[dict[str, Any]] = []
    active_wave_ids: set[int] = set()
    remote_copy_rows = 0
    ordered_entries: list[dict[str, Any]] = []
    if callable(timeline_hook):
        timeline_hook(
            "before_async_p2p_phase",
            phase=context.phase,
            tensor_role=tensor_role,
            plan_hash=plan.plan_hash,
            recv_op_count=0,
            send_op_count=0,
        )
    plan_task_lookup: dict[str, tuple[int, BucketTask, Any]] = {}
    frontier_tasks: list[ReleaseBatchTask] = []
    previous_task_id = ""
    peer_sequence = 0
    for wave in plan.waves:
        for task in wave.bucket_tasks:
            payload = next((item for item in task.payload_slices if item.tensor_role == tensor_role), None)
            if payload is None or int(payload.row_count) <= 0:
                continue
            plan_task_lookup[str(task.task_id)] = (int(wave.wave_id), task, payload)
            if emit_detailed_artifacts:
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
                execution_entries.append(
                    {
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
                )
            if int(task.src_rank) == int(task.dst_rank):
                continue
            active_wave_ids.add(int(wave.wave_id))
            deps = (previous_task_id,) if previous_task_id else ()
            frontier_tasks.append(
                ReleaseBatchTask(
                    task_id=str(task.task_id),
                    phase=str(context.phase),
                    src_rank=int(task.src_rank),
                    dst_rank=int(task.dst_rank),
                    row_count=int(payload.row_count),
                    sender_offset=int(payload.sender_offset_rows),
                    receiver_offset=int(payload.receiver_offset_rows),
                    tensor_role=str(tensor_role),
                    peer_sequence=int(peer_sequence),
                    dependency_ids=deps,
                    plan_digest=str(plan.plan_hash),
                    plan_version=int((plan.metrics or {}).get("plan_version", 0) or 0),
                )
            )
            previous_task_id = str(task.task_id)
            peer_sequence += 1
            if int(task.dst_rank) == rank:
                remote_copy_rows += int(payload.row_count)
    session = rank_context.get("async_phase_session") if isinstance(rank_context, dict) else None
    is_primary_payload = bool(rank_context.get("is_primary_payload", tensor_role == "hidden_states")) if isinstance(rank_context, dict) else (tensor_role == "hidden_states")
    precomputed_task_order = ()
    if isinstance(session, dict) and not is_primary_payload:
        precomputed_task_order = tuple(str(task_id) for task_id in (session.get("final_task_order") or ()))
    if precomputed_task_order:
        lookup = {str(task.task_id): task for task in frontier_tasks}
        ordered: list[ReleaseBatchTask] = []
        for task_id in precomputed_task_order:
            task = lookup.pop(str(task_id), None)
            if task is not None:
                ordered.append(task)
        ordered.extend(lookup.values())
        frontier_tasks = ordered
    frontier = ReleaseBatchFrontier(
        tasks=frontier_tasks,
        max_inflight_release_batches=int((plan.metrics or {}).get("max_inflight_release_batches", 1) or 1),
    )
    late_suffix_provider = rank_context.get("late_suffix_provider") if isinstance(rank_context, dict) else None
    if not is_primary_payload:
        late_suffix_provider = None
    on_release_batch_completed = rank_context.get("on_release_batch_completed") if isinstance(rank_context, dict) else None
    suffix_splice_count = 0
    retained_tensors: list[torch.Tensor] = []
    total_send_ops = 0
    total_recv_ops = 0
    batch_isend_irecv_call_count = 0
    work_handle_count = 0
    op_build_us = 0.0
    batch_submit_us = 0.0
    wait_us = 0.0
    batch_count = 0
    first_transport_submit_ns = 0
    last_transport_complete_ns = 0
    op_build_begin_ns = 0
    op_build_end_ns = 0
    submit_begin_ns = 0
    first_request_submitted_ns = 0
    last_request_submitted_ns = 0
    first_request_completed_ns = 0
    all_requests_completed_ns = 0
    while True:
        batch = frontier.commit_batch(limit=1)
        if not batch:
            if frontier.pending_count() <= 0:
                break
            ready = frontier.ready_batch(limit=1)
            if not ready:
                break
            batch = frontier.commit_batch(limit=1)
            if not batch:
                break
        batch_count += 1
        frontier.mark_in_flight([task.task_id for task in batch])
        op_build_start_ns = time.perf_counter_ns()
        if op_build_begin_ns <= 0:
            op_build_begin_ns = int(op_build_start_ns)
        ops: list[Any] = []
        batch_send = 0
        batch_recv = 0
        for release_task in batch:
            lookup = plan_task_lookup.get(str(release_task.task_id))
            if lookup is not None:
                wave_id, task, payload = lookup
                seq = _sequence_entry(
                    context=context,
                    phase=str(context.phase),
                    tensor_role=str(tensor_role),
                    wave_id=int(wave_id),
                    task=task,
                    row_count=int(payload.row_count),
                    dtype=str(payload.dtype),
                    shape_suffix=tuple(int(v) for v in payload.shape_suffix),
                )
                payload_byte_count = int(payload.payload_byte_count)
            else:
                wave_id = -1
                seq = (
                    int(context.forward_epoch),
                    int(rank),
                    0,
                    int(release_task.src_rank),
                    int(release_task.dst_rank),
                    int(release_task.peer_sequence),
                    int(release_task.row_count),
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                payload_byte_count = int(release_task.row_count)
            entry = {
                "phase": str(context.phase),
                "wave_id": int(wave_id),
                "task_id": str(release_task.task_id),
                "src_rank": int(release_task.src_rank),
                "dst_rank": int(release_task.dst_rank),
                "tensor_role": str(tensor_role),
                "sender_offset_rows": int(release_task.sender_offset),
                "receiver_offset_rows": int(release_task.receiver_offset),
                "row_count": int(release_task.row_count),
                "byte_count": int(payload_byte_count),
                "sequence_key": list(seq),
                "record_type": "task",
                "execution_mode": "joint_window_async_p2p",
            }
            if int(release_task.dst_rank) == rank:
                recv_tensor = output.narrow(0, int(release_task.receiver_offset), int(release_task.row_count))
                retained_tensors.append(recv_tensor)
                ops.append(dist.P2POp(dist.irecv, recv_tensor, int(release_task.src_rank), group=world_group))
                batch_recv += 1
                if emit_detailed_artifacts:
                    ordered_entries.append({**entry, "op_kind": "recv", "release_epoch": int(frontier.release_epoch)})
            if int(release_task.src_rank) == rank:
                send_tensor = input_tensor.narrow(0, int(release_task.sender_offset), int(release_task.row_count))
                retained_tensors.append(send_tensor)
                ops.append(dist.P2POp(dist.isend, send_tensor, int(release_task.dst_rank), group=world_group))
                batch_send += 1
                if emit_detailed_artifacts:
                    ordered_entries.append({**entry, "op_kind": "send", "release_epoch": int(frontier.release_epoch)})
        op_build_end_ns = time.perf_counter_ns()
        op_build_end_ns = int(op_build_end_ns)
        batch_submit_start_ns = time.perf_counter_ns()
        if submit_begin_ns <= 0:
            submit_begin_ns = int(batch_submit_start_ns)
        if first_transport_submit_ns <= 0 and ops:
            first_transport_submit_ns = int(batch_submit_start_ns)
        if first_request_submitted_ns <= 0 and ops:
            first_request_submitted_ns = int(batch_submit_start_ns)
        work_handles: list[Any] = list(dist.batch_isend_irecv(ops)) if ops else []
        batch_submit_end_ns = time.perf_counter_ns()
        if ops:
            last_request_submitted_ns = int(batch_submit_end_ns)
        wait_start_ns = time.perf_counter_ns()
        for work in work_handles:
            work.wait()
        wait_end_ns = time.perf_counter_ns()
        if ops:
            if first_request_completed_ns <= 0:
                first_request_completed_ns = int(wait_end_ns)
            all_requests_completed_ns = int(wait_end_ns)
            last_transport_complete_ns = int(wait_end_ns)
        frontier.mark_completed([task.task_id for task in batch])
        op_build_us += float((op_build_end_ns - op_build_start_ns) / 1000.0)
        batch_submit_us += float((batch_submit_end_ns - batch_submit_start_ns) / 1000.0)
        wait_us += float((wait_end_ns - wait_start_ns) / 1000.0)
        total_send_ops += int(batch_send)
        total_recv_ops += int(batch_recv)
        batch_isend_irecv_call_count += int(1 if ops else 0)
        work_handle_count += int(len(work_handles))
        if callable(on_release_batch_completed):
            on_release_batch_completed(
                int(frontier.release_epoch),
                {
                    "frontier_digest": str(frontier.frontier_digest()),
                    "immutable_prefix_ids": list(frontier.immutable_prefix_ids()),
                    "replaceable_suffix_ids": list(frontier.replaceable_suffix_ids()),
                    "release_epoch": int(frontier.release_epoch),
                },
            )
        if callable(late_suffix_provider) and frontier.pending_count() > 0:
            suffix_result = late_suffix_provider(
                context=context,
                plan=plan,
                tensor_role=tensor_role,
                frontier=frontier,
                release_epoch=int(frontier.release_epoch),
            )
            if isinstance(suffix_result, dict) and suffix_result.get("apply_suffix"):
                suffix_tasks = list(suffix_result.get("suffix_tasks", []))
                frontier.apply_late_suffix(
                    new_plan_version=int(suffix_result.get("new_plan_version", 1)),
                    suffix_tasks=suffix_tasks,
                    plan_origin="late_spliced",
                    parent_plan_version=int(suffix_result.get("parent_plan_version", 0)),
                    agreement_token=dict(suffix_result.get("agreement_token", {})),
                )
                suffix_splice_count += 1
    total_end_ns = time.perf_counter_ns()
    active_transport_critical_path_us = (
        float((all_requests_completed_ns - first_request_submitted_ns) / 1000.0)
        if first_request_submitted_ns > 0 and all_requests_completed_ns > 0
        else None
    )

    if callable(timeline_hook):
        timeline_hook(
            "after_async_p2p_phase",
            phase=context.phase,
            tensor_role=tensor_role,
            plan_hash=plan.plan_hash,
            recv_op_count=total_recv_ops,
            send_op_count=total_send_ops,
        )

    final_task_ids = [str(task.task_id) for task in frontier.tasks]
    if isinstance(session, dict):
        if is_primary_payload:
            session["final_task_order"] = list(final_task_ids)
            session["suffix_splice_count"] = int(suffix_splice_count)
            session["final_frontier_digest"] = str(frontier.frontier_digest())
            session["lineage"] = [item.to_dict() for item in frontier.lineage]
        else:
            session["secondary_replayed"] = True

    execution_entries.append(
        {
            "record_type": "async_phase_summary",
            "phase": str(context.phase),
            "tensor_role": str(tensor_role),
            "execution_mode": "joint_window_async_p2p",
            "recv_op_count": int(total_recv_ops),
            "send_op_count": int(total_send_ops),
            "retained_tensor_count": len(retained_tensors),
            "work_handle_count": int(work_handle_count),
            "coalescing_enabled": False,
            "coalesced_task_count": 0,
            "local_copy_task_count": int(local_copy_task_count),
            "local_copy_row_count": int(local_copy_rows),
            "local_copy_us": float((local_copy_end_ns - local_copy_start_ns) / 1000.0),
            "op_build_us": float(op_build_us),
            "batch_submit_us": float(batch_submit_us),
            "wait_us": float(wait_us),
            "total_us": float((total_end_ns - total_start_ns) / 1000.0),
            "op_build_begin_ns": int(op_build_begin_ns),
            "op_build_end_ns": int(op_build_end_ns),
            "submit_begin_ns": int(submit_begin_ns),
            "first_request_submitted_ns": int(first_request_submitted_ns),
            "last_request_submitted_ns": int(last_request_submitted_ns),
            "first_request_completed_ns": int(first_request_completed_ns),
            "all_requests_completed_ns": int(all_requests_completed_ns),
            "submit_queue_us": (
                float((first_request_submitted_ns - submit_begin_ns) / 1000.0)
                if submit_begin_ns > 0 and first_request_submitted_ns > 0
                else None
            ),
            "submit_span_us": (
                float((last_request_submitted_ns - first_request_submitted_ns) / 1000.0)
                if first_request_submitted_ns > 0 and last_request_submitted_ns > 0
                else None
            ),
            "request_wait_us": (
                float((all_requests_completed_ns - last_request_submitted_ns) / 1000.0)
                if last_request_submitted_ns > 0 and all_requests_completed_ns > 0
                else None
            ),
            "active_transport_sum_us": float(batch_submit_us + wait_us),
            "active_transport_critical_path_us": active_transport_critical_path_us,
            "batch_isend_irecv_call_count": int(batch_isend_irecv_call_count),
            "all_work_completed": True,
            "first_transport_submit_ns": int(first_transport_submit_ns),
            "last_transport_complete_ns": int(last_transport_complete_ns),
            "p0_first_submit_ns": int(first_transport_submit_ns if str(context.phase).upper() == "P0" else 0),
            "p0_last_complete_ns": int(last_transport_complete_ns if str(context.phase).upper() == "P0" else 0),
            "p1_first_submit_ns": int(first_transport_submit_ns if str(context.phase).upper() == "P1" else 0),
            "p1_last_complete_ns": int(last_transport_complete_ns if str(context.phase).upper() == "P1" else 0),
            "frontier_release_batch_count": int(batch_count),
            "suffix_splice_count": int(suffix_splice_count),
            "frontier_digest": str(frontier.frontier_digest()),
            "immutable_prefix_ids": list(frontier.immutable_prefix_ids()),
            "replaceable_suffix_ids": list(frontier.replaceable_suffix_ids()),
            "final_task_ids": list(final_task_ids),
            "lineage": [item.to_dict() for item in frontier.lineage],
            "is_primary_payload": bool(is_primary_payload),
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
