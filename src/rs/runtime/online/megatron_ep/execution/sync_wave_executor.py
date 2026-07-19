"""同步 wave 执行器。

主要函数：
- execute_scheduled_phase_tensor()
它按 PhaseExecutionPlan 的 wave 顺序驱动实际 collective 调用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.phase import PhaseExecutionPlan, PhaseReadyContext


@dataclass(frozen=True)
class PhaseExecutionResult:
    phase: str
    tensor_role: str
    wave_count: int
    bucket_count: int
    active_wave_count: int
    local_copy_rows: int
    remote_copy_rows: int
    output_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "tensor_role": self.tensor_role,
            "wave_count": self.wave_count,
            "bucket_count": self.bucket_count,
            "active_wave_count": self.active_wave_count,
            "local_copy_rows": self.local_copy_rows,
            "remote_copy_rows": self.remote_copy_rows,
            "output_shape": list(self.output_shape),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhaseExecutionResult":
        return cls(
            phase=str(payload["phase"]),
            tensor_role=str(payload["tensor_role"]),
            wave_count=int(payload["wave_count"]),
            bucket_count=int(payload["bucket_count"]),
            active_wave_count=int(payload["active_wave_count"]),
            local_copy_rows=int(payload["local_copy_rows"]),
            remote_copy_rows=int(payload["remote_copy_rows"]),
            output_shape=tuple(int(v) for v in payload["output_shape"]),
        )


def _empty_like_rows(reference: torch.Tensor, rows: int) -> torch.Tensor:
    if reference.ndim == 1:
        return reference.new_empty((rows,))
    return reference.new_empty((rows, *reference.shape[1:]))


def _copy_segment(output: torch.Tensor, input_tensor: torch.Tensor, *, recv_offset: int, send_offset: int, rows: int) -> None:
    if rows <= 0:
        return
    output.narrow(0, int(recv_offset), int(rows)).copy_(input_tensor.narrow(0, int(send_offset), int(rows)))


def execute_scheduled_phase_tensor(
    *,
    context: PhaseReadyContext,
    plan: PhaseExecutionPlan,
    tensor_role: str,
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup | None,
    timeline_hook: Any | None = None,
) -> tuple[torch.Tensor, PhaseExecutionResult, list[dict[str, Any]]]:
    total_start_ns = time.monotonic_ns()
    world_group = group if group is not None else dist.group.WORLD
    peer_count = len(context.ep_group_ranks)
    rank = int(context.global_rank)
    local_peer_index = context.ep_group_ranks.index(rank)
    total_recv_rows = int(sum(context.recv_splits))
    output = _empty_like_rows(input_tensor, total_recv_rows)

    incoming_by_src = {int(slot.src_rank): slot for slot in context.incoming_slots}
    local_copy_rows = 0
    local_copy_start_ns = time.monotonic_ns()
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
    local_copy_end_ns = time.monotonic_ns()

    remote_copy_rows = 0
    active_wave_count = 0
    execution_entries: list[dict[str, Any]] = []
    wave_concat_ns = 0
    wave_collective_ns = 0
    wave_scatter_ns = 0
    first_transport_submit_ns = 0
    last_transport_complete_ns = 0
    for wave in plan.waves:
        if callable(timeline_hook):
            timeline_hook(
                "before_wave",
                phase=context.phase,
                wave_id=int(wave.wave_id),
                tensor_role=tensor_role,
                plan_hash=plan.plan_hash,
            )
        send_splits = [0 for _ in range(peer_count)]
        recv_splits = [0 for _ in range(peer_count)]
        send_slices: list[torch.Tensor] = []
        recv_task = None
        outgoing = None
        concat_start_ns = time.monotonic_ns()
        for task in wave.bucket_tasks:
            if int(task.src_rank) == rank:
                outgoing = task
                send_splits[int(task.destination_peer_index)] = int(task.row_count)
                payload = next(payload for payload in task.payload_slices if payload.tensor_role == tensor_role)
                send_slices.append(input_tensor.narrow(0, int(payload.sender_offset_rows), int(payload.row_count)))
            if int(task.dst_rank) == rank:
                recv_task = task
                recv_splits[int(task.source_peer_index)] = int(task.row_count)
        if not any(send_splits) and not any(recv_splits):
            # Non-participating ranks must still enter the collective with zero-sized tensors.
            input_buffer = _empty_like_rows(input_tensor, 0)
            output_buffer = _empty_like_rows(input_tensor, 0)
        else:
            active_wave_count += 1
            input_buffer = send_slices[0] if len(send_slices) == 1 else (torch.cat(send_slices, dim=0) if send_slices else _empty_like_rows(input_tensor, 0))
            output_buffer = _empty_like_rows(input_tensor, int(sum(recv_splits)))
        concat_end_ns = time.monotonic_ns()
        wave_concat_ns += int(concat_end_ns - concat_start_ns)
        if callable(timeline_hook):
            timeline_hook(
                "before_payload_collective",
                phase=context.phase,
                wave_id=int(wave.wave_id),
                tensor_role=tensor_role,
                input_split_sizes=list(send_splits),
                output_split_sizes=list(recv_splits),
                plan_hash=plan.plan_hash,
            )
        for task in wave.bucket_tasks:
            payload = next(payload for payload in task.payload_slices if payload.tensor_role == tensor_role)
            execution_entries.append(
                {
                    "phase": context.phase,
                    "wave_id": int(wave.wave_id),
                    "task_id": str(task.task_id),
                    "bucket_id": str(task.task_id),
                    "src_rank": int(task.src_rank),
                    "dst_rank": int(task.dst_rank),
                    "sender_offset_rows": int(payload.sender_offset_rows),
                    "receiver_offset_rows": int(payload.receiver_offset_rows),
                    "row_count": int(payload.row_count),
                    "byte_count": int(payload.payload_byte_count),
                    "tensor_role": str(tensor_role),
                    "policy_name": plan.policy_name,
                    "plan_hash": plan.plan_hash,
                }
            )
        collective_start_ns = time.monotonic_ns()
        if first_transport_submit_ns <= 0:
            first_transport_submit_ns = int(collective_start_ns)
        dist.all_to_all_single(
            output_buffer,
            input_buffer,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=world_group,
        )
        collective_end_ns = time.monotonic_ns()
        last_transport_complete_ns = int(collective_end_ns)
        wave_collective_ns += int(collective_end_ns - collective_start_ns)
        if callable(timeline_hook):
            timeline_hook(
                "after_payload_collective",
                phase=context.phase,
                wave_id=int(wave.wave_id),
                tensor_role=tensor_role,
                input_split_sizes=list(send_splits),
                output_split_sizes=list(recv_splits),
                plan_hash=plan.plan_hash,
            )
        scatter_start_ns = time.monotonic_ns()
        if recv_task is not None and int(sum(recv_splits)) > 0:
            payload = next(payload for payload in recv_task.payload_slices if payload.tensor_role == tensor_role)
            _copy_segment(
                output,
                output_buffer,
                recv_offset=int(payload.receiver_offset_rows),
                send_offset=0,
                rows=int(payload.row_count),
            )
            remote_copy_rows += int(payload.row_count)
        scatter_end_ns = time.monotonic_ns()
        wave_scatter_ns += int(scatter_end_ns - scatter_start_ns)
        if callable(timeline_hook):
            timeline_hook(
                "after_wave",
                phase=context.phase,
                wave_id=int(wave.wave_id),
                tensor_role=tensor_role,
                plan_hash=plan.plan_hash,
            )

    result = PhaseExecutionResult(
        phase=context.phase,
        tensor_role=tensor_role,
        wave_count=len(plan.waves),
        bucket_count=sum(len(wave.bucket_tasks) for wave in plan.waves),
        active_wave_count=active_wave_count,
        local_copy_rows=local_copy_rows,
        remote_copy_rows=remote_copy_rows,
        output_shape=tuple(int(dim) for dim in output.shape),
    )
    if callable(timeline_hook):
        timeline_hook(
            "after_phase",
            phase=context.phase,
            wave_id=-1,
            tensor_role=tensor_role,
            plan_hash=plan.plan_hash,
        )
    total_end_ns = time.monotonic_ns()
    execution_entries.append(
        {
            "record_type": "phase_sync_summary",
            "phase": str(context.phase),
            "tensor_role": str(tensor_role),
            "execution_mode": "phase_sync_wave",
            "local_copy_task_count": int(sum(1 for bundle in context.transport_bundles if bool(bundle.outgoing_segment.is_local) and int(bundle.outgoing_segment.row_count) > 0)),
            "local_copy_row_count": int(local_copy_rows),
            "wave_count": int(len(plan.waves)),
            "collective_count": int(len(plan.waves)),
            "wave_concat_us": float(wave_concat_ns / 1000.0),
            "wave_collective_us": float(wave_collective_ns / 1000.0),
            "wave_scatter_us": float(wave_scatter_ns / 1000.0),
            "local_copy_us": float((local_copy_end_ns - local_copy_start_ns) / 1000.0),
            "total_us": float((total_end_ns - total_start_ns) / 1000.0),
            "idle_barrier_wait_us": 0.0,
            "first_transport_submit_ns": int(first_transport_submit_ns),
            "last_transport_complete_ns": int(last_transport_complete_ns),
            "p0_first_submit_ns": int(first_transport_submit_ns if str(context.phase).upper() == "P0" else 0),
            "p0_last_complete_ns": int(last_transport_complete_ns if str(context.phase).upper() == "P0" else 0),
            "p1_first_submit_ns": int(first_transport_submit_ns if str(context.phase).upper() == "P1" else 0),
            "p1_last_complete_ns": int(last_transport_complete_ns if str(context.phase).upper() == "P1" else 0),
        }
    )
    return output, result, execution_entries
