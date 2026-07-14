from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from rs.core.contracts.execution import ExecutionContext, ExecutionOutcome, MaterializedPlan, ValidationResult
from rs.runtime.online.megatron_ep.phase import PhaseReadyContext


def _coerce_phase_ready_context(payload: object) -> PhaseReadyContext:
    if isinstance(payload, PhaseReadyContext):
        return payload
    if isinstance(payload, dict):
        return PhaseReadyContext.from_dict(dict(payload))
    raise ValueError("phase_ready_context metadata is required")


@dataclass(frozen=True)
class PayloadInvocation:
    run_id: str
    forward_generation: int
    layer_id: str
    phase: str
    payload_role: str
    shape: tuple[int, ...]
    dtype: str
    layout_digest: str
    invocation_id: str
    input_tensor: torch.Tensor | None = None
    process_group: dist.ProcessGroup | None = None
    rank_context: dict[str, Any] = field(default_factory=dict)
    event_sink: Any | None = None

    def validate(self) -> None:
        if not str(self.run_id):
            raise ValueError("run_id must be non-empty")
        if int(self.forward_generation) < 0:
            raise ValueError("forward_generation must be >= 0")
        if not str(self.layer_id) or not str(self.phase) or not str(self.payload_role):
            raise ValueError("payload invocation identity must be non-empty")
        if not str(self.dtype):
            raise ValueError("dtype must be non-empty")
        if not str(self.layout_digest):
            raise ValueError("layout_digest must be non-empty")
        if not str(self.invocation_id):
            raise ValueError("invocation_id must be non-empty")
        for dim in self.shape:
            if int(dim) < 0:
                raise ValueError("shape dims must be >= 0")


class CommonExecutionGuard:
    def __init__(self) -> None:
        self._active_reservations: set[str] = set()
        self._consumed_invocations: set[str] = set()

    def _validate_common(self, *, plan: MaterializedPlan, invocation: PayloadInvocation, context: ExecutionContext) -> ValidationResult:
        try:
            plan.validate()
            invocation.validate()
            context.validate()
        except Exception as exc:
            return ValidationResult(valid=False, stage="guard", reason=str(exc))
        if str(context.run_id) != str(invocation.run_id):
            return ValidationResult(valid=False, stage="guard", reason="run_id_mismatch")
        if int(context.forward_generation) != int(invocation.forward_generation):
            return ValidationResult(valid=False, stage="guard", reason="forward_generation_mismatch")
        if str(context.layer_id) != str(invocation.layer_id):
            return ValidationResult(valid=False, stage="guard", reason="layer_id_mismatch")
        if str(context.phase) != str(invocation.phase):
            return ValidationResult(valid=False, stage="guard", reason="phase_mismatch")
        if str(plan.phase) != str(invocation.phase):
            return ValidationResult(valid=False, stage="guard", reason="plan_phase_mismatch")
        if str(plan.layout_digest) != str(invocation.layout_digest):
            return ValidationResult(valid=False, stage="guard", reason="layout_digest_mismatch")
        roles = {str(item.payload_role) for item in plan.payload_specs}
        if str(invocation.payload_role) not in roles:
            return ValidationResult(valid=False, stage="guard", reason="payload_role_mismatch")
        matching_spec = next(item for item in plan.payload_specs if str(item.payload_role) == str(invocation.payload_role))
        if str(invocation.dtype) != str(matching_spec.dtype):
            return ValidationResult(valid=False, stage="guard", reason="dtype_mismatch")
        if invocation.input_tensor is not None and tuple(int(dim) for dim in invocation.input_tensor.shape) != tuple(int(dim) for dim in invocation.shape):
            return ValidationResult(valid=False, stage="guard", reason="shape_mismatch")
        return ValidationResult(valid=True, stage="guard")

    def validate(self, *, plan: MaterializedPlan, invocation: PayloadInvocation, context: ExecutionContext) -> ValidationResult:
        result = self._validate_common(plan=plan, invocation=invocation, context=context)
        if not result.valid:
            return result
        invocation_id = str(invocation.invocation_id)
        if invocation_id in self._consumed_invocations:
            return ValidationResult(valid=False, stage="guard", reason="duplicate_invocation")
        self._consumed_invocations.add(invocation_id)
        return ValidationResult(valid=True, stage="guard")

    def reserve(self, *, plan: MaterializedPlan, invocation: PayloadInvocation, context: ExecutionContext) -> ValidationResult:
        result = self._validate_common(plan=plan, invocation=invocation, context=context)
        if not result.valid:
            return result
        invocation_id = str(invocation.invocation_id)
        if invocation_id in self._consumed_invocations or invocation_id in self._active_reservations:
            return ValidationResult(valid=False, stage="guard", reason="duplicate_invocation")
        self._active_reservations.add(invocation_id)
        return ValidationResult(valid=True, stage="guard")

    def commit(self, invocation_id: str) -> None:
        invocation_key = str(invocation_id)
        self._active_reservations.discard(invocation_key)
        self._consumed_invocations.add(invocation_key)

    def rollback(self, invocation_id: str) -> None:
        self._active_reservations.discard(str(invocation_id))


def _payload_spec_map(plan: MaterializedPlan) -> dict[str, Any]:
    return {str(spec.payload_role): spec for spec in plan.payload_specs}


def _peer_rows_total(rows_by_peer: dict[int, int]) -> int:
    return int(sum(int(value) for value in rows_by_peer.values()))


def _peer_base_offset(rows_by_peer: dict[int, int], peer_group_rank: int) -> int:
    return int(sum(int(rows_by_peer.get(index, 0)) for index in range(int(peer_group_rank))))


def _make_output_tensor(invocation: PayloadInvocation, *, total_rows: int) -> torch.Tensor:
    if invocation.input_tensor is None:
        raise ValueError("input_tensor required")
    shape_suffix = tuple(int(dim) for dim in invocation.input_tensor.shape[1:])
    if invocation.input_tensor.ndim == 1:
        return invocation.input_tensor.new_empty((int(total_rows),))
    return invocation.input_tensor.new_empty((int(total_rows), *shape_suffix))


def _copy_rows(target: torch.Tensor, target_offset: int, source: torch.Tensor, source_offset: int, row_count: int) -> None:
    if int(row_count) <= 0:
        return
    target.narrow(0, int(target_offset), int(row_count)).copy_(source.narrow(0, int(source_offset), int(row_count)))


def _source_input_offset(plan: MaterializedPlan, payload_role: str, dst_group_rank: int, peer_local_offset: int) -> int:
    rows_by_peer = {int(peer): int(value) for peer, value in plan.expected_outgoing_rows[str(payload_role)].items()}
    return int(_peer_base_offset(rows_by_peer, int(dst_group_rank)) + int(peer_local_offset))


def _target_output_offset(plan: MaterializedPlan, payload_role: str, src_group_rank: int, peer_local_offset: int) -> int:
    rows_by_peer = {int(peer): int(value) for peer, value in plan.expected_incoming_rows[str(payload_role)].items()}
    return int(_peer_base_offset(rows_by_peer, int(src_group_rank)) + int(peer_local_offset))


def _slices_for_role(plan: MaterializedPlan, payload_role: str) -> list[Any]:
    return [
        item
        for batch in plan.batches
        for item in batch.slices
        if str(item.payload_role) == str(payload_role)
    ]


def _submitted_task_ids_for_role(plan: MaterializedPlan, payload_role: str) -> tuple[str, ...]:
    return tuple(str(item.task_id) for item in _slices_for_role(plan, payload_role))


def _p2p_tag_for_slice(item: Any) -> int:
    payload_offset = {
        "hidden_states": 1,
        "routing_probs": 2,
        "p1_hidden_states": 3,
    }.get(str(item.payload_role), 9)
    return int(
        int(item.src_group_rank) * 100000
        + int(item.dst_group_rank) * 10000
        + payload_offset * 1000
        + (sum(ord(ch) for ch in str(item.flow_id)) % 1000)
    )


class _BaseExecutor:
    backend_id = ""

    def execute(self, *, plan: MaterializedPlan, invocation: PayloadInvocation, context: ExecutionContext) -> ExecutionOutcome:
        if invocation.input_tensor is None:
            return ExecutionOutcome(
                success=False,
                output_payload=None,
                submitted_task_ids=tuple(),
                completed_task_ids=tuple(),
                failed_task_ids=tuple(),
                unresolved_task_ids=tuple(),
                executed_batch_count=0,
                all_work_completed=False,
                failure_code="missing_input_tensor",
                details={},
            )
        submitted_task_ids = _submitted_task_ids_for_role(plan, invocation.payload_role)
        if not submitted_task_ids:
            return ExecutionOutcome(
                success=True,
                output_payload=invocation.input_tensor.clone(),
                submitted_task_ids=tuple(),
                completed_task_ids=tuple(),
                failed_task_ids=tuple(),
                unresolved_task_ids=tuple(),
                executed_batch_count=0,
                all_work_completed=True,
                details={"backend_id": str(self.backend_id), "reason": "no_matching_payload_role"},
            )
        raise NotImplementedError


class PhaseSyncExecutor(_BaseExecutor):
    backend_id = "phase_sync"

    def execute(self, *, plan: MaterializedPlan, invocation: PayloadInvocation, context: ExecutionContext) -> ExecutionOutcome:
        base = super().execute(plan=plan, invocation=invocation, context=context) if not _submitted_task_ids_for_role(plan, invocation.payload_role) else None
        if base is not None:
            return base
        payload_role = str(invocation.payload_role)
        input_tensor = invocation.input_tensor
        assert input_tensor is not None
        process_group = invocation.process_group if invocation.process_group is not None else dist.group.WORLD
        world_size = int(plan.rank_map.world_size)
        expected_incoming = {int(peer): int(value) for peer, value in plan.expected_incoming_rows[payload_role].items()}
        output = _make_output_tensor(invocation, total_rows=_peer_rows_total(expected_incoming))
        completed_task_ids: list[str] = []
        all_to_all_call_count = 0
        for batch in plan.batches:
            role_slices = [item for item in batch.slices if str(item.payload_role) == payload_role]
            if not role_slices:
                continue
            send_splits = [0 for _ in range(world_size)]
            recv_splits = [0 for _ in range(world_size)]
            remote_send_slices = []
            incoming_local = []
            incoming_remote = []
            for item in role_slices:
                if int(item.src_global_rank) == int(plan.local_global_rank):
                    if int(item.dst_global_rank) == int(plan.local_global_rank):
                        local_input_offset = _source_input_offset(plan, payload_role, int(item.dst_group_rank), int(item.send_offset_rows))
                        local_output_offset = _target_output_offset(plan, payload_role, int(item.src_group_rank), int(item.recv_offset_rows))
                        _copy_rows(output, local_output_offset, input_tensor, local_input_offset, int(item.row_count))
                        completed_task_ids.append(str(item.task_id))
                    else:
                        send_splits[int(item.dst_group_rank)] += int(item.row_count)
                        remote_send_slices.append(item)
                if int(item.dst_global_rank) == int(plan.local_global_rank) and int(item.src_global_rank) != int(plan.local_global_rank):
                    recv_splits[int(item.src_group_rank)] += int(item.row_count)
                    incoming_remote.append(item)
            total_send = int(sum(send_splits))
            total_recv = int(sum(recv_splits))
            if total_send > 0 or total_recv > 0:
                send_buffer = input_tensor.new_empty((total_send, *input_tensor.shape[1:])) if input_tensor.ndim > 1 else input_tensor.new_empty((total_send,))
                recv_buffer = input_tensor.new_empty((total_recv, *input_tensor.shape[1:])) if input_tensor.ndim > 1 else input_tensor.new_empty((total_recv,))
                pack_cursor = {peer: _peer_base_offset({int(idx): int(value) for idx, value in enumerate(send_splits)}, peer) for peer in range(world_size)}
                for item in remote_send_slices:
                    src_input_offset = _source_input_offset(plan, payload_role, int(item.dst_group_rank), int(item.send_offset_rows))
                    target_offset = int(pack_cursor[int(item.dst_group_rank)])
                    _copy_rows(send_buffer, target_offset, input_tensor, src_input_offset, int(item.row_count))
                    pack_cursor[int(item.dst_group_rank)] += int(item.row_count)
                dist.all_to_all_single(
                    recv_buffer,
                    send_buffer,
                    output_split_sizes=recv_splits,
                    input_split_sizes=send_splits,
                    group=process_group,
                )
                all_to_all_call_count += 1
                recv_base = {peer: _peer_base_offset({int(idx): int(value) for idx, value in enumerate(recv_splits)}, peer) for peer in range(world_size)}
                for item in incoming_remote:
                    output_offset = _target_output_offset(plan, payload_role, int(item.src_group_rank), int(item.recv_offset_rows))
                    recv_offset = int(recv_base[int(item.src_group_rank)])
                    _copy_rows(output, output_offset, recv_buffer, recv_offset, int(item.row_count))
                    recv_base[int(item.src_group_rank)] += int(item.row_count)
                completed_task_ids.extend(str(item.task_id) for item in remote_send_slices)
                completed_task_ids.extend(str(item.task_id) for item in incoming_remote)
        submitted_task_ids = _submitted_task_ids_for_role(plan, payload_role)
        completed_unique = tuple(dict.fromkeys(completed_task_ids))
        unresolved = tuple(task_id for task_id in submitted_task_ids if task_id not in set(completed_unique))
        return ExecutionOutcome(
            success=not unresolved,
            output_payload=output,
            submitted_task_ids=submitted_task_ids,
            completed_task_ids=completed_unique,
            failed_task_ids=tuple(),
            unresolved_task_ids=unresolved,
            executed_batch_count=int(len([batch for batch in plan.batches if any(str(item.payload_role) == payload_role for item in batch.slices)])),
            all_work_completed=not unresolved,
            failure_code=None if not unresolved else "unresolved_task",
            details={
                "backend_id": str(self.backend_id),
                "submitted_task_count": int(len(submitted_task_ids)),
                "distributed_operation_count": int(all_to_all_call_count),
            },
        )


class P2PReleaseExecutor(_BaseExecutor):
    backend_id = "async_release"

    def execute(self, *, plan: MaterializedPlan, invocation: PayloadInvocation, context: ExecutionContext) -> ExecutionOutcome:
        submitted = _submitted_task_ids_for_role(plan, invocation.payload_role)
        if not submitted:
            return super().execute(plan=plan, invocation=invocation, context=context)
        payload_role = str(invocation.payload_role)
        input_tensor = invocation.input_tensor
        assert input_tensor is not None
        process_group = invocation.process_group if invocation.process_group is not None else dist.group.WORLD
        world_size = int(plan.rank_map.world_size)
        expected_incoming = {int(peer): int(value) for peer, value in plan.expected_incoming_rows[payload_role].items()}
        output = _make_output_tensor(invocation, total_rows=_peer_rows_total(expected_incoming))
        max_inflight_batches = int(dict(context.metadata).get("max_inflight_batches", 2) or 2)
        satisfied_release_ids = {str(value) for value in context.satisfied_release_dependency_ids}
        pending_batches = []
        for batch in plan.batches:
            role_slices = [item for item in batch.slices if str(item.payload_role) == payload_role]
            if role_slices:
                pending_batches.append((batch, role_slices))
        inflight: deque[tuple[str, list[Any], list[tuple[Any, torch.Tensor]], list[torch.Tensor], list[Any]]] = deque()
        completed_batches: list[str] = []
        completed_task_ids: list[str] = []
        submitted_task_ids: list[str] = []
        peak_inflight = 0
        p2p_operation_count = 0

        def _batch_ready(role_slices: list[Any]) -> tuple[bool, str | None]:
            for item in role_slices:
                for dep in item.dependency_ids:
                    if str(dep).startswith("release:") and str(dep) not in satisfied_release_ids:
                        return False, str(dep)
            return True, None

        def _submit_batch(batch_id: str, role_slices: list[Any]) -> tuple[str, list[Any], list[tuple[Any, torch.Tensor]], list[torch.Tensor], list[Any]]:
            handles: list[Any] = []
            recv_payloads: list[tuple[Any, torch.Tensor]] = []
            retained_send_tensors: list[torch.Tensor] = []
            local_completions: list[Any] = []
            for item in role_slices:
                if int(item.src_global_rank) == int(plan.local_global_rank) and int(item.dst_global_rank) == int(plan.local_global_rank):
                    local_input_offset = _source_input_offset(plan, payload_role, int(item.dst_group_rank), int(item.send_offset_rows))
                    local_output_offset = _target_output_offset(plan, payload_role, int(item.src_group_rank), int(item.recv_offset_rows))
                    _copy_rows(output, local_output_offset, input_tensor, local_input_offset, int(item.row_count))
                    local_completions.append(item)
                    continue
                shape_suffix = tuple(int(dim) for dim in input_tensor.shape[1:])
                if int(item.dst_global_rank) == int(plan.local_global_rank) and int(item.src_global_rank) != int(plan.local_global_rank):
                    recv_tensor = input_tensor.new_empty((int(item.row_count), *shape_suffix)) if input_tensor.ndim > 1 else input_tensor.new_empty((int(item.row_count),))
                    handles.append(dist.irecv(recv_tensor, src=int(item.src_global_rank), group=process_group, tag=_p2p_tag_for_slice(item)))
                    recv_payloads.append((item, recv_tensor))
                if int(item.src_global_rank) == int(plan.local_global_rank) and int(item.dst_global_rank) != int(plan.local_global_rank):
                    src_input_offset = _source_input_offset(plan, payload_role, int(item.dst_group_rank), int(item.send_offset_rows))
                    send_tensor = input_tensor.narrow(0, int(src_input_offset), int(item.row_count)).contiguous()
                    retained_send_tensors.append(send_tensor)
                    handles.append(dist.isend(send_tensor, dst=int(item.dst_global_rank), group=process_group, tag=_p2p_tag_for_slice(item)))
            return (str(batch_id), handles, recv_payloads, retained_send_tensors, role_slices + local_completions)

        remaining = list(pending_batches)
        stalled_rounds = 0
        while remaining or inflight:
            ready_index = None
            blocked_reason = None
            for index, (batch, role_slices) in enumerate(remaining):
                ready, missing = _batch_ready(role_slices)
                if ready:
                    ready_index = index
                    break
                blocked_reason = missing
            while ready_index is not None and len(inflight) < max_inflight_batches:
                batch, role_slices = remaining.pop(ready_index)
                inflight.append(_submit_batch(str(batch.batch_id), role_slices))
                submitted_task_ids.extend(str(item.task_id) for item in role_slices)
                p2p_operation_count += int(len(inflight[-1][1]) > 0)
                peak_inflight = max(peak_inflight, len(inflight))
                completed_batches.append(str(batch.batch_id))
                ready_index = None
                blocked_reason = None
                for index, (next_batch, next_slices) in enumerate(remaining):
                    ready, missing = _batch_ready(next_slices)
                    if ready:
                        ready_index = index
                        break
                    blocked_reason = missing
            if inflight:
                batch_id, handles, recv_payloads, retained_send_tensors, role_slices = inflight.popleft()
                try:
                    for handle in handles:
                        handle.wait()
                except Exception as exc:
                    return ExecutionOutcome(
                        success=False,
                        output_payload=None,
                        submitted_task_ids=tuple(dict.fromkeys(submitted_task_ids)),
                        completed_task_ids=tuple(dict.fromkeys(completed_task_ids)),
                        failed_task_ids=tuple(str(item.task_id) for item in role_slices),
                        unresolved_task_ids=tuple(str(item.task_id) for _, slices in remaining for item in slices),
                        executed_batch_count=int(len(completed_batches)),
                        all_work_completed=False,
                        failure_code=f"work_wait_failed:{type(exc).__name__}",
                        details={"backend_id": str(self.backend_id)},
                    )
                for item, recv_tensor in recv_payloads:
                    output_offset = _target_output_offset(plan, payload_role, int(item.src_group_rank), int(item.recv_offset_rows))
                    _copy_rows(output, output_offset, recv_tensor, 0, int(item.row_count))
                for item in role_slices:
                    completed_task_ids.append(str(item.task_id))
                    if str(item.task_id).startswith("release:p0_inbound_complete") or str(item.task_id).startswith("release:p1_inbound_complete"):
                        satisfied_release_ids.add(str(item.task_id))
                stalled_rounds = 0
            elif remaining:
                stalled_rounds += 1
                if stalled_rounds > 0:
                    return ExecutionOutcome(
                        success=False,
                        output_payload=None,
                        submitted_task_ids=tuple(dict.fromkeys(submitted_task_ids)),
                        completed_task_ids=tuple(dict.fromkeys(completed_task_ids)),
                        failed_task_ids=tuple(),
                        unresolved_task_ids=tuple(str(item.task_id) for _, slices in remaining for item in slices),
                        executed_batch_count=int(len(completed_batches)),
                        all_work_completed=False,
                        failure_code=f"unresolved_dependency_or_cycle:{blocked_reason or 'unknown'}",
                        details={"backend_id": str(self.backend_id)},
                    )
        submitted_tuple = tuple(dict.fromkeys(submitted_task_ids))
        completed_tuple = tuple(dict.fromkeys(completed_task_ids))
        unresolved = tuple(task_id for task_id in submitted if task_id not in set(completed_tuple))
        return ExecutionOutcome(
            success=not unresolved,
            output_payload=output,
            submitted_task_ids=submitted_tuple,
            completed_task_ids=completed_tuple,
            failed_task_ids=tuple(),
            unresolved_task_ids=unresolved,
            executed_batch_count=int(len(completed_batches)),
            all_work_completed=not unresolved,
            failure_code=None if not unresolved else "unresolved_task",
            details={
                "backend_id": str(self.backend_id),
                "submitted_task_count": int(len(submitted_tuple)),
                "completed_batches": tuple(completed_batches),
                "max_inflight_batches": max_inflight_batches,
                "peak_inflight_batches": int(peak_inflight),
                "distributed_operation_count": int(p2p_operation_count),
            },
        )


class GlooFunctionalExecutor(_BaseExecutor):
    backend_id = "gloo_functional"

    def execute(self, *, plan: MaterializedPlan, invocation: PayloadInvocation, context: ExecutionContext) -> ExecutionOutcome:
        return PhaseSyncExecutor().execute(plan=plan, invocation=invocation, context=context)
