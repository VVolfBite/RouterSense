from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.execution.async_release_backend import execute_async_phase_tensor
from rs.runtime.online.megatron_ep.execution.sync_wave_executor import execute_scheduled_phase_tensor
from rs.scheduling.phase_execution import PhaseExecutionPlan, PhaseReadyContext


@dataclass(frozen=True)
class ExecutionRequest:
    execution_plan: PhaseExecutionPlan
    phase_context: PhaseReadyContext
    tensor_role: str
    input_tensor: torch.Tensor
    process_group: dist.ProcessGroup | None
    rank_context: dict[str, Any]
    event_sink: Any | None = None
    requested_backend_id: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    output_tensor: torch.Tensor
    execution_plan_digest: str
    send_op_count: int
    recv_op_count: int
    local_copy_task_count: int
    local_copy_row_count: int
    enqueue_us: float
    wait_us: float
    total_us: float
    fallback_used: bool
    timeout: bool
    raw_summary: dict[str, Any]
    execution_entries: tuple[dict[str, Any], ...]
    requested_backend_id: str = ""
    backend_id: str = ""
    executed_backend_id: str = ""
    batch_isend_irecv_call_count: int = 0
    preflight_collective_count: int = 0
    preflight_passed: bool = True
    fallback_reason: str = ""
    all_work_completed: bool = True
    timing_us: dict[str, float] | None = None
    phase_metrics: dict[str, Any] | None = None
    failure_code: str = ""
    session_poisoned: bool = False
    blocked_release_tokens: tuple[str, ...] = ()
    first_transport_submit_ns: int = 0
    last_transport_complete_ns: int = 0
    p0_first_submit_ns: int = 0
    p0_last_complete_ns: int = 0
    p1_first_submit_ns: int = 0
    p1_last_complete_ns: int = 0

    def __post_init__(self) -> None:
        if not self.executed_backend_id and self.backend_id:
            object.__setattr__(self, "executed_backend_id", str(self.backend_id))
        if not self.backend_id and self.executed_backend_id:
            object.__setattr__(self, "backend_id", str(self.executed_backend_id))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_tensor"] = {
            "shape": tuple(int(v) for v in self.output_tensor.shape),
            "dtype": str(self.output_tensor.dtype),
            "device": str(self.output_tensor.device),
        }
        return payload


class TransportExecutor(Protocol):
    backend_id: str

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        ...


class PhaseSyncTransportExecutor:
    backend_id = "phase_sync"

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        start_ns = time.perf_counter_ns()
        output, summary, entries = execute_scheduled_phase_tensor(
            context=request.phase_context,
            plan=request.execution_plan,
            tensor_role=request.tensor_role,
            input_tensor=request.input_tensor,
            group=request.process_group,
            timeline_hook=request.event_sink,
        )
        end_ns = time.perf_counter_ns()
        summary_row = next((row for row in entries if row.get("record_type") == "phase_sync_summary"), {})
        return ExecutionResult(
            output_tensor=output,
            requested_backend_id=str(request.requested_backend_id or self.backend_id),
            executed_backend_id=self.backend_id,
            execution_plan_digest=str(request.execution_plan.plan_hash),
            send_op_count=0,
            recv_op_count=0,
            local_copy_task_count=int(summary_row.get("local_copy_task_count", 0) or 0),
            local_copy_row_count=int(summary.local_copy_rows),
            batch_isend_irecv_call_count=0,
            preflight_collective_count=0,
            preflight_passed=True,
            enqueue_us=float(summary_row.get("wave_collective_us", 0.0) or 0.0),
            wait_us=0.0,
            total_us=(end_ns - start_ns) / 1000.0,
            fallback_used=False,
            fallback_reason="",
            timeout=False,
            all_work_completed=True,
            timing_us={
                "local_copy_us": float(summary_row.get("local_copy_us", 0.0) or 0.0),
                "wave_concat_us": float(summary_row.get("wave_concat_us", 0.0) or 0.0),
                "wave_collective_us": float(summary_row.get("wave_collective_us", 0.0) or 0.0),
                "wave_scatter_us": float(summary_row.get("wave_scatter_us", 0.0) or 0.0),
                "idle_barrier_wait_us": float(summary_row.get("idle_barrier_wait_us", 0.0) or 0.0),
                "submit_us": float(summary_row.get("wave_collective_us", 0.0) or 0.0),
                "wait_us": 0.0,
                "communication_us": float(summary_row.get("total_us", (end_ns - start_ns) / 1000.0) or 0.0),
                "total_forward_us": float((end_ns - start_ns) / 1000.0),
            },
            phase_metrics={
                "wave_count": int(summary_row.get("wave_count", len(request.execution_plan.waves)) or 0),
                "collective_count": int(summary_row.get("collective_count", len(request.execution_plan.waves)) or 0),
            },
            first_transport_submit_ns=int(summary_row.get("first_transport_submit_ns", 0) or 0),
            last_transport_complete_ns=int(summary_row.get("last_transport_complete_ns", 0) or 0),
            p0_first_submit_ns=int(summary_row.get("p0_first_submit_ns", 0) or 0),
            p0_last_complete_ns=int(summary_row.get("p0_last_complete_ns", 0) or 0),
            p1_first_submit_ns=int(summary_row.get("p1_first_submit_ns", 0) or 0),
            p1_last_complete_ns=int(summary_row.get("p1_last_complete_ns", 0) or 0),
            raw_summary=summary.to_dict(),
            execution_entries=tuple(entries),
        )


class AsyncReleaseTransportExecutor:
    backend_id = "async_release"

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        result = execute_async_phase_tensor(
            context=request.phase_context,
            plan=request.execution_plan,
            tensor_role=request.tensor_role,
            input_tensor=request.input_tensor,
            process_group=request.process_group,
            rank_context=request.rank_context,
            timeline_hook=request.event_sink,
        )
        summary_row = next((row for row in result.execution_entries if row.get("record_type") == "async_phase_summary"), {})
        plan_tasks = [
            task
            for wave in request.execution_plan.waves
            for task in wave.bucket_tasks
        ]
        task_rows = [
            int(payload.row_count)
            for task in plan_tasks
            for payload in task.payload_slices
            if str(payload.tensor_role) == str(request.tensor_role)
        ]
        task_wire_bytes = [
            int(payload.payload_byte_count)
            for task in plan_tasks
            for payload in task.payload_slices
            if str(payload.tensor_role) == str(request.tensor_role)
        ]
        return ExecutionResult(
            output_tensor=result.output,
            requested_backend_id=str(request.requested_backend_id or self.backend_id),
            executed_backend_id=self.backend_id,
            execution_plan_digest=str(request.execution_plan.plan_hash),
            send_op_count=int(summary_row.get("send_op_count", 0) or 0),
            recv_op_count=int(summary_row.get("recv_op_count", 0) or 0),
            local_copy_task_count=int(summary_row.get("local_copy_task_count", 0) or 0),
            local_copy_row_count=int(summary_row.get("local_copy_row_count", 0) or 0),
            batch_isend_irecv_call_count=int(summary_row.get("batch_isend_irecv_call_count", 0) or 0),
            preflight_collective_count=0,
            preflight_passed=True,
            enqueue_us=float(summary_row.get("batch_submit_us", 0.0) or 0.0),
            wait_us=float(summary_row.get("wait_us", 0.0) or 0.0),
            total_us=float(summary_row.get("total_us", 0.0) or 0.0),
            fallback_used=False,
            fallback_reason="",
            timeout=False,
            all_work_completed=bool(summary_row.get("all_work_completed", True)),
            timing_us={
                "local_copy_us": float(summary_row.get("local_copy_us", 0.0) or 0.0),
                "op_build_us": float(summary_row.get("op_build_us", 0.0) or 0.0),
                "batch_submit_us": float(summary_row.get("batch_submit_us", 0.0) or 0.0),
                "work_wait_us": float(summary_row.get("wait_us", 0.0) or 0.0),
                "wait_us": float(summary_row.get("wait_us", 0.0) or 0.0),
                "submit_queue_us": float(summary_row.get("submit_queue_us") or 0.0),
                "submit_span_us": float(summary_row.get("submit_span_us") or 0.0),
                "request_wait_us": float(summary_row.get("request_wait_us") or 0.0),
                "active_transport_sum_us": float(summary_row.get("active_transport_sum_us") or 0.0),
                "active_transport_critical_path_us": float(
                    summary_row.get("active_transport_critical_path_us") or 0.0
                ),
                "communication_us": float(summary_row.get("total_us", 0.0) or 0.0),
                "total_forward_us": float(summary_row.get("total_us", 0.0) or 0.0),
            },
            phase_metrics={
                "wave_count": int(len(request.execution_plan.waves)),
                "task_count": int(len(plan_tasks)),
                "tensor_role_task_count": int(len(task_rows)),
                "total_rows": int(sum(task_rows)),
                "total_wire_bytes": int(sum(task_wire_bytes)),
                "send_task_count": int(summary_row.get("send_op_count", 0) or 0),
                "recv_task_count": int(summary_row.get("recv_op_count", 0) or 0),
                "p2p_op_count": int(summary_row.get("send_op_count", 0) or 0) + int(summary_row.get("recv_op_count", 0) or 0),
                "work_handle_count": int(summary_row.get("work_handle_count", 0) or 0),
                "op_build_begin_ns": int(summary_row.get("op_build_begin_ns", 0) or 0),
                "op_build_end_ns": int(summary_row.get("op_build_end_ns", 0) or 0),
                "submit_begin_ns": int(summary_row.get("submit_begin_ns", 0) or 0),
                "first_request_submitted_ns": int(summary_row.get("first_request_submitted_ns", 0) or 0),
                "last_request_submitted_ns": int(summary_row.get("last_request_submitted_ns", 0) or 0),
                "first_request_completed_ns": int(summary_row.get("first_request_completed_ns", 0) or 0),
                "all_requests_completed_ns": int(summary_row.get("all_requests_completed_ns", 0) or 0),
            },
            first_transport_submit_ns=int(summary_row.get("first_transport_submit_ns", 0) or 0),
            last_transport_complete_ns=int(summary_row.get("last_transport_complete_ns", 0) or 0),
            p0_first_submit_ns=int(summary_row.get("p0_first_submit_ns", 0) or 0),
            p0_last_complete_ns=int(summary_row.get("p0_last_complete_ns", 0) or 0),
            p1_first_submit_ns=int(summary_row.get("p1_first_submit_ns", 0) or 0),
            p1_last_complete_ns=int(summary_row.get("p1_last_complete_ns", 0) or 0),
            raw_summary=result.summary.to_dict(),
            execution_entries=tuple(result.execution_entries),
            failure_code=str(result.failure_code),
            session_poisoned=bool(result.session_poisoned),
            blocked_release_tokens=tuple(str(value) for value in result.blocked_release_tokens),
        )


def execute_transport(request: ExecutionRequest, *, backend: str) -> ExecutionResult:
    if backend == "phase_sync":
        return PhaseSyncTransportExecutor().execute(request)
    if backend == "async_release":
        return AsyncReleaseTransportExecutor().execute(request)
    raise ValueError(f"unsupported transport backend {backend!r}")


__all__ = [
    "AsyncReleaseTransportExecutor",
    "ExecutionRequest",
    "ExecutionResult",
    "PhaseSyncTransportExecutor",
    "TransportExecutor",
    "execute_transport",
]
