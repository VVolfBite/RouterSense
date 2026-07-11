from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
import torch.distributed as dist

from rs.runtime.online.megatron_ep.execution.async_p2p_executor import execute_async_phase_tensor
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


@dataclass(frozen=True)
class ExecutionResult:
    output_tensor: torch.Tensor
    backend_id: str
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
        return ExecutionResult(
            output_tensor=output,
            backend_id=self.backend_id,
            execution_plan_digest=str(request.execution_plan.plan_hash),
            send_op_count=0,
            recv_op_count=0,
            local_copy_task_count=int(summary.local_copy_tasks),
            local_copy_row_count=int(summary.local_copy_rows),
            enqueue_us=0.0,
            wait_us=0.0,
            total_us=(end_ns - start_ns) / 1000.0,
            fallback_used=False,
            timeout=False,
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
        return ExecutionResult(
            output_tensor=result.output,
            backend_id=self.backend_id,
            execution_plan_digest=str(request.execution_plan.plan_hash),
            send_op_count=int(summary_row.get("send_op_count", 0) or 0),
            recv_op_count=int(summary_row.get("recv_op_count", 0) or 0),
            local_copy_task_count=int(summary_row.get("local_copy_task_count", 0) or 0),
            local_copy_row_count=int(summary_row.get("local_copy_row_count", 0) or 0),
            enqueue_us=float(summary_row.get("enqueue_us", 0.0) or 0.0),
            wait_us=float(summary_row.get("wait_us", 0.0) or 0.0),
            total_us=float(summary_row.get("total_us", 0.0) or 0.0),
            fallback_used=False,
            timeout=False,
            raw_summary=result.summary.to_dict(),
            execution_entries=tuple(result.execution_entries),
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
