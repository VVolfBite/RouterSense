from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .manifest import DispatchPlan
from .nccl_executor import NCCLExecutionResult, NCCLExecutor


@dataclass
class CollectiveRecord:
    op_name: str
    layer_id: int
    rank: int
    send_counts: list[int]
    recv_counts: list[int]
    send_bytes: int
    recv_bytes: int
    async_op: bool


class CollectiveOps:
    """Model-agnostic NCCL collective wrapper and accounting surface."""

    def __init__(self, bytes_per_row: int = 0) -> None:
        self.bytes_per_row = bytes_per_row
        self.records: list[CollectiveRecord] = []

    def dispatch(
        self,
        payload: Any,
        *,
        plan: DispatchPlan,
        rank: int,
        async_op: bool = False,
    ) -> Any:
        send_counts = plan.send_counts_for_rank(rank)
        recv_counts = plan.recv_counts_for_rank(rank)
        self.records.append(
            CollectiveRecord(
                op_name="dispatch",
                layer_id=plan.layer_id,
                rank=rank,
                send_counts=send_counts,
                recv_counts=recv_counts,
                send_bytes=sum(send_counts) * self.bytes_per_row,
                recv_bytes=sum(recv_counts) * self.bytes_per_row,
                async_op=async_op,
            )
        )
        return payload

    def return_results(
        self,
        payload: Any,
        *,
        plan: DispatchPlan,
        rank: int,
        async_op: bool = False,
    ) -> Any:
        send_counts = plan.recv_counts_for_rank(rank)
        recv_counts = plan.send_counts_for_rank(rank)
        self.records.append(
            CollectiveRecord(
                op_name="return",
                layer_id=plan.layer_id,
                rank=rank,
                send_counts=send_counts,
                recv_counts=recv_counts,
                send_bytes=sum(send_counts) * self.bytes_per_row,
                recv_bytes=sum(recv_counts) * self.bytes_per_row,
                async_op=async_op,
            )
        )
        return payload

    def execute_scheduled_phase(
        self,
        *,
        plan: DispatchPlan,
        rank: int,
        schedule: list[dict[str, Any]],
        phase: int,
        direction: str,
        executor: NCCLExecutor,
        hidden_size: int,
        payload: Any = None,
    ) -> NCCLExecutionResult:
        """Execute one scheduled phase for the current rank."""

        my_sends = [
            (int(chunk["dst_gpu"]), int(chunk["size"]))
            for chunk in schedule
            if int(chunk.get("phase", phase)) == phase and int(chunk["src_gpu"]) == rank
        ]
        my_recvs = [
            (int(chunk["src_gpu"]), int(chunk["size"]))
            for chunk in schedule
            if int(chunk.get("phase", phase)) == phase and int(chunk["dst_gpu"]) == rank
        ]
        result = executor.execute_phase(
            send_chunks=my_sends,
            recv_chunks=my_recvs,
            hidden_size=hidden_size,
            phase=phase,
            direction=direction,
            payload=payload,
        )
        self.records.append(
            CollectiveRecord(
                op_name=f"scheduled_{direction}",
                layer_id=plan.layer_id,
                rank=rank,
                send_counts=[size for _peer, size in my_sends],
                recv_counts=[size for _peer, size in my_recvs],
                send_bytes=sum(size for _peer, size in my_sends) * self.bytes_per_row,
                recv_bytes=sum(size for _peer, size in my_recvs) * self.bytes_per_row,
                async_op=True,
            )
        )
        return result
