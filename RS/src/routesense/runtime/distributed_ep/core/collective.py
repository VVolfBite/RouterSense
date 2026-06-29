from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .manifest import DispatchPlan


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
