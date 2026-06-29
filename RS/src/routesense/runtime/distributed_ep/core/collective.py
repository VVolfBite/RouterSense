from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CollectiveRecord:
    op_name: str
    send_bytes: int
    recv_bytes: int
    async_op: bool


class CollectiveOps:
    """Model-agnostic NCCL collective wrapper."""

    def __init__(self) -> None:
        self.records: list[CollectiveRecord] = []

    def dispatch(
        self,
        payload: Any,
        *,
        send_counts: list[int],
        recv_counts: list[int],
        async_op: bool = False,
    ) -> Any:
        self.records.append(
            CollectiveRecord(
                op_name="dispatch",
                send_bytes=sum(send_counts),
                recv_bytes=sum(recv_counts),
                async_op=async_op,
            )
        )
        return payload

    def return_results(
        self,
        payload: Any,
        *,
        send_counts: list[int],
        recv_counts: list[int],
        async_op: bool = False,
    ) -> Any:
        self.records.append(
            CollectiveRecord(
                op_name="return",
                send_bytes=sum(send_counts),
                recv_bytes=sum(recv_counts),
                async_op=async_op,
            )
        )
        return payload
