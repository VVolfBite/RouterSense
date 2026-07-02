from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NCCLOpRecord:
    op: str
    peer_rank: int
    tensor_size: int
    start_us: float
    end_us: float
    duration_us: float


@dataclass
class NCCLExecutionResult:
    ops: list[NCCLOpRecord] = field(default_factory=list)
    total_wall_time_us: float = 0.0
    phase: int = 0
    direction: str = "dispatch"


class NCCLExecutor:
    """Execute point-to-point NCCL ops following a given chunk order."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        dtype=None,
        device: str | Any | None = None,
    ) -> None:
        import os
        import torch  # type: ignore

        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype if dtype is not None else torch.float16
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

    def execute_phase(
        self,
        send_chunks: list[tuple[int, int]],
        recv_chunks: list[tuple[int, int]],
        *,
        hidden_size: int,
        phase: int = 0,
        direction: str = "dispatch",
        payload=None,
    ) -> NCCLExecutionResult:
        import torch  # type: ignore
        import torch.distributed as dist  # type: ignore

        ops: list[NCCLOpRecord] = []
        t0 = time.perf_counter()

        recv_buffers: dict[int, torch.Tensor] = {}
        recv_handles: list[tuple[int, int, Any]] = []
        for src_rank, num_rows in recv_chunks:
            buffer = torch.empty(num_rows * hidden_size, dtype=self.dtype, device=self.device)
            recv_buffers[src_rank] = buffer
            handle = dist.irecv(buffer, src=src_rank)
            recv_handles.append((src_rank, num_rows, handle))

        send_handles: list[tuple[int, int, Any, Any, float]] = []
        send_offset = 0
        for dst_rank, num_rows in send_chunks:
            elements = num_rows * hidden_size
            if payload is not None:
                send_tensor = payload[send_offset : send_offset + elements].contiguous()
                send_offset += elements
            else:
                send_tensor = torch.ones(elements, dtype=self.dtype, device=self.device)
            handle = dist.isend(send_tensor, dst=dst_rank)
            send_handles.append((dst_rank, num_rows, send_tensor, handle, time.perf_counter() * 1e6))

        for src_rank, num_rows, handle in recv_handles:
            start_us = time.perf_counter() * 1e6
            handle.wait()
            end_us = time.perf_counter() * 1e6
            ops.append(
                NCCLOpRecord(
                    op="recv",
                    peer_rank=src_rank,
                    tensor_size=num_rows * hidden_size,
                    start_us=start_us,
                    end_us=end_us,
                    duration_us=end_us - start_us,
                )
            )

        for dst_rank, num_rows, _send_tensor, handle, start_us in send_handles:
            handle.wait()
            end_us = time.perf_counter() * 1e6
            ops.append(
                NCCLOpRecord(
                    op="send",
                    peer_rank=dst_rank,
                    tensor_size=num_rows * hidden_size,
                    start_us=start_us,
                    end_us=end_us,
                    duration_us=end_us - start_us,
                )
            )

        total_us = (time.perf_counter() - t0) * 1e6
        return NCCLExecutionResult(
            ops=ops,
            total_wall_time_us=total_us,
            phase=phase,
            direction=direction,
        )
