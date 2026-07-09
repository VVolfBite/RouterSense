"""Tensorized traffic-matrix helpers for online dispatch/prediction paths.

This module is intentionally limited to tensor/NCCL-friendly collectives.
It must not use Python object collectives in the online hot/control path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TrafficMatrixBundle:
    matrix: tuple[tuple[int, ...], ...]
    matrix_source: str
    is_global: bool
    world_size: int
    dtype: str
    device_type: str
    gather_time_us: float
    gather_call_count: int
    row_sums: tuple[int, ...]
    col_sums: tuple[int, ...]
    total_bytes: int
    nonzero_edge_count: int
    shape: tuple[int, int]
    matrix_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_source": self.matrix_source,
            "is_global": self.is_global,
            "world_size": self.world_size,
            "dtype": self.dtype,
            "device_type": self.device_type,
            "gather_time_us": self.gather_time_us,
            "gather_call_count": self.gather_call_count,
            "row_sums": list(self.row_sums),
            "col_sums": list(self.col_sums),
            "total_bytes": self.total_bytes,
            "nonzero_edge_count": self.nonzero_edge_count,
            "shape": list(self.shape),
            "matrix_digest": self.matrix_digest,
        }


def build_local_peer_bytes_tensor(
    per_peer_bytes: tuple[int, ...] | list[int],
    world_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    values = [int(value) for value in per_peer_bytes]
    if len(values) < int(world_size):
        values.extend([0] * (int(world_size) - len(values)))
    elif len(values) > int(world_size):
        values = values[: int(world_size)]
    return torch.tensor(values, dtype=torch.int64, device=torch.device(device))


def gather_global_peer_bytes_matrix(
    local_peer_bytes: torch.Tensor,
    *,
    group: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if local_peer_bytes.ndim != 1:
        raise ValueError("local_peer_bytes must be rank-1")
    world_size = int(local_peer_bytes.numel())
    gather_call_count = 0
    gather_start_ns = time.monotonic_ns()

    if world_size <= 1:
        matrix = local_peer_bytes.reshape(1, world_size)
        source = "single_rank_fallback"
        is_global = False
    elif not dist.is_available() or not dist.is_initialized():
        matrix = _replicated_local_row(local_peer_bytes)
        source = "replicated_local_row_fallback"
        is_global = False
    else:
        gathered = _all_gather_matrix(local_peer_bytes=local_peer_bytes, world_size=world_size, group=group)
        matrix = gathered
        source = "tensor_all_gather"
        is_global = True
        gather_call_count = 1

    gather_end_ns = time.monotonic_ns()
    row_sums = tuple(int(value) for value in matrix.sum(dim=1).detach().cpu().tolist())
    col_sums = tuple(int(value) for value in matrix.sum(dim=0).detach().cpu().tolist())
    total_bytes = int(sum(row_sums))
    nonzero_edge_count = int(((matrix > 0) & ~torch.eye(matrix.shape[0], matrix.shape[1], device=matrix.device, dtype=torch.bool)).sum().item())
    metadata = {
        "matrix_source": source,
        "is_global": is_global,
        "world_size": int(matrix.shape[0]),
        "dtype": str(matrix.dtype),
        "device_type": str(matrix.device.type),
        "gather_time_us": max(0.0, float(gather_end_ns - gather_start_ns) / 1000.0),
        "gather_call_count": gather_call_count,
        "row_sums": row_sums,
        "col_sums": col_sums,
        "total_bytes": total_bytes,
        "nonzero_edge_count": nonzero_edge_count,
    }
    return matrix, metadata


def build_traffic_matrix_bundle(
    *,
    per_peer_bytes: tuple[int, ...] | list[int],
    world_size: int,
    device: torch.device | str,
    group: Any | None = None,
) -> TrafficMatrixBundle:
    local_tensor = build_local_peer_bytes_tensor(per_peer_bytes, world_size, device)
    matrix_tensor, metadata = gather_global_peer_bytes_matrix(local_tensor, group=group)
    matrix = tuple(
        tuple(int(value) for value in row)
        for row in matrix_tensor.detach().cpu().tolist()
    )
    return TrafficMatrixBundle(
        matrix=matrix,
        matrix_source=str(metadata["matrix_source"]),
        is_global=bool(metadata["is_global"]),
        world_size=int(metadata["world_size"]),
        dtype=str(metadata["dtype"]),
        device_type=str(metadata["device_type"]),
        gather_time_us=float(metadata["gather_time_us"]),
        gather_call_count=int(metadata["gather_call_count"]),
        row_sums=tuple(int(value) for value in metadata["row_sums"]),
        col_sums=tuple(int(value) for value in metadata["col_sums"]),
        total_bytes=int(metadata["total_bytes"]),
        nonzero_edge_count=int(metadata["nonzero_edge_count"]),
        shape=(int(matrix_tensor.shape[0]), int(matrix_tensor.shape[1])),
        matrix_digest=_matrix_digest(matrix),
    )


def _all_gather_matrix(
    *,
    local_peer_bytes: torch.Tensor,
    world_size: int,
    group: Any | None,
) -> torch.Tensor:
    if hasattr(dist, "all_gather_into_tensor"):
        flat = torch.empty(world_size * world_size, dtype=local_peer_bytes.dtype, device=local_peer_bytes.device)
        dist.all_gather_into_tensor(flat, local_peer_bytes.contiguous(), group=group)
        return flat.view(world_size, world_size)
    gathered = [torch.empty_like(local_peer_bytes) for _ in range(world_size)]
    dist.all_gather(gathered, local_peer_bytes.contiguous(), group=group)
    return torch.stack(gathered, dim=0)


def _replicated_local_row(local_peer_bytes: torch.Tensor) -> torch.Tensor:
    world_size = int(local_peer_bytes.numel())
    matrix = local_peer_bytes.unsqueeze(0).repeat(world_size, 1)
    diag = torch.arange(world_size, device=local_peer_bytes.device)
    matrix[diag, diag] = 0
    return matrix


def _matrix_digest(matrix: tuple[tuple[int, ...], ...]) -> str:
    payload = "|".join(",".join(str(int(value)) for value in row) for row in matrix)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
