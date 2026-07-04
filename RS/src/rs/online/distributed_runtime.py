from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedDeviceInfo:
    rank: int
    local_rank: int | None
    world_size: int
    backend: str
    hostname: str
    cuda_device_count: int
    cuda_device_index: int | None
    cuda_device_name: str | None
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributedStageStatus:
    rank: int
    stage: str
    ok: bool
    backend: str
    device: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DistributedStageError(RuntimeError):
    def __init__(self, *, stage: str, failures: list[dict[str, Any]]):
        self.stage = stage
        self.failures = failures
        summary = "; ".join(
            f"rank={item['rank']} backend={item['backend']} device={item['device']} error={item.get('error')}"
            for item in failures
        )
        super().__init__(f"distributed stage '{stage}' failed: {summary}")


def resolve_backend(*, world_size: int, requested_backend: str | None) -> str:
    if requested_backend:
        backend = str(requested_backend).lower()
        if backend not in {"nccl", "gloo"}:
            raise RuntimeError(f"unsupported backend {requested_backend!r}; expected 'nccl' or 'gloo'")
        return backend
    if int(world_size) > 1 and torch.cuda.is_available():
        return "nccl"
    return "gloo"


def require_ws2_native_ep_backend(
    *,
    world_size: int,
    backend: str,
    route_partition_only: bool = False,
    hidden_dispatch_only: bool = False,
) -> None:
    if int(world_size) != 2:
        return
    if route_partition_only or hidden_dispatch_only:
        return
    if str(backend).lower() != "nccl":
        raise RuntimeError(
            "WS=2 native MoE-layer harness requires NCCL and two CUDA devices. "
            "Gloo is supported only for CPU protocol tests."
        )
    if not torch.cuda.is_available() or int(torch.cuda.device_count()) < 2:
        raise RuntimeError(
            "WS=2 native MoE-layer harness requires NCCL and two CUDA devices. "
            "Gloo is supported only for CPU protocol tests."
        )


def distributed_timeout() -> timedelta:
    seconds = int(os.environ.get("RS_DIST_TIMEOUT_SECONDS", "120"))
    return timedelta(seconds=max(seconds, 30))


def init_process_group_checked(*, backend: str) -> None:
    if dist.is_initialized():
        return
    dist.init_process_group(backend=backend, timeout=distributed_timeout())


def destroy_process_group_checked() -> None:
    if not dist.is_initialized():
        return
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def run_distributed_stage(
    stage: str,
    *,
    backend: str,
    rank_device: torch.device,
    fn,
):
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before distributed stage execution")
    status = DistributedStageStatus(
        rank=int(dist.get_rank()),
        stage=str(stage),
        ok=False,
        backend=str(backend).lower(),
        device=str(rank_device),
        error=None,
    )
    result = None
    try:
        result = fn()
        status = DistributedStageStatus(
            rank=int(dist.get_rank()),
            stage=str(stage),
            ok=True,
            backend=str(backend).lower(),
            device=str(rank_device),
            error=None,
        )
    except Exception as exc:
        status = DistributedStageStatus(
            rank=int(dist.get_rank()),
            stage=str(stage),
            ok=False,
            backend=str(backend).lower(),
            device=str(rank_device),
            error=f"{type(exc).__name__}: {exc}",
        )
    gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, status.to_dict())
    failures = [item for item in gathered if item and not bool(item.get("ok"))]
    if failures:
        raise DistributedStageError(stage=str(stage), failures=failures)
    return result


def resolve_distributed_device(
    requested_device_index: int | None,
    world_size: int,
    backend: str,
) -> torch.device:
    backend = str(backend).lower()
    if int(world_size) == 1:
        if requested_device_index is None:
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(requested_device_index)}")
        if int(requested_device_index) != 0:
            raise RuntimeError("CUDA unavailable; only device-index 0 is valid for CPU mode")
        return torch.device("cpu")

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("backend=nccl requires CUDA availability")
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is None:
            raise RuntimeError("backend=nccl with world_size>1 requires LOCAL_RANK")
        local_rank = int(local_rank_env)
        cuda_device_count = int(torch.cuda.device_count())
        if int(world_size) > cuda_device_count:
            raise RuntimeError(
                f"world_size {world_size} exceeds visible CUDA device count {cuda_device_count}"
            )
        if requested_device_index is not None and int(requested_device_index) != local_rank:
            raise RuntimeError(
                f"requested device index {requested_device_index} does not match LOCAL_RANK {local_rank}"
            )
        return torch.device(f"cuda:{local_rank}")

    if requested_device_index is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{int(requested_device_index)}")
    return torch.device("cpu")


def collective_device_for_backend(backend: str, rank_device: torch.device) -> torch.device:
    backend = str(backend).lower()
    if backend == "nccl":
        if rank_device.type != "cuda":
            raise RuntimeError("backend=nccl requires CUDA rank device for collectives")
        return rank_device
    return torch.device("cpu")


def capture_distributed_device_info(
    *,
    rank: int,
    world_size: int,
    backend: str,
    rank_device: torch.device,
) -> DistributedDeviceInfo:
    local_rank_env = os.environ.get("LOCAL_RANK")
    local_rank = int(local_rank_env) if local_rank_env is not None else None
    cuda_device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    cuda_device_index: int | None = None
    cuda_device_name: str | None = None
    if rank_device.type == "cuda":
        cuda_device_index = int(rank_device.index if rank_device.index is not None else 0)
        cuda_device_name = str(torch.cuda.get_device_name(cuda_device_index))
    return DistributedDeviceInfo(
        rank=int(rank),
        local_rank=local_rank,
        world_size=int(world_size),
        backend=str(backend).lower(),
        hostname=str(socket.gethostname()),
        cuda_device_count=cuda_device_count,
        cuda_device_index=cuda_device_index,
        cuda_device_name=cuda_device_name,
        device=str(rank_device),
    )


def assert_distinct_cuda_device_mapping(
    *,
    backend: str,
    rank_device: torch.device,
    world_size: int,
) -> list[int]:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before verifying CUDA device mapping")
    if str(backend).lower() != "nccl":
        return []
    if rank_device.type != "cuda":
        raise RuntimeError("backend=nccl requires CUDA rank device before verifying mapping")
    collective_device = collective_device_for_backend(backend, rank_device)
    local_index = int(rank_device.index if rank_device.index is not None else 0)
    local_tensor = torch.tensor([local_index], dtype=torch.int64, device=collective_device)
    gathered = [torch.empty_like(local_tensor) for _ in range(int(world_size))]
    dist.all_gather(gathered, local_tensor)
    mapped = [int(item.item()) for item in gathered]
    if len(set(mapped)) != len(mapped):
        raise RuntimeError(f"multiple ranks mapped to the same CUDA device: {mapped}")
    return mapped
