from __future__ import annotations

import pytest
import torch

from rs.online.distributed_runtime import collective_device_for_backend


def test_collective_device_uses_cuda_for_nccl() -> None:
    device = collective_device_for_backend("nccl", torch.device("cuda:1"))
    assert str(device) == "cuda:1"


def test_collective_device_uses_cpu_for_gloo() -> None:
    device = collective_device_for_backend("gloo", torch.device("cuda:0"))
    assert str(device) == "cpu"


def test_collective_device_rejects_cpu_rank_device_for_nccl() -> None:
    with pytest.raises(RuntimeError, match="requires CUDA rank device"):
        collective_device_for_backend("nccl", torch.device("cpu"))
