from __future__ import annotations

import pytest
import torch

from rs.online.distributed_runtime import resolve_distributed_device


def test_nccl_world_size_two_uses_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    device = resolve_distributed_device(requested_device_index=None, world_size=2, backend="nccl")
    assert str(device) == "cuda:1"


def test_nccl_rejects_explicit_device_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="does not match LOCAL_RANK"):
        resolve_distributed_device(requested_device_index=0, world_size=2, backend="nccl")


def test_nccl_rejects_world_size_exceeding_gpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="exceeds visible CUDA device count"):
        resolve_distributed_device(requested_device_index=None, world_size=2, backend="nccl")


def test_ws2_nccl_device_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    device = resolve_distributed_device(requested_device_index=0, world_size=2, backend="nccl")
    assert str(device) == "cuda:0"
