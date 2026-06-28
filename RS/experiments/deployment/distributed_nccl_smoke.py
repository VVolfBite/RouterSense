#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _snapshot() -> dict[str, object]:
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    torch.cuda.set_device(device)
    results = {"rank": rank, "world_size": world_size, "device": str(device), "hostname": socket.gethostname()}

    def run_all_reduce(size: int) -> dict[str, object]:
        tensor = torch.arange(size, device=device, dtype=torch.float32) + rank
        expected = tensor.clone()
        dist.all_reduce(tensor)
        expected = expected * world_size
        return {"size": size, "ok": bool(torch.allclose(tensor, expected))}

    def run_all_gather(size: int) -> dict[str, object]:
        tensor = torch.arange(size, device=device, dtype=torch.float32) + rank
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        ok = len(gathered) == world_size and all(chunk.shape == tensor.shape for chunk in gathered)
        return {"size": size, "ok": ok}

    def run_all_to_all(size: int) -> dict[str, object]:
        send = torch.arange(size * world_size, device=device, dtype=torch.float32).reshape(world_size, size) + rank * 1000
        recv = torch.zeros_like(send)
        dist.all_to_all_single(recv, send)
        ok = bool(recv.shape == send.shape)
        return {"size": size, "ok": ok}

    results["all_reduce"] = [run_all_reduce(256), run_all_reduce(16384), run_all_reduce(262144)]
    results["all_gather"] = [run_all_gather(256), run_all_gather(16384), run_all_gather(262144)]
    results["all_to_all_single"] = [run_all_to_all(256), run_all_to_all(16384), run_all_to_all(262144)]
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real 4-rank NCCL smoke.")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts" / "deployment" / "phase0c_nccl_smoke"))
    args = parser.parse_args(argv)
    import torch.distributed as dist  # type: ignore

    dist.init_process_group(backend="nccl")
    try:
        payload = _snapshot()
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "nccl_smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
