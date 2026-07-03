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
    cuda_props = torch.cuda.get_device_properties(device)
    nccl_version = getattr(torch.cuda.nccl, "version", lambda: None)()
    results = {
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "hostname": socket.gethostname(),
        "gpu": {
            "name": cuda_props.name,
            "total_memory_gb": round(cuda_props.total_memory / (1024**3), 2),
            "multi_processor_count": int(cuda_props.multi_processor_count),
        },
        "nccl_version": nccl_version,
    }

    def run_all_reduce(size: int) -> dict[str, object]:
        tensor = torch.arange(size, device=device, dtype=torch.float32) + rank
        dist.all_reduce(tensor)
        expected = torch.arange(size, device=device, dtype=torch.float32) * world_size + sum(range(world_size))
        return {"size": size, "ok": bool(torch.allclose(tensor, expected))}

    def run_all_gather(size: int) -> dict[str, object]:
        tensor = torch.arange(size, device=device, dtype=torch.float32) + rank
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        ok = len(gathered) == world_size and all(
            torch.allclose(chunk, torch.arange(size, device=device, dtype=torch.float32) + source_rank)
            for source_rank, chunk in enumerate(gathered)
        )
        return {"size": size, "ok": ok}

    def run_all_to_all(size: int) -> dict[str, object]:
        send = torch.arange(size * world_size, device=device, dtype=torch.float32).reshape(world_size, size) + rank * 1000
        recv = torch.zeros_like(send)
        dist.all_to_all_single(recv, send)
        expected = torch.stack(
            [
                torch.arange(size, device=device, dtype=torch.float32) + source_rank * 1000 + rank * size
                for source_rank in range(world_size)
            ],
            dim=0,
        )
        ok = bool(recv.shape == send.shape and torch.allclose(recv, expected))
        return {"size": size, "ok": ok}

    def run_all_to_all_asymmetric() -> dict[str, object]:
        send_counts = [rank + peer + 1 for peer in range(world_size)]
        send_chunks = [
            torch.full((count,), rank * 100 + peer, dtype=torch.float32, device=device)
            for peer, count in enumerate(send_counts)
        ]
        send_tensor = torch.cat(send_chunks) if send_chunks else torch.empty((0,), dtype=torch.float32, device=device)
        recv_counts = [peer + rank + 1 for peer in range(world_size)]
        recv_tensor = torch.empty(sum(recv_counts), dtype=torch.float32, device=device)
        dist.all_to_all_single(
            recv_tensor,
            send_tensor,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
        )
        expected = []
        for src in range(world_size):
            count = src + rank + 1
            expected.extend([src * 100 + rank] * count)
        expected_tensor = torch.tensor(expected, dtype=torch.float32, device=device)
        ok = bool(recv_tensor.shape == expected_tensor.shape and torch.allclose(recv_tensor, expected_tensor))
        return {
            "send_counts": send_counts,
            "recv_counts": recv_counts,
            "ok": ok,
        }

    results["all_reduce"] = [run_all_reduce(256), run_all_reduce(16384), run_all_reduce(262144)]
    results["all_gather"] = [run_all_gather(256), run_all_gather(16384), run_all_gather(262144)]
    results["all_to_all_single"] = [run_all_to_all(256), run_all_to_all(16384), run_all_to_all(262144)]
    results["all_to_all_single_asymmetric"] = run_all_to_all_asymmetric()
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
