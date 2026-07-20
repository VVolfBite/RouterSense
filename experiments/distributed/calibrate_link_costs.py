#!/usr/bin/env python3
from __future__ import annotations

"""Measure rank-pair GPU transfer costs and emit a planner cost profile.

Run only under torchrun.  Every rank follows the same pair/size schedule, so a
failed measurement aborts instead of leaving a partially valid profile.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from rs.topology import fit_affine_row_cost, infer_model_row_contract, write_link_cost_profile


def _parse_int_csv(value: str) -> tuple[int, ...]:
    rows = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not rows or min(rows) <= 0:
        raise argparse.ArgumentTypeError("row sizes must be positive integers")
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--row-bytes", type=int, default=0)
    parser.add_argument("--precision-bytes", type=int, default=2)
    parser.add_argument("--rows", type=_parse_int_csv, default=(1, 16, 64, 256))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args(argv)



def _measure_direction(
    *,
    src: int,
    dst: int,
    size_bytes: int,
    warmup: int,
    repeats: int,
    rank: int,
    device: torch.device,
) -> float | None:
    send_tensor = torch.empty(size_bytes, dtype=torch.uint8, device=device) if rank == src else None
    recv_tensor = torch.empty(size_bytes, dtype=torch.uint8, device=device) if rank == dst else None
    durations: list[float] = []
    for iteration in range(int(warmup) + int(repeats)):
        dist.barrier()
        if rank == src:
            torch.cuda.synchronize(device)
            started_ns = time.perf_counter_ns()
            work = dist.isend(send_tensor, dst=dst)
            work.wait()
            torch.cuda.synchronize(device)
            elapsed_us = float(time.perf_counter_ns() - started_ns) / 1000.0
            if iteration >= int(warmup):
                durations.append(elapsed_us)
        elif rank == dst:
            work = dist.irecv(recv_tensor, src=src)
            work.wait()
            torch.cuda.synchronize(device)
        dist.barrier()
    if rank != src:
        return None
    return float(statistics.median(durations))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for link calibration")
    dist.init_process_group(backend="nccl")
    try:
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        if local_world_size <= 0 or world_size % local_world_size != 0:
            raise RuntimeError("uniform LOCAL_WORLD_SIZE is required")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if int(args.row_bytes) > 0:
            model_contract = {
                "model_path": str(Path(args.model_path).expanduser().resolve()) if args.model_path else "",
                "config_path": "",
                "config_sha256": "",
                "hidden_size": 0,
                "element_bytes": int(args.precision_bytes),
                "row_bytes": int(args.row_bytes),
            }
        else:
            model_contract = infer_model_row_contract(
                str(args.model_path),
                element_bytes=int(args.precision_bytes),
            )
        row_bytes = int(model_contract["row_bytes"])
        row_counts = tuple(int(value) for value in args.rows)
        raw: dict[str, list[dict[str, float | int]]] = {}
        pair_fits: dict[str, tuple[float, float]] = {}
        for src in range(world_size):
            for dst in range(world_size):
                if src == dst:
                    continue
                samples: list[tuple[int, float]] = []
                for rows in row_counts:
                    size_bytes = int(rows) * int(row_bytes)
                    local_duration = _measure_direction(
                        src=src,
                        dst=dst,
                        size_bytes=size_bytes,
                        warmup=int(args.warmup),
                        repeats=int(args.repeats),
                        rank=rank,
                        device=device,
                    )
                    holder = torch.tensor(
                        [float(local_duration or 0.0)],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.broadcast(holder, src=src)
                    duration = float(holder.item())
                    samples.append((size_bytes, duration))
                key = f"{src}->{dst}"
                raw[key] = [
                    {"rows": int(size // row_bytes), "bytes": int(size), "median_us": float(duration)}
                    for size, duration in samples
                ]
                pair_fits[key] = fit_affine_row_cost(samples, row_bytes=row_bytes)

        slopes = [value[0] for value in pair_fits.values()]
        intercepts = [value[1] for value in pair_fits.values()]
        default_slope = float(statistics.median(slopes)) if slopes else 1.0
        default_intercept = float(statistics.median(intercepts)) if intercepts else 0.0
        slope_matrix = [[default_slope for _ in range(world_size)] for _ in range(world_size)]
        intercept_matrix = [[default_intercept for _ in range(world_size)] for _ in range(world_size)]
        for src in range(world_size):
            for dst in range(world_size):
                if src == dst:
                    continue
                slope_matrix[src][dst], intercept_matrix[src][dst] = pair_fits[f"{src}->{dst}"]
        rank_to_node = [rank_id // local_world_size for rank_id in range(world_size)]
        if rank == 0:
            output = Path(args.output)
            profile = write_link_cost_profile(
                output,
                {
                    "world_size": world_size,
                    "ranks_per_node": local_world_size,
                    "rank_to_node": rank_to_node,
                    "row_bytes": row_bytes,
                    "edge_slope_us_per_row": slope_matrix,
                    "edge_intercept_us": intercept_matrix,
                    "wave_launch_us": 0.0,
                    "source": "cuda_nccl_pairwise_warmup",
                    "metadata": {
                        "rows": list(row_counts),
                        "warmup": int(args.warmup),
                        "repeats": int(args.repeats),
                        "raw_pair_measurements": raw,
                        "model_contract": model_contract,
                        "wave_launch_semantics": "not_separately_measured_zero",
                        "torch_version": str(torch.__version__),
                        "cuda_version": str(torch.version.cuda),
                    },
                },
            )
            print(json.dumps({"status": "PASS", "output": str(output), "profile": profile.to_dict()}, indent=2))
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
