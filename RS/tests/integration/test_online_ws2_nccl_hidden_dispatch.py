from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rs.contracts import TraceOrigin
from rs.online.distributed_runtime import assert_distinct_cuda_device_mapping
from rs.online.olmoe_ep import (
    build_online_expert_placement,
    build_online_route_partition,
    build_request_identity_tables,
    build_request_protocol_hash,
    build_rank_manifest,
    execute_ws2_hidden_dispatch_only,
    run_distributed_count_agreement,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _hidden_states(rank: int, device: torch.device) -> torch.Tensor:
    base = 100.0 if rank == 0 else 200.0
    return torch.tensor([[base + 1.0, base + 2.0], [base + 3.0, base + 4.0]], dtype=torch.float16, device=device)


def _router_logits(rank: int, device: torch.device) -> torch.Tensor:
    if rank == 0:
        values = [[9.0, 8.0, 1.0, 0.0], [0.0, 1.0, 7.0, 6.0]]
    else:
        values = [[8.0, 9.0, 0.0, 1.0], [7.0, 6.0, 0.0, 1.0]]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _nccl_worker(rank: int, port: int, out_dir: str) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        mapped = assert_distinct_cuda_device_mapping(backend="nccl", rank_device=device, world_size=2)
        prompts_by_rank = ["rank0 prompt", "rank1 prompt"]
        request_id_table, microbatch_id_table, request_table_hash = build_request_identity_tables(
            prompts_by_rank=prompts_by_rank,
        )
        placement = build_online_expert_placement(world_size=2, expert_count=4, rank_to_node_id=[0, 0])
        partition = build_online_route_partition(
            run_id="nccl-run",
            request_id=request_id_table[rank],
            microbatch_id=microbatch_id_table[0],
            request_numeric_id=rank,
            microbatch_numeric_id=0,
            layer_id=0,
            source_rank=rank,
            source_node_id=0,
            hidden_states=_hidden_states(rank, device),
            router_logits=_router_logits(rank, device),
            placement=placement,
            top_k=2,
            trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_ROUTE_PARTITION,
        )
        manifest = build_rank_manifest(
            partition=partition,
            placement=placement,
            prompt_text=prompts_by_rank[rank],
            request_protocol_hash=build_request_protocol_hash(
                prompts_by_rank=prompts_by_rank,
                microbatch_id=microbatch_id_table[0],
                layer_id=0,
            ),
            request_table_hash=request_table_hash,
        )
        agreement = run_distributed_count_agreement(
            partition=partition,
            manifest=manifest,
            placement=placement,
            validate_metadata=True,
            rank_device=device,
        )
        dispatch = execute_ws2_hidden_dispatch_only(
            hidden_states=_hidden_states(rank, device),
            partition=partition,
            manifest=manifest,
            placement=placement,
            agreement=agreement,
            request_id_table=request_id_table,
            microbatch_id_table=microbatch_id_table,
        )
        payload = {
            "rank": rank,
            "device": str(device),
            "mapped": mapped,
            "correctness_status": dispatch.validation.correctness_status,
            "transport": dispatch.transport_record.to_dict(),
            "received_route_count": len(dispatch.received_routes),
        }
        Path(out_dir, f"nccl-rank-{rank}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    (not torch.cuda.is_available()) or torch.cuda.device_count() < 2 or (not dist.is_nccl_available()),
    reason="requires 2 visible CUDA devices and NCCL support",
)
def test_online_ws2_nccl_hidden_dispatch(tmp_path) -> None:
    out_dir = tmp_path / "nccl"
    out_dir.mkdir()
    mp.spawn(_nccl_worker, args=(_free_port(), str(out_dir)), nprocs=2, join=True)
    rank0 = json.loads((out_dir / "nccl-rank-0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "nccl-rank-1.json").read_text(encoding="utf-8"))
    assert rank0["device"] == "cuda:0"
    assert rank1["device"] == "cuda:1"
    assert rank0["mapped"] == [0, 1]
    assert rank1["mapped"] == [0, 1]
    assert rank0["correctness_status"] == "metadata_passed"
    assert rank1["correctness_status"] == "metadata_passed"
    assert rank0["transport"]["backend"] == "nccl"
    assert rank1["transport"]["backend"] == "nccl"
    assert rank0["transport"]["verified_backend"] == "nccl_gpu"
    assert rank1["transport"]["verified_backend"] == "nccl_gpu"
    assert rank0["transport"]["hidden_payload_transferred"] is True
    assert rank1["transport"]["hidden_payload_transferred"] is True
