from __future__ import annotations

import json
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rs.contracts import TraceOrigin
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


def _hidden_states(rank: int) -> torch.Tensor:
    base = 5.0 if rank == 0 else 10.0
    return torch.tensor([[base, base + 1.0], [base + 2.0, base + 3.0]], dtype=torch.float32)


def _router_logits(rank: int) -> torch.Tensor:
    if rank == 0:
        return torch.tensor([[9.0, 8.0, 1.0, 0.0], [0.0, 1.0, 7.0, 6.0]], dtype=torch.float32)
    return torch.tensor([[8.0, 9.0, 0.0, 1.0], [7.0, 6.0, 0.0, 1.0]], dtype=torch.float32)


def _identity_worker(rank: int, port: int, out_dir: str) -> None:
    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        prompts_by_rank = ["rank0 prompt", "rank1 prompt"]
        request_id_table, microbatch_id_table, request_table_hash = build_request_identity_tables(
            prompts_by_rank=prompts_by_rank,
        )
        placement = build_online_expert_placement(world_size=2, expert_count=4, rank_to_node_id=[0, 0])
        partition = build_online_route_partition(
            run_id="identity-run",
            request_id=request_id_table[rank],
            microbatch_id=microbatch_id_table[0],
            request_numeric_id=rank,
            microbatch_numeric_id=0,
            layer_id=0,
            source_rank=rank,
            source_node_id=0,
            hidden_states=_hidden_states(rank),
            router_logits=_router_logits(rank),
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
            rank_device=torch.device("cpu"),
        )
        dispatch = execute_ws2_hidden_dispatch_only(
            hidden_states=_hidden_states(rank),
            partition=partition,
            manifest=manifest,
            placement=placement,
            agreement=agreement,
            request_id_table=request_id_table,
            microbatch_id_table=microbatch_id_table,
        )
        payload = {
            "rank": rank,
            "received_routes": [route.to_dict() for route in dispatch.received_routes],
        }
        Path(out_dir, f"identity-rank-{rank}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def test_online_ws2_request_identity_transport_round_trip(tmp_path) -> None:
    out_dir = tmp_path / "identity"
    out_dir.mkdir()
    mp.spawn(_identity_worker, args=(_free_port(), str(out_dir)), nprocs=2, join=True)
    rank0 = json.loads((out_dir / "identity-rank-0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "identity-rank-1.json").read_text(encoding="utf-8"))
    assert all(route["identity"]["request_id"] == "request-1" for route in rank0["received_routes"])
    assert all(route["identity"]["request_numeric_id"] == 1 for route in rank0["received_routes"])
    assert all(route["identity"]["microbatch_id"] == "ws2-mb-0" for route in rank0["received_routes"])
    assert all(route["identity"]["request_id"] == "request-0" for route in rank1["received_routes"])
    assert all(route["identity"]["request_numeric_id"] == 0 for route in rank1["received_routes"])
