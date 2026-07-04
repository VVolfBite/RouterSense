from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rs.contracts import TraceOrigin
from rs.online.observer_io import write_online_trace_artifacts
from rs.online.olmoe_ep import (
    build_online_expert_placement,
    build_online_route_partition,
    build_request_identity_tables,
    build_rank_manifest,
    build_request_protocol_hash,
    build_ws2_hidden_dispatch_trace,
    execute_ws2_hidden_dispatch_only,
    run_distributed_count_agreement,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _hidden_states(rank: int) -> torch.Tensor:
    base = 10.0 if rank == 0 else 20.0
    return torch.tensor([[base + 1.0, base + 2.0], [base + 3.0, base + 4.0]], dtype=torch.float32)


def _router_logits(rank: int) -> torch.Tensor:
    if rank == 0:
        return torch.tensor([[9.0, 8.0, 1.0, 0.0], [0.0, 1.0, 7.0, 6.0]], dtype=torch.float32)
    return torch.tensor([[8.0, 9.0, 0.0, 1.0], [7.0, 6.0, 0.0, 1.0]], dtype=torch.float32)


def _dispatch_worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        placement = build_online_expert_placement(world_size=world_size, expert_count=4, rank_to_node_id=[0, 0])
        request_id_table, microbatch_id_table, request_table_hash = build_request_identity_tables(
            prompts_by_rank=["prompt-0", "prompt-1"],
        )
        partition = build_online_route_partition(
            run_id="dispatch-run",
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
            prompt_text=f"prompt-{rank}",
            request_protocol_hash=build_request_protocol_hash(
                prompts_by_rank=["prompt-0", "prompt-1"],
                microbatch_id="mb-0",
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
        trace = build_ws2_hidden_dispatch_trace(
            partition=partition,
            placement=placement,
            manifest=manifest,
            hidden_dispatch=dispatch,
        )
        write_online_trace_artifacts(
            output_dir=out_dir,
            run_id=f"dispatch-rank-{rank}",
            trace=trace,
            metadata={
                "pipeline": "online",
                "trace_origin": TraceOrigin.OBSERVED_ONLINE_WS2_HIDDEN_DISPATCH,
                "world_size": world_size,
                "is_real_ep_runtime": False,
                "is_real_ep_transport": True,
                "transport_backend": "torch_distributed_hidden_dispatch_only",
            },
        )
        payload = {
            "rank": rank,
            "correctness_status": dispatch.validation.correctness_status,
            "transport": dispatch.transport_record.to_dict(),
            "received_route_count": len(dispatch.received_routes),
            "received_hidden": dispatch.received_hidden_states.tolist(),
        }
        Path(out_dir, f"dispatch-rank-{rank}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if dispatch.validation.correctness_status != "metadata_passed":
            raise RuntimeError(str(dispatch.validation.details))
    finally:
        dist.destroy_process_group()


def test_online_ws2_hidden_dispatch(tmp_path) -> None:
    out_dir = tmp_path / "dispatch"
    out_dir.mkdir()
    mp.spawn(_dispatch_worker, args=(2, _free_port(), str(out_dir)), nprocs=2, join=True)
    rank0 = json.loads((out_dir / "dispatch-rank-0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "dispatch-rank-1.json").read_text(encoding="utf-8"))
    assert rank0["correctness_status"] == "metadata_passed"
    assert rank1["correctness_status"] == "metadata_passed"
    assert rank0["transport"]["hidden_payload_transferred"] is True
    assert rank1["transport"]["hidden_payload_transferred"] is True
    assert rank0["transport"]["phase"] == "dispatch_only"
    assert rank1["transport"]["phase"] == "dispatch_only"
    assert rank0["received_route_count"] == rank0["transport"]["recv_rows"]
    assert rank1["received_route_count"] == rank1["transport"]["recv_rows"]
    assert rank0["transport"]["send_counts"][1] == rank1["transport"]["recv_counts"][0]
    assert rank1["transport"]["send_counts"][0] == rank0["transport"]["recv_counts"][1]
