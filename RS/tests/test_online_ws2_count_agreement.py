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
    build_ws2_partition_trace,
    run_distributed_count_agreement,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_hidden(rank: int) -> torch.Tensor:
    return torch.tensor([[1.0 + rank, 0.0], [0.0, 2.0 + rank]], dtype=torch.float32)


def _rank_router_logits(rank: int) -> torch.Tensor:
    if rank == 0:
        return torch.tensor([[9.0, 8.0, 1.0, 0.0], [0.0, 1.0, 7.0, 6.0]], dtype=torch.float32)
    return torch.tensor([[8.0, 9.0, 0.0, 1.0], [7.0, 6.0, 0.0, 1.0]], dtype=torch.float32)


def _ws2_worker(rank: int, world_size: int, port: int, out_dir: str, mismatch: str) -> None:
    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        prompts_by_rank = ["prompt-rank0", "prompt-rank1"]
        rank_to_node_id = [0, 0]
        if mismatch == "placement" and rank == 1:
            rank_to_node_id = [0, 1]
        placement = build_online_expert_placement(world_size=world_size, expert_count=4, rank_to_node_id=rank_to_node_id)
        run_id = "shared-run"
        if mismatch == "run_id" and rank == 1:
            run_id = "mismatched-run"
        request_protocol_hash = build_request_protocol_hash(
            prompts_by_rank=prompts_by_rank if mismatch != "protocol" or rank == 0 else ["prompt-rank0", "tampered"],
            microbatch_id="mb-0",
            layer_id=0,
        )
        request_id_table, microbatch_id_table, request_table_hash = build_request_identity_tables(
            prompts_by_rank=prompts_by_rank,
        )
        if mismatch == "request_table" and rank == 1:
            request_table_hash = "0" * 64
        partition = build_online_route_partition(
            run_id=run_id,
            request_id=request_id_table[rank],
            microbatch_id=microbatch_id_table[0],
            request_numeric_id=rank,
            microbatch_numeric_id=0,
            layer_id=0,
            source_rank=rank,
            source_node_id=0,
            hidden_states=_rank_hidden(rank),
            router_logits=_rank_router_logits(rank),
            placement=placement,
            top_k=2,
            trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_ROUTE_PARTITION,
        )
        manifest = build_rank_manifest(
            partition=partition,
            placement=placement,
            prompt_text=prompts_by_rank[rank],
            request_protocol_hash=request_protocol_hash,
            request_table_hash=request_table_hash,
        )
        agreement = run_distributed_count_agreement(
            partition=partition,
            manifest=manifest,
            placement=placement,
            validate_metadata=True,
            rank_device=torch.device("cpu"),
        )
        trace = build_ws2_partition_trace(
            partition=partition,
            placement=placement,
            manifest=manifest,
            agreement=agreement,
        )
        write_online_trace_artifacts(
            output_dir=out_dir,
            run_id=f"rank-{rank}",
            trace=trace,
            metadata={
                "pipeline": "online",
                "trace_origin": TraceOrigin.OBSERVED_ONLINE_WS2_ROUTE_PARTITION,
                "world_size": world_size,
                "is_real_ep_runtime": False,
                "is_real_ep_transport": False,
                "transport_backend": "torch_distributed_metadata_agreement",
            },
        )
        payload = {
            "rank": rank,
            "manifest_hash": manifest.manifest_hash,
            "placement_hash": placement.placement_hash,
            "local_route_count": len(partition.local_routes),
            "remote_route_count": len(partition.remote_send_routes),
            "per_peer_send_rows": partition.per_peer_send_rows,
            "transport": agreement.transport_record.to_dict(),
            "validation": agreement.validation.to_dict(),
            "gathered_send_count_matrix": agreement.gathered_send_count_matrix,
        }
        Path(out_dir, f"rank-{rank}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if agreement.validation.correctness_status != "metadata_passed":
            raise RuntimeError(str(agreement.validation.details))
    finally:
        dist.destroy_process_group()


def test_online_ws2_count_agreement(tmp_path) -> None:
    out_dir = tmp_path / "ok"
    out_dir.mkdir()
    mp.spawn(_ws2_worker, args=(2, _free_port(), str(out_dir), "none"), nprocs=2, join=True)
    rank0 = json.loads((out_dir / "rank-0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "rank-1.json").read_text(encoding="utf-8"))
    assert rank0["validation"]["correctness_status"] == "metadata_passed"
    assert rank1["validation"]["correctness_status"] == "metadata_passed"
    assert rank0["manifest_hash"] != rank1["manifest_hash"]
    assert rank0["placement_hash"] == rank1["placement_hash"]
    assert rank0["transport"]["send_counts"][1] == rank1["transport"]["recv_counts"][0]
    assert rank1["transport"]["send_counts"][0] == rank0["transport"]["recv_counts"][1]


@pytest.mark.parametrize("mismatch", ["run_id", "placement", "protocol", "request_table"])
def test_online_ws2_mismatch_fails_fast(tmp_path, mismatch: str) -> None:
    out_dir = tmp_path / mismatch
    out_dir.mkdir()
    with pytest.raises(Exception):
        mp.spawn(_ws2_worker, args=(2, _free_port(), str(out_dir), mismatch), nprocs=2, join=True)


def test_ws2_request_table_hash_mismatch_fails(tmp_path) -> None:
    out_dir = tmp_path / "request_table"
    out_dir.mkdir()
    with pytest.raises(Exception):
        mp.spawn(_ws2_worker, args=(2, _free_port(), str(out_dir), "request_table"), nprocs=2, join=True)
