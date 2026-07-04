from __future__ import annotations

import json

import torch
import torch.distributed as dist

from rs.contracts import TraceOrigin
from rs.offline.calibration import assert_online_native_ep_observation
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


def test_offline_gate_rejects_ws2_hidden_dispatch_trace(tmp_path) -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method=f"file:///{(tmp_path / 'pg_init_hidden_gate').as_posix()}",
            rank=0,
            world_size=1,
        )
    try:
        placement = build_online_expert_placement(world_size=1, expert_count=2, rank_to_node_id=[0])
        hidden_states = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        router_logits = torch.tensor([[3.0, 1.0]], dtype=torch.float32)
        request_id_table, microbatch_id_table, request_table_hash = build_request_identity_tables(
            prompts_by_rank=["prompt-0"],
        )
        partition = build_online_route_partition(
            run_id="run-0",
            request_id=request_id_table[0],
            microbatch_id=microbatch_id_table[0],
            request_numeric_id=0,
            microbatch_numeric_id=0,
            layer_id=0,
            source_rank=0,
            source_node_id=0,
            hidden_states=hidden_states,
            router_logits=router_logits,
            placement=placement,
            top_k=1,
            trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_ROUTE_PARTITION,
        )
        manifest = build_rank_manifest(
            partition=partition,
            placement=placement,
            prompt_text="prompt-0",
            request_protocol_hash=build_request_protocol_hash(
                prompts_by_rank=["prompt-0"],
                microbatch_id="mb-0",
                layer_id=0,
            ),
            request_table_hash=request_table_hash,
        )
        agreement = run_distributed_count_agreement(
            partition=partition,
            manifest=manifest,
            placement=placement,
            validate_metadata=False,
            rank_device=torch.device("cpu"),
        )
        dispatch = execute_ws2_hidden_dispatch_only(
            hidden_states=hidden_states,
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
        _jsonl_path, metadata_path = write_online_trace_artifacts(
            output_dir=tmp_path,
            run_id="ws2-hidden-dispatch",
            trace=trace,
            metadata={
                "pipeline": "online",
                "trace_origin": TraceOrigin.OBSERVED_ONLINE_WS2_HIDDEN_DISPATCH,
                "world_size": 1,
                "is_real_ep_runtime": False,
                "is_real_ep_transport": True,
                "transport_backend": "torch_distributed_hidden_dispatch_only",
            },
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            assert_online_native_ep_observation(metadata, metadata_path)
            raise AssertionError("expected offline gate to reject ws2 hidden dispatch trace")
        except RuntimeError as exc:
            assert "missing expert compute, inverse combine" in str(exc)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
