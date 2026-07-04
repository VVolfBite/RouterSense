from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rs.online.olmoe_ep.ws2_native_ep import (
    WS2LocalLayerObservation,
    assert_single_node_hostnames,
    execute_ws2_native_ep_layer_from_observation,
)
from rs.runtime.distributed_ep.adapter.expert_store import LocalExpertWeights


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _local_weights_for_rank(rank: int) -> LocalExpertWeights:
    if rank == 0:
        gate_up = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        down = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
        return LocalExpertWeights(
            local_expert_ids=[0],
            gate_up_proj=gate_up,
            down_proj=down,
            hidden_dim=2,
            intermediate_dim=2,
            activation_name="silu",
        )
    gate_up = torch.tensor(
        [[[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    down = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    return LocalExpertWeights(
        local_expert_ids=[1],
        gate_up_proj=gate_up,
        down_proj=down,
        hidden_dim=2,
        intermediate_dim=2,
        activation_name="silu",
    )


def _apply_single_expert(hidden: torch.Tensor, weights: LocalExpertWeights, expert_id: int) -> torch.Tensor:
    local_index = weights.local_expert_ids.index(expert_id)
    gate_up = weights.gate_up_proj[local_index]
    down = weights.down_proj[local_index]
    gate_up_value = F.linear(hidden.unsqueeze(0), gate_up)
    gate, up = gate_up_value.chunk(2, dim=-1)
    current_hidden = F.silu(gate) * up
    current_hidden = F.linear(current_hidden, down)
    return current_hidden[0]


def _compute_reference_output(hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
    weights0 = _local_weights_for_rank(0)
    weights1 = _local_weights_for_rank(1)
    probs = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_experts = torch.topk(probs, k=2, dim=-1)
    output = torch.zeros_like(hidden_states)
    for token_index in range(int(hidden_states.shape[0])):
        hidden = hidden_states[token_index]
        for slot in range(2):
            expert_id = int(topk_experts[token_index, slot].item())
            route_weight = float(topk_weights[token_index, slot].item())
            expert_output = (
                _apply_single_expert(hidden, weights0, expert_id)
                if expert_id == 0
                else _apply_single_expert(hidden, weights1, expert_id)
            )
            output[token_index] += expert_output * route_weight
    return output


def _positive_hidden(rank: int) -> torch.Tensor:
    if rank == 0:
        return torch.tensor([[1.0, 0.5], [0.75, 1.25]], dtype=torch.float32)
    return torch.tensor([[0.5, 1.5], [1.25, 0.75]], dtype=torch.float32)


def _router_logits_remote(rank: int) -> torch.Tensor:
    if rank == 0:
        return torch.tensor([[5.0, 4.5], [3.5, 4.0]], dtype=torch.float32)
    return torch.tensor([[4.0, 5.0], [4.5, 3.5]], dtype=torch.float32)


def _router_logits_local_only(rank: int) -> torch.Tensor:
    if rank == 0:
        return torch.tensor([[8.0, 1.0], [7.0, 0.5]], dtype=torch.float32)
    return torch.tensor([[1.0, 8.0], [0.5, 7.0]], dtype=torch.float32)


def _build_observation(rank: int, *, remote: bool) -> WS2LocalLayerObservation:
    hidden_states = _positive_hidden(rank)
    router_logits = _router_logits_remote(rank) if remote else _router_logits_local_only(rank)
    reference_output = _compute_reference_output(hidden_states, router_logits)
    return WS2LocalLayerObservation(
        hidden_states=hidden_states,
        router_logits=router_logits,
        reference_output=reference_output,
        layer_id=0,
        top_k=2 if remote else 1,
        expert_count=2,
        probe={"supported": True},
        model_revision=None,
        resolved_device="cpu",
        dtype="float32",
        experts_module=object(),
        local_weights_override=_local_weights_for_rank(rank),
    )


def _native_ep_worker(rank: int, port: int, out_dir: str, remote: bool) -> None:
    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        result = execute_ws2_native_ep_layer_from_observation(
            run_id="ws2-native-ep",
            prompts_by_rank=["prompt-a", "prompt-b"],
            prompt_text="prompt-a" if rank == 0 else "prompt-b",
            observation=_build_observation(rank, remote=remote),
            backend="gloo",
            rank_device=torch.device("cpu"),
            validate=True,
            require_remote_route=False,
        )
        payload = {
            "rank": rank,
            "correctness_status": result.validation.correctness_status,
            "numerical_correctness_pass": result.validation.numerical_correctness_pass,
            "transport_exercised": result.transport_exercised,
            "remote_route_count": result.remote_route_count,
            "dispatch_rows": result.dispatch_rows,
            "combine_rows": result.combine_rows,
            "max_abs_error": result.validation.max_abs_error,
            "mean_abs_error": result.validation.mean_abs_error,
            "cosine_similarity": result.validation.cosine_similarity,
            "transport_operations": [record.to_dict() for record in result.trace.transport_operations],
            "expert_buckets": [bucket.to_dict() for bucket in result.trace.expert_buckets],
            "output": None if result.output is None else result.output.tolist(),
        }
        Path(out_dir, f"rank-{rank}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _require_remote_worker(rank: int, port: int) -> None:
    dist.init_process_group(backend="gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        execute_ws2_native_ep_layer_from_observation(
            run_id="ws2-native-ep",
            prompts_by_rank=["prompt-a", "prompt-b"],
            prompt_text="prompt-a" if rank == 0 else "prompt-b",
            observation=_build_observation(rank, remote=False),
            backend="gloo",
            rank_device=torch.device("cpu"),
            validate=True,
            require_remote_route=True,
        )
    finally:
        dist.destroy_process_group()


def test_ws2_zero_remote_route_is_not_transport_verified(tmp_path) -> None:
    out_dir = tmp_path / "zero-remote"
    out_dir.mkdir()
    mp.spawn(_native_ep_worker, args=(_free_port(), str(out_dir), False), nprocs=2, join=True)
    rank0 = json.loads((out_dir / "rank-0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "rank-1.json").read_text(encoding="utf-8"))
    assert rank0["correctness_status"] == "skipped_no_remote_route"
    assert rank1["correctness_status"] == "skipped_no_remote_route"
    assert rank0["transport_exercised"] is False
    assert rank1["transport_exercised"] is False
    assert rank0["dispatch_rows"] == 0
    assert rank1["dispatch_rows"] == 0


def test_ws2_require_remote_route_fails_when_no_remote() -> None:
    with pytest.raises(Exception):
        mp.spawn(_require_remote_worker, args=(_free_port(),), nprocs=2, join=True)


def test_ws2_distributed_layer_numerical_parity(tmp_path) -> None:
    out_dir = tmp_path / "remote"
    out_dir.mkdir()
    mp.spawn(_native_ep_worker, args=(_free_port(), str(out_dir), True), nprocs=2, join=True)
    rank0 = json.loads((out_dir / "rank-0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "rank-1.json").read_text(encoding="utf-8"))
    assert rank0["correctness_status"] == "passed"
    assert rank1["correctness_status"] == "passed"
    assert rank0["numerical_correctness_pass"] is True
    assert rank1["numerical_correctness_pass"] is True
    assert rank0["remote_route_count"] > 0
    assert rank1["remote_route_count"] > 0
    assert rank0["dispatch_rows"] > 0
    assert rank1["dispatch_rows"] > 0
    assert rank0["combine_rows"] > 0
    assert rank1["combine_rows"] > 0


def test_ws2_local_and_remote_route_coverage(tmp_path) -> None:
    out_dir = tmp_path / "coverage"
    out_dir.mkdir()
    mp.spawn(_native_ep_worker, args=(_free_port(), str(out_dir), True), nprocs=2, join=True)
    rank0 = json.loads((out_dir / "rank-0.json").read_text(encoding="utf-8"))
    assert len(rank0["expert_buckets"]) >= 1
    assert any(record["phase"] == "dispatch" for record in rank0["transport_operations"])
    assert any(record["phase"] == "combine" for record in rank0["transport_operations"])


def test_ws2_owner_rank_expert_compute(tmp_path) -> None:
    out_dir = tmp_path / "owner"
    out_dir.mkdir()
    mp.spawn(_native_ep_worker, args=(_free_port(), str(out_dir), True), nprocs=2, join=True)
    rank0 = json.loads((out_dir / "rank-0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "rank-1.json").read_text(encoding="utf-8"))
    assert all(bucket["expert_id"] == 0 for bucket in rank0["expert_buckets"])
    assert all(bucket["expert_id"] == 1 for bucket in rank1["expert_buckets"])


def test_ws2_inverse_combine_route_completeness(tmp_path) -> None:
    out_dir = tmp_path / "combine"
    out_dir.mkdir()
    mp.spawn(_native_ep_worker, args=(_free_port(), str(out_dir), True), nprocs=2, join=True)
    rank0 = json.loads((out_dir / "rank-0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((out_dir / "rank-1.json").read_text(encoding="utf-8"))
    assert len(rank0["output"]) == 2
    assert len(rank1["output"]) == 2
    assert rank0["combine_rows"] == rank0["remote_route_count"]
    assert rank1["combine_rows"] == rank1["remote_route_count"]


def test_ws2_multinode_rejected() -> None:
    with pytest.raises(RuntimeError, match="UnsupportedMultiNode"):
        assert_single_node_hostnames(["host-a", "host-b"])
