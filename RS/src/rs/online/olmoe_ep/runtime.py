from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import torch

from ...contracts import LayerRouteTrace, RouteIdentity, RouteRecord
from ...runtime.distributed_ep.adapter.expert_store import LocalExpertWeights
from ...runtime.distributed_ep.adapter.olmoe_adapter import execute_local_experts, probe_olmoe_adapter_config
from ...runtime.distributed_ep.core.manifest import RouteItem


@dataclass(frozen=True)
class InputPartition:
    run_id: str
    request_id: str
    microbatch_id: str
    source_rank: int
    prompt_text: str
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OnlineRoutePartition:
    layer_trace: LayerRouteTrace
    local_route_items: list[RouteItem]
    remote_route_records: list[RouteRecord]


@dataclass
class WorldSizeOneExecution:
    output: torch.Tensor
    route_partition: OnlineRoutePartition


def require_online_native_ep_runtime() -> None:
    raise NotImplementedError(
        "online_native_a2a_ep is not fully implemented yet; use execute_world_size_one_local_layer() "
        "for the current validated local-only Phase 2 scaffold"
    )


def build_input_partition(
    *,
    run_id: str,
    request_id: str,
    microbatch_id: str,
    source_rank: int,
    prompt_text: str,
    token_count: int,
) -> InputPartition:
    return InputPartition(
        run_id=run_id,
        request_id=request_id,
        microbatch_id=microbatch_id,
        source_rank=int(source_rank),
        prompt_text=prompt_text,
        token_count=int(token_count),
    )


def feature_probe_online_olmoe_runtime(model: Any) -> dict[str, Any]:
    adapter = probe_olmoe_adapter_config(model)
    first_layer = getattr(getattr(model, "model", None), "layers", [None])[0]
    mlp = getattr(first_layer, "mlp", None)
    experts = getattr(mlp, "experts", None)
    required = {
        "has_mlp": mlp is not None,
        "has_gate": bool(mlp is not None and hasattr(mlp, "gate")),
        "has_experts": experts is not None,
        "has_gate_up_proj": bool(experts is not None and hasattr(experts, "gate_up_proj")),
        "has_down_proj": bool(experts is not None and hasattr(experts, "down_proj")),
    }
    supported = all(required.values())
    return {
        "supported": supported,
        "reason": None if supported else "missing required OLMoE MLP/expert attributes for online adapter",
        "adapter": adapter.to_dict(),
        "requirements": required,
    }


def compute_plan_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def assert_plan_hash_agreement(plan_hashes: list[str]) -> None:
    unique = sorted(set(str(value) for value in plan_hashes))
    if len(unique) != 1:
        raise RuntimeError(f"plan hash mismatch across ranks: {unique}")


def build_route_partition_for_layer(
    *,
    run_id: str,
    request_id: str,
    microbatch_id: str,
    layer_id: int,
    source_rank: int,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    owner_by_expert: dict[int, int],
    top_k: int,
) -> OnlineRoutePartition:
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be [tokens, hidden], got {tuple(hidden_states.shape)}")
    if router_logits.ndim != 2:
        raise ValueError(f"router_logits must be [tokens, experts], got {tuple(router_logits.shape)}")
    if hidden_states.shape[0] != router_logits.shape[0]:
        raise ValueError("hidden_states token rows must match router_logits token rows")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    probs = torch.softmax(router_logits.detach().float(), dim=-1)
    weights, experts = torch.topk(probs, k=min(int(top_k), int(probs.shape[-1])), dim=-1)

    route_records: list[RouteRecord] = []
    local_route_items: list[RouteItem] = []
    remote_route_records: list[RouteRecord] = []
    hidden_size = int(hidden_states.shape[-1])
    bytes_per_row = int(hidden_size * hidden_states.element_size())

    for token_index in range(int(hidden_states.shape[0])):
        for topk_slot, (weight, expert_id) in enumerate(
            zip(weights[token_index].tolist(), experts[token_index].tolist(), strict=False)
        ):
            destination_rank = int(owner_by_expert[int(expert_id)])
            record = RouteRecord(
                identity=RouteIdentity(
                    run_id=run_id,
                    request_id=request_id,
                    microbatch_id=microbatch_id,
                    layer_id=int(layer_id),
                    source_rank=int(source_rank),
                    destination_rank=destination_rank,
                    expert_id=int(expert_id),
                    token_index_local=int(token_index),
                    topk_slot=int(topk_slot),
                ),
                routing_weight=float(weight),
                payload_rows=1,
                payload_bytes=bytes_per_row,
                is_local_route=destination_rank == int(source_rank),
                is_remote_route=destination_rank != int(source_rank),
            )
            route_records.append(record)
            route_item = RouteItem(
                request_id=request_id,
                generation_step=0,
                layer_id=int(layer_id),
                token_flat_index=int(token_index),
                route_rank_within_topk=int(topk_slot),
                origin_rank=int(source_rank),
                destination_rank=destination_rank,
                expert_id=int(expert_id),
                payload_rows=1,
                routing_weight=float(weight),
                is_cross_node=destination_rank != int(source_rank),
            )
            if record.is_local_route:
                local_route_items.append(route_item)
            else:
                remote_route_records.append(record)

    return OnlineRoutePartition(
        layer_trace=LayerRouteTrace(
            trace_origin="observed_online_native_ep",
            future_information_mode="none",
            layer_id=int(layer_id),
            route_records=route_records,
        ),
        local_route_items=local_route_items,
        remote_route_records=remote_route_records,
    )


def aggregate_local_route_outputs(
    routed_outputs: torch.Tensor,
    route_items: list[RouteItem],
    *,
    token_count: int,
    hidden_size: int,
) -> torch.Tensor:
    output = torch.zeros((token_count, hidden_size), dtype=routed_outputs.dtype, device=routed_outputs.device)
    for row_index, route_item in enumerate(route_items):
        output[int(route_item.token_flat_index)] += routed_outputs[row_index]
    return output


def execute_world_size_one_local_layer(
    *,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    local_expert_weights: LocalExpertWeights,
    layer_id: int,
    partition: InputPartition,
    top_k: int,
) -> WorldSizeOneExecution:
    owner_by_expert = {int(expert_id): 0 for expert_id in local_expert_weights.local_expert_ids}
    route_partition = build_route_partition_for_layer(
        run_id=partition.run_id,
        request_id=partition.request_id,
        microbatch_id=partition.microbatch_id,
        layer_id=layer_id,
        source_rank=partition.source_rank,
        hidden_states=hidden_states,
        router_logits=router_logits,
        owner_by_expert=owner_by_expert,
        top_k=int(top_k),
    )
    if route_partition.remote_route_records:
        raise RuntimeError("world_size=1 execution must not produce remote routes")
    routed_hidden = torch.stack(
        [hidden_states[int(item.token_flat_index)] for item in route_partition.local_route_items],
        dim=0,
    ) if route_partition.local_route_items else hidden_states.new_empty((0, hidden_states.shape[-1]))
    routed_outputs = execute_local_experts(
        routed_hidden,
        route_partition.local_route_items,
        local_expert_weights,
    )
    combined = aggregate_local_route_outputs(
        routed_outputs,
        route_partition.local_route_items,
        token_count=int(hidden_states.shape[0]),
        hidden_size=int(hidden_states.shape[-1]),
    )
    return WorldSizeOneExecution(output=combined, route_partition=route_partition)
