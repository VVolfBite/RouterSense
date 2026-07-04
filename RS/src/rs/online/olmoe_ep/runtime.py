from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import torch

from ...contracts import EpExecutionTrace, LayerRouteTrace, RankStageTiming, RouteIdentity, RouteRecord, TraceOrigin, FutureInformationMode
from ...runtime import load_model_and_tokenizer
from ...runtime.distributed_ep.adapter.expert_store import extract_local_expert_weights
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


@dataclass
class WorldSizeOneParityResult:
    layer_id: int
    token_count: int
    top_k: int
    max_abs_error: float
    mean_abs_error: float
    numerical_correctness_pass: bool
    route_count: int
    local_route_count: int
    remote_route_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorldSizeOneObservedTrace:
    execution_trace: EpExecutionTrace
    parity: WorldSizeOneParityResult
    metadata: dict[str, Any]


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
    has_packed_experts = bool(experts is not None and hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"))
    has_modulelist_experts = bool(
        experts is not None
        and hasattr(experts, "__getitem__")
        and len(experts) > 0
        and hasattr(experts[0], "gate_proj")
        and hasattr(experts[0], "up_proj")
        and hasattr(experts[0], "down_proj")
    )
    required = {
        "has_mlp": mlp is not None,
        "has_gate": bool(mlp is not None and hasattr(mlp, "gate")),
        "has_experts": experts is not None,
        "has_packed_experts": has_packed_experts,
        "has_modulelist_experts": has_modulelist_experts,
    }
    supported = required["has_mlp"] and required["has_gate"] and required["has_experts"] and (
        required["has_packed_experts"] or required["has_modulelist_experts"]
    )
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


def run_world_size_one_native_parity(
    *,
    model_id: str,
    model_path: str | None,
    prompt_text: str,
    layer_index: int = 0,
    precision: str = "fp16",
    device_index: int = 0,
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> dict[str, Any]:
    model, tokenizer, resolved_revision, resolved_device, dtype = load_model_and_tokenizer(
        model_id=model_id,
        model_path=model_path,
        precision=precision,
        device_index=device_index,
    )
    probe = feature_probe_online_olmoe_runtime(model)
    if not probe["supported"]:
        raise RuntimeError(str(probe["reason"]))

    encoded = tokenizer(prompt_text, return_tensors="pt")
    target_device = next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")
    encoded = {key: value.to(target_device) for key, value in encoded.items()}
    moe_layer_ids = [
        layer_id
        for layer_id, layer in enumerate(model.model.layers)
        if hasattr(getattr(layer, "mlp", None), "gate") and hasattr(getattr(layer, "mlp", None), "experts")
    ]
    if layer_index < 0 or layer_index >= len(moe_layer_ids):
        raise RuntimeError(f"layer_index {layer_index} out of range for {len(moe_layer_ids)} OLMoE MoE layers")
    layer_id = int(moe_layer_ids[layer_index])
    layer = model.model.layers[layer_id]
    mlp = layer.mlp
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, inputs, output):
        hidden = inputs[0]
        captured["mlp_input"] = hidden.detach()
        captured["mlp_output"] = output[0].detach()
        captured["router_logits"] = output[1].detach()

    handle = mlp.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model(**encoded, output_router_logits=True, output_hidden_states=True, return_dict=True, use_cache=False)
    finally:
        handle.remove()

    if {"mlp_input", "mlp_output", "router_logits"} - set(captured):
        raise RuntimeError("failed to capture MoE layer input/output during parity run")

    mlp_input = captured["mlp_input"].squeeze(0)
    mlp_output = captured["mlp_output"].squeeze(0)
    router_logits = captured["router_logits"]
    if router_logits.ndim == 3:
        router_logits = router_logits.squeeze(0)

    partition = build_input_partition(
        run_id="world-size-1-parity",
        request_id="parity-request-0",
        microbatch_id="parity-mb-0",
        source_rank=0,
        prompt_text=prompt_text,
        token_count=int(mlp_input.shape[0]),
    )
    local_weights = extract_local_expert_weights(mlp.experts, list(range(int(model.config.num_experts))))
    execution = execute_world_size_one_local_layer(
        hidden_states=mlp_input,
        router_logits=router_logits,
        local_expert_weights=local_weights,
        layer_id=layer_id,
        partition=partition,
        top_k=int(model.config.num_experts_per_tok),
    )
    reference = mlp_output.to(execution.output.dtype)
    abs_error = (execution.output - reference).abs()
    max_abs_error = float(abs_error.max().item()) if abs_error.numel() else 0.0
    mean_abs_error = float(abs_error.mean().item()) if abs_error.numel() else 0.0
    numerical_pass = bool(torch.allclose(execution.output, reference, atol=atol, rtol=rtol))
    parity = WorldSizeOneParityResult(
        layer_id=layer_id,
        token_count=int(mlp_input.shape[0]),
        top_k=int(model.config.num_experts_per_tok),
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        numerical_correctness_pass=numerical_pass,
        route_count=len(execution.route_partition.layer_trace.route_records),
        local_route_count=len(execution.route_partition.local_route_items),
        remote_route_count=len(execution.route_partition.remote_route_records),
    )
    return {
        "model_id": model_id,
        "model_path": model_path,
        "model_revision": resolved_revision,
        "device": resolved_device,
        "dtype": dtype,
        "probe": probe,
        "parity": parity.to_dict(),
    }


def collect_world_size_one_observed_native_ep_trace(
    *,
    model_id: str,
    model_path: str | None,
    prompt_text: str,
    layer_index: int = 0,
    precision: str = "fp16",
    device_index: int = 0,
) -> WorldSizeOneObservedTrace:
    model, tokenizer, resolved_revision, resolved_device, dtype = load_model_and_tokenizer(
        model_id=model_id,
        model_path=model_path,
        precision=precision,
        device_index=device_index,
    )
    probe = feature_probe_online_olmoe_runtime(model)
    if not probe["supported"]:
        raise RuntimeError(str(probe["reason"]))

    encoded = tokenizer(prompt_text, return_tensors="pt")
    target_device = next(parameter.device for parameter in model.parameters() if parameter.device.type != "meta")
    encoded = {key: value.to(target_device) for key, value in encoded.items()}
    moe_layer_ids = [
        current_layer_id
        for current_layer_id, layer in enumerate(model.model.layers)
        if hasattr(getattr(layer, "mlp", None), "gate") and hasattr(getattr(layer, "mlp", None), "experts")
    ]
    if layer_index < 0 or layer_index >= len(moe_layer_ids):
        raise RuntimeError(f"layer_index {layer_index} out of range for {len(moe_layer_ids)} OLMoE MoE layers")
    layer_id = int(moe_layer_ids[layer_index])
    layer = model.model.layers[layer_id]
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, inputs, output):
        captured["mlp_input"] = inputs[0].detach()
        captured["mlp_output"] = output[0].detach()
        captured["router_logits"] = output[1].detach()

    handle = layer.mlp.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model(**encoded, output_router_logits=True, output_hidden_states=True, return_dict=True, use_cache=False)
    finally:
        handle.remove()

    if {"mlp_input", "mlp_output", "router_logits"} - set(captured):
        raise RuntimeError("failed to capture MoE layer input/output during observed trace collection")

    mlp_input = captured["mlp_input"].squeeze(0)
    mlp_output = captured["mlp_output"].squeeze(0)
    router_logits = captured["router_logits"]
    if router_logits.ndim == 3:
        router_logits = router_logits.squeeze(0)

    partition = build_input_partition(
        run_id="world-size-1-observed",
        request_id="observed-request-0",
        microbatch_id="observed-mb-0",
        source_rank=0,
        prompt_text=prompt_text,
        token_count=int(mlp_input.shape[0]),
    )
    local_weights = extract_local_expert_weights(layer.mlp.experts, list(range(int(model.config.num_experts))))
    execution = execute_world_size_one_local_layer(
        hidden_states=mlp_input,
        router_logits=router_logits,
        local_expert_weights=local_weights,
        layer_id=layer_id,
        partition=partition,
        top_k=int(model.config.num_experts_per_tok),
    )
    abs_error = (execution.output - mlp_output.to(execution.output.dtype)).abs()
    parity = WorldSizeOneParityResult(
        layer_id=layer_id,
        token_count=int(mlp_input.shape[0]),
        top_k=int(model.config.num_experts_per_tok),
        max_abs_error=float(abs_error.max().item()) if abs_error.numel() else 0.0,
        mean_abs_error=float(abs_error.mean().item()) if abs_error.numel() else 0.0,
        numerical_correctness_pass=bool(torch.allclose(execution.output, mlp_output.to(execution.output.dtype), atol=5e-3, rtol=5e-3)),
        route_count=len(execution.route_partition.layer_trace.route_records),
        local_route_count=len(execution.route_partition.local_route_items),
        remote_route_count=len(execution.route_partition.remote_route_records),
    )
    execution_trace = EpExecutionTrace(
        trace_origin=TraceOrigin.OBSERVED_ONLINE_NATIVE_EP,
        future_information_mode=FutureInformationMode.NONE,
        route_traces=[execution.route_partition.layer_trace],
        stage_timings=[
            RankStageTiming(rank=0, stage="router_observation", wall_ms=0.0),
            RankStageTiming(rank=0, stage="local_expert_compute", wall_ms=0.0),
            RankStageTiming(rank=0, stage="combine", wall_ms=0.0),
        ],
        metadata={
            "world_size": 1,
            "model_id": model_id,
            "model_revision": resolved_revision,
            "device": resolved_device,
            "dtype": dtype,
            "layer_id": layer_id,
            "token_count": int(mlp_input.shape[0]),
            "top_k": int(model.config.num_experts_per_tok),
        },
    )
    return WorldSizeOneObservedTrace(
        execution_trace=execution_trace,
        parity=parity,
        metadata={
            "model_id": model_id,
            "model_path": model_path,
            "model_revision": resolved_revision,
            "device": resolved_device,
            "dtype": dtype,
            "probe": probe,
            "parity": parity.to_dict(),
        },
    )
