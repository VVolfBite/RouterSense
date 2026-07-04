from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ...contracts import (
    EpExecutionTrace,
    ExpertBucketRecord,
    FutureInformationMode,
    OnlineExpertPlacement,
    OnlineRoutePartition,
    OnlineRouteRecord,
    RankManifest,
    RankStageTiming,
    TraceOrigin,
    TransportOperationRecord,
    ValidationResult,
)
from ..distributed_runtime import (
    DistributedDeviceInfo,
    assert_distinct_cuda_device_mapping,
    capture_distributed_device_info,
    collective_device_for_backend,
    run_distributed_stage,
)
from ...runtime import load_model_and_tokenizer
from ...runtime.distributed_ep.adapter.expert_store import (
    LocalExpertWeights,
    extract_local_expert_weights,
)
from ...runtime.distributed_ep.adapter.olmoe_adapter import probe_olmoe_adapter_config
from .observer import export_ws2_native_ep_moe_layer_harness_trace_artifacts
from .ws2_stage1 import (
    WS2CountAgreementResult,
    build_online_expert_placement,
    build_online_layer_route_trace,
    build_online_route_partition,
    build_rank_manifest,
    build_request_identity_tables,
    build_request_protocol_hash,
    run_distributed_count_agreement,
)
from .ws2_stage2 import (
    _decode_received_routes,
    _route_metadata_int_tensor,
    _route_metadata_weight_tensor,
)


ATOL_FP16 = 5e-3
RTOL_FP16 = 5e-3
MAX_ABS_FP16 = 5e-3
MEAN_ABS_FP16 = 1e-3
COSINE_FP16 = 0.999


@dataclass(frozen=True)
class WS2LocalLayerObservation:
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    reference_output: torch.Tensor
    layer_id: int
    top_k: int
    expert_count: int
    probe: dict[str, Any]
    model_revision: str | None
    resolved_device: str
    dtype: str
    experts_module: Any
    local_weights_override: LocalExpertWeights | None = None


@dataclass(frozen=True)
class ExpertBucket:
    expert_id: int
    routes: list[OnlineRouteRecord]
    hidden_rows: torch.Tensor


@dataclass(frozen=True)
class WS2NativeEPMoELayerResult:
    rank: int
    backend: str
    device_info: DistributedDeviceInfo
    distinct_cuda_device_indices: list[int]
    placement: OnlineExpertPlacement
    manifest: RankManifest
    agreement: WS2CountAgreementResult
    partition: OnlineRoutePartition
    trace: EpExecutionTrace
    validation: ValidationResult
    output: torch.Tensor | None
    reference_output: torch.Tensor
    local_route_count: int
    remote_route_count: int
    dispatch_rows: int
    combine_rows: int
    transport_exercised: bool


def _hash_text_tensor(value: str, *, device: torch.device) -> torch.Tensor:
    payload = value.encode("utf-8")[:64]
    fixed = payload + (b"\0" * (64 - len(payload)))
    return torch.tensor(list(fixed), dtype=torch.uint8, device=device)


def _tensor_to_text(tensor: torch.Tensor) -> str:
    return bytes(int(item) for item in tensor.tolist()).split(b"\0", 1)[0].decode("utf-8")


def assert_single_node_hostnames(hostnames: list[str]) -> None:
    if len(set(str(hostname) for hostname in hostnames)) != 1:
        raise RuntimeError(f"UnsupportedMultiNode: hostnames={hostnames}")


def assert_single_node_only(*, backend: str, rank_device: torch.device) -> None:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before verifying node topology")
    collective_device = collective_device_for_backend(backend, rank_device)
    hostname = socket.gethostname()
    local = _hash_text_tensor(hostname, device=collective_device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    hostnames = [_tensor_to_text(item.detach().cpu()) for item in gathered]
    assert_single_node_hostnames(hostnames)


def _capture_local_layer_reference(
    *,
    model_id: str,
    model_path: str | None,
    prompt_text: str,
    layer_index: int,
    precision: str,
    device_index: int,
) -> WS2LocalLayerObservation:
    model, tokenizer, resolved_revision, resolved_device, dtype = load_model_and_tokenizer(
        model_id=model_id,
        model_path=model_path,
        precision=precision,
        device_index=device_index,
    )
    probe = probe_olmoe_adapter_config(model)
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
    logical_moe_index = int(layer_index)
    layer_id = int(moe_layer_ids[layer_index])
    layer = model.model.layers[layer_id]
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, inputs, output):
        captured["mlp_input"] = inputs[0].detach()
        captured["mlp_output"] = output.detach()

    handle = layer.mlp.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_router_logits=True,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
    finally:
        handle.remove()
    router_logits_by_layer = getattr(outputs, "router_logits", None)
    if not router_logits_by_layer:
        raise RuntimeError("model forward did not return router_logits")
    router_logits = router_logits_by_layer[logical_moe_index].detach()
    if {"mlp_input", "mlp_output"} - set(captured):
        raise RuntimeError("failed to capture local MoE layer input/output/router for ws2 native ep")
    if router_logits.ndim == 3:
        router_logits = router_logits.squeeze(0)
    return WS2LocalLayerObservation(
        hidden_states=captured["mlp_input"].squeeze(0),
        router_logits=router_logits,
        reference_output=captured["mlp_output"].squeeze(0),
        layer_id=layer_id,
        top_k=int(model.config.num_experts_per_tok),
        expert_count=int(model.config.num_experts),
        probe=probe.to_dict(),
        model_revision=resolved_revision,
        resolved_device=resolved_device,
        dtype=dtype,
        experts_module=layer.mlp.experts,
    )


def _route_identity_key(route: OnlineRouteRecord) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(route.identity.request_numeric_id),
        int(route.identity.microbatch_numeric_id),
        int(route.identity.layer_id),
        int(route.identity.source_rank),
        int(route.identity.local_token_index),
        int(route.identity.topk_slot),
        int(route.identity.expert_id),
    )


def _build_remote_hidden_tensor(hidden_states: torch.Tensor, routes: list[OnlineRouteRecord]) -> torch.Tensor:
    if not routes:
        return hidden_states.new_empty((0, hidden_states.shape[-1]))
    return torch.stack(
        [hidden_states[int(route.identity.local_token_index)] for route in routes],
        dim=0,
    )


def _run_a2a_with_metadata(
    *,
    run_id: str,
    backend: str,
    hidden_payload: torch.Tensor,
    routes: list[OnlineRouteRecord],
    request_id_table: list[str],
    microbatch_id_table: list[str],
    send_splits: list[int],
    recv_splits: list[int],
    operation_id: str,
    phase: str,
    local_rank: int | None,
) -> tuple[torch.Tensor, list[OnlineRouteRecord], TransportOperationRecord]:
    collective_device = collective_device_for_backend(backend, hidden_payload.device)
    send_hidden = hidden_payload.to(device=collective_device)
    send_metadata_int = _route_metadata_int_tensor(routes).to(device=collective_device)
    send_metadata_weight = _route_metadata_weight_tensor(routes).to(device=collective_device)
    hidden_size = int(send_hidden.shape[-1]) if send_hidden.ndim == 2 else 0
    recv_hidden = torch.empty(
        (sum(recv_splits), hidden_size),
        dtype=send_hidden.dtype,
        device=collective_device,
    )
    recv_metadata_int = torch.empty((sum(recv_splits), 12), dtype=torch.int64, device=collective_device)
    recv_metadata_weight = torch.empty((sum(recv_splits),), dtype=torch.float32, device=collective_device)

    cuda_elapsed_ms: float | None = None
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None
    if backend == "nccl":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    started = time.perf_counter()
    dist.all_to_all_single(recv_hidden, send_hidden, output_split_sizes=recv_splits, input_split_sizes=send_splits)
    dist.all_to_all_single(
        recv_metadata_int,
        send_metadata_int,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )
    dist.all_to_all_single(
        recv_metadata_weight,
        send_metadata_weight,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )
    wall_elapsed_ms = (time.perf_counter() - started) * 1000.0
    if backend == "nccl" and start_event is not None and end_event is not None:
        end_event.record()
        end_event.synchronize()
        cuda_elapsed_ms = float(start_event.elapsed_time(end_event))

    received_routes = _decode_received_routes(
        run_id=run_id,
        request_id_table=request_id_table,
        microbatch_id_table=microbatch_id_table,
        metadata_int=recv_metadata_int.detach().cpu(),
        metadata_weight=recv_metadata_weight.detach().cpu(),
    )
    transport_record = TransportOperationRecord(
        run_id=run_id,
        rank=int(dist.get_rank()),
        local_rank=local_rank,
        world_size=int(dist.get_world_size()),
        operation_id=operation_id,
        phase=phase,
        backend=backend,
        verified_backend="nccl_gpu" if backend == "nccl" else "gloo_cpu_test_only",
        device=str(collective_device),
        operation_kind="variable_size_hidden_and_metadata_all_to_all",
        hidden_payload_transferred=bool(sum(send_splits) > 0 or sum(recv_splits) > 0),
        send_counts=list(send_splits),
        recv_counts=list(recv_splits),
        send_rows=int(sum(send_splits)),
        recv_rows=int(sum(recv_splits)),
        send_bytes=int(send_hidden.numel() * send_hidden.element_size()),
        recv_bytes=int(recv_hidden.numel() * recv_hidden.element_size()),
        wall_elapsed_ms=wall_elapsed_ms,
        success=True,
        hidden_bytes=int(send_hidden.numel() * send_hidden.element_size()),
        metadata_int_bytes=int(send_metadata_int.numel() * send_metadata_int.element_size()),
        metadata_float_bytes=int(send_metadata_weight.numel() * send_metadata_weight.element_size()),
        pack_ms=0.0,
        post_ms=None,
        wait_ms=wall_elapsed_ms,
        collective_post_ms=None,
        collective_wait_ms=wall_elapsed_ms,
        unpack_ms=0.0,
        cuda_elapsed_ms=cuda_elapsed_ms,
        details={},
    )
    return recv_hidden, received_routes, transport_record


def _execute_online_expert_rows(
    hidden_rows: torch.Tensor,
    routes: list[OnlineRouteRecord],
    local_weights: LocalExpertWeights,
) -> torch.Tensor:
    if hidden_rows.shape[0] != len(routes):
        raise RuntimeError("hidden row count must match route count for owner-rank expert compute")
    if hidden_rows.shape[0] == 0:
        return hidden_rows.new_empty((0, local_weights.hidden_dim))
    local_index_by_expert = {expert_id: index for index, expert_id in enumerate(local_weights.local_expert_ids)}
    output = torch.zeros(
        (hidden_rows.shape[0], local_weights.hidden_dim),
        dtype=hidden_rows.dtype,
        device=hidden_rows.device,
    )
    for row_index, route in enumerate(routes):
        local_index = local_index_by_expert.get(int(route.identity.expert_id))
        if local_index is None:
            raise RuntimeError(f"expert {route.identity.expert_id} is not resident on this owner rank")
        current_state = hidden_rows[row_index : row_index + 1]
        gate_up = local_weights.gate_up_proj[local_index]
        down = local_weights.down_proj[local_index]
        gate_up_value = F.linear(current_state, gate_up)
        gate, up = gate_up_value.chunk(2, dim=-1)
        current_hidden = F.silu(gate) * up
        current_hidden = F.linear(current_hidden, down)
        output[row_index] = current_hidden[0].to(output.dtype)
    return output


def _group_owner_buckets(
    *,
    rank: int,
    hidden_states: torch.Tensor,
    partition: OnlineRoutePartition,
    received_remote_routes: list[OnlineRouteRecord],
    received_hidden_states: torch.Tensor,
    placement: OnlineExpertPlacement,
) -> tuple[list[ExpertBucket], int, int]:
    routes: list[OnlineRouteRecord] = []
    route_rows: list[torch.Tensor] = []
    for route in partition.local_routes:
        if int(route.identity.destination_rank) != int(rank):
            raise RuntimeError("local route destination must equal current rank")
        expert_owner = int(placement.owner_rank_by_expert[int(route.identity.expert_id)])
        if expert_owner != int(rank):
            raise RuntimeError(f"local route expert {route.identity.expert_id} is not owned by rank {rank}")
        routes.append(route)
        route_rows.append(hidden_states[int(route.identity.local_token_index)])
    if int(received_hidden_states.shape[0]) != len(received_remote_routes):
        raise RuntimeError("received remote hidden rows must match received remote route count")
    for row_index, route in enumerate(received_remote_routes):
        if int(route.identity.destination_rank) != int(rank):
            raise RuntimeError("received remote route destination must equal current rank")
        expert_owner = int(placement.owner_rank_by_expert[int(route.identity.expert_id)])
        if expert_owner != int(rank):
            raise RuntimeError(f"remote route expert {route.identity.expert_id} is not owned by rank {rank}")
        routes.append(route)
        route_rows.append(received_hidden_states[row_index])
    if route_rows:
        stacked = torch.stack(route_rows, dim=0)
    else:
        stacked = hidden_states.new_empty((0, hidden_states.shape[-1]))
    by_expert: dict[int, list[int]] = {}
    for row_index, route in enumerate(routes):
        by_expert.setdefault(int(route.identity.expert_id), []).append(row_index)
    buckets: list[ExpertBucket] = []
    local_rows = 0
    remote_rows = 0
    for expert_id, indices in sorted(by_expert.items()):
        bucket_routes = [routes[index] for index in indices]
        bucket_hidden = stacked.index_select(
            0,
            torch.tensor(indices, dtype=torch.long, device=stacked.device),
        )
        local_rows += sum(1 for route in bucket_routes if int(route.identity.source_rank) == int(rank))
        remote_rows += sum(1 for route in bucket_routes if int(route.identity.source_rank) != int(rank))
        buckets.append(ExpertBucket(expert_id=expert_id, routes=bucket_routes, hidden_rows=bucket_hidden))
    return buckets, local_rows, remote_rows


def _execute_owner_buckets(
    *,
    rank: int,
    buckets: list[ExpertBucket],
    local_weights: LocalExpertWeights,
    layer_id: int,
) -> tuple[list[OnlineRouteRecord], torch.Tensor, list[ExpertBucketRecord], float]:
    outputs: list[torch.Tensor] = []
    ordered_routes: list[OnlineRouteRecord] = []
    bucket_records: list[ExpertBucketRecord] = []
    started = time.perf_counter()
    for bucket in buckets:
        bucket_output = _execute_online_expert_rows(bucket.hidden_rows, bucket.routes, local_weights)
        outputs.append(bucket_output)
        ordered_routes.extend(bucket.routes)
        bucket_records.append(
            ExpertBucketRecord(
                rank=int(rank),
                layer_id=int(layer_id),
                expert_id=int(bucket.expert_id),
                bucket_rows=int(len(bucket.routes)),
                bucket_bytes=int(bucket.hidden_rows.numel() * bucket.hidden_rows.element_size()),
            )
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if outputs:
        combined = torch.cat(outputs, dim=0)
    elif buckets:
        combined = buckets[0].hidden_rows.new_empty((0, local_weights.hidden_dim))
    else:
        combined = torch.empty((0, local_weights.hidden_dim), dtype=local_weights.gate_up_proj.dtype, device=local_weights.gate_up_proj.device)
    return ordered_routes, combined, bucket_records, elapsed_ms


def _scatter_local_output(
    *,
    rank: int,
    partition: OnlineRoutePartition,
    local_owner_routes: list[OnlineRouteRecord],
    local_owner_outputs: torch.Tensor,
    returned_remote_routes: list[OnlineRouteRecord],
    returned_remote_outputs: torch.Tensor,
    reference_output: torch.Tensor,
) -> tuple[torch.Tensor, ValidationResult]:
    token_count = int(reference_output.shape[0])
    hidden_size = int(reference_output.shape[-1])
    output = reference_output.new_zeros((token_count, hidden_size))
    all_expected = {_route_identity_key(route): route for route in partition.all_routes}
    contributions: list[tuple[OnlineRouteRecord, torch.Tensor]] = []
    for row_index, route in enumerate(local_owner_routes):
        contributions.append((route, local_owner_outputs[row_index]))
    for row_index, route in enumerate(returned_remote_routes):
        contributions.append((route, returned_remote_outputs[row_index]))
    seen: dict[tuple[int, int, int, int, int, int, int], int] = {}
    for route, current_output in contributions:
        key = _route_identity_key(route)
        seen[key] = seen.get(key, 0) + 1
        if int(route.identity.source_rank) != int(rank):
            raise RuntimeError(f"combine received contribution for wrong source rank {route.identity.source_rank} on rank {rank}")
        output[int(route.identity.local_token_index)] += current_output * float(route.routing_weight)
    if set(seen) != set(all_expected):
        missing = sorted(set(all_expected) - set(seen))
        extra = sorted(set(seen) - set(all_expected))
        raise RuntimeError(f"route completeness mismatch on rank {rank}: missing={missing} extra={extra}")
    duplicates = [key for key, count in seen.items() if count != 1]
    if duplicates:
        raise RuntimeError(f"duplicate route contributions on rank {rank}: {duplicates}")
    abs_error = (output - reference_output.to(output.dtype)).abs()
    max_abs_error = float(abs_error.max().item()) if abs_error.numel() else 0.0
    mean_abs_error = float(abs_error.mean().item()) if abs_error.numel() else 0.0
    denom = float(reference_output.abs().mean().item()) if reference_output.numel() else 0.0
    relative_error = float(mean_abs_error / max(denom, 1e-12))
    flat_output = output.reshape(-1).float()
    flat_reference = reference_output.reshape(-1).float()
    cosine_similarity = float(F.cosine_similarity(flat_output.unsqueeze(0), flat_reference.unsqueeze(0)).item())
    numerical_pass = (
        max_abs_error <= MAX_ABS_FP16
        and mean_abs_error <= MEAN_ABS_FP16
        and cosine_similarity >= COSINE_FP16
        and bool(torch.allclose(output, reference_output.to(output.dtype), atol=ATOL_FP16, rtol=RTOL_FP16))
    )
    return output, ValidationResult(
        correctness_status="passed" if numerical_pass else "failed",
        numerical_correctness_pass=numerical_pass,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        relative_error=relative_error,
        cosine_similarity=cosine_similarity,
        details={
            "token_count": token_count,
            "hidden_size": hidden_size,
            "route_count": len(partition.all_routes),
        },
    )


def execute_ws2_native_ep_layer_from_observation(
    *,
    run_id: str,
    prompts_by_rank: list[str],
    prompt_text: str,
    observation: WS2LocalLayerObservation,
    backend: str,
    rank_device: torch.device,
    validate: bool,
    require_remote_route: bool,
) -> WS2NativeEPMoELayerResult:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before ws2 native ep layer execution")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if int(world_size) != 2:
        raise RuntimeError(f"ws2 native ep layer requires world_size=2, received {world_size}")
    if str(backend).lower() == "nccl":
        torch.cuda.set_device(rank_device)
    assert_single_node_only(backend=backend, rank_device=rank_device)
    distinct_cuda_device_indices = run_distributed_stage(
        "device_mapping",
        backend=backend,
        rank_device=rank_device,
        fn=lambda: assert_distinct_cuda_device_mapping(
            backend=backend,
            rank_device=rank_device,
            world_size=world_size,
        ),
    )
    device_info = capture_distributed_device_info(
        rank=rank,
        world_size=world_size,
        backend=backend,
        rank_device=rank_device,
    )
    request_id_table, microbatch_id_table, request_table_hash = build_request_identity_tables(
        prompts_by_rank=prompts_by_rank,
    )
    request_numeric_id = int(rank)
    microbatch_numeric_id = 0
    placement = build_online_expert_placement(
        world_size=world_size,
        expert_count=int(observation.expert_count),
        rank_to_node_id=[0, 0],
    )
    request_id = request_id_table[request_numeric_id]
    microbatch_id = microbatch_id_table[microbatch_numeric_id]
    request_protocol_hash = build_request_protocol_hash(
        prompts_by_rank=prompts_by_rank,
        microbatch_id=microbatch_id,
        layer_id=observation.layer_id,
    )
    partition = run_distributed_stage(
        "route_partition",
        backend=backend,
        rank_device=rank_device,
        fn=lambda: build_online_route_partition(
            run_id=run_id,
            request_id=request_id,
            microbatch_id=microbatch_id,
            request_numeric_id=request_numeric_id,
            microbatch_numeric_id=microbatch_numeric_id,
            layer_id=observation.layer_id,
            source_rank=rank,
            source_node_id=0,
            hidden_states=observation.hidden_states,
            router_logits=observation.router_logits,
            placement=placement,
            top_k=int(observation.top_k),
        ),
    )
    manifest = build_rank_manifest(
        partition=partition,
        placement=placement,
        prompt_text=prompt_text,
        request_protocol_hash=request_protocol_hash,
        request_table_hash=request_table_hash,
    )
    agreement = run_distributed_stage(
        "count_agreement",
        backend=backend,
        rank_device=rank_device,
        fn=lambda: run_distributed_count_agreement(
            partition=partition,
            manifest=manifest,
            placement=placement,
            validate_metadata=True,
            rank_device=rank_device,
        ),
    )
    collective_device = collective_device_for_backend(backend, rank_device)
    local_remote_rows = torch.tensor(
        [sum(int(value) for value in partition.per_peer_send_rows.values())],
        dtype=torch.int64,
        device=collective_device,
    )
    dist.all_reduce(local_remote_rows, op=dist.ReduceOp.SUM)
    total_remote_route_count = int(local_remote_rows.item())
    if total_remote_route_count == 0:
        if require_remote_route:
            raise RuntimeError("require_remote_route=true but total_remote_route_count == 0")
        validation = ValidationResult(
            correctness_status="skipped_no_remote_route",
            numerical_correctness_pass=None,
            details={"total_remote_route_count": 0},
        )
        trace = EpExecutionTrace(
            trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_MOE_LAYER_HARNESS,
            future_information_mode=FutureInformationMode.NONE,
            online_route_traces=[build_online_layer_route_trace(partition=partition, trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_MOE_LAYER_HARNESS)],
            rank_manifests=[manifest],
            expert_placements=[placement],
            transport_operations=[agreement.transport_record],
            validation_results=[validation],
            metadata={
                "execution_mode": "online_ws2_native_ep_moe_layer_harness",
                "transport_exercised": False,
                "validation_reference": "hf_single_rank_layer_output",
                "claim_scope": "ws2_distributed_moe_layer_correctness_only",
                "is_real_ep_runtime": False,
                "is_complete_ep_dispatch": False,
            },
        )
        return WS2NativeEPMoELayerResult(
            rank=rank,
            backend=backend,
            device_info=device_info,
            distinct_cuda_device_indices=distinct_cuda_device_indices,
            placement=placement,
            manifest=manifest,
            agreement=agreement,
            partition=partition,
            trace=trace,
            validation=validation,
            output=None,
            reference_output=observation.reference_output,
            local_route_count=len(partition.local_routes),
            remote_route_count=len(partition.remote_send_routes),
            dispatch_rows=0,
            combine_rows=0,
            transport_exercised=False,
        )

    owner_rank = int(rank)
    local_expert_ids = [
        expert_id
        for expert_id, expert_owner in enumerate(placement.owner_rank_by_expert)
        if int(expert_owner) == owner_rank
    ]
    local_weights = (
        observation.local_weights_override
        if observation.local_weights_override is not None
        else extract_local_expert_weights(observation.experts_module, local_expert_ids)
    )
    remote_send_routes = sorted(
        partition.remote_send_routes,
        key=lambda route: (
            int(route.identity.destination_rank),
            int(route.identity.local_token_index),
            int(route.identity.topk_slot),
            int(route.identity.expert_id),
        ),
    )
    send_hidden = _build_remote_hidden_tensor(observation.hidden_states, remote_send_routes)
    dispatch_recv_splits = [int(value) for value in agreement.transport_record.recv_counts]
    dispatch_send_splits = [int(partition.per_peer_send_rows.get(peer, 0)) for peer in range(world_size)]
    dispatch_recv_hidden, raw_received_remote_routes, dispatch_record = run_distributed_stage(
        "dispatch",
        backend=backend,
        rank_device=rank_device,
        fn=lambda: _run_a2a_with_metadata(
            run_id=partition.run_id,
            backend=backend,
            hidden_payload=send_hidden,
            routes=remote_send_routes,
            request_id_table=request_id_table,
            microbatch_id_table=microbatch_id_table,
            send_splits=dispatch_send_splits,
            recv_splits=dispatch_recv_splits,
            operation_id="dispatch-hidden-0",
            phase="dispatch",
            local_rank=device_info.local_rank,
        ),
    )
    received_remote_routes = raw_received_remote_routes
    for route in received_remote_routes:
        if int(route.identity.destination_rank) != rank:
            raise RuntimeError(f"dispatch delivered route to wrong destination on rank {rank}")
    buckets, local_expert_rows, remote_expert_rows = _group_owner_buckets(
        rank=rank,
        hidden_states=observation.hidden_states,
        partition=partition,
        received_remote_routes=received_remote_routes,
        received_hidden_states=dispatch_recv_hidden,
        placement=placement,
    )
    owner_routes, owner_outputs, bucket_records, expert_compute_ms = run_distributed_stage(
        "expert_compute",
        backend=backend,
        rank_device=rank_device,
        fn=lambda: _execute_owner_buckets(
            rank=rank,
            buckets=buckets,
            local_weights=local_weights,
            layer_id=observation.layer_id,
        ),
    )
    local_owner_routes = [route for route in owner_routes if int(route.identity.source_rank) == rank]
    local_owner_outputs = (
        torch.stack(
            [owner_outputs[index] for index, route in enumerate(owner_routes) if int(route.identity.source_rank) == rank],
            dim=0,
        )
        if local_owner_routes
        else owner_outputs.new_empty((0, owner_outputs.shape[-1]))
    )
    return_routes = [route for route in owner_routes if int(route.identity.source_rank) != rank]
    return_outputs = (
        torch.stack(
            [owner_outputs[index] for index, route in enumerate(owner_routes) if int(route.identity.source_rank) != rank],
            dim=0,
        )
        if return_routes
        else owner_outputs.new_empty((0, owner_outputs.shape[-1]))
    )
    combine_send_splits = [0 for _ in range(world_size)]
    for route in return_routes:
        combine_send_splits[int(route.identity.source_rank)] += int(route.payload_rows)
    combine_recv_splits = [int(partition.per_peer_send_rows.get(peer, 0)) for peer in range(world_size)]
    combine_recv_outputs, raw_returned_routes, combine_record = run_distributed_stage(
        "combine",
        backend=backend,
        rank_device=rank_device,
        fn=lambda: _run_a2a_with_metadata(
            run_id=partition.run_id,
            backend=backend,
            hidden_payload=return_outputs,
            routes=return_routes,
            request_id_table=request_id_table,
            microbatch_id_table=microbatch_id_table,
            send_splits=combine_send_splits,
            recv_splits=combine_recv_splits,
            operation_id="combine-return-0",
            phase="combine",
            local_rank=device_info.local_rank,
        ),
    )
    returned_remote_routes = raw_returned_routes
    output, validation = run_distributed_stage(
        "validation",
        backend=backend,
        rank_device=rank_device,
        fn=lambda: _scatter_local_output(
            rank=rank,
            partition=partition,
            local_owner_routes=local_owner_routes,
            local_owner_outputs=local_owner_outputs,
            returned_remote_routes=returned_remote_routes,
            returned_remote_outputs=combine_recv_outputs,
            reference_output=observation.reference_output,
        ),
    )
    correctness_status = validation.correctness_status if validate else "metadata_passed"
    final_validation = ValidationResult(
        correctness_status=correctness_status,
        numerical_correctness_pass=validation.numerical_correctness_pass if validate else None,
        max_abs_error=validation.max_abs_error if validate else None,
        mean_abs_error=validation.mean_abs_error if validate else None,
        relative_error=validation.relative_error if validate else None,
        cosine_similarity=validation.cosine_similarity if validate else None,
        details={
            **(validation.details or {}),
            "validation_reference": "hf_single_rank_layer_output",
            "local_expert_rows": local_expert_rows,
            "remote_expert_rows": remote_expert_rows,
        },
    )
    trace = EpExecutionTrace(
        trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_MOE_LAYER_HARNESS,
        future_information_mode=FutureInformationMode.NONE,
        stage_timings=[
            RankStageTiming(rank=rank, stage="count_exchange", wall_ms=float(agreement.transport_record.wall_elapsed_ms)),
            RankStageTiming(rank=rank, stage="dispatch", wall_ms=float(dispatch_record.wall_elapsed_ms), cuda_event_ms=dispatch_record.cuda_elapsed_ms),
            RankStageTiming(rank=rank, stage="expert_compute", wall_ms=float(expert_compute_ms)),
            RankStageTiming(rank=rank, stage="combine", wall_ms=float(combine_record.wall_elapsed_ms), cuda_event_ms=combine_record.cuda_elapsed_ms),
        ],
        expert_buckets=bucket_records,
        online_route_traces=[build_online_layer_route_trace(partition=partition, trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_MOE_LAYER_HARNESS)],
        rank_manifests=[manifest],
        expert_placements=[placement],
        transport_operations=[agreement.transport_record, dispatch_record, combine_record],
        validation_results=[final_validation],
        metadata={
            "execution_mode": "online_ws2_native_ep_moe_layer_harness",
            "validation_reference": "hf_single_rank_layer_output",
            "transport_exercised": True,
            "backend": backend,
            "request_table_hash": request_table_hash,
            "claim_scope": "ws2_distributed_moe_layer_correctness_only",
            "is_real_ep_runtime": False,
            "is_complete_ep_dispatch": True,
        },
    )
    return WS2NativeEPMoELayerResult(
        rank=rank,
        backend=backend,
        device_info=device_info,
        distinct_cuda_device_indices=distinct_cuda_device_indices,
        placement=placement,
        manifest=manifest,
        agreement=agreement,
        partition=partition,
        trace=trace,
        validation=final_validation,
        output=output,
        reference_output=observation.reference_output,
        local_route_count=len(partition.local_routes),
        remote_route_count=len(partition.remote_send_routes),
        dispatch_rows=int(dispatch_record.send_rows),
        combine_rows=int(combine_record.recv_rows),
        transport_exercised=True,
    )


def run_world_size_two_native_ep_moe_layer(
    *,
    run_id: str,
    model_id: str,
    model_path: str | None,
    prompts_by_rank: list[str],
    layer_index: int,
    precision: str,
    rank_device: torch.device,
    backend: str,
    require_remote_route: bool,
    validate: bool,
    output_dir: str | None = None,
) -> WS2NativeEPMoELayerResult:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before ws2 native ep layer run")
    rank = dist.get_rank()
    prompt_text = str(prompts_by_rank[rank])
    observation = run_distributed_stage(
        "layer_capture",
        backend=backend,
        rank_device=rank_device,
        fn=lambda: _capture_local_layer_reference(
            model_id=model_id,
            model_path=model_path,
            prompt_text=prompt_text,
            layer_index=layer_index,
            precision=precision,
            device_index=(rank_device.index if rank_device.type == "cuda" else 0),
        ),
    )
    result = execute_ws2_native_ep_layer_from_observation(
        run_id=run_id,
        prompts_by_rank=prompts_by_rank,
        prompt_text=prompt_text,
        observation=observation,
        backend=backend,
        rank_device=rank_device,
        validate=validate,
        require_remote_route=require_remote_route,
    )
    if output_dir is not None:
        export_ws2_native_ep_moe_layer_harness_trace_artifacts(
            output_dir=output_dir,
            run_id=f"{run_id}-rank{rank}",
            trace=result.trace,
            extra_metadata={
                "execution_mode": "online_ws2_native_ep_moe_layer_harness",
                "claim_scope": "ws2_distributed_moe_layer_correctness_only",
                "backend": backend,
                "verified_backend": "nccl_gpu" if backend == "nccl" else "gloo_cpu_test_only",
                "is_real_ep_transport": bool(result.transport_exercised),
                "is_complete_ep_dispatch": bool(result.transport_exercised),
                "is_real_ep_runtime": False,
                "trace_origin": TraceOrigin.OBSERVED_ONLINE_WS2_MOE_LAYER_HARNESS,
                "expert_residency_mode": "full_checkpoint_then_local_extract",
                "checkpoint_loading_is_memory_efficient": False,
                "correctness_status": result.validation.correctness_status,
                "numerical_correctness_pass": result.validation.numerical_correctness_pass,
                "performance_claim_eligible": False,
                "transport_exercised": result.transport_exercised,
                "device_info": result.device_info.to_dict(),
                "distinct_cuda_device_indices": result.distinct_cuda_device_indices,
                "validation_reference": "hf_single_rank_layer_output",
            },
        )
    return result
