from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from ...contracts import (
    EpExecutionTrace,
    FutureInformationMode,
    OnlineExpertPlacement,
    OnlineLayerRouteTrace,
    OnlineRouteIdentity,
    OnlineRoutePartition,
    OnlineRouteRecord,
    RankManifest,
    RankStageTiming,
    TraceOrigin,
    TransportOperationRecord,
    ValidationResult,
    stable_hash_dict,
)
from ...runtime import load_model_and_tokenizer
from ...runtime.distributed_ep.adapter.olmoe_adapter import probe_olmoe_adapter_config


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_to_tensor(hash_hex: str) -> torch.Tensor:
    if len(hash_hex) != 64:
        raise ValueError(f"expected sha256 hex length 64, got {len(hash_hex)}")
    return torch.tensor(list(hash_hex.encode("ascii")), dtype=torch.uint8)


def _tensor_to_hash(tensor: torch.Tensor) -> str:
    return bytes(int(value) for value in tensor.tolist()).decode("ascii")


def _all_gather_fixed_hash(local_hash: str) -> list[str]:
    world_size = dist.get_world_size()
    local = _hash_to_tensor(local_hash)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return [_tensor_to_hash(item) for item in gathered]


def _all_gather_int64_vector(local_vector: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    gathered = [torch.empty_like(local_vector) for _ in range(world_size)]
    dist.all_gather(gathered, local_vector)
    return torch.stack(gathered, dim=0)


def build_online_expert_placement(
    *,
    world_size: int,
    expert_count: int,
    rank_to_node_id: list[int],
) -> OnlineExpertPlacement:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if expert_count <= 0:
        raise ValueError("expert_count must be positive")
    if len(rank_to_node_id) != world_size:
        raise ValueError("rank_to_node_id length must equal world_size")
    owner_rank_by_expert = [int(expert_id % world_size) for expert_id in range(expert_count)]
    owner_node_id_by_expert = [int(rank_to_node_id[owner_rank]) for owner_rank in owner_rank_by_expert]
    placement_hash = stable_hash_dict(
        {
            "world_size": int(world_size),
            "expert_count": int(expert_count),
            "owner_rank_by_expert": owner_rank_by_expert,
            "owner_node_id_by_expert": owner_node_id_by_expert,
            "placement_mode": "expert_id_mod_world_size",
            "residency_mode": "full_checkpoint_then_local_extract",
        }
    )
    return OnlineExpertPlacement(
        world_size=int(world_size),
        expert_count=int(expert_count),
        owner_rank_by_expert=owner_rank_by_expert,
        owner_node_id_by_expert=owner_node_id_by_expert,
        placement_hash=placement_hash,
    )


def build_request_protocol_hash(*, prompts_by_rank: list[str], microbatch_id: str, layer_id: int) -> str:
    return stable_hash_dict(
        {
            "prompts_by_rank": [_sha256_text(prompt) for prompt in prompts_by_rank],
            "microbatch_id": microbatch_id,
            "layer_id": int(layer_id),
        }
    )


def build_online_route_partition(
    *,
    run_id: str,
    request_id: str,
    microbatch_id: str,
    layer_id: int,
    source_rank: int,
    source_node_id: int,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    placement: OnlineExpertPlacement,
    top_k: int,
    trace_origin: str = TraceOrigin.OBSERVED_ONLINE_WS2_ROUTE_PARTITION,
) -> OnlineRoutePartition:
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be [tokens, hidden], got {tuple(hidden_states.shape)}")
    if router_logits.ndim != 2:
        raise ValueError(f"router_logits must be [tokens, experts], got {tuple(router_logits.shape)}")
    if hidden_states.shape[0] != router_logits.shape[0]:
        raise ValueError("hidden_states token rows must match router_logits token rows")
    if router_logits.shape[-1] != placement.expert_count:
        raise ValueError("router_logits expert dimension must match placement expert_count")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    probs = torch.softmax(router_logits.detach().float(), dim=-1)
    weights, experts = torch.topk(probs, k=min(int(top_k), int(probs.shape[-1])), dim=-1)
    hidden_size = int(hidden_states.shape[-1])
    bytes_per_row = int(hidden_size * hidden_states.element_size())

    local_routes: list[OnlineRouteRecord] = []
    remote_send_routes: list[OnlineRouteRecord] = []
    all_routes: list[OnlineRouteRecord] = []
    per_peer_send_rows: dict[int, int] = {rank: 0 for rank in range(placement.world_size)}
    per_peer_send_bytes: dict[int, int] = {rank: 0 for rank in range(placement.world_size)}
    per_expert_local_bucket_rows: dict[int, int] = {}

    for token_index in range(int(hidden_states.shape[0])):
        for topk_slot, (weight, expert_id) in enumerate(
            zip(weights[token_index].tolist(), experts[token_index].tolist(), strict=False)
        ):
            destination_rank = int(placement.owner_rank_by_expert[int(expert_id)])
            destination_node_id = int(placement.owner_node_id_by_expert[int(expert_id)])
            record = OnlineRouteRecord(
                identity=OnlineRouteIdentity(
                    run_id=run_id,
                    request_id=request_id,
                    microbatch_id=microbatch_id,
                    layer_id=int(layer_id),
                    source_rank=int(source_rank),
                    destination_rank=destination_rank,
                    source_node_id=int(source_node_id),
                    destination_node_id=destination_node_id,
                    local_token_index=int(token_index),
                    topk_slot=int(topk_slot),
                    expert_id=int(expert_id),
                ),
                routing_weight=float(weight),
                payload_rows=1,
                payload_bytes=bytes_per_row,
                is_local_route=destination_rank == int(source_rank),
                is_remote_route=destination_rank != int(source_rank),
                is_cross_rank=destination_rank != int(source_rank),
                is_cross_node=destination_node_id != int(source_node_id),
            )
            all_routes.append(record)
            if record.is_local_route:
                local_routes.append(record)
                per_expert_local_bucket_rows[int(expert_id)] = per_expert_local_bucket_rows.get(int(expert_id), 0) + 1
            else:
                remote_send_routes.append(record)
                per_peer_send_rows[destination_rank] += int(record.payload_rows)
                per_peer_send_bytes[destination_rank] += int(record.payload_bytes)

    local_rows = sum(int(record.payload_rows) for record in local_routes)
    remote_rows = sum(int(record.payload_rows) for record in remote_send_routes)
    all_rows = sum(int(record.payload_rows) for record in all_routes)
    if local_rows + remote_rows != all_rows:
        raise RuntimeError("local/remote route rows do not cover all routes")

    return OnlineRoutePartition(
        run_id=run_id,
        request_id=request_id,
        microbatch_id=microbatch_id,
        layer_id=int(layer_id),
        rank=int(source_rank),
        world_size=int(placement.world_size),
        node_id=int(source_node_id),
        local_routes=local_routes,
        remote_send_routes=remote_send_routes,
        all_routes=all_routes,
        per_peer_send_rows=per_peer_send_rows,
        per_peer_send_bytes=per_peer_send_bytes,
        per_expert_local_bucket_rows=per_expert_local_bucket_rows,
    )


def build_online_layer_route_trace(
    *,
    partition: OnlineRoutePartition,
    trace_origin: str = TraceOrigin.OBSERVED_ONLINE_WS2_ROUTE_PARTITION,
) -> OnlineLayerRouteTrace:
    return OnlineLayerRouteTrace(
        trace_origin=trace_origin,
        future_information_mode=FutureInformationMode.NONE,
        layer_id=partition.layer_id,
        local_routes=list(partition.local_routes),
        remote_send_routes=list(partition.remote_send_routes),
        all_routes=list(partition.all_routes),
    )


def build_rank_manifest(
    *,
    partition: OnlineRoutePartition,
    placement: OnlineExpertPlacement,
    prompt_text: str,
    request_protocol_hash: str,
) -> RankManifest:
    payload = {
        "run_id": partition.run_id,
        "request_id": partition.request_id,
        "microbatch_id": partition.microbatch_id,
        "layer_id": partition.layer_id,
        "rank": partition.rank,
        "world_size": partition.world_size,
        "node_id": partition.node_id,
        "placement_hash": placement.placement_hash,
        "request_protocol_hash": request_protocol_hash,
        "prompt_digest": _sha256_text(prompt_text),
        "route_count": len(partition.all_routes),
        "local_route_count": len(partition.local_routes),
        "remote_route_count": len(partition.remote_send_routes),
        "remote_send_row_count": sum(int(record.payload_rows) for record in partition.remote_send_routes),
    }
    manifest_hash = stable_hash_dict(payload)
    return RankManifest(
        run_id=partition.run_id,
        request_id=partition.request_id,
        microbatch_id=partition.microbatch_id,
        layer_id=partition.layer_id,
        rank=partition.rank,
        world_size=partition.world_size,
        node_id=partition.node_id,
        placement_hash=placement.placement_hash,
        request_protocol_hash=request_protocol_hash,
        prompt_digest=_sha256_text(prompt_text),
        route_count=len(partition.all_routes),
        local_route_count=len(partition.local_routes),
        remote_route_count=len(partition.remote_send_routes),
        remote_send_row_count=sum(int(record.payload_rows) for record in partition.remote_send_routes),
        manifest_hash=manifest_hash,
    )


def _pair_digest(partition: OnlineRoutePartition, *, destination_rank: int) -> str:
    payload = [
        {
            "source_rank": record.identity.source_rank,
            "destination_rank": record.identity.destination_rank,
            "local_token_index": record.identity.local_token_index,
            "topk_slot": record.identity.topk_slot,
            "expert_id": record.identity.expert_id,
            "routing_weight": record.routing_weight,
            "payload_rows": record.payload_rows,
            "payload_bytes": record.payload_bytes,
        }
        for record in partition.remote_send_routes
        if int(record.identity.destination_rank) == int(destination_rank)
    ]
    return stable_hash_dict({"destination_rank": int(destination_rank), "routes": payload})


@dataclass(frozen=True)
class WS2CountAgreementResult:
    transport_record: TransportOperationRecord
    gathered_run_id_hashes: list[str]
    gathered_manifest_hashes: list[str]
    gathered_request_protocol_hashes: list[str]
    gathered_placement_hashes: list[str]
    gathered_layer_ids: list[int]
    gathered_send_count_matrix: list[list[int]]
    gathered_send_bytes_matrix: list[list[int]]
    gathered_pair_digests: list[list[str]]
    validation: ValidationResult


@dataclass(frozen=True)
class WS2RoutePartitionOnlyResult:
    rank: int
    trace: EpExecutionTrace
    partition: OnlineRoutePartition
    placement: OnlineExpertPlacement
    manifest: RankManifest
    agreement: WS2CountAgreementResult
    metadata: dict[str, Any]


def run_distributed_count_agreement(
    *,
    partition: OnlineRoutePartition,
    manifest: RankManifest,
    placement: OnlineExpertPlacement,
    validate_metadata: bool,
    operation_id: str = "count-exchange-0",
) -> WS2CountAgreementResult:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for count agreement")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != partition.world_size:
        raise RuntimeError(f"partition world_size {partition.world_size} does not match dist world_size {world_size}")
    if rank != partition.rank:
        raise RuntimeError(f"partition rank {partition.rank} does not match dist rank {rank}")

    send_counts_tensor = torch.tensor(
        [int(partition.per_peer_send_rows.get(peer, 0)) for peer in range(world_size)],
        dtype=torch.int64,
    )
    send_bytes_tensor = torch.tensor(
        [int(partition.per_peer_send_bytes.get(peer, 0)) for peer in range(world_size)],
        dtype=torch.int64,
    )
    digest_row = torch.stack([_hash_to_tensor(_pair_digest(partition, destination_rank=peer)) for peer in range(world_size)], dim=0)

    started = time.perf_counter()
    gathered_send_counts = _all_gather_int64_vector(send_counts_tensor)
    gathered_send_bytes = _all_gather_int64_vector(send_bytes_tensor)
    gathered_run_id_hashes = _all_gather_fixed_hash(_sha256_text(partition.run_id))
    gathered_manifest_hashes = _all_gather_fixed_hash(manifest.manifest_hash)
    gathered_request_protocol_hashes = _all_gather_fixed_hash(manifest.request_protocol_hash)
    gathered_placement_hashes = _all_gather_fixed_hash(placement.placement_hash)
    gathered_layer_ids = _all_gather_int64_vector(torch.tensor([int(partition.layer_id)], dtype=torch.int64))
    gathered_digest_rows = [torch.empty_like(digest_row) for _ in range(world_size)]
    dist.all_gather(gathered_digest_rows, digest_row)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    recv_counts_tensor = gathered_send_counts[:, rank].to(dtype=torch.int64)
    recv_bytes_tensor = gathered_send_bytes[:, rank].to(dtype=torch.int64)
    send_rows = int(send_counts_tensor.sum().item())
    recv_rows = int(recv_counts_tensor.sum().item())
    send_bytes = int(send_bytes_tensor.sum().item())
    recv_bytes = int(recv_bytes_tensor.sum().item())

    details: dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "run_id_hashes": gathered_run_id_hashes,
        "manifest_hashes": gathered_manifest_hashes,
        "placement_hashes": gathered_placement_hashes,
        "request_protocol_hashes": gathered_request_protocol_hashes,
        "layer_ids": [int(value[0].item()) for value in gathered_layer_ids],
    }

    try:
        if manifest.run_id != partition.run_id:
            raise RuntimeError(
                f"manifest run_id mismatch on rank {rank}: manifest={manifest.run_id} partition={partition.run_id}"
            )
        if manifest.placement_hash != placement.placement_hash:
            raise RuntimeError(
                f"placement hash mismatch on rank {rank}: manifest={manifest.placement_hash} placement={placement.placement_hash}"
            )
        if len(set(gathered_request_protocol_hashes)) != 1:
            raise RuntimeError(
                f"request protocol hash mismatch on rank {rank}: {gathered_request_protocol_hashes}"
            )
        if len(set(gathered_run_id_hashes)) != 1:
            raise RuntimeError(f"run id mismatch across ranks on rank {rank}: {gathered_run_id_hashes}")
        if len(set(gathered_placement_hashes)) != 1:
            raise RuntimeError(f"placement hash mismatch across ranks on rank {rank}: {gathered_placement_hashes}")
        if len({int(value[0].item()) for value in gathered_layer_ids}) != 1:
            raise RuntimeError(
                f"layer id mismatch across ranks on rank {rank}: {[int(value[0].item()) for value in gathered_layer_ids]}"
            )
        for peer_rank, expected_hash in enumerate(gathered_manifest_hashes):
            if len(expected_hash) != 64:
                raise RuntimeError(f"invalid manifest hash length from peer {peer_rank}: {expected_hash!r}")
        if int(sum(int(record.payload_rows) for record in partition.remote_send_routes)) != send_rows:
            raise RuntimeError(
                f"remote route row mismatch on rank {rank}: routes={sum(int(record.payload_rows) for record in partition.remote_send_routes)} "
                f"send_counts_sum={send_rows}"
            )
        for record in partition.remote_send_routes:
            if record.identity.destination_rank == rank:
                raise RuntimeError(f"remote send contains self route on rank {rank}: {record.to_dict()}")
            if record.identity.destination_rank < 0 or record.identity.destination_rank >= world_size:
                raise RuntimeError(
                    f"illegal destination rank on rank {rank}: {record.identity.destination_rank}"
                )
        for peer_rank in range(world_size):
            actual_recv = int(recv_counts_tensor[peer_rank].item())
            expected_recv = int(gathered_send_counts[peer_rank, rank].item())
            if actual_recv != expected_recv:
                raise RuntimeError(
                    f"recv count mismatch on rank {rank} from peer {peer_rank}: expected {expected_recv} actual {actual_recv} "
                    f"run_id={partition.run_id} placement_hash={placement.placement_hash} layer_id={partition.layer_id}"
                )
        if validate_metadata:
            for peer_rank in range(world_size):
                if peer_rank == rank:
                    continue
                incoming_rows = int(recv_counts_tensor[peer_rank].item())
                incoming_digest = _tensor_to_hash(gathered_digest_rows[peer_rank][rank])
                if incoming_rows > 0 and not incoming_digest.strip("0"):
                    raise RuntimeError(
                        f"expected receive digest missing on rank {rank} from peer {peer_rank}: expected count {incoming_rows} actual digest {incoming_digest!r} "
                        f"run_id={partition.run_id} placement_hash={placement.placement_hash} layer_id={partition.layer_id}"
                    )
    except Exception as exc:
        validation = ValidationResult(
            correctness_status="metadata_failed",
            numerical_correctness_pass=None,
            details={"error": str(exc), **details},
        )
        transport_record = TransportOperationRecord(
            run_id=partition.run_id,
            rank=rank,
            world_size=world_size,
            operation_id=operation_id,
            phase="count_exchange",
            backend=str(dist.get_backend()),
            operation_kind="metadata_agreement",
            hidden_payload_transferred=False,
            send_counts=[int(value) for value in send_counts_tensor.tolist()],
            recv_counts=[int(value) for value in recv_counts_tensor.tolist()],
            send_rows=send_rows,
            recv_rows=recv_rows,
            send_bytes=send_bytes,
            recv_bytes=recv_bytes,
            post_ms=elapsed_ms,
            wait_ms=0.0,
            wall_elapsed_ms=elapsed_ms,
            success=False,
            details={"error": str(exc)},
        )
        return WS2CountAgreementResult(
            transport_record=transport_record,
            gathered_run_id_hashes=gathered_run_id_hashes,
            gathered_manifest_hashes=gathered_manifest_hashes,
            gathered_request_protocol_hashes=gathered_request_protocol_hashes,
            gathered_placement_hashes=gathered_placement_hashes,
            gathered_layer_ids=[int(value[0].item()) for value in gathered_layer_ids],
            gathered_send_count_matrix=[[int(value) for value in row.tolist()] for row in gathered_send_counts],
            gathered_send_bytes_matrix=[[int(value) for value in row.tolist()] for row in gathered_send_bytes],
            gathered_pair_digests=[
                [_tensor_to_hash(gathered_digest_rows[src][dst]) for dst in range(world_size)]
                for src in range(world_size)
            ],
            validation=validation,
        )

    validation = ValidationResult(
        correctness_status="metadata_passed",
        numerical_correctness_pass=None,
        details=details,
    )
    transport_record = TransportOperationRecord(
        run_id=partition.run_id,
        rank=rank,
        world_size=world_size,
        operation_id=operation_id,
        phase="count_exchange",
        backend=str(dist.get_backend()),
        operation_kind="metadata_agreement",
        hidden_payload_transferred=False,
        send_counts=[int(value) for value in send_counts_tensor.tolist()],
        recv_counts=[int(value) for value in recv_counts_tensor.tolist()],
        send_rows=send_rows,
        recv_rows=recv_rows,
        send_bytes=send_bytes,
        recv_bytes=recv_bytes,
        post_ms=elapsed_ms,
        wait_ms=0.0,
        wall_elapsed_ms=elapsed_ms,
        success=True,
        details={
            "placement_hash": placement.placement_hash,
            "manifest_hash": manifest.manifest_hash,
            "request_protocol_hash": manifest.request_protocol_hash,
        },
    )
    return WS2CountAgreementResult(
        transport_record=transport_record,
        gathered_run_id_hashes=gathered_run_id_hashes,
        gathered_manifest_hashes=gathered_manifest_hashes,
        gathered_request_protocol_hashes=gathered_request_protocol_hashes,
        gathered_placement_hashes=gathered_placement_hashes,
        gathered_layer_ids=[int(value[0].item()) for value in gathered_layer_ids],
        gathered_send_count_matrix=[[int(value) for value in row.tolist()] for row in gathered_send_counts],
        gathered_send_bytes_matrix=[[int(value) for value in row.tolist()] for row in gathered_send_bytes],
        gathered_pair_digests=[
            [_tensor_to_hash(gathered_digest_rows[src][dst]) for dst in range(world_size)]
            for src in range(world_size)
        ],
        validation=validation,
    )


def build_ws2_partition_trace(
    *,
    partition: OnlineRoutePartition,
    placement: OnlineExpertPlacement,
    manifest: RankManifest,
    agreement: WS2CountAgreementResult,
) -> EpExecutionTrace:
    return EpExecutionTrace(
        trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_ROUTE_PARTITION,
        future_information_mode=FutureInformationMode.NONE,
        stage_timings=[
            RankStageTiming(
                rank=partition.rank,
                stage="count_exchange",
                wall_ms=float(agreement.transport_record.wall_elapsed_ms),
            )
        ],
        metadata={
            "world_size": partition.world_size,
            "layer_id": partition.layer_id,
            "request_id": partition.request_id,
            "microbatch_id": partition.microbatch_id,
            "node_id": partition.node_id,
            "placement_hash": placement.placement_hash,
            "manifest_hash": manifest.manifest_hash,
        },
        online_route_traces=[build_online_layer_route_trace(partition=partition)],
        rank_manifests=[manifest],
        expert_placements=[placement],
        transport_operations=[agreement.transport_record],
        validation_results=[agreement.validation],
    )


def _capture_local_layer_observation(
    *,
    model_id: str,
    model_path: str | None,
    prompt_text: str,
    layer_index: int,
    precision: str,
    device_index: int,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
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
    layer_id = int(moe_layer_ids[layer_index])
    layer = model.model.layers[layer_id]
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, inputs, output):
        captured["mlp_input"] = inputs[0].detach()
        captured["router_logits"] = output[1].detach()

    handle = layer.mlp.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model(**encoded, output_router_logits=True, output_hidden_states=True, return_dict=True, use_cache=False)
    finally:
        handle.remove()
    if {"mlp_input", "router_logits"} - set(captured):
        raise RuntimeError("failed to capture MoE layer input/router during ws2 route partition collection")
    hidden_states = captured["mlp_input"].squeeze(0)
    router_logits = captured["router_logits"]
    if router_logits.ndim == 3:
        router_logits = router_logits.squeeze(0)
    return hidden_states, router_logits, layer_id, {
        "model_revision": resolved_revision,
        "device": resolved_device,
        "dtype": dtype,
        "adapter_probe": probe.to_dict(),
        "expert_count": int(model.config.num_experts),
        "top_k": int(model.config.num_experts_per_tok),
    }


def run_world_size_two_route_partition_only(
    *,
    run_id: str,
    model_id: str,
    model_path: str | None,
    prompts_by_rank: list[str],
    layer_index: int,
    precision: str,
    device_index: int,
    validate_metadata: bool,
    allow_identical_prompts: bool = False,
) -> WS2RoutePartitionOnlyResult:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before running ws2 route partition")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if world_size != 2:
        raise RuntimeError(f"ws2 route partition only path requires world_size=2, received {world_size}")
    if len(prompts_by_rank) != world_size:
        raise RuntimeError(f"prompts_by_rank length must equal world_size ({world_size})")
    if not allow_identical_prompts and prompts_by_rank[0] == prompts_by_rank[1]:
        raise RuntimeError("prompt-rank0 and prompt-rank1 must differ unless allow_identical_prompts=true")

    prompt_text = prompts_by_rank[rank]
    hidden_states, router_logits, layer_id, observation_metadata = _capture_local_layer_observation(
        model_id=model_id,
        model_path=model_path,
        prompt_text=prompt_text,
        layer_index=layer_index,
        precision=precision,
        device_index=device_index,
    )
    rank_to_node_id = [0 for _ in range(world_size)]
    placement = build_online_expert_placement(
        world_size=world_size,
        expert_count=int(router_logits.shape[-1]),
        rank_to_node_id=rank_to_node_id,
    )
    request_id = f"rank-{rank}-request"
    microbatch_id = "ws2-mb-0"
    request_protocol_hash = build_request_protocol_hash(
        prompts_by_rank=prompts_by_rank,
        microbatch_id=microbatch_id,
        layer_id=layer_id,
    )
    partition = build_online_route_partition(
        run_id=run_id,
        request_id=request_id,
        microbatch_id=microbatch_id,
        layer_id=layer_id,
        source_rank=rank,
        source_node_id=rank_to_node_id[rank],
        hidden_states=hidden_states,
        router_logits=router_logits,
        placement=placement,
        top_k=int(observation_metadata["top_k"]),
    )
    manifest = build_rank_manifest(
        partition=partition,
        placement=placement,
        prompt_text=prompt_text,
        request_protocol_hash=request_protocol_hash,
    )
    agreement = run_distributed_count_agreement(
        partition=partition,
        manifest=manifest,
        placement=placement,
        validate_metadata=validate_metadata,
    )
    trace = build_ws2_partition_trace(
        partition=partition,
        placement=placement,
        manifest=manifest,
        agreement=agreement,
    )
    metadata = {
        **observation_metadata,
        "execution_mode": "online_ws2_route_partition_only",
        "claim_scope": "distributed_route_partition_and_count_agreement_only",
        "is_real_ep_runtime": False,
        "is_real_ep_transport": False,
        "is_transport_calibration_trace": False,
        "correctness_status": agreement.validation.correctness_status,
        "source_ownership_mode": "dist_rank_local_prompt",
        "expert_residency_mode": "full_checkpoint_then_local_extract",
        "transport_backend": "torch_distributed_metadata_agreement",
        "route_partition_only": True,
        "validate_metadata": bool(validate_metadata),
    }
    return WS2RoutePartitionOnlyResult(
        rank=rank,
        trace=trace,
        partition=partition,
        placement=placement,
        manifest=manifest,
        agreement=agreement,
        metadata=metadata,
    )
