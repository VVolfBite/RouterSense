from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from ...contracts import (
    EpExecutionTrace,
    FutureInformationMode,
    OnlineExpertPlacement,
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
from .ws2_stage1 import WS2CountAgreementResult, build_online_layer_route_trace


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_to_tensor(hash_hex: str) -> torch.Tensor:
    return torch.tensor(list(hash_hex.encode("ascii")), dtype=torch.uint8)


def _tensor_to_hash(tensor: torch.Tensor) -> str:
    return bytes(int(value) for value in tensor.tolist()).decode("ascii")


def _all_gather_hash_matrix(local_hashes: list[str]) -> list[list[str]]:
    local_tensor = torch.stack([_hash_to_tensor(value) for value in local_hashes], dim=0)
    gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_tensor)
    return [
        [_tensor_to_hash(gathered[src][dst]) for dst in range(local_tensor.shape[0])]
        for src in range(len(gathered))
    ]


def _ordered_remote_routes(partition: OnlineRoutePartition) -> list[OnlineRouteRecord]:
    return sorted(
        partition.remote_send_routes,
        key=lambda record: (
            int(record.identity.destination_rank),
            int(record.identity.local_token_index),
            int(record.identity.topk_slot),
            int(record.identity.expert_id),
        ),
    )


def _route_metadata_int_tensor(routes: list[OnlineRouteRecord]) -> torch.Tensor:
    if not routes:
        return torch.empty((0, 8), dtype=torch.int64)
    rows = [
        [
            int(route.identity.source_rank),
            int(route.identity.destination_rank),
            int(route.identity.source_node_id),
            int(route.identity.destination_node_id),
            int(route.identity.local_token_index),
            int(route.identity.topk_slot),
            int(route.identity.expert_id),
            int(route.payload_bytes),
        ]
        for route in routes
    ]
    return torch.tensor(rows, dtype=torch.int64)


def _route_metadata_weight_tensor(routes: list[OnlineRouteRecord]) -> torch.Tensor:
    if not routes:
        return torch.empty((0,), dtype=torch.float32)
    return torch.tensor([float(route.routing_weight) for route in routes], dtype=torch.float32)


def _build_send_hidden_tensor(hidden_states: torch.Tensor, routes: list[OnlineRouteRecord]) -> torch.Tensor:
    if not routes:
        return hidden_states.new_empty((0, hidden_states.shape[-1]))
    return torch.stack(
        [hidden_states[int(route.identity.local_token_index)] for route in routes],
        dim=0,
    )


def _tensor_digest_by_peer(
    *,
    payload: torch.Tensor,
    split_rows: list[int],
) -> list[str]:
    digests: list[str] = []
    start = 0
    for rows in split_rows:
        rows = int(rows)
        segment = payload[start : start + rows]
        digests.append(_sha256_bytes(segment.detach().cpu().contiguous().numpy().tobytes()))
        start += rows
    return digests


def _route_digest_by_peer(
    *,
    routes: list[OnlineRouteRecord],
    world_size: int,
) -> list[str]:
    by_peer: list[list[dict[str, Any]]] = [[] for _ in range(world_size)]
    for route in routes:
        by_peer[int(route.identity.destination_rank)].append(route.to_dict())
    return [stable_hash_dict({"peer": peer, "routes": payload}) for peer, payload in enumerate(by_peer)]


def _decode_received_routes(
    *,
    run_id: str,
    microbatch_id: str,
    layer_id: int,
    metadata_int: torch.Tensor,
    metadata_weight: torch.Tensor,
) -> list[OnlineRouteRecord]:
    if metadata_int.shape[0] != metadata_weight.shape[0]:
        raise RuntimeError("received metadata int/weight row counts do not match")
    received: list[OnlineRouteRecord] = []
    for row_index in range(int(metadata_int.shape[0])):
        row = metadata_int[row_index]
        source_rank = int(row[0].item())
        destination_rank = int(row[1].item())
        source_node_id = int(row[2].item())
        destination_node_id = int(row[3].item())
        local_token_index = int(row[4].item())
        topk_slot = int(row[5].item())
        expert_id = int(row[6].item())
        payload_bytes = int(row[7].item())
        received.append(
            OnlineRouteRecord(
                identity=OnlineRouteIdentity(
                    run_id=run_id,
                    request_id=f"rank-{source_rank}-request",
                    microbatch_id=microbatch_id,
                    layer_id=layer_id,
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    source_node_id=source_node_id,
                    destination_node_id=destination_node_id,
                    local_token_index=local_token_index,
                    topk_slot=topk_slot,
                    expert_id=expert_id,
                ),
                routing_weight=float(metadata_weight[row_index].item()),
                payload_rows=1,
                payload_bytes=payload_bytes,
                is_local_route=source_rank == destination_rank,
                is_remote_route=source_rank != destination_rank,
                is_cross_rank=source_rank != destination_rank,
                is_cross_node=source_node_id != destination_node_id,
            )
        )
    return received


@dataclass(frozen=True)
class WS2HiddenDispatchResult:
    transport_record: TransportOperationRecord
    received_routes: list[OnlineRouteRecord]
    received_hidden_states: torch.Tensor
    validation: ValidationResult
    gathered_hidden_digests: list[list[str]]
    gathered_route_digests: list[list[str]]


def execute_ws2_hidden_dispatch_only(
    *,
    hidden_states: torch.Tensor,
    partition: OnlineRoutePartition,
    manifest: RankManifest,
    placement: OnlineExpertPlacement,
    agreement: WS2CountAgreementResult,
    operation_id: str = "dispatch-hidden-0",
) -> WS2HiddenDispatchResult:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized for hidden dispatch")
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be [tokens, hidden], got {tuple(hidden_states.shape)}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ordered_routes = _ordered_remote_routes(partition)
    send_hidden = _build_send_hidden_tensor(hidden_states, ordered_routes)
    send_metadata_int = _route_metadata_int_tensor(ordered_routes)
    send_metadata_weight = _route_metadata_weight_tensor(ordered_routes)
    send_splits = [int(partition.per_peer_send_rows.get(peer, 0)) for peer in range(world_size)]
    recv_splits = [int(value) for value in agreement.transport_record.recv_counts]
    hidden_size = int(hidden_states.shape[-1])
    recv_hidden = hidden_states.new_empty((sum(recv_splits), hidden_size))
    recv_metadata_int = torch.empty((sum(recv_splits), 8), dtype=torch.int64)
    recv_metadata_weight = torch.empty((sum(recv_splits),), dtype=torch.float32)

    local_hidden_digests = _tensor_digest_by_peer(payload=send_hidden, split_rows=send_splits)
    local_route_digests = _route_digest_by_peer(routes=ordered_routes, world_size=world_size)
    gathered_hidden_digests = _all_gather_hash_matrix(local_hidden_digests)
    gathered_route_digests = _all_gather_hash_matrix(local_route_digests)

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
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    received_routes = _decode_received_routes(
        run_id=partition.run_id,
        microbatch_id=partition.microbatch_id,
        layer_id=partition.layer_id,
        metadata_int=recv_metadata_int.cpu(),
        metadata_weight=recv_metadata_weight.cpu(),
    )
    details: dict[str, Any] = {
        "rank": rank,
        "manifest_hash": manifest.manifest_hash,
        "placement_hash": placement.placement_hash,
        "received_route_count": len(received_routes),
    }
    try:
        if len(received_routes) != sum(recv_splits):
            raise RuntimeError(
                f"received route count mismatch on rank {rank}: routes={len(received_routes)} recv_splits_sum={sum(recv_splits)}"
            )
        if int(recv_hidden.shape[0]) != len(received_routes):
            raise RuntimeError(
                f"received hidden row mismatch on rank {rank}: hidden_rows={int(recv_hidden.shape[0])} routes={len(received_routes)}"
            )
        start = 0
        for source_rank, rows in enumerate(recv_splits):
            rows = int(rows)
            hidden_digest = _sha256_bytes(recv_hidden[start : start + rows].detach().cpu().contiguous().numpy().tobytes())
            route_digest = stable_hash_dict(
                {
                    "peer": int(rank),
                    "routes": [route.to_dict() for route in received_routes[start : start + rows]],
                }
            )
            expected_hidden_digest = gathered_hidden_digests[source_rank][rank]
            expected_route_digest = gathered_route_digests[source_rank][rank]
            if hidden_digest != expected_hidden_digest:
                raise RuntimeError(
                    f"received hidden digest mismatch on rank {rank} from peer {source_rank}: expected {expected_hidden_digest} actual {hidden_digest}"
                )
            if route_digest != expected_route_digest:
                raise RuntimeError(
                    f"received route digest mismatch on rank {rank} from peer {source_rank}: expected {expected_route_digest} actual {route_digest}"
                )
            for route in received_routes[start : start + rows]:
                if int(route.identity.destination_rank) != rank:
                    raise RuntimeError(
                        f"received route destination mismatch on rank {rank}: {route.identity.destination_rank}"
                    )
            start += rows
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
            phase="dispatch",
            backend=str(dist.get_backend()),
            operation_kind="hidden_dispatch_only",
            hidden_payload_transferred=True,
            send_counts=list(send_splits),
            recv_counts=list(recv_splits),
            send_rows=int(sum(send_splits)),
            recv_rows=int(sum(recv_splits)),
            send_bytes=int(send_hidden.numel() * send_hidden.element_size()),
            recv_bytes=int(recv_hidden.numel() * recv_hidden.element_size()),
            post_ms=elapsed_ms,
            wait_ms=0.0,
            wall_elapsed_ms=elapsed_ms,
            success=False,
            details={"error": str(exc)},
        )
        return WS2HiddenDispatchResult(
            transport_record=transport_record,
            received_routes=received_routes,
            received_hidden_states=recv_hidden,
            validation=validation,
            gathered_hidden_digests=gathered_hidden_digests,
            gathered_route_digests=gathered_route_digests,
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
        phase="dispatch",
        backend=str(dist.get_backend()),
        operation_kind="hidden_dispatch_only",
        hidden_payload_transferred=True,
        send_counts=list(send_splits),
        recv_counts=list(recv_splits),
        send_rows=int(sum(send_splits)),
        recv_rows=int(sum(recv_splits)),
        send_bytes=int(send_hidden.numel() * send_hidden.element_size()),
        recv_bytes=int(recv_hidden.numel() * recv_hidden.element_size()),
        post_ms=elapsed_ms,
        wait_ms=0.0,
        wall_elapsed_ms=elapsed_ms,
        success=True,
        details=details,
    )
    return WS2HiddenDispatchResult(
        transport_record=transport_record,
        received_routes=received_routes,
        received_hidden_states=recv_hidden,
        validation=validation,
        gathered_hidden_digests=gathered_hidden_digests,
        gathered_route_digests=gathered_route_digests,
    )


def build_ws2_hidden_dispatch_trace(
    *,
    partition: OnlineRoutePartition,
    placement: OnlineExpertPlacement,
    manifest: RankManifest,
    hidden_dispatch: WS2HiddenDispatchResult,
) -> EpExecutionTrace:
    return EpExecutionTrace(
        trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_HIDDEN_DISPATCH,
        future_information_mode=FutureInformationMode.NONE,
        stage_timings=[
            RankStageTiming(
                rank=partition.rank,
                stage="dispatch",
                wall_ms=float(hidden_dispatch.transport_record.wall_elapsed_ms),
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
        online_route_traces=[build_online_layer_route_trace(partition=partition, trace_origin=TraceOrigin.OBSERVED_ONLINE_WS2_HIDDEN_DISPATCH)],
        rank_manifests=[manifest],
        expert_placements=[placement],
        transport_operations=[hidden_dispatch.transport_record],
        validation_results=[hidden_dispatch.validation],
    )
