"""Runtime helper for optional expert-route trace capture.

Default capture stays compact:
- aggregate source_rank x expert counts
- aggregate weighted counts when available
- no per-token expert rows in the default JSONL path

Heavy per-token trace is intentionally not enabled here.
"""

from __future__ import annotations

from typing import Any

import torch

from rs.scheduling.traffic_matrix import canonicalize_remote_matrix

from .expert_to_traffic import compare_reconstructed_traffic, source_expert_counts_to_traffic_matrix
from .expert_trace import ExpertRouteRecord, SourceExpertCountMatrix, aggregate_route_records


def maybe_capture_expert_route_trace(
    *,
    recorder,
    layer_id: int,
    rank: int,
    source_rank: int,
    dispatcher: Any,
    selected_experts: Any,
    routing_weights: Any,
    top_k: int,
    token_count: int,
    hidden_shape: tuple[int, ...] | None,
    bytes_per_token: int,
    per_peer_bytes: tuple[int, ...] | None,
    ep_group_ranks: tuple[int, ...],
    enabled: bool,
) -> None:
    if not enabled or recorder is None:
        return
    expert_ids, weights, warnings = _extract_expert_rows(
        dispatcher=dispatcher,
        selected_experts=selected_experts,
        routing_weights=routing_weights,
    )
    num_experts = _infer_num_experts(dispatcher=dispatcher, expert_ids=expert_ids)
    world_size = int(len(ep_group_ranks) or 1)
    if not expert_ids or num_experts <= 0:
        for message in warnings or ("selected_experts_unavailable",):
            recorder.record_expert_trace_warning(
                {
                    "layer_id": int(layer_id),
                    "rank": int(rank),
                    "source_rank": int(source_rank),
                    "warning": str(message),
                    "selected_experts_available": False,
                    "routing_weights_available": weights is not None,
                }
            )
        return
    route_record = ExpertRouteRecord(
        layer_id=int(layer_id),
        rank=int(rank),
        token_count=int(token_count if token_count > 0 else len(expert_ids)),
        top_k=int(top_k),
        expert_ids=expert_ids,
        routing_weights=weights,
        source_rank=int(source_rank),
    )
    counts = aggregate_route_records((route_record,), world_size=world_size, num_experts=num_experts, use_routing_weights=weights is not None)
    expert_to_rank_map = _infer_expert_to_rank_map(
        num_experts=num_experts,
        world_size=world_size,
        ep_group_ranks=ep_group_ranks,
        dispatcher=dispatcher,
    )
    expert_to_global_rank_map = tuple(
        int(ep_group_ranks[local_rank]) if 0 <= int(local_rank) < len(ep_group_ranks) else int(local_rank)
        for local_rank in expert_to_rank_map
    )
    counts = SourceExpertCountMatrix(
        layer_id=counts.layer_id,
        world_size=counts.world_size,
        num_experts=counts.num_experts,
        counts=counts.counts,
        weighted_counts=counts.weighted_counts,
        rank=int(rank),
        source_rank=int(source_rank),
        expert_to_rank_map=tuple(int(v) for v in expert_to_rank_map),
        tokens_per_source_rank=counts.tokens_per_source_rank,
        bytes_per_token=max(1, int(bytes_per_token)),
        selected_experts_available=True,
        routing_weights_available=weights is not None,
    )
    actual_matrix = _local_row_actual_matrix(per_peer_bytes=per_peer_bytes, world_size=world_size, source_rank=source_rank)
    reconstructed = source_expert_counts_to_traffic_matrix(
        counts,
        {expert_id: int(dst_rank) for expert_id, dst_rank in enumerate(expert_to_rank_map)},
        bytes_per_token=int(counts.bytes_per_token),
        top_k=int(top_k),
    )
    audit = compare_reconstructed_traffic(reconstructed, actual_matrix)
    recorder.record_expert_route_trace(
        {
            "layer_id": int(layer_id),
            "rank": int(rank),
            "source_rank": int(source_rank),
            "world_size": int(world_size),
            "num_experts": int(num_experts),
            "token_count": int(route_record.token_count),
            "top_k": int(route_record.top_k),
            "selected_experts_available": True,
            "routing_weights_available": weights is not None,
            "hidden_shape": list(hidden_shape) if hidden_shape is not None else None,
            "expert_to_rank_map": list(expert_to_rank_map),
            "expert_to_ep_local_rank_map": list(expert_to_rank_map),
            "expert_to_global_rank_map": list(expert_to_global_rank_map),
            "source_expert_counts": [list(row) for row in counts.counts],
            "weighted_source_expert_counts": None
            if counts.weighted_counts is None
            else [[float(value) for value in row] for row in counts.weighted_counts],
            "tokens_per_source_rank": None
            if counts.tokens_per_source_rank is None
            else [int(value) for value in counts.tokens_per_source_rank],
            "bytes_per_token": int(counts.bytes_per_token),
            "heavy_debug_trace": False,
        }
    )
    source_count_payload = counts.to_dict()
    source_count_payload["top_k"] = int(top_k)
    source_count_payload["source_expert_counts"] = [list(row) for row in counts.counts]
    source_count_payload["weighted_source_expert_counts"] = None
    if counts.weighted_counts is not None:
        source_count_payload["weighted_source_expert_counts"] = [
            [float(value) for value in row] for row in counts.weighted_counts
        ]
    recorder.record_source_expert_counts(source_count_payload)
    recorder.record_expert_to_traffic_audit(
        {
            "layer_id": int(layer_id),
            "rank": int(rank),
            "source_rank": int(source_rank),
            "scope": "local_row_only",
            "expert_to_rank_map": list(expert_to_rank_map),
            "expert_to_ep_local_rank_map": list(expert_to_rank_map),
            "expert_to_global_rank_map": list(expert_to_global_rank_map),
            "actual_matrix": [list(row) for row in canonicalize_remote_matrix(actual_matrix)],
            "reconstructed_matrix": [list(row) for row in reconstructed],
            **audit.to_dict(),
        }
    )
    for message in warnings:
        recorder.record_expert_trace_warning(
            {
                "layer_id": int(layer_id),
                "rank": int(rank),
                "source_rank": int(source_rank),
                "warning": str(message),
                "selected_experts_available": True,
                "routing_weights_available": weights is not None,
            }
        )


def _extract_expert_rows(*, dispatcher: Any, selected_experts: Any, routing_weights: Any) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...] | None, tuple[str, ...]]:
    warnings: list[str] = []
    selected = _to_tensor(selected_experts)
    weights = _to_tensor(routing_weights)
    if selected is None and getattr(dispatcher, "_comm_manager", None) is not None:
        selected = _to_tensor(getattr(dispatcher._comm_manager, "token_indices", None))
        weights = weights if weights is not None else _to_tensor(getattr(dispatcher._comm_manager, "token_probs", None))
    if selected is not None:
        if selected.ndim == 1:
            selected = selected.view(-1, 1)
        expert_rows = tuple(tuple(int(value) for value in row) for row in selected.detach().cpu().tolist())
        weight_rows = None
        if weights is not None:
            if weights.ndim == 1:
                weights = weights.view(-1, 1)
            weight_rows = tuple(tuple(float(value) for value in row) for row in weights.detach().cpu().tolist())
        return expert_rows, weight_rows, tuple(warnings)
    routing_map = _to_tensor(getattr(dispatcher, "routing_map", None))
    probs = _to_tensor(getattr(dispatcher, "probs", None))
    if routing_map is None and getattr(dispatcher, "_comm_manager", None) is not None:
        routing_map = _to_tensor(getattr(dispatcher._comm_manager, "routing_map", None))
        probs = probs if probs is not None else _to_tensor(getattr(dispatcher._comm_manager, "token_probs", None))
    if routing_map is None:
        warnings.append("selected_experts_and_routing_map_unavailable")
        return (), None, tuple(warnings)
    if routing_map.ndim > 2:
        routing_map = routing_map.reshape(routing_map.shape[0], -1)
    routing_map = routing_map.detach().cpu()
    if probs is not None and probs.ndim > 2:
        probs = probs.reshape(probs.shape[0], -1)
    expert_rows: list[tuple[int, ...]] = []
    weight_rows: list[tuple[float, ...]] = []
    for token_idx in range(int(routing_map.shape[0])):
        expert_idx = torch.nonzero(routing_map[token_idx], as_tuple=False).reshape(-1)
        expert_rows.append(tuple(int(value) for value in expert_idx.tolist()))
        if probs is not None:
            probs_cpu = probs.detach().cpu()
            weight_rows.append(tuple(float(probs_cpu[token_idx, int(idx)]) for idx in expert_idx.tolist()))
    if probs is None:
        warnings.append("routing_weights_unavailable")
    return tuple(expert_rows), None if probs is None else tuple(weight_rows), tuple(warnings)


def _infer_num_experts(*, dispatcher: Any, expert_ids: tuple[tuple[int, ...], ...]) -> int:
    max_from_rows = max((max(row) for row in expert_ids if row), default=-1) + 1
    local_indices = getattr(dispatcher, "local_expert_indices", None) or ()
    max_local = max((int(value) for value in local_indices), default=-1) + 1
    comm_manager = getattr(dispatcher, "_comm_manager", None)
    from_comm = int(getattr(comm_manager, "num_experts", 0) or 0) if comm_manager is not None else 0
    return int(max(max_from_rows, max_local, from_comm, 0))


def _infer_expert_to_rank_map(*, num_experts: int, world_size: int, ep_group_ranks: tuple[int, ...], dispatcher: Any) -> tuple[int, ...]:
    local_indices = tuple(int(v) for v in (getattr(dispatcher, "local_expert_indices", None) or ()))
    num_local_experts = int(len(local_indices) or max(1, num_experts // max(1, world_size)))
    mapping: list[int] = []
    for expert_id in range(int(num_experts)):
        rank_index = min(max(0, expert_id // max(1, num_local_experts)), max(0, world_size - 1))
        mapping.append(int(rank_index))
    return tuple(mapping)


def _local_row_actual_matrix(*, per_peer_bytes: tuple[int, ...] | None, world_size: int, source_rank: int) -> tuple[tuple[int, ...], ...]:
    matrix = [[0 for _ in range(world_size)] for _ in range(world_size)]
    if per_peer_bytes is not None:
        for dst_rank in range(min(world_size, len(per_peer_bytes))):
            matrix[int(source_rank)][dst_rank] = int(per_peer_bytes[dst_rank])
    return canonicalize_remote_matrix(tuple(tuple(int(value) for value in row) for row in matrix))


def _to_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    return None


__all__ = ["maybe_capture_expert_route_trace"]
