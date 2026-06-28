from __future__ import annotations

import json
import hashlib
import math
import os
import random
import socket
import statistics
import time
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .runner import RunnerConfig, _collect_microbatch_routing, _load_model_bundle, _load_prompts
from .scheduler import BucketRecord, RuntimeState, SchedulerInput, create_scheduler


PLACEMENT_POLICIES = {"modulo", "balanced", "hotspot"}
EMPTY_RUNTIME_STATE = RuntimeState()
STRATEGIES = [
    "fifo",
    "random-order",
    "state-only",
    "strong-state",
    "strong-state-sort",
    "strong-state-packer",
    "oracle-minimax-packer",
    "lina-inspired",
    "dependency-only",
    "shuffled-dependency",
    "full",
    "full-with-shuffled-dependency",
]


@dataclass
class ProtocolConfig:
    model_key: str
    model_path: str
    output_dir: str
    artifact_dir: str
    placement_policy: str = "balanced"
    prompt_path: str | None = None
    layer: str = "auto"
    microbatch_size: int = 8
    microbatch_count: int = 16
    max_length: int = 128
    precision: str = "bf16"
    top_k: int | None = None
    seed: int = 42
    world_size: int = 1
    warmup_repetitions: int = 3
    repetitions: int = 8
    strategy_order_mode: str = "counterbalanced"
    compute_scale: float = 1.0
    hidden_dim_scale: float = 0.5
    intermediate_scale: float = 2.0
    release_round_size: int = 4
    scenario: str = "natural"
    regime: str = "balanced"
    hotspot_expert_fraction: float = 0.25
    audit_plan_id: str | None = None
    audit_strategy_subset: list[str] | None = None
    plan_snapshot_path: str | None = None
    origin_sharding: str = "round-robin"
    require_remote_traffic: bool = True


@dataclass
class PlacementEntry:
    expert_id: int
    destination_rank: int


@dataclass
class WorkloadBucket:
    bucket_id: str
    route_id: str
    token_ids: list[int]
    token_positions: list[int]
    origin_rank: int
    destination_id: int
    destination_rank: int
    expert_id: int
    layer_id: int
    microbatch_id: int
    token_count: int
    payload_rows: int
    hidden_dim: int
    intermediate_dim: int
    estimated_service_units: float
    payload_bytes: int
    source_count: int
    source_coverage: float
    coactive_peer_degree: float
    coactive_event_density: float
    position_spread: float
    bridge_score: float
    route_share: float
    density_over_mean: float
    size_norm: float
    inverse_size_rank_norm: float
    is_hot_bucket: bool
    route_item_indices: list[int] = field(default_factory=list)
    route_item_ids: list[str] = field(default_factory=list)
    route_ranks: list[int] = field(default_factory=list)

    def to_scheduler_record(self) -> BucketRecord:
        flow_key = f"{self.origin_rank}->{self.destination_rank}"
        return_key = f"{self.destination_rank}->{self.origin_rank}"
        estimated_dispatch_cost = self.payload_bytes / float(1 << 20)
        estimated_compute_cost = float(self.payload_rows)
        estimated_return_cost = self.payload_bytes / float(1 << 20)
        return BucketRecord(
            bucket_id=self.bucket_id,
            route_item_ids=[self.route_id],
            route_item_indices=list(self.route_item_indices),
            origin_rank=self.origin_rank,
            destination_id=self.destination_id,
            destination_rank=self.destination_rank,
            expert_id=self.expert_id,
            arrival_index=0,
            token_count=self.token_count,
            payload_rows=self.payload_rows,
            payload_bytes=self.payload_bytes,
            token_positions=list(self.token_positions),
            source_count=self.source_count,
            source_coverage=self.source_coverage,
            coactive_peer_degree=self.coactive_peer_degree,
            coactive_event_density=self.coactive_event_density,
            position_spread=self.position_spread,
            bridge_score=self.bridge_score,
            route_share=self.route_share,
            density_over_mean=self.density_over_mean,
            size_norm=self.size_norm,
            inverse_size_rank_norm=self.inverse_size_rank_norm,
            is_hot_bucket=self.is_hot_bucket,
            estimated_service_units=self.estimated_service_units,
            source_outbound_pending_bytes=0.0,
            destination_inbound_pending_bytes=0.0,
            flow_pending_bytes=0.0,
            return_flow_pending_bytes=0.0,
            rank_compute_queue_rows=0.0,
            expert_pending_rows=0.0,
            estimated_dispatch_cost=estimated_dispatch_cost,
            estimated_compute_cost=estimated_compute_cost,
            estimated_return_cost=estimated_return_cost,
        )


@dataclass
class WorkloadPlan:
    plan_id: str
    model_key: str
    seed: int
    microbatch_id: int
    layer_id: int
    layer_path: str
    hidden_dim: int
    intermediate_dim: int
    placement_policy: str
    origin_sharding: str
    placement_mapping: list[PlacementEntry]
    bucket_workloads: list[WorkloadBucket]
    routing_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "model_key": self.model_key,
            "seed": self.seed,
            "microbatch_id": self.microbatch_id,
            "layer_id": self.layer_id,
            "layer_path": self.layer_path,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "placement_policy": self.placement_policy,
            "origin_sharding": self.origin_sharding,
            "placement_mapping": [asdict(entry) for entry in self.placement_mapping],
            "bucket_workloads": [asdict(bucket) for bucket in self.bucket_workloads],
            "routing_summary": self.routing_summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkloadPlan":
        return cls(
            plan_id=str(payload["plan_id"]),
            model_key=str(payload["model_key"]),
            seed=int(payload["seed"]),
            microbatch_id=int(payload["microbatch_id"]),
            layer_id=int(payload["layer_id"]),
            layer_path=str(payload["layer_path"]),
            hidden_dim=int(payload["hidden_dim"]),
            intermediate_dim=int(payload["intermediate_dim"]),
            placement_policy=str(payload["placement_policy"]),
            origin_sharding=str(payload.get("origin_sharding", "round-robin")),
            placement_mapping=[PlacementEntry(**entry) for entry in payload["placement_mapping"]],
            bucket_workloads=[WorkloadBucket(**bucket) for bucket in payload["bucket_workloads"]],
            routing_summary=dict(payload["routing_summary"]),
        )


@dataclass
class GlobalRuntimeState:
    destination_pending_work: dict[int, float] = field(default_factory=dict)
    destination_queue_depth: dict[int, int] = field(default_factory=dict)
    rank_backlog_work: dict[int, float] = field(default_factory=dict)
    rank_queue_depth: dict[int, int] = field(default_factory=dict)
    destination_hotness: dict[int, float] = field(default_factory=dict)
    destination_latency_history: dict[int, list[float]] = field(default_factory=dict)
    source_outbound_pending_bytes: dict[int, int] = field(default_factory=dict)
    destination_inbound_pending_bytes: dict[int, int] = field(default_factory=dict)
    flow_pending_bytes: dict[str, int] = field(default_factory=dict)
    return_flow_pending_bytes: dict[str, int] = field(default_factory=dict)
    rank_compute_queue_rows: dict[int, int] = field(default_factory=dict)
    expert_pending_rows: dict[int, int] = field(default_factory=dict)
    rank_dispatch_time_ms: dict[int, float] = field(default_factory=dict)
    rank_compute_time_ms: dict[int, float] = field(default_factory=dict)
    rank_return_time_ms: dict[int, float] = field(default_factory=dict)
    rank_local_completion_time_ms: dict[int, float] = field(default_factory=dict)
    global_completion_time_ms: float = 0.0
    state_semantics: str = "historical_plus_projected"
    batch_index: int = 0

    def to_scheduler_state(self) -> RuntimeState:
        return RuntimeState(
            destination_pending_work=dict(self.destination_pending_work),
            destination_queue_depth=dict(self.destination_queue_depth),
            rank_backlog_work=dict(self.rank_backlog_work),
            rank_queue_depth=dict(self.rank_queue_depth),
            destination_hotness=dict(self.destination_hotness),
            destination_latency_history={key: list(values) for key, values in self.destination_latency_history.items()},
            source_outbound_pending_bytes=dict(self.source_outbound_pending_bytes),
            destination_inbound_pending_bytes=dict(self.destination_inbound_pending_bytes),
            flow_pending_bytes=dict(self.flow_pending_bytes),
            return_flow_pending_bytes=dict(self.return_flow_pending_bytes),
            rank_compute_queue_rows=dict(self.rank_compute_queue_rows),
            expert_pending_rows=dict(self.expert_pending_rows),
            rank_dispatch_time_ms=dict(self.rank_dispatch_time_ms),
            rank_compute_time_ms=dict(self.rank_compute_time_ms),
            rank_return_time_ms=dict(self.rank_return_time_ms),
            batch_index=self.batch_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalRuntimeState":
        return cls(**payload)


@dataclass
class GlobalDispatchPlan:
    plan_id: str
    workload_plan_id: str
    policy_name: str
    policy_seed: int
    microbatch_id: int
    release_order: list[str]
    release_rounds: list[list[str]]
    total_dispatch_matrix: list[list[int]]
    per_round_dispatch_matrices: list[list[list[int]]]
    decision_hash: str
    scheduler_state_snapshot: dict[str, Any]
    scheduler_decision: dict[str, Any]
    round_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    round_pressure_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalDispatchPlan":
        return cls(**payload)


def canonical_global_plan_payload(plan: GlobalDispatchPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "workload_plan_id": plan.workload_plan_id,
        "policy_name": plan.policy_name,
        "policy_seed": plan.policy_seed,
        "microbatch_id": plan.microbatch_id,
        "release_order": list(plan.release_order),
        "release_rounds": [list(item) for item in plan.release_rounds],
        "total_dispatch_matrix": [list(row) for row in plan.total_dispatch_matrix],
        "per_round_dispatch_matrices": [
            [list(row) for row in matrix] for matrix in plan.per_round_dispatch_matrices
        ],
        "scheduler_state_snapshot": dict(plan.scheduler_state_snapshot),
        "scheduler_decision": dict(plan.scheduler_decision),
        "round_diagnostics": list(plan.round_diagnostics),
        "round_pressure_score": float(plan.round_pressure_score),
    }


def canonical_global_plan_hash(plan: GlobalDispatchPlan) -> str:
    return _hash_payload(canonical_global_plan_payload(plan))


@dataclass
class ExecutionCache:
    payload_by_bucket_id: dict[str, torch.Tensor]
    metadata_by_bucket_id: dict[str, torch.Tensor]
    expert_weight1: dict[int, torch.Tensor]
    expert_weight2: dict[int, torch.Tensor]
    correctness_mode: bool


@dataclass(frozen=True)
class CanonicalRoundSegment:
    round_index: int
    source_rank: int
    destination_rank: int
    expert_id: int
    bucket_id: str
    route_item_ids: list[str]
    token_ids: list[int]
    route_ranks: list[int]
    payload_row_offset: int
    payload_row_count: int
    metadata_row_offset: int
    metadata_row_count: int
    receive_row_offset: int
    receive_metadata_row_offset: int
    return_origin_rank: int


@dataclass(frozen=True)
class RouteItemMetadata:
    token_id: int
    global_route_item_index: int
    origin_rank: int
    destination_rank: int
    expert_id: int
    route_rank: int

    def to_dict(self) -> dict[str, int]:
        return {
            "token_id": self.token_id,
            "global_route_item_index": self.global_route_item_index,
            "origin_rank": self.origin_rank,
            "destination_rank": self.destination_rank,
            "expert_id": self.expert_id,
            "route_rank": self.route_rank,
        }


@dataclass
class RoundEvaluation:
    strategy: str
    objective: float
    round_count: int
    per_round: list[dict[str, Any]]
    release_rounds: list[list[str]]


def dependency_score_for_bucket(bucket: WorkloadBucket) -> float:
    return (
        0.25 * bucket.source_coverage
        + 0.30 * bucket.coactive_peer_degree
        + 0.15 * bucket.coactive_event_density
        + 0.15 * bucket.position_spread
        + 0.15 * bucket.bridge_score
    )


def state_score_for_bucket(bucket: WorkloadBucket, runtime_state: RuntimeState | None = None) -> float:
    pending = 0.0
    queue = 0.0
    rank_pending = 0.0
    rank_queue = 0.0
    hotness = 0.0
    latency = 0.0
    if runtime_state is not None:
        pending = float(runtime_state.destination_pending_work.get(bucket.destination_id, 0.0))
        queue = float(runtime_state.destination_queue_depth.get(bucket.destination_id, 0.0))
        rank_pending = float(runtime_state.rank_backlog_work.get(bucket.destination_rank, 0.0))
        rank_queue = float(runtime_state.rank_queue_depth.get(bucket.destination_rank, 0.0))
        hotness = float(runtime_state.destination_hotness.get(bucket.destination_id, 0.0))
        history = runtime_state.destination_latency_history.get(bucket.destination_id, [])
        latency = float(sum(history) / len(history)) if history else 0.0
    pending_norm = min(1.0, pending / max(1.0, bucket.estimated_service_units * 4.0))
    queue_norm = min(1.0, queue / 8.0)
    rank_backlog_norm = min(1.0, rank_pending / max(1.0, bucket.estimated_service_units * 6.0))
    rank_queue_norm = min(1.0, rank_queue / 8.0)
    hotness_norm = min(1.0, hotness / 6.0)
    latency_norm = min(1.0, latency / 0.2) if latency > 0 else 0.0
    return (
        0.28 * bucket.size_norm
        + 0.14 * bucket.inverse_size_rank_norm
        + 0.12 * bucket.route_share
        + 0.10 * bucket.density_over_mean
        + 0.08 * (1.0 if bucket.is_hot_bucket else 0.0)
        - 0.12 * pending_norm
        - 0.06 * queue_norm
        - 0.05 * rank_backlog_norm
        - 0.03 * rank_queue_norm
        + 0.01 * hotness_norm
        - 0.01 * latency_norm
    )


def strong_state_score_for_bucket(bucket: WorkloadBucket, runtime_state: RuntimeState | None = None) -> float:
    pending = float(runtime_state.rank_backlog_work.get(bucket.destination_rank, 0.0)) if runtime_state else 0.0
    queue = float(runtime_state.rank_queue_depth.get(bucket.destination_rank, 0.0)) if runtime_state else 0.0
    history = runtime_state.destination_latency_history.get(bucket.destination_id, []) if runtime_state else []
    latency = float(sum(history) / len(history)) if history else 0.0
    source_outbound = float(runtime_state.source_outbound_pending_bytes.get(bucket.origin_rank, 0.0)) if runtime_state else 0.0
    destination_inbound = float(runtime_state.destination_inbound_pending_bytes.get(bucket.destination_rank, 0.0)) if runtime_state else 0.0
    flow_pending = float(runtime_state.flow_pending_bytes.get(f"{bucket.origin_rank}->{bucket.destination_rank}", 0.0)) if runtime_state else 0.0
    return_pending = float(runtime_state.return_flow_pending_bytes.get(f"{bucket.destination_rank}->{bucket.origin_rank}", 0.0)) if runtime_state else 0.0
    rank_compute_rows = float(runtime_state.rank_compute_queue_rows.get(bucket.destination_rank, 0.0)) if runtime_state else 0.0
    expert_pending_rows = float(runtime_state.expert_pending_rows.get(bucket.expert_id, 0.0)) if runtime_state else 0.0
    dispatch_history = float(runtime_state.rank_dispatch_time_ms.get(bucket.origin_rank, 0.0)) if runtime_state else 0.0
    compute_history = float(runtime_state.rank_compute_time_ms.get(bucket.destination_rank, 0.0)) if runtime_state else 0.0
    return_history = float(runtime_state.rank_return_time_ms.get(bucket.destination_rank, 0.0)) if runtime_state else 0.0
    rank_pressure = min(1.0, pending / max(1.0, bucket.estimated_service_units * 4.0))
    queue_norm = min(1.0, queue / 8.0)
    latency_norm = min(1.0, latency / 0.2) if latency > 0 else 0.0
    projected_bytes = min(1.0, bucket.payload_bytes / float(1 << 20) / 8.0)
    source_norm = min(1.0, source_outbound / float(1 << 20) / 16.0)
    destination_norm = min(1.0, destination_inbound / float(1 << 20) / 16.0)
    flow_norm = min(1.0, flow_pending / float(1 << 20) / 8.0)
    return_norm = min(1.0, return_pending / float(1 << 20) / 8.0)
    compute_queue_norm = min(1.0, rank_compute_rows / max(1.0, bucket.payload_rows * 8.0))
    expert_queue_norm = min(1.0, expert_pending_rows / max(1.0, bucket.payload_rows * 8.0))
    history_norm = min(1.0, (dispatch_history + compute_history + return_history) / 100.0)
    return (
        0.12 * bucket.size_norm
        + 0.10 * bucket.route_share
        + 0.12 * projected_bytes
        + 0.10 * bucket.payload_rows / max(1.0, bucket.payload_rows + 8.0)
        - 0.08 * rank_pressure
        - 0.05 * queue_norm
        - 0.04 * latency_norm
        - 0.10 * source_norm
        - 0.10 * destination_norm
        - 0.12 * flow_norm
        - 0.08 * return_norm
        - 0.05 * compute_queue_norm
        - 0.05 * expert_queue_norm
        - 0.03 * history_norm
    )


def _bucket_projected_state_features(bucket: WorkloadBucket | BucketRecord, runtime_state: RuntimeState | None = None) -> dict[str, float]:
    flow_key = f"{bucket.origin_rank}->{bucket.destination_rank}"
    return_key = f"{bucket.destination_rank}->{bucket.origin_rank}"
    runtime_state = runtime_state if runtime_state is not None else EMPTY_RUNTIME_STATE
    source_outbound = max(float(getattr(bucket, "source_outbound_pending_bytes", 0.0)), float(runtime_state.source_outbound_pending_bytes.get(bucket.origin_rank, 0.0)))
    destination_inbound = max(float(getattr(bucket, "destination_inbound_pending_bytes", 0.0)), float(runtime_state.destination_inbound_pending_bytes.get(bucket.destination_rank, 0.0)))
    flow_pending = max(float(getattr(bucket, "flow_pending_bytes", 0.0)), float(runtime_state.flow_pending_bytes.get(flow_key, 0.0)))
    return_flow = max(float(getattr(bucket, "return_flow_pending_bytes", 0.0)), float(runtime_state.return_flow_pending_bytes.get(return_key, 0.0)))
    rank_compute_rows = max(float(getattr(bucket, "rank_compute_queue_rows", 0.0)), float(runtime_state.rank_compute_queue_rows.get(bucket.destination_rank, 0.0)))
    expert_pending_rows = max(float(getattr(bucket, "expert_pending_rows", 0.0)), float(runtime_state.expert_pending_rows.get(bucket.expert_id, 0.0)))
    dispatch_history = float(runtime_state.rank_dispatch_time_ms.get(bucket.origin_rank, 0.0))
    compute_history = float(runtime_state.rank_compute_time_ms.get(bucket.destination_rank, 0.0))
    return_history = float(runtime_state.rank_return_time_ms.get(bucket.destination_rank, 0.0))
    return {
        "source_outbound_pending_bytes": source_outbound,
        "destination_inbound_pending_bytes": destination_inbound,
        "flow_pending_bytes": flow_pending,
        "return_flow_pending_bytes": return_flow,
        "rank_compute_queue_rows": rank_compute_rows,
        "expert_pending_rows": expert_pending_rows,
        "dispatch_history_ms": dispatch_history,
        "compute_history_ms": compute_history,
        "return_history_ms": return_history,
    }


def _stable_bucket_order_key(bucket: WorkloadBucket | BucketRecord) -> tuple[int, str]:
    token_anchor = min(getattr(bucket, "token_positions", []) or [0])
    return (int(getattr(bucket, "arrival_index", token_anchor)), str(bucket.bucket_id))


def _arrival_jitter_offset(bucket: WorkloadBucket, *, mode: str, seed: int) -> int:
    normalized = str(mode).strip().lower()
    if normalized == "none":
        return 0
    scale = {"mild": 3, "moderate": 7}.get(normalized)
    if scale is None:
        raise ValueError(f"unsupported source arrival jitter mode: {mode}")
    first_token_ordinal = min(bucket.token_positions) if bucket.token_positions else 0
    jitter_seed = hash((seed, bucket.origin_rank, bucket.microbatch_id, first_token_ordinal))
    rng = random.Random(jitter_seed)
    return int(rng.randint(0, scale))


def _with_arrival_jitter(
    buckets: list[WorkloadBucket],
    *,
    mode: str,
    seed: int,
) -> list[WorkloadBucket]:
    if mode == "none":
        return list(buckets)
    adjusted = [WorkloadBucket(**asdict(bucket)) for bucket in buckets]
    ordered = sorted(
        adjusted,
        key=lambda bucket: (
            bucket.origin_rank,
            bucket.microbatch_id,
            min(bucket.token_positions) if bucket.token_positions else 0,
            bucket.bucket_id,
        ),
    )
    for index, bucket in enumerate(ordered):
        bucket.arrival_index = index + _arrival_jitter_offset(bucket, mode=mode, seed=seed)
    return sorted(ordered, key=_stable_bucket_order_key)


def _pack_rounds_from_order(bucket_ids: list[str], round_size: int) -> list[list[str]]:
    normalized = max(1, int(round_size))
    return [bucket_ids[index : index + normalized] for index in range(0, len(bucket_ids), normalized)]


def _round_pressure_components(
    round_buckets: list[WorkloadBucket | BucketRecord],
    runtime_state: RuntimeState | None = None,
) -> dict[str, Any]:
    source_increment: dict[int, float] = {}
    destination_increment: dict[int, float] = {}
    flow_increment: dict[str, float] = {}
    expert_increment: dict[int, float] = {}
    rank_increment: dict[int, float] = {}
    all_sources: set[int] = set()
    all_destinations: set[int] = set()
    active_flows: set[tuple[int, int]] = set()
    dest_fanin: dict[int, set[int]] = {}
    source_fanout: dict[int, set[int]] = {}
    total_payload_bytes = float(sum(max(1, int(bucket.payload_bytes)) for bucket in round_buckets))
    total_payload_rows = float(sum(max(1, int(bucket.payload_rows)) for bucket in round_buckets))
    total_flow_bytes = float(sum(max(1, int(bucket.payload_bytes)) for bucket in round_buckets))
    runtime_state = runtime_state if runtime_state is not None else EMPTY_RUNTIME_STATE
    for bucket in round_buckets:
        source_increment[bucket.origin_rank] = source_increment.get(bucket.origin_rank, 0.0) + float(bucket.payload_bytes)
        destination_increment[bucket.destination_rank] = destination_increment.get(bucket.destination_rank, 0.0) + float(bucket.payload_bytes)
        flow_key = f"{bucket.origin_rank}->{bucket.destination_rank}"
        flow_increment[flow_key] = flow_increment.get(flow_key, 0.0) + float(bucket.payload_bytes)
        expert_increment[bucket.expert_id] = expert_increment.get(bucket.expert_id, 0.0) + float(bucket.payload_rows)
        rank_increment[bucket.destination_rank] = rank_increment.get(bucket.destination_rank, 0.0) + float(bucket.payload_rows)
        all_sources.add(int(bucket.origin_rank))
        all_destinations.add(int(bucket.destination_rank))
        active_flows.add((int(bucket.origin_rank), int(bucket.destination_rank)))
        dest_fanin.setdefault(int(bucket.destination_rank), set()).add(int(bucket.origin_rank))
        source_fanout.setdefault(int(bucket.origin_rank), set()).add(int(bucket.destination_rank))

    historical_source = {int(key): float(value) for key, value in runtime_state.source_outbound_pending_bytes.items()}
    historical_destination = {int(key): float(value) for key, value in runtime_state.destination_inbound_pending_bytes.items()}
    historical_flow = {str(key): float(value) for key, value in runtime_state.flow_pending_bytes.items()}
    historical_expert = {int(key): float(value) for key, value in runtime_state.expert_pending_rows.items()}
    historical_rank = {int(key): float(value) for key, value in runtime_state.rank_compute_queue_rows.items()}
    historical_return = {str(key): float(value) for key, value in runtime_state.return_flow_pending_bytes.items()}

    source_totals = {key: historical_source.get(key, 0.0) + source_increment.get(key, 0.0) for key in set(historical_source) | set(source_increment)}
    destination_totals = {key: historical_destination.get(key, 0.0) + destination_increment.get(key, 0.0) for key in set(historical_destination) | set(destination_increment)}
    flow_totals = {
        key: historical_flow.get(key, 0.0) + flow_increment.get(key, 0.0)
        for key in set(historical_flow) | set(flow_increment)
    }
    expert_totals = {key: historical_expert.get(key, 0.0) + expert_increment.get(key, 0.0) for key in set(historical_expert) | set(expert_increment)}
    rank_totals = {key: historical_rank.get(key, 0.0) + rank_increment.get(key, 0.0) for key in set(historical_rank) | set(rank_increment)}

    normalization = {
        "bytes": max(total_payload_bytes, 1.0),
        "rows": max(total_payload_rows, 1.0),
        "flow_bytes": max(total_flow_bytes, 1.0),
        "world_size": float(max(1, len(set(all_sources) | set(all_destinations) | set(runtime_state.source_outbound_pending_bytes.keys()) | set(runtime_state.destination_inbound_pending_bytes.keys())))),
    }
    whole_batch_projected_totals = {
        "source_outbound_bytes": {
            int(bucket.origin_rank): sum(float(item.payload_bytes) for item in round_buckets if int(item.origin_rank) == int(bucket.origin_rank))
            for bucket in round_buckets
        },
        "destination_inbound_bytes": {
            int(bucket.destination_rank): sum(float(item.payload_bytes) for item in round_buckets if int(item.destination_rank) == int(bucket.destination_rank))
            for bucket in round_buckets
        },
        "flow_bytes": {
            f"{bucket.origin_rank}->{bucket.destination_rank}": sum(
                float(item.payload_bytes)
                for item in round_buckets
                if int(item.origin_rank) == int(bucket.origin_rank) and int(item.destination_rank) == int(bucket.destination_rank)
            )
            for bucket in round_buckets
        },
        "expert_rows": {
            int(bucket.expert_id): sum(float(item.payload_rows) for item in round_buckets if int(item.expert_id) == int(bucket.expert_id))
            for bucket in round_buckets
        },
        "rank_compute_rows": {
            int(bucket.destination_rank): sum(float(item.payload_rows) for item in round_buckets if int(item.destination_rank) == int(bucket.destination_rank))
            for bucket in round_buckets
        },
    }
    scarcity_components = {
        "source_outbound_bytes": {
            str(key): source_totals.get(key, 0.0) / max(1.0, whole_batch_projected_totals["source_outbound_bytes"].get(key, 0.0))
            for key in source_totals
        },
        "destination_inbound_bytes": {
            str(key): destination_totals.get(key, 0.0) / max(1.0, whole_batch_projected_totals["destination_inbound_bytes"].get(key, 0.0))
            for key in destination_totals
        },
        "flow_bytes": {
            str(key): flow_totals.get(key, 0.0) / max(1.0, whole_batch_projected_totals["flow_bytes"].get(key, 0.0))
            for key in flow_totals
        },
        "expert_rows": {
            str(key): expert_totals.get(key, 0.0) / max(1.0, whole_batch_projected_totals["expert_rows"].get(key, 0.0))
            for key in expert_totals
        },
        "rank_compute_rows": {
            str(key): rank_totals.get(key, 0.0) / max(1.0, whole_batch_projected_totals["rank_compute_rows"].get(key, 0.0))
            for key in rank_totals
        },
    }
    world_size = normalization["world_size"]
    active_flow_count = float(len(active_flows))
    max_dest_source_fanin = float(max((len(value) for value in dest_fanin.values()), default=0))
    max_source_dest_fanout = float(max((len(value) for value in source_fanout.values()), default=0))
    active_destination_count = float(len(all_destinations))
    active_source_count = float(len(all_sources))
    flow_density = active_flow_count / max(1.0, float(len(round_buckets)))
    flow_fragmentation = (
        flow_density
        + active_flow_count / max(1.0, world_size * world_size)
        + 0.25 * (max_dest_source_fanin / max(1.0, world_size))
        + 0.25 * (max_source_dest_fanout / max(1.0, world_size))
    )
    return {
        "round_increment_components": {
            "source_outbound_bytes": source_increment,
            "destination_inbound_bytes": destination_increment,
            "flow_bytes": flow_increment,
            "expert_rows": expert_increment,
            "rank_compute_rows": rank_increment,
            "return_flow_bytes": {},
        },
        "historical_baseline_components": {
            "source_outbound_bytes": historical_source,
            "destination_inbound_bytes": historical_destination,
            "flow_bytes": historical_flow,
            "expert_rows": historical_expert,
            "rank_compute_rows": historical_rank,
            "return_flow_bytes": historical_return,
        },
        "whole_batch_projected_totals": whole_batch_projected_totals,
        "scarcity_components": scarcity_components,
        "fragmentation_components": {
            "active_flow_count": active_flow_count,
            "max_dest_source_fanin": max_dest_source_fanin,
            "max_source_dest_fanout": max_source_dest_fanout,
            "active_destination_count": active_destination_count,
            "active_source_count": active_source_count,
        },
        "round_pressure_components": {
            "max_source_outbound_bytes_norm": max(source_totals.values(), default=0.0) / normalization["bytes"],
            "max_destination_inbound_bytes_norm": max(destination_totals.values(), default=0.0) / normalization["bytes"],
            "max_flow_bytes_norm": max(flow_totals.values(), default=0.0) / normalization["flow_bytes"],
            "max_expert_rows_norm": max(expert_totals.values(), default=0.0) / normalization["rows"],
            "max_rank_compute_rows_norm": max(rank_totals.values(), default=0.0) / normalization["rows"],
            "flow_fragmentation_penalty_norm": flow_fragmentation,
        },
        "normalization_constants": normalization,
    }


def _round_pressure_objective(
    round_buckets: list[WorkloadBucket | BucketRecord],
    runtime_state: RuntimeState | None = None,
) -> tuple[float, dict[str, float]]:
    components = _round_pressure_components(round_buckets, runtime_state)
    objective = sum(float(value) for value in components["round_pressure_components"].values())
    return objective, components


def evaluate_release_rounds(
    plan: WorkloadPlan,
    release_rounds: list[list[str]],
    runtime_state: RuntimeState | None = None,
    *,
    strategy: str = "unknown",
) -> RoundEvaluation:
    bucket_lookup = {bucket.bucket_id: bucket for bucket in plan.bucket_workloads}
    per_round: list[dict[str, Any]] = []
    for round_index, bucket_ids in enumerate(release_rounds):
        round_buckets = [bucket_lookup[bucket_id] for bucket_id in bucket_ids]
        objective, components = _round_pressure_objective(round_buckets, runtime_state)
        per_round.append(
            {
                "round_index": round_index,
                "bucket_ids": list(bucket_ids),
                "objective": float(objective),
                "round_pressure_components": {
                    key: float(value) for key, value in components["round_pressure_components"].items()
                },
                "historical_baseline_components": components["historical_baseline_components"],
                "round_increment_components": components["round_increment_components"],
                "whole_batch_projected_totals": components["whole_batch_projected_totals"],
                "scarcity_components": components["scarcity_components"],
                "fragmentation_components": components["fragmentation_components"],
                "normalization_constants": components["normalization_constants"],
            }
        )
    objective = max((item["objective"] for item in per_round), default=0.0)
    return RoundEvaluation(
        strategy=strategy,
        objective=float(objective),
        round_count=len(release_rounds),
        per_round=per_round,
        release_rounds=[list(item) for item in release_rounds],
    )


def _greedy_minimax_round_pack(
    buckets: list[WorkloadBucket | BucketRecord],
    *,
    round_size: int,
    runtime_state: RuntimeState | None,
    policy_seed: int,
    arrival_jitter_mode: str = "none",
    oracle: bool = False,
) -> dict[str, Any]:
    def _record_to_workload(item: WorkloadBucket | BucketRecord) -> WorkloadBucket:
        if isinstance(item, WorkloadBucket):
            return item
        return WorkloadBucket(
            bucket_id=item.bucket_id,
            route_id=item.route_item_ids[0] if item.route_item_ids else item.bucket_id,
            token_ids=[],
            token_positions=[],
            origin_rank=item.origin_rank,
            destination_id=item.destination_id,
            destination_rank=item.destination_rank,
            expert_id=item.expert_id,
            layer_id=0,
            microbatch_id=0,
            token_count=item.token_count,
            payload_rows=item.payload_rows,
            hidden_dim=0,
            intermediate_dim=0,
            estimated_service_units=item.estimated_service_units,
            payload_bytes=item.payload_bytes,
            source_count=item.source_count,
            source_coverage=item.source_coverage,
            coactive_peer_degree=item.coactive_peer_degree,
            coactive_event_density=item.coactive_event_density,
            position_spread=item.position_spread,
            bridge_score=item.bridge_score,
            route_share=item.route_share,
            density_over_mean=item.density_over_mean,
            size_norm=item.size_norm,
            inverse_size_rank_norm=item.inverse_size_rank_norm,
            is_hot_bucket=item.is_hot_bucket,
        )

    materialized = list(buckets)
    if materialized and isinstance(materialized[0], WorkloadBucket):
        candidate_buckets = _with_arrival_jitter(materialized, mode=arrival_jitter_mode, seed=policy_seed)
    else:
        candidate_buckets = sorted(materialized, key=_stable_bucket_order_key)
    remaining = list(candidate_buckets)
    rounds: list[list[WorkloadBucket | BucketRecord]] = []
    effective_state = runtime_state if runtime_state is not None else EMPTY_RUNTIME_STATE
    eval_plan = WorkloadPlan(
        plan_id="candidate",
        model_key="candidate",
        seed=policy_seed,
        microbatch_id=0,
        layer_id=0,
        layer_path="candidate",
        hidden_dim=0,
        intermediate_dim=0,
        placement_policy="candidate",
        origin_sharding="round-robin",
        placement_mapping=[],
        bucket_workloads=[_record_to_workload(item) for item in materialized],
        routing_summary={},
    )
    while remaining:
        current_round: list[WorkloadBucket] = []
        while remaining and len(current_round) < max(1, int(round_size)):
            best_bucket: WorkloadBucket | BucketRecord | None = None
            best_key: tuple[float, float, int, str] | None = None
            for bucket in remaining:
                trial_round = current_round + [bucket]
                trial_eval = evaluate_release_rounds(
                    eval_plan,
                    [[item.bucket_id for item in trial_round]],
                    effective_state,
                    strategy="candidate",
                )
                objective = trial_eval.objective
                components = trial_eval.per_round[0]
                imbalance_focus = max(
                    components["round_pressure_components"]["max_destination_inbound_bytes_norm"],
                    components["round_pressure_components"]["max_source_outbound_bytes_norm"],
                    components["round_pressure_components"]["max_flow_bytes_norm"],
                    components["round_pressure_components"]["max_expert_rows_norm"],
                    components["round_pressure_components"]["max_rank_compute_rows_norm"],
                )
                if oracle:
                    future_bytes = sum(float(other.payload_bytes) for other in remaining if other.bucket_id != bucket.bucket_id and other.destination_rank == bucket.destination_rank)
                    future_rows = sum(float(other.payload_rows) for other in remaining if other.bucket_id != bucket.bucket_id and other.expert_id == bucket.expert_id)
                    objective += future_bytes / max(1.0, sum(float(item.payload_bytes) for item in remaining))
                    objective += future_rows / max(1.0, sum(float(item.payload_rows) for item in remaining))
                key = (
                    float(objective),
                    float(imbalance_focus),
                    int(getattr(bucket, "arrival_index", 0)),
                    str(bucket.bucket_id),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_bucket = bucket
            assert best_bucket is not None
            current_round.append(best_bucket)
            remaining = [bucket for bucket in remaining if bucket.bucket_id != best_bucket.bucket_id]
        rounds.append(sorted(current_round, key=_stable_bucket_order_key))
    release_rounds = [[bucket.bucket_id for bucket in round_buckets] for round_buckets in rounds]
    release_order = [bucket_id for round_ids in release_rounds for bucket_id in round_ids]
    evaluation = evaluate_release_rounds(eval_plan, release_rounds, effective_state, strategy="strong-state-packer")
    return {
        "release_order": release_order,
        "release_rounds": release_rounds,
        "round_diagnostics": evaluation.per_round,
        "round_pressure_score": evaluation.objective,
    }


def lina_inspired_score_for_bucket(bucket: WorkloadBucket, runtime_state: RuntimeState | None = None) -> float:
    hotness = float(runtime_state.destination_hotness.get(bucket.destination_id, 0.0)) if runtime_state else 0.0
    pending = float(runtime_state.rank_backlog_work.get(bucket.destination_rank, 0.0)) if runtime_state else 0.0
    popularity_ema = min(1.0, hotness / 10.0)
    projected_bytes = min(1.0, (bucket.payload_rows * bucket.hidden_dim * 2) / float(1 << 20) / 8.0)
    imbalance = min(1.0, pending / max(1.0, bucket.estimated_service_units * 5.0))
    return (
        0.28 * bucket.route_share
        + 0.24 * popularity_ema
        + 0.22 * projected_bytes
        + 0.16 * bucket.size_norm
        - 0.10 * imbalance
    )


def shuffled_dependency_order(
    buckets: list[WorkloadBucket],
    runtime_state: RuntimeState | None,
    *,
    seed: int,
    combine_with_state: bool,
) -> list[str]:
    rng = random.Random(seed)
    dep_scores = [dependency_score_for_bucket(bucket) for bucket in buckets]
    shuffled = list(dep_scores)
    rng.shuffle(shuffled)
    scored = []
    for index, bucket in enumerate(buckets):
        dep = shuffled[index]
        state = state_score_for_bucket(bucket, runtime_state)
        selected = dep if not combine_with_state else 0.5 * state + 0.5 * dep
        scored.append((selected, index, bucket.destination_id, bucket.bucket_id))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in scored]


def random_order(bucket_ids: list[str], seed: int) -> list[str]:
    ordered = list(bucket_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def available_audit_strategies() -> list[str]:
    return ["fifo", "random-order", "strong-state", "full"]


def release_rounds_for_order(release_order: list[str], round_size: int) -> list[list[str]]:
    normalized = max(1, int(round_size))
    return [release_order[index : index + normalized] for index in range(0, len(release_order), normalized)]


def build_placement_map(expert_count: int, world_size: int, policy: str) -> list[PlacementEntry]:
    if policy not in PLACEMENT_POLICIES:
        raise ValueError(f"unsupported placement policy: {policy}")
    entries: list[PlacementEntry] = []
    if policy == "modulo":
        for expert_id in range(expert_count):
            entries.append(PlacementEntry(expert_id=expert_id, destination_rank=expert_id % max(1, world_size)))
        return entries

    if policy == "hotspot":
        hotspot_rank_span = max(1, min(world_size, 2))
        hotspot_cutoff = max(1, int(expert_count * 0.5))
        for expert_id in range(expert_count):
            if expert_id < hotspot_cutoff:
                rank = expert_id % hotspot_rank_span
            else:
                tail = max(1, world_size - hotspot_rank_span)
                rank = hotspot_rank_span + ((expert_id - hotspot_cutoff) % tail) if tail > 0 else 0
                rank = min(world_size - 1, rank)
            entries.append(PlacementEntry(expert_id=expert_id, destination_rank=rank))
        return entries

    experts_per_rank = math.ceil(expert_count / max(1, world_size))
    for expert_id in range(expert_count):
        rank = min(world_size - 1, expert_id // max(1, experts_per_rank))
        entries.append(PlacementEntry(expert_id=expert_id, destination_rank=rank))
    return entries


def apply_placement(bucket_records: list[BucketRecord], placement: list[PlacementEntry]) -> list[BucketRecord]:
    mapping = {entry.expert_id: entry.destination_rank for entry in placement}
    updated: list[BucketRecord] = []
    for index, bucket in enumerate(bucket_records):
        record = BucketRecord(**bucket.to_dict())
        record.arrival_index = index
        record.destination_rank = int(mapping.get(record.destination_id, record.destination_id % max(1, len(set(mapping.values())) or 1)))
        updated.append(record)
    return updated


def assign_origin_rank(token_id: int, token_count: int, world_size: int, mode: str) -> int:
    raise RuntimeError("assign_origin_rank(token_id, ...) is deprecated; use assign_origin_rank_from_ordinal")


def assign_origin_rank_from_ordinal(token_ordinal: int, token_count: int, world_size: int, mode: str) -> int:
    normalized = str(mode).strip().lower()
    if world_size <= 1:
        return 0
    if normalized == "round-robin":
        return int(token_ordinal) % world_size
    if normalized == "contiguous":
        if token_count <= 0:
            return 0
        shard_size = max(1, math.ceil(token_count / world_size))
        return min(world_size - 1, int(token_ordinal) // shard_size)
    raise ValueError(f"unsupported origin sharding mode: {mode}")


def build_workload_plan(
    *,
    config: ProtocolConfig,
    microbatch_id: int,
    routing: dict[str, Any],
    hidden_dim: int,
    intermediate_dim: int,
    placement: list[PlacementEntry],
) -> WorkloadPlan:
    placement_map = {entry.expert_id: entry.destination_rank for entry in placement}
    route_items = list(routing.get("route_items", []))
    token_ids = list(routing.get("routing_summary", {}).get("token_ids", []))
    if not route_items:
        raise RuntimeError("routing payload is missing route_items; cannot build origin-aware workload plan")
    token_count = len(token_ids)
    token_ordinals = {int(token_id): index for index, token_id in enumerate(sorted(set(int(token_id) for token_id in token_ids)))}
    route_item_index_map = dict(routing.get("route_item_index_map", {}))
    if not route_item_index_map:
        raise RuntimeError("routing payload is missing route_item_index_map; cannot build globally unique route identities")
    grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
    bucket_workloads: list[WorkloadBucket] = []
    bucket_record_map = {bucket.destination_id: bucket for bucket in routing["bucket_records"]}
    for route in route_items:
        expert_id = int(route["expert_id"])
        token_id = int(route["token_id"])
        destination_rank = int(placement_map.get(expert_id, expert_id % config.world_size))
        token_ordinal = int(token_ordinals[token_id])
        origin_rank = assign_origin_rank_from_ordinal(token_ordinal, token_count, config.world_size, config.origin_sharding)
        route_tuple = (
            int(routing["layer_id"]),
            int(microbatch_id),
            int(route["token_pos"]),
            token_id,
            int(route["route_rank"]),
            expert_id,
            str(route["route_id"]),
        )
        route_item_index = int(route_item_index_map[str(route_tuple)])
        key = (origin_rank, destination_rank, expert_id)
        token_pos = int(route["token_pos"])
        current = grouped.setdefault(
            key,
            {
                "route_tuples": [],
            },
        )
        current["route_tuples"].append(
            {
                "route_item_index": route_item_index,
                "token_id": token_id,
                "token_pos": token_pos,
                "route_rank": int(route["route_rank"]),
                "expert_id": expert_id,
                "route_id": str(route["route_id"]),
            }
        )

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][1],
            item[0][0],
            item[0][2],
            min(int(value["token_pos"]) for value in item[1]["route_tuples"]),
        ),
    )
    for index, ((origin_rank, destination_rank, expert_id), item) in enumerate(ordered_groups):
        route_tuple_items = sorted(
            item["route_tuples"],
            key=lambda value: (
                int(value["token_pos"]),
                int(value["token_id"]),
                int(value["route_rank"]),
                int(value["expert_id"]),
                int(value["route_item_index"]),
            ),
        )
        base_bucket = bucket_record_map[expert_id]
        token_ids_in_bucket = [int(value["token_id"]) for value in route_tuple_items]
        token_positions = [int(value["token_pos"]) for value in route_tuple_items]
        token_count_in_bucket = len(token_ids_in_bucket)
        payload_rows = max(1, int(round(token_count_in_bucket * max(1.0, config.compute_scale))))
        payload_bytes = payload_rows * hidden_dim * 2
        route_suffix = "_".join(str(value["route_id"]) for value in route_tuple_items[:2])
        bucket_workloads.append(
            WorkloadBucket(
                bucket_id=f"mb{microbatch_id}_o{origin_rank}_d{destination_rank}_e{expert_id}_{index}",
                route_id=route_suffix,
                route_item_indices=[int(value["route_item_index"]) for value in route_tuple_items],
                token_ids=token_ids_in_bucket,
                token_positions=token_positions,
                origin_rank=origin_rank,
                destination_id=expert_id,
                destination_rank=destination_rank,
                expert_id=expert_id,
                layer_id=int(routing["layer_id"]),
                microbatch_id=microbatch_id,
                token_count=token_count_in_bucket,
                payload_rows=payload_rows,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                estimated_service_units=max(1.0, float(base_bucket.estimated_service_units) * (token_count_in_bucket / max(1, base_bucket.token_count))),
                payload_bytes=payload_bytes,
                source_count=base_bucket.source_count,
                source_coverage=base_bucket.source_coverage,
                coactive_peer_degree=base_bucket.coactive_peer_degree,
                coactive_event_density=base_bucket.coactive_event_density,
                position_spread=base_bucket.position_spread,
                bridge_score=base_bucket.bridge_score,
                route_share=token_count_in_bucket / max(1, len(route_items)),
                density_over_mean=base_bucket.density_over_mean,
                size_norm=base_bucket.size_norm,
                inverse_size_rank_norm=1.0 - (index / max(1, len(ordered_groups) - 1)) if ordered_groups else 0.0,
                is_hot_bucket=token_count_in_bucket > 1,
                route_item_ids=[str(value["route_id"]) for value in route_tuple_items],
                route_ranks=[int(value["route_rank"]) for value in route_tuple_items],
            )
        )

    token_origin_summary: dict[int, list[int]] = {}
    token_origin_detail: dict[str, dict[str, int]] = {}
    for token_id in token_ids:
        ordinal = int(token_ordinals[int(token_id)])
        rank = assign_origin_rank_from_ordinal(ordinal, token_count, config.world_size, config.origin_sharding)
        token_origin_summary.setdefault(rank, []).append(int(token_id))
        token_origin_detail[str(token_id)] = {"token_id": int(token_id), "token_ordinal": ordinal, "origin_rank": rank}

    routing_summary = dict(routing["routing_summary"])
    routing_summary["origin_sharding"] = config.origin_sharding
    routing_summary["token_to_origin_rank"] = {str(token_id): value["origin_rank"] for token_id, value in token_origin_detail.items()}
    routing_summary["token_origin_rank_detail"] = token_origin_detail
    routing_summary["token_origin_rank_summary"] = {str(rank): values for rank, values in token_origin_summary.items()}

    return WorkloadPlan(
        plan_id=f"{config.model_key}_mb{microbatch_id}_seed{config.seed}",
        model_key=config.model_key,
        seed=config.seed,
        microbatch_id=microbatch_id,
        layer_id=int(routing["layer_id"]),
        layer_path=str(routing["layer_path"]),
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        placement_policy=config.placement_policy,
        origin_sharding=config.origin_sharding,
        placement_mapping=placement,
        bucket_workloads=bucket_workloads,
        routing_summary=routing_summary,
    )


def plan_conflict_score(plan: WorkloadPlan) -> float:
    buckets = list(plan.bucket_workloads)
    if not buckets:
        return 0.0
    dep_order = _order_by_score(buckets, dependency_score_for_bucket)
    state_order = _order_by_score(buckets, lambda bucket: state_score_for_bucket(bucket, None))
    mismatch = sum(1 for index, bucket_id in enumerate(dep_order) if state_order[index] != bucket_id)
    hotspot = sum(bucket.token_count for bucket in buckets if bucket.destination_rank in {0, 1})
    total = sum(bucket.token_count for bucket in buckets)
    hotspot_ratio = (hotspot / total) if total else 0.0
    return float(mismatch) / max(1, len(dep_order)) + 0.5 * hotspot_ratio


def select_plans_for_scenario(plans: list[WorkloadPlan], scenario: str) -> list[WorkloadPlan]:
    normalized = str(scenario).strip().lower()
    if normalized == "natural":
        return plans
    if normalized == "conflict":
        ranked = sorted(plans, key=plan_conflict_score, reverse=True)
        keep = max(1, len(ranked) // 2)
        return ranked[:keep]
    raise ValueError(f"unsupported scenario: {scenario}")


def counterbalanced_orders(strategies: list[str], repetitions: int) -> list[list[str]]:
    orders: list[list[str]] = []
    for rep in range(repetitions):
        shift = rep % len(strategies)
        orders.append(strategies[shift:] + strategies[:shift])
    return orders


def randomized_orders(strategies: list[str], repetitions: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    orders: list[list[str]] = []
    for _ in range(repetitions):
        current = list(strategies)
        rng.shuffle(current)
        orders.append(current)
    return orders


def latin_square_orders(strategies: list[str], repetitions: int) -> list[list[str]]:
    if repetitions % max(1, len(strategies)) != 0:
        raise ValueError("repetitions must be a multiple of the strategy count for Latin square ordering")
    return counterbalanced_orders(strategies, repetitions)


def resolve_strategy_orders(
    mode: str,
    repetitions: int,
    seed: int,
    *,
    strategies: list[str] | None = None,
) -> list[list[str]]:
    ordered_strategies = list(strategies or STRATEGIES)
    if mode == "fixed":
        return [list(ordered_strategies) for _ in range(repetitions)]
    if mode == "randomized":
        return randomized_orders(ordered_strategies, repetitions, seed)
    if mode == "latin-square":
        return latin_square_orders(ordered_strategies, repetitions)
    if mode == "counterbalanced":
        return counterbalanced_orders(ordered_strategies, repetitions)
    raise ValueError(f"unsupported strategy-order-mode: {mode}")


def mark_state_meaningful(microbatch_count: int) -> bool:
    return microbatch_count >= 2


def init_nccl() -> dict[str, int]:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def cleanup_nccl() -> None:
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


def _exchange_sizes(send_counts: torch.Tensor, group: Any = None) -> torch.Tensor:
    recv_counts = torch.zeros_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    return recv_counts


def _prepare_round_payload(
    rank: int,
    world_size: int,
    round_buckets: list[WorkloadBucket],
    hidden_dim: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    raise RuntimeError("_prepare_round_payload is deprecated and must not be used by correctness or benchmark paths")


def _all_to_all_variable(
    payload: torch.Tensor,
    send_rows: torch.Tensor,
    hidden_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    recv_rows = _exchange_sizes(send_rows)
    send_splits = send_rows.tolist()
    recv_splits = recv_rows.tolist()
    recv_total = int(sum(recv_splits))
    recv = torch.empty((recv_total, hidden_dim), device=payload.device, dtype=payload.dtype)
    send_flat = payload.reshape(-1)
    recv_flat = recv.reshape(-1)
    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=[int(value) * hidden_dim for value in recv_splits],
        input_split_sizes=[int(value) * hidden_dim for value in send_splits],
    )
    return recv, recv_rows, [int(v) for v in send_splits], [int(v) for v in recv_splits]


def _all_to_all_metadata_variable(
    metadata: torch.Tensor,
    send_rows: torch.Tensor,
    metadata_width: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    recv_rows = _exchange_sizes(send_rows)
    send_splits = [int(value) for value in send_rows.tolist()]
    recv_splits = [int(value) for value in recv_rows.tolist()]
    recv_total = int(sum(recv_splits))
    recv = torch.empty((recv_total, metadata_width), device=metadata.device, dtype=metadata.dtype)
    dist.all_to_all_single(
        recv.reshape(-1),
        metadata.reshape(-1),
        output_split_sizes=[rows * metadata_width for rows in recv_splits],
        input_split_sizes=[rows * metadata_width for rows in send_splits],
    )
    return recv, recv_rows, send_splits, recv_splits


def _bucket_effective_rows(bucket: WorkloadBucket, *, correctness_mode: bool) -> int:
    return max(1, int(bucket.token_count)) if correctness_mode else int(bucket.payload_rows)


def _build_bucket_metadata_dict(bucket: WorkloadBucket, rows: int) -> dict[str, Any]:
    route_item_ids = list(bucket.route_item_ids) if bucket.route_item_ids else [bucket.route_id]
    route_item_indices = list(bucket.route_item_indices) if bucket.route_item_indices else [0 for _ in route_item_ids]
    route_ranks = list(bucket.route_ranks) if bucket.route_ranks else [0 for _ in route_item_ids]
    token_ids = list(bucket.token_ids)
    return {
        "bucket_id": bucket.bucket_id,
        "route_id": bucket.route_id,
        "route_item_ids": route_item_ids,
        "route_item_indices": route_item_indices,
        "route_ranks": route_ranks,
        "origin_rank": int(bucket.origin_rank),
        "destination_rank": int(bucket.destination_rank),
        "rows": int(rows),
        "destination_id": int(bucket.destination_id),
        "expert_id": int(bucket.expert_id),
        "estimated_service_units": float(bucket.estimated_service_units),
        "token_count": int(bucket.token_count),
        "token_ids": token_ids,
        "token_positions": list(bucket.token_positions),
        "payload_bytes": int(rows * bucket.hidden_dim * 2),
    }


def _canonical_source_bucket_order(buckets: list[WorkloadBucket]) -> list[WorkloadBucket]:
    return sorted(
        buckets,
        key=lambda bucket: (
            int(bucket.destination_rank),
            min(bucket.token_positions) if bucket.token_positions else 0,
            bucket.bucket_id,
        ),
    )


def build_canonical_round_layout(
    round_buckets: list[WorkloadBucket],
    world_size: int,
    *,
    correctness_mode: bool,
    round_index: int,
) -> dict[str, Any]:
    by_source: dict[int, list[WorkloadBucket]] = {rank: [] for rank in range(world_size)}
    for bucket in round_buckets:
        by_source[int(bucket.origin_rank)].append(bucket)

    source_segments: dict[int, list[CanonicalRoundSegment]] = {rank: [] for rank in range(world_size)}
    destination_segments: dict[int, list[CanonicalRoundSegment]] = {rank: [] for rank in range(world_size)}
    send_splits_by_source: dict[int, list[int]] = {rank: [0 for _ in range(world_size)] for rank in range(world_size)}
    recv_splits_by_destination: dict[int, list[int]] = {rank: [0 for _ in range(world_size)] for rank in range(world_size)}

    for source_rank in range(world_size):
        payload_offset = 0
        metadata_offset = 0
        ordered = _canonical_source_bucket_order(by_source[source_rank])
        for bucket in ordered:
            rows = _bucket_effective_rows(bucket, correctness_mode=correctness_mode)
            segment = CanonicalRoundSegment(
                round_index=round_index,
                source_rank=source_rank,
                destination_rank=int(bucket.destination_rank),
                expert_id=int(bucket.expert_id),
                bucket_id=bucket.bucket_id,
                route_item_ids=list(bucket.route_item_ids) if bucket.route_item_ids else [bucket.route_id],
                token_ids=list(bucket.token_ids),
                route_ranks=list(bucket.route_ranks) if bucket.route_ranks else [0],
                payload_row_offset=payload_offset,
                payload_row_count=rows,
                metadata_row_offset=metadata_offset,
                metadata_row_count=rows,
                receive_row_offset=0,
                receive_metadata_row_offset=0,
                return_origin_rank=int(bucket.origin_rank),
            )
            source_segments[source_rank].append(segment)
            destination_segments[int(bucket.destination_rank)].append(segment)
            send_splits_by_source[source_rank][int(bucket.destination_rank)] += rows
            recv_splits_by_destination[int(bucket.destination_rank)][source_rank] += rows
            payload_offset += rows
            metadata_offset += rows

    normalized_destination_segments: dict[int, list[CanonicalRoundSegment]] = {rank: [] for rank in range(world_size)}
    for destination_rank in range(world_size):
        offset = 0
        metadata_offset = 0
        ordered: list[CanonicalRoundSegment] = []
        for source_rank in range(world_size):
            ordered.extend(
                sorted(
                    [segment for segment in destination_segments[destination_rank] if segment.source_rank == source_rank],
                    key=lambda item: (
                        item.destination_rank,
                        item.payload_row_offset,
                        item.bucket_id,
                    ),
                )
            )
        for segment in ordered:
            normalized = CanonicalRoundSegment(
                round_index=segment.round_index,
                source_rank=segment.source_rank,
                destination_rank=segment.destination_rank,
                expert_id=segment.expert_id,
                bucket_id=segment.bucket_id,
                route_item_ids=list(segment.route_item_ids),
                token_ids=list(segment.token_ids),
                route_ranks=list(segment.route_ranks),
                payload_row_offset=segment.payload_row_offset,
                payload_row_count=segment.payload_row_count,
                metadata_row_offset=segment.metadata_row_offset,
                metadata_row_count=segment.metadata_row_count,
                receive_row_offset=offset,
                receive_metadata_row_offset=metadata_offset,
                return_origin_rank=segment.return_origin_rank,
            )
            normalized_destination_segments[destination_rank].append(normalized)
            offset += segment.payload_row_count
            metadata_offset += segment.metadata_row_count
    return {
        "round_index": round_index,
        "source_segments": source_segments,
        "destination_segments": normalized_destination_segments,
        "send_splits_by_source": send_splits_by_source,
        "recv_splits_by_destination": recv_splits_by_destination,
    }


def _pack_round_payload_and_metadata(
    rank: int,
    round_layout: dict[str, Any],
    execution_cache: ExecutionCache,
    bucket_lookup: dict[str, WorkloadBucket],
    hidden_dim: int,
    metadata_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    source_segments: list[CanonicalRoundSegment] = list(round_layout["source_segments"][rank])
    send_splits = list(round_layout["send_splits_by_source"][rank])
    payload_segments: list[torch.Tensor] = []
    metadata_segments: list[torch.Tensor] = []
    metadata_records: list[dict[str, Any]] = []
    packing_metadata: list[dict[str, Any]] = []
    for segment in source_segments:
        bucket = bucket_lookup[segment.bucket_id]
        payload = execution_cache.payload_by_bucket_id[segment.bucket_id]
        metadata = execution_cache.metadata_by_bucket_id[segment.bucket_id]
        if int(payload.shape[0]) != int(segment.payload_row_count):
            raise RuntimeError(f"payload rows mismatch for bucket {segment.bucket_id}: {payload.shape[0]} vs {segment.payload_row_count}")
        if int(metadata.shape[0]) != int(segment.metadata_row_count):
            raise RuntimeError(f"metadata rows mismatch for bucket {segment.bucket_id}: {metadata.shape[0]} vs {segment.metadata_row_count}")
        payload_segments.append(payload)
        metadata_segments.append(metadata)
        meta_dict = _build_bucket_metadata_dict(
            bucket,
            int(segment.payload_row_count),
        )
        metadata_records.append(meta_dict)
        packing_metadata.append(
            {
                "bucket_id": bucket.bucket_id,
                "route_id": bucket.route_id,
                "route_item_ids": list(meta_dict["route_item_ids"]),
                "origin_rank": int(bucket.origin_rank),
                "destination_rank": int(bucket.destination_rank),
                "expert_id": int(bucket.expert_id),
                "rows": int(segment.payload_row_count),
                "bytes": int(segment.payload_row_count * hidden_dim * 2),
                "token_ids": list(bucket.token_ids),
            }
        )
    if payload_segments:
        payload_tensor = torch.cat(payload_segments, dim=0)
        metadata_tensor = torch.cat(metadata_segments, dim=0)
    else:
        payload_tensor = torch.zeros((0, hidden_dim), device=device, dtype=torch.float16)
        metadata_tensor = torch.zeros((0, metadata_width), device=device, dtype=torch.int32)
    if sum(send_splits) != int(payload_tensor.shape[0]) or int(payload_tensor.shape[0]) != int(metadata_tensor.shape[0]):
        raise RuntimeError("canonical source packing produced inconsistent rows")
    return payload_tensor, metadata_tensor, torch.tensor(send_splits, dtype=torch.int64, device=device), metadata_records, packing_metadata


def _pack_return_segments(
    result_payload: torch.Tensor,
    received_metadata: list[dict[str, Any]],
    world_size: int,
) -> tuple[torch.Tensor, list[int], list[dict[str, Any]]]:
    grouped_payload: dict[int, list[torch.Tensor]] = {rank: [] for rank in range(world_size)}
    grouped_metadata: dict[int, list[dict[str, Any]]] = {rank: [] for rank in range(world_size)}
    offset = 0
    for item in received_metadata:
        rows = int(item["rows"])
        segment = result_payload[offset : offset + rows]
        origin_rank = int(item["origin_rank"])
        grouped_payload[origin_rank].append(segment)
        grouped_metadata[origin_rank].append(dict(item))
        offset += rows
    packed_segments: list[torch.Tensor] = []
    packed_metadata: list[dict[str, Any]] = []
    send_splits: list[int] = []
    for origin_rank in range(world_size):
        metadata_items = grouped_metadata[origin_rank]
        payload_items = grouped_payload[origin_rank]
        rows = sum(int(item["rows"]) for item in metadata_items)
        send_splits.append(rows)
        if rows:
            packed_segments.append(torch.cat(payload_items, dim=0) if len(payload_items) > 1 else payload_items[0])
            packed_metadata.extend(metadata_items)
    if offset != int(result_payload.shape[0]):
        raise RuntimeError("pack_return_segments: metadata rows do not cover result payload rows")
    if sum(send_splits) != int(result_payload.shape[0]):
        raise RuntimeError("pack_return_segments: send_splits do not match result payload rows")
    packed_payload = (
        torch.cat(packed_segments, dim=0)
        if packed_segments
        else torch.zeros((0, result_payload.shape[1]), device=result_payload.device, dtype=result_payload.dtype)
    )
    return packed_payload, send_splits, packed_metadata


def _pack_return_payload_and_metadata(
    result_payload: torch.Tensor,
    result_metadata: torch.Tensor,
    received_metadata: list[dict[str, Any]],
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[str, Any]]]:
    if int(result_payload.shape[0]) != int(result_metadata.shape[0]):
        raise RuntimeError("return payload and metadata row count mismatch")
    grouped_payload: dict[int, list[torch.Tensor]] = {rank: [] for rank in range(world_size)}
    grouped_metadata_tensors: dict[int, list[torch.Tensor]] = {rank: [] for rank in range(world_size)}
    grouped_metadata_dicts: dict[int, list[dict[str, Any]]] = {rank: [] for rank in range(world_size)}
    offset = 0
    for item in received_metadata:
        rows = int(item["rows"])
        origin_rank = int(item["origin_rank"])
        grouped_payload[origin_rank].append(result_payload[offset : offset + rows])
        grouped_metadata_tensors[origin_rank].append(result_metadata[offset : offset + rows])
        grouped_metadata_dicts[origin_rank].append(dict(item))
        offset += rows
    if offset != int(result_payload.shape[0]):
        raise RuntimeError("return metadata rows do not cover result rows")
    packed_payload_parts: list[torch.Tensor] = []
    packed_metadata_parts: list[torch.Tensor] = []
    packed_metadata_dicts: list[dict[str, Any]] = []
    send_splits: list[int] = []
    for origin_rank in range(world_size):
        dict_items = grouped_metadata_dicts[origin_rank]
        tensor_items = grouped_metadata_tensors[origin_rank]
        payload_items = grouped_payload[origin_rank]
        rows = sum(int(item["rows"]) for item in dict_items)
        send_splits.append(rows)
        if rows:
            if rows != sum(int(tensor.shape[0]) for tensor in tensor_items):
                raise RuntimeError("return metadata tensor rows do not match dict rows")
            if rows != sum(int(tensor.shape[0]) for tensor in payload_items):
                raise RuntimeError("return payload tensor rows do not match dict rows")
            packed_payload_parts.append(torch.cat(payload_items, dim=0) if len(payload_items) > 1 else payload_items[0])
            packed_metadata_parts.append(torch.cat(tensor_items, dim=0) if len(tensor_items) > 1 else tensor_items[0])
            packed_metadata_dicts.extend(dict_items)
    if sum(send_splits) != int(result_payload.shape[0]):
        raise RuntimeError("return send splits do not match result payload rows")
    packed_payload = (
        torch.cat(packed_payload_parts, dim=0)
        if packed_payload_parts
        else torch.zeros((0, result_payload.shape[1]), device=result_payload.device, dtype=result_payload.dtype)
    )
    packed_metadata = (
        torch.cat(packed_metadata_parts, dim=0)
        if packed_metadata_parts
        else torch.zeros((0, result_metadata.shape[1]), device=result_metadata.device, dtype=result_metadata.dtype)
    )
    if int(packed_payload.shape[0]) != int(packed_metadata.shape[0]):
        raise RuntimeError("packed return payload/metadata rows mismatch")
    return packed_payload, packed_metadata, send_splits, packed_metadata_dicts


def _run_local_compute(
    payload: torch.Tensor,
    execution_cache: ExecutionCache,
) -> tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if payload.numel() == 0:
        result = payload
    else:
        hidden = torch.nn.functional.linear(payload, execution_cache.weight1)
        hidden = torch.nn.functional.gelu(hidden)
        result = torch.nn.functional.linear(hidden, execution_cache.weight2)
    end.record()
    torch.cuda.synchronize(payload.device)
    return result, float(start.elapsed_time(end))


def _run_expert_grouped_compute(
    recv_payload: torch.Tensor,
    recv_metadata: list[dict[str, Any]],
    execution_cache: ExecutionCache,
) -> tuple[torch.Tensor, float, list[dict[str, Any]]]:
    if recv_payload.numel() == 0:
        return recv_payload, 0.0, []
    offset = 0
    expert_groups: dict[int, list[tuple[int, int]]] = {}
    for item in recv_metadata:
        rows = int(item["rows"])
        expert_id = int(item["expert_id"])
        expert_groups.setdefault(expert_id, []).append((offset, rows))
        offset += rows
    result = torch.empty_like(recv_payload)
    total_ms = 0.0
    expert_records: list[dict[str, Any]] = []
    for expert_id, spans in sorted(expert_groups.items()):
        rows = sum(length for _, length in spans)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        segments = [recv_payload[begin : begin + length] for begin, length in spans]
        packed = torch.cat(segments, dim=0) if len(segments) > 1 else segments[0]
        hidden = torch.nn.functional.linear(packed, execution_cache.expert_weight1[expert_id])
        hidden = torch.nn.functional.gelu(hidden)
        expert_result = torch.nn.functional.linear(hidden, execution_cache.expert_weight2[expert_id])
        cursor = 0
        for begin, length in spans:
            result[begin : begin + length].copy_(expert_result[cursor : cursor + length])
            cursor += length
        end.record()
        torch.cuda.synchronize(recv_payload.device)
        elapsed = float(start.elapsed_time(end))
        total_ms += elapsed
        expert_records.append({"expert_id": expert_id, "rows": rows, "compute_time_ms": elapsed, "span_count": len(spans)})
    return result, total_ms, expert_records


def _return_to_origin(
    rank: int,
    world_size: int,
    result_payload: torch.Tensor,
    result_metadata: torch.Tensor,
    origin_metadata: list[dict[str, Any]],
    hidden_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int], torch.Tensor]:
    metadata_width = int(result_metadata.shape[1]) if result_metadata.ndim == 2 else 0
    packed_payload, packed_metadata_tensor, send_splits, _ = _pack_return_payload_and_metadata(
        result_payload,
        result_metadata,
        origin_metadata,
        world_size,
    )
    return_counts = torch.tensor(send_splits, dtype=torch.int64, device=result_payload.device)
    recv_return_rows = _exchange_sizes(return_counts)
    recv_splits = recv_return_rows.tolist()
    recv_total = int(sum(recv_splits))
    recv_back = torch.empty((recv_total, hidden_dim), device=result_payload.device, dtype=result_payload.dtype)
    recv_metadata = torch.empty((recv_total, metadata_width), device=result_metadata.device, dtype=result_metadata.dtype)
    dist.all_to_all_single(
        recv_back.reshape(-1),
        packed_payload.reshape(-1),
        output_split_sizes=[int(value) * hidden_dim for value in recv_splits],
        input_split_sizes=[int(value) * hidden_dim for value in send_splits],
    )
    dist.all_to_all_single(
        recv_metadata.reshape(-1),
        packed_metadata_tensor.reshape(-1),
        output_split_sizes=[int(value) * metadata_width for value in recv_splits],
        input_split_sizes=[int(value) * metadata_width for value in send_splits],
    )
    return recv_back, recv_metadata, recv_return_rows, [int(v) for v in send_splits], [int(v) for v in recv_splits], packed_metadata_tensor


def _calc_release_rounds(release_order: list[str], round_size: int) -> list[list[str]]:
    return release_rounds_for_order(release_order, round_size)


def _bucket_lookup(plan: WorkloadPlan) -> dict[str, WorkloadBucket]:
    return {bucket.bucket_id: bucket for bucket in plan.bucket_workloads}


def _matrix_zeros(world_size: int) -> list[list[int]]:
    return [[0 for _ in range(world_size)] for _ in range(world_size)]


def _split_to_matrix(source_rank: int, splits: list[int], hidden_dim: int) -> tuple[list[list[int]], list[list[int]]]:
    bytes_matrix = _matrix_zeros(len(splits))
    rows_matrix = _matrix_zeros(len(splits))
    for destination_rank, rows in enumerate(splits):
        rows_matrix[source_rank][destination_rank] = int(rows)
        bytes_matrix[source_rank][destination_rank] = int(rows) * hidden_dim * 2
    return rows_matrix, bytes_matrix


def _planned_round_matrix(
    round_buckets: list[WorkloadBucket],
    world_size: int,
    hidden_dim: int,
    *,
    correctness_mode: bool = False,
) -> tuple[list[list[int]], list[list[int]]]:
    rows = _matrix_zeros(world_size)
    bytes_matrix = _matrix_zeros(world_size)
    for bucket in round_buckets:
        effective_rows = int(bucket.payload_rows) if not correctness_mode else max(1, int(bucket.token_count))
        rows[bucket.origin_rank][bucket.destination_rank] += effective_rows
        bytes_matrix[bucket.origin_rank][bucket.destination_rank] += effective_rows * hidden_dim * 2
    return rows, bytes_matrix


def _sum_matrix(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    size = len(left)
    return [[int(left[i][j]) + int(right[i][j]) for j in range(size)] for i in range(size)]


def _transpose_matrix(matrix: list[list[int]]) -> list[list[int]]:
    size = len(matrix)
    return [[int(matrix[j][i]) for j in range(size)] for i in range(size)]


def _off_diagonal_sum(matrix: list[list[int]]) -> int:
    total = 0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if i != j:
                total += int(value)
    return total


def _nonzero_off_diagonal_entries(matrix: list[list[int]]) -> int:
    count = 0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if i != j and int(value) > 0:
                count += 1
    return count


def _all_gather_object(value: Any) -> list[Any]:
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def summarize_route_manifests(
    expected_manifest: list[dict[str, Any]],
    dispatch_manifest: list[dict[str, Any]],
    receive_manifest: list[dict[str, Any]],
    return_manifest: list[dict[str, Any]],
    origin_verified_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    def _index(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(int(record["route_item_index"]), []).append(record)
        return grouped

    dispatch_idx = _index(dispatch_manifest)
    receive_idx = _index(receive_manifest)
    return_idx = _index(return_manifest)
    verified_idx = _index(origin_verified_manifest)
    expected_idx = {int(item["global_route_item_index"]): item for item in expected_manifest}
    expected = sorted(expected_idx)
    duplicate_dispatch = sum(max(0, len(items) - 1) for items in dispatch_idx.values())
    duplicate_receive = sum(max(0, len(items) - 1) for items in receive_idx.values())
    duplicate_return = sum(max(0, len(items) - 1) for items in return_idx.values())
    duplicate_verified = sum(max(0, len(items) - 1) for items in verified_idx.values())
    missing_dispatch = 0
    missing_receive = 0
    missing_return = 0
    missing_verified = 0
    unexpected_dispatch = 0
    unexpected_receive = 0
    unexpected_return = 0
    unexpected_verified = 0
    wrong_origin_count = 0
    wrong_destination_count = 0
    wrong_expert_count = 0
    wrong_route_rank_count = 0
    wrong_token_id_count = 0
    wrong_microbatch_count = 0
    for route_item_index in dispatch_idx:
        if route_item_index not in expected_idx:
            unexpected_dispatch += 1
    for route_item_index in receive_idx:
        if route_item_index not in expected_idx:
            unexpected_receive += 1
    for route_item_index in return_idx:
        if route_item_index not in expected_idx:
            unexpected_return += 1
    for route_item_index in verified_idx:
        if route_item_index not in expected_idx:
            unexpected_verified += 1
    for route_item_index in expected:
        expected_item = expected_idx[route_item_index]
        dispatch_item = dispatch_idx.get(route_item_index, [])
        receive_item = receive_idx.get(route_item_index, [])
        return_item = return_idx.get(route_item_index, [])
        verified_item = verified_idx.get(route_item_index, [])
        if len(dispatch_item) != 1:
            missing_dispatch += 1
        if len(receive_item) != 1:
            missing_receive += 1
        if len(return_item) != 1:
            missing_return += 1
        if len(verified_item) != 1:
            missing_verified += 1
        if not (len(dispatch_item) == len(receive_item) == len(return_item) == len(verified_item) == 1):
            continue
        for item in (dispatch_item[0], receive_item[0], return_item[0], verified_item[0]):
            if int(item["token_id"]) != int(expected_item["token_id"]):
                wrong_token_id_count += 1
            if int(item["origin_rank"]) != int(expected_item["origin_rank"]):
                wrong_origin_count += 1
            if int(item["destination_rank"]) != int(expected_item["destination_rank"]):
                wrong_destination_count += 1
            if int(item["expert_id"]) != int(expected_item["expert_id"]):
                wrong_expert_count += 1
            if int(item.get("route_rank", 0)) != int(expected_item.get("route_rank", 0)):
                wrong_route_rank_count += 1
            if int(item.get("microbatch_id", expected_item["microbatch_id"])) != int(expected_item["microbatch_id"]):
                wrong_microbatch_count += 1
    return {
        "expected_route_item_count": len(expected),
        "dispatch_route_item_count": len(dispatch_manifest),
        "receive_route_item_count": len(receive_manifest),
        "return_route_item_count": len(return_manifest),
        "verified_route_item_count": len(origin_verified_manifest),
        "missing_dispatch_count": missing_dispatch,
        "missing_receive_count": missing_receive,
        "missing_return_count": missing_return,
        "missing_verified_count": missing_verified,
        "unexpected_dispatch_count": unexpected_dispatch,
        "unexpected_receive_count": unexpected_receive,
        "unexpected_return_count": unexpected_return,
        "unexpected_verified_count": unexpected_verified,
        "duplicate_dispatch_count": duplicate_dispatch,
        "duplicate_receive_count": duplicate_receive,
        "duplicate_return_count": duplicate_return,
        "duplicate_verified_count": duplicate_verified,
        "wrong_token_id_count": wrong_token_id_count,
        "wrong_origin_count": wrong_origin_count,
        "wrong_destination_count": wrong_destination_count,
        "wrong_expert_count": wrong_expert_count,
        "wrong_route_rank_count": wrong_route_rank_count,
        "wrong_microbatch_count": wrong_microbatch_count,
        "route_identity_scope": "global_across_all_microbatches",
        "manifest_truth_source": "gpu_metadata_sidecar",
    }


def manifest_is_valid(summary: dict[str, Any]) -> bool:
    required_zero = [
        "missing_dispatch_count",
        "missing_receive_count",
        "missing_return_count",
        "missing_verified_count",
        "unexpected_dispatch_count",
        "unexpected_receive_count",
        "unexpected_return_count",
        "unexpected_verified_count",
        "duplicate_dispatch_count",
        "duplicate_receive_count",
        "duplicate_return_count",
        "duplicate_verified_count",
        "wrong_token_id_count",
        "wrong_origin_count",
        "wrong_destination_count",
        "wrong_expert_count",
        "wrong_route_rank_count",
        "wrong_microbatch_count",
    ]
    return all(int(summary.get(key, 0)) == 0 for key in required_zero)


def _rank_phase_log(
    artifact_dir: str,
    rank: int,
    phase: str,
    round_index: int,
    *,
    rows: int = 0,
    send_splits: list[int] | None = None,
    recv_splits: list[int] | None = None,
) -> None:
    target = Path(artifact_dir) / "logs"
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "rank": rank,
        "phase": phase,
        "round": round_index,
        "rows": int(rows),
        "send_splits": list(send_splits or []),
        "recv_splits": list(recv_splits or []),
    }
    with (target / f"rank{rank}.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _policy_feature_contract(strategy: str) -> dict[str, Any]:
    contracts = {
        "fifo": {
            "allowed_feature_groups": ["arrival_index"],
            "actual_feature_names_read": ["arrival_index"],
            "forbidden_feature_names_checked": ["dependency_features", "runtime_state", "payload_bytes"],
        },
        "random-order": {
            "allowed_feature_groups": ["random_seed", "bucket_ids"],
            "actual_feature_names_read": ["policy_seed", "bucket_id"],
            "forbidden_feature_names_checked": ["dependency_features", "runtime_state", "payload_bytes"],
        },
        "state-only": {
            "allowed_feature_groups": ["global_state", "payload_shape", "placement", "projected_load", "service_history"],
            "actual_feature_names_read": [
                "destination_pending_work",
                "destination_queue_depth",
                "rank_backlog_work",
                "rank_queue_depth",
                "destination_hotness",
                "destination_latency_history",
                "payload_rows",
                "payload_bytes",
            ],
            "forbidden_feature_names_checked": [
                "source_coverage",
                "coactive_peer_degree",
                "coactive_event_density",
                "position_spread",
                "bridge_score",
            ],
        },
        "strong-state": {
            "allowed_feature_groups": ["source_pressure", "destination_pressure", "flow_pressure", "return_pressure", "expert_pressure", "history"],
            "actual_feature_names_read": [
                "source_outbound_pending_bytes",
                "destination_inbound_pending_bytes",
                "flow_pending_bytes",
                "return_flow_pending_bytes",
                "rank_compute_queue_rows",
                "expert_pending_rows",
                "rank_dispatch_time_ms",
                "rank_compute_time_ms",
                "rank_return_time_ms",
                "payload_rows",
                "payload_bytes",
            ],
            "forbidden_feature_names_checked": [
                "source_coverage",
                "coactive_peer_degree",
                "coactive_event_density",
                "position_spread",
                "bridge_score",
            ],
        },
        "dependency-only": {
            "allowed_feature_groups": ["dependency_features"],
            "actual_feature_names_read": [
                "source_coverage",
                "coactive_peer_degree",
                "coactive_event_density",
                "position_spread",
                "bridge_score",
            ],
            "forbidden_feature_names_checked": [
                "source_outbound_pending_bytes",
                "destination_inbound_pending_bytes",
                "flow_pending_bytes",
                "return_flow_pending_bytes",
            ],
        },
        "full": {
            "allowed_feature_groups": ["state_features", "dependency_features"],
            "actual_feature_names_read": [
                "state_only_score",
                "dependency_only_score",
            ],
            "forbidden_feature_names_checked": [],
        },
    }
    return contracts.get(strategy, {"allowed_feature_groups": [], "actual_feature_names_read": [], "forbidden_feature_names_checked": []})


def benchmark_preflight(
    protocol: ProtocolConfig,
    plan: WorkloadPlan,
    execution_cache: ExecutionCache,
    *,
    allowed_pids: list[int] | None = None,
    allowed_cmd_substrings: list[str] | None = None,
) -> dict[str, Any]:
    gpu_processes = []
    unexpected_gpu_processes = []
    allowed_pid_set = {int(pid) for pid in (allowed_pids or [])}
    allowed_cmd_tokens = [str(token) for token in (allowed_cmd_substrings or []) if str(token)]
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if result:
            gpu_processes = [line.strip() for line in result.splitlines() if line.strip()]
    except Exception:
        gpu_processes = []
    for line in gpu_processes:
        pid_text = line.split(",", 1)[0].strip()
        try:
            pid_value = int(pid_text)
        except ValueError:
            unexpected_gpu_processes.append(line)
            continue
        if pid_value not in allowed_pid_set:
            cmdline = ""
            try:
                cmdline = subprocess.check_output(
                    ["ps", "-p", str(pid_value), "-o", "cmd="],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                cmdline = ""
            if allowed_cmd_tokens and any(token in cmdline for token in allowed_cmd_tokens):
                continue
            unexpected_gpu_processes.append(line)
    if unexpected_gpu_processes:
        raise RuntimeError(f"benchmark preflight failed: unexpected GPU compute processes detected: {unexpected_gpu_processes}")
    first_bucket = plan.bucket_workloads[0] if plan.bucket_workloads else None
    payload_checksum = None
    metadata_checksum = None
    weight_checksum = None
    if first_bucket is not None and first_bucket.bucket_id in execution_cache.payload_by_bucket_id:
        payload_checksum = float(execution_cache.payload_by_bucket_id[first_bucket.bucket_id].sum().item())
        metadata_checksum = int(execution_cache.metadata_by_bucket_id[first_bucket.bucket_id].sum().item())
    if execution_cache.expert_weight1:
        expert_id = sorted(execution_cache.expert_weight1)[0]
        weight_checksum = float(execution_cache.expert_weight1[expert_id].sum().item() + execution_cache.expert_weight2[expert_id].sum().item())
    trace_hash = _hash_payload(
        {
            "model_key": plan.model_key,
            "microbatch_id": plan.microbatch_id,
            "layer_id": plan.layer_id,
            "routing_summary": plan.routing_summary,
        }
    )
    plan_hash = _hash_payload(plan.to_dict())
    placement_hash = _hash_payload([asdict(entry) for entry in plan.placement_mapping])
    token_origin_map_hash = _hash_payload(plan.routing_summary.get("token_origin_rank_detail", {}))
    route_item_records = [
        {
            "route_item_index": int(route_item_index),
            "token_id": int(token_id),
            "origin_rank": int(bucket.origin_rank),
            "destination_rank": int(bucket.destination_rank),
            "expert_id": int(bucket.expert_id),
            "route_rank": int(route_rank),
        }
        for bucket in plan.bucket_workloads
        for route_item_index, token_id, route_rank in zip(bucket.route_item_indices, bucket.token_ids, bucket.route_ranks)
    ]
    route_item_records.sort(
        key=lambda item: (
            int(item["route_item_index"]),
            int(item["token_id"]),
            int(item["origin_rank"]),
            int(item["destination_rank"]),
            int(item["expert_id"]),
            int(item["route_rank"]),
        )
    )
    total_route_item_set_hash = _hash_payload(route_item_records)
    return {
        "payload_checksum": payload_checksum,
        "metadata_checksum": metadata_checksum,
        "expert_weight_checksum": weight_checksum,
        "plan_hash": plan_hash,
        "trace_hash": trace_hash,
        "placement_hash": placement_hash,
        "token_origin_map_hash": token_origin_map_hash,
        "total_route_item_set_hash": total_route_item_set_hash,
        "global_route_identity_scope": "global_across_all_microbatches",
        "used_python_metadata_gather_expected": False,
        "legacy_payload_function_allowed": False,
        "gpu_processes": gpu_processes,
        "unexpected_gpu_processes": unexpected_gpu_processes,
        "allowed_pids": sorted(allowed_pid_set),
        "allowed_cmd_substrings": allowed_cmd_tokens,
        "metric_semantics_version": "poc2.5-final",
        "state_semantics": "historical_plus_projected",
    }


def expected_route_item_manifest(plans: list[WorkloadPlan]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for plan in plans:
        for bucket in plan.bucket_workloads:
            for route_item_index, token_id, route_rank in zip(bucket.route_item_indices, bucket.token_ids, bucket.route_ranks):
                expected.append(
                    {
                        "global_route_item_index": int(route_item_index),
                        "token_id": int(token_id),
                        "origin_rank": int(bucket.origin_rank),
                        "destination_rank": int(bucket.destination_rank),
                        "expert_id": int(bucket.expert_id),
                        "route_rank": int(route_rank),
                        "microbatch_id": int(plan.microbatch_id),
                        "layer_id": int(plan.layer_id),
                    }
                )
    expected.sort(key=lambda item: int(item["global_route_item_index"]))
    return expected


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_flow_key(source_rank: int, destination_rank: int) -> str:
    return f"{source_rank}->{destination_rank}"


def preallocate_execution_cache(
    plan: WorkloadPlan,
    rank: int,
    device: torch.device,
    *,
    correctness_mode: bool,
) -> ExecutionCache:
    payload_by_bucket_id: dict[str, torch.Tensor] = {}
    metadata_by_bucket_id: dict[str, torch.Tensor] = {}
    expert_ids = sorted({bucket.expert_id for bucket in plan.bucket_workloads if bucket.destination_rank == rank or bucket.origin_rank == rank})
    for bucket in plan.bucket_workloads:
        if bucket.origin_rank != rank:
            continue
        rows = bucket.payload_rows if not correctness_mode else max(1, bucket.token_count)
        route_item_ids = list(bucket.route_item_ids) if bucket.route_item_ids else [bucket.route_id]
        route_item_indices = list(bucket.route_item_indices) if bucket.route_item_indices else list(range(rows))
        route_ranks = list(bucket.route_ranks) if bucket.route_ranks else [0 for _ in route_item_ids]
        if correctness_mode:
            payload = torch.zeros((rows, plan.hidden_dim), device=device, dtype=torch.float16)
            if plan.hidden_dim > 0:
                payload[:, 0] = torch.tensor(
                    [float(bucket.token_ids[min(index, len(bucket.token_ids) - 1)]) for index in range(rows)],
                    device=device,
                    dtype=torch.float16,
                )
            if plan.hidden_dim > 1:
                payload[:, 1] = float(bucket.origin_rank)
            if plan.hidden_dim > 2:
                payload[:, 2] = float(bucket.destination_rank)
            if plan.hidden_dim > 3:
                payload[:, 3] = float(bucket.expert_id)
        else:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(plan.seed + rank * 1009 + bucket.destination_rank * 97 + sum(bucket.token_ids) * 13)
            payload = torch.randn((rows, plan.hidden_dim), device=device, dtype=torch.float16, generator=generator)
        payload_by_bucket_id[bucket.bucket_id] = payload
        metadata = torch.zeros((rows, 6), device=device, dtype=torch.int32)
        token_rows = [int(bucket.token_ids[min(index, len(bucket.token_ids) - 1)]) for index in range(rows)]
        route_row_indices = [int(route_item_indices[min(index, len(route_item_indices) - 1)]) for index in range(rows)]
        route_rank_rows = [int(route_ranks[min(index, len(route_ranks) - 1)]) for index in range(rows)]
        metadata[:, 0] = torch.tensor(token_rows, device=device, dtype=torch.int32)
        metadata[:, 1] = torch.tensor(route_row_indices, device=device, dtype=torch.int32)
        metadata[:, 2] = int(bucket.origin_rank)
        metadata[:, 3] = int(bucket.destination_rank)
        metadata[:, 4] = int(bucket.expert_id)
        metadata[:, 5] = torch.tensor(route_rank_rows, device=device, dtype=torch.int32)
        metadata_by_bucket_id[bucket.bucket_id] = metadata
    expert_weight1: dict[int, torch.Tensor] = {}
    expert_weight2: dict[int, torch.Tensor] = {}
    for expert_id in expert_ids:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(plan.seed + expert_id * 7919 + rank * 101)
        expert_weight1[expert_id] = torch.randn((plan.intermediate_dim, plan.hidden_dim), device=device, dtype=torch.float16, generator=generator)
        expert_weight2[expert_id] = torch.randn((plan.hidden_dim, plan.intermediate_dim), device=device, dtype=torch.float16, generator=generator)
    if plan.hidden_dim > 0:
        warm = torch.zeros((1, plan.hidden_dim), device=device, dtype=torch.float16)
        for expert_id in expert_ids[:1]:
            hidden = torch.nn.functional.linear(warm, expert_weight1[expert_id])
            _ = torch.nn.functional.linear(torch.nn.functional.gelu(hidden), expert_weight2[expert_id])
        torch.cuda.synchronize(device)
    return ExecutionCache(
        payload_by_bucket_id=payload_by_bucket_id,
        metadata_by_bucket_id=metadata_by_bucket_id,
        expert_weight1=expert_weight1,
        expert_weight2=expert_weight2,
        correctness_mode=correctness_mode,
    )


def _received_metadata_for_rank(
    all_source_metadata: list[list[dict[str, Any]]],
    destination_rank: int,
) -> list[dict[str, Any]]:
    received: list[dict[str, Any]] = []
    for source_items in all_source_metadata:
        for item in source_items:
            if int(item["destination_rank"]) == destination_rank:
                received.append(dict(item))
    return received


def _planned_received_metadata_for_rank(
    round_layout: dict[str, Any],
    bucket_lookup: dict[str, WorkloadBucket],
    destination_rank: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for segment in round_layout["destination_segments"][destination_rank]:
        bucket = bucket_lookup[segment.bucket_id]
        items.append(_build_bucket_metadata_dict(bucket, int(segment.payload_row_count)))
    return items


def decode_metadata_tensor_rows(metadata_tensor: torch.Tensor) -> list[RouteItemMetadata]:
    rows = metadata_tensor.cpu().tolist() if int(metadata_tensor.shape[0]) else []
    decoded: list[RouteItemMetadata] = []
    for row in rows:
        if len(row) != 6:
            raise RuntimeError(f"metadata row width mismatch: expected 6, got {len(row)}")
        decoded.append(
            RouteItemMetadata(
                token_id=int(row[0]),
                global_route_item_index=int(row[1]),
                origin_rank=int(row[2]),
                destination_rank=int(row[3]),
                expert_id=int(row[4]),
                route_rank=int(row[5]),
            )
        )
    return decoded


def _metadata_tensor_to_dicts(
    metadata_tensor: torch.Tensor,
    planned_metadata: list[dict[str, Any]],
    *,
    current_destination_rank: int | None = None,
) -> list[dict[str, Any]]:
    rows = int(metadata_tensor.shape[0])
    if rows != sum(int(item["rows"]) for item in planned_metadata):
        raise RuntimeError("metadata tensor rows do not match planned metadata rows")
    decoded_rows = decode_metadata_tensor_rows(metadata_tensor)
    decoded: list[dict[str, Any]] = []
    cursor = 0
    for item in planned_metadata:
        item_rows = int(item["rows"])
        slice_rows = decoded_rows[cursor : cursor + item_rows]
        if len(slice_rows) != item_rows:
            raise RuntimeError("metadata tensor row slicing mismatch")
        if any(row.origin_rank != int(item["origin_rank"]) for row in slice_rows):
            raise RuntimeError("metadata origin_rank does not match planned segment source rank")
        if any(row.destination_rank != int(item["destination_rank"]) for row in slice_rows):
            raise RuntimeError("metadata destination_rank does not match planned segment destination rank")
        if current_destination_rank is not None and any(row.destination_rank != int(current_destination_rank) for row in slice_rows):
            raise RuntimeError("metadata destination_rank does not match current destination rank")
        if any(row.expert_id != int(item["expert_id"]) for row in slice_rows):
            raise RuntimeError("metadata expert_id does not match planned segment expert_id")
        decoded.append(
            {
                **item,
                "token_ids": [row.token_id for row in slice_rows],
                "route_item_indices": [row.global_route_item_index for row in slice_rows],
                "route_ranks": [row.route_rank for row in slice_rows],
                "metadata_rows": [row.to_dict() for row in slice_rows],
            }
        )
        cursor += item_rows
    if cursor != rows:
        raise RuntimeError("metadata tensor decoding did not consume all rows")
    return decoded


def build_global_dispatch_plan(
    *,
    protocol: ProtocolConfig,
    plan: WorkloadPlan,
    strategy: str,
    global_state: GlobalRuntimeState,
    repetition_index: int,
) -> GlobalDispatchPlan:
    if dist.get_rank() != 0:
        raise RuntimeError("only rank0 may generate a global dispatch plan")
    scheduler_state = global_state.to_scheduler_state()
    planned_source_outbound: dict[int, float] = {}
    planned_destination_inbound: dict[int, float] = {}
    planned_flow_pending: dict[str, float] = {}
    planned_return_flow: dict[str, float] = {}
    planned_rank_compute_rows: dict[int, float] = {}
    planned_expert_rows: dict[int, float] = {}
    for bucket in plan.bucket_workloads:
        planned_source_outbound[bucket.origin_rank] = planned_source_outbound.get(bucket.origin_rank, 0.0) + float(bucket.payload_bytes)
        planned_destination_inbound[bucket.destination_rank] = planned_destination_inbound.get(bucket.destination_rank, 0.0) + float(bucket.payload_bytes)
        planned_flow_pending[_make_flow_key(bucket.origin_rank, bucket.destination_rank)] = (
            planned_flow_pending.get(_make_flow_key(bucket.origin_rank, bucket.destination_rank), 0.0) + float(bucket.payload_bytes)
        )
        planned_return_flow[_make_flow_key(bucket.destination_rank, bucket.origin_rank)] = (
            planned_return_flow.get(_make_flow_key(bucket.destination_rank, bucket.origin_rank), 0.0) + float(bucket.payload_bytes)
        )
        planned_rank_compute_rows[bucket.destination_rank] = planned_rank_compute_rows.get(bucket.destination_rank, 0.0) + float(bucket.payload_rows)
        planned_expert_rows[bucket.expert_id] = planned_expert_rows.get(bucket.expert_id, 0.0) + float(bucket.payload_rows)
    bucket_records = []
    for bucket in plan.bucket_workloads:
        record = bucket.to_scheduler_record()
        flow_key = f"{bucket.origin_rank}->{bucket.destination_rank}"
        return_key = f"{bucket.destination_rank}->{bucket.origin_rank}"
        record.source_outbound_pending_bytes = float(planned_source_outbound.get(bucket.origin_rank, 0.0))
        record.destination_inbound_pending_bytes = float(planned_destination_inbound.get(bucket.destination_rank, 0.0))
        record.flow_pending_bytes = float(planned_flow_pending.get(flow_key, 0.0))
        record.return_flow_pending_bytes = float(planned_return_flow.get(return_key, 0.0))
        record.rank_compute_queue_rows = float(planned_rank_compute_rows.get(bucket.destination_rank, 0.0))
        record.expert_pending_rows = float(planned_expert_rows.get(bucket.expert_id, 0.0))
        record.estimated_dispatch_cost = bucket.payload_bytes / float(1 << 20)
        record.estimated_compute_cost = float(bucket.payload_rows)
        record.estimated_return_cost = bucket.payload_bytes / float(1 << 20)
        bucket_records.append(record)
    scheduler_input = SchedulerInput(
        strategy=strategy,
        microbatch_id=plan.microbatch_id,
        layer_id=plan.layer_id,
        layer_path=plan.layer_path,
        bucket_records=bucket_records,
        runtime_state=scheduler_state,
        metadata={
            "plan_id": plan.plan_id,
            "release_round_size": protocol.release_round_size,
            "arrival_jitter_mode": plan.routing_summary.get("arrival_jitter_mode", "none"),
        },
    )
    decision = _build_policy_decision(strategy, scheduler_input, plan, scheduler_state, repetition_index)
    release_rounds = [list(item) for item in getattr(decision, "release_rounds", _calc_release_rounds(decision.release_order, protocol.release_round_size))]
    lookup = _bucket_lookup(plan)
    per_round_dispatch_matrices = []
    total_dispatch_matrix = _matrix_zeros(protocol.world_size)
    for round_ids in release_rounds:
        round_buckets = [lookup[bucket_id] for bucket_id in round_ids]
        rows_matrix, _ = _planned_round_matrix(
            round_buckets,
            protocol.world_size,
            plan.hidden_dim,
            correctness_mode=False,
        )
        per_round_dispatch_matrices.append(rows_matrix)
        total_dispatch_matrix = _sum_matrix(total_dispatch_matrix, rows_matrix)
    plan_payload = GlobalDispatchPlan(
        plan_id=f"{plan.plan_id}_{strategy}_rep{repetition_index}",
        workload_plan_id=plan.plan_id,
        policy_name=strategy,
        policy_seed=plan.seed + repetition_index,
        microbatch_id=plan.microbatch_id,
        release_order=list(decision.release_order),
        release_rounds=[list(item) for item in release_rounds],
        total_dispatch_matrix=total_dispatch_matrix,
        per_round_dispatch_matrices=per_round_dispatch_matrices,
        decision_hash="",
        scheduler_state_snapshot=global_state.to_dict(),
        scheduler_decision=decision.to_dict(),
        round_diagnostics=list(getattr(decision, "round_diagnostics", [])),
        round_pressure_score=float(getattr(decision, "round_pressure_score", 0.0)),
    )
    plan_payload.decision_hash = canonical_global_plan_hash(plan_payload)
    return plan_payload


def broadcast_global_dispatch_plan(global_plan: GlobalDispatchPlan | None) -> GlobalDispatchPlan:
    payloads: list[Any] = [global_plan.to_dict() if global_plan is not None else None]
    dist.broadcast_object_list(payloads, src=0)
    if payloads[0] is None:
        raise RuntimeError("global dispatch plan broadcast failed")
    plan = GlobalDispatchPlan.from_dict(payloads[0])
    local_hash = canonical_global_plan_hash(plan)
    gathered = _all_gather_object(local_hash)
    if any(item != plan.decision_hash for item in gathered):
        raise RuntimeError(f"global dispatch plan hash mismatch across ranks: {gathered} vs {plan.decision_hash}")
    return plan


def run_policy_on_plan(
    *,
    protocol: ProtocolConfig,
    plan: WorkloadPlan,
    global_plan: GlobalDispatchPlan,
    execution_cache: ExecutionCache,
    repetition_index: int,
    warmup: bool,
    local_rank: int,
) -> dict[str, Any]:
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    strategy = global_plan.policy_name
    release_rounds = [list(item) for item in global_plan.release_rounds]
    lookup = _bucket_lookup(plan)

    microbatch_start_wall = time.perf_counter()
    round_records = []
    sent_rows_total = 0
    recv_rows_total = 0
    total_pack_ms = 0.0
    total_dispatch_ms = 0.0
    total_compute_ms = 0.0
    total_return_ms = 0.0
    total_control_metadata_ms = 0.0
    total_sync_ms = 0.0
    received_tokens = 0
    sent_tokens = 0
    total_dispatch_rows_matrix = _matrix_zeros(world_size)
    total_dispatch_bytes_matrix = _matrix_zeros(world_size)
    total_return_rows_matrix = _matrix_zeros(world_size)
    total_return_bytes_matrix = _matrix_zeros(world_size)
    local_dispatched_bucket_ids: set[str] = set()
    dispatch_manifest: list[dict[str, Any]] = []
    receive_manifest: list[dict[str, Any]] = []
    return_manifest: list[dict[str, Any]] = []
    origin_verified_manifest: list[dict[str, Any]] = []
    used_python_metadata_gather = False
    _rank_phase_log(protocol.artifact_dir, rank, "preallocate", -1)

    for round_index, round_ids in enumerate(release_rounds):
        round_buckets = [lookup[bucket_id] for bucket_id in round_ids]
        local_origin_buckets = [bucket for bucket in round_buckets if bucket.origin_rank == rank]
        local_destination_buckets = [bucket for bucket in round_buckets if bucket.destination_rank == rank]
        round_layout = build_canonical_round_layout(
            round_buckets,
            world_size,
            correctness_mode=execution_cache.correctness_mode,
            round_index=round_index,
        )
        planned_rows_matrix, planned_bytes_matrix = _planned_round_matrix(
            round_buckets,
            world_size,
            plan.hidden_dim,
            correctness_mode=execution_cache.correctness_mode,
        )
        pack_start = torch.cuda.Event(enable_timing=True)
        pack_end = torch.cuda.Event(enable_timing=True)
        _rank_phase_log(protocol.artifact_dir, rank, "dispatch_payload", round_index)
        pack_start.record()
        payload, metadata_tensor, send_rows, metadata, packing_metadata = _pack_round_payload_and_metadata(
            rank,
            round_layout,
            execution_cache,
            lookup,
            plan.hidden_dim,
            6,
            device,
        )
        pack_end.record()
        torch.cuda.synchronize(device)
        pack_time_ms = float(pack_start.elapsed_time(pack_end))
        total_pack_ms += pack_time_ms
        sent_rows_total += int(send_rows.sum().item())
        sent_tokens += sum(bucket.token_count for bucket in local_origin_buckets)
        for bucket in local_origin_buckets:
            if bucket.bucket_id in local_dispatched_bucket_ids:
                raise RuntimeError(f"bucket dispatched more than once on origin rank {rank}: {bucket.bucket_id}")
            local_dispatched_bucket_ids.add(bucket.bucket_id)

        dispatch_start = torch.cuda.Event(enable_timing=True)
        dispatch_end = torch.cuda.Event(enable_timing=True)
        dispatch_start.record()
        recv_payload, recv_rows, planned_send_splits, actual_recv_splits = _all_to_all_variable(payload, send_rows, plan.hidden_dim)
        _rank_phase_log(
            protocol.artifact_dir,
            rank,
            "dispatch_metadata",
            round_index,
            rows=int(metadata_tensor.shape[0]),
            send_splits=planned_send_splits,
            recv_splits=actual_recv_splits,
        )
        recv_metadata_tensor, recv_metadata_rows, metadata_send_splits, metadata_recv_splits = _all_to_all_metadata_variable(
            metadata_tensor,
            send_rows,
            6,
        )
        dispatch_end.record()
        torch.cuda.synchronize(device)
        dispatch_time_ms = float(dispatch_start.elapsed_time(dispatch_end))
        if [int(value) for value in recv_metadata_rows.tolist()] != [int(value) for value in recv_rows.tolist()]:
            raise RuntimeError("metadata recv rows do not match payload recv rows")
        if metadata_send_splits != planned_send_splits or metadata_recv_splits != actual_recv_splits:
            raise RuntimeError("metadata splits do not match payload splits")
        recv_rows_total += int(recv_rows.sum().item())
        received_tokens += sum(bucket.token_count for bucket in local_destination_buckets)
        control_metadata_start = time.perf_counter()
        planned_received_metadata = _planned_received_metadata_for_rank(round_layout, lookup, rank)
        received_metadata = _metadata_tensor_to_dicts(recv_metadata_tensor, planned_received_metadata)
        total_control_metadata_ms += (time.perf_counter() - control_metadata_start) * 1000.0
        if sum(int(item["rows"]) for item in received_metadata) != int(recv_rows.sum().item()):
            raise RuntimeError("received metadata rows do not match received payload rows")
        if execution_cache.correctness_mode:
            for item in metadata:
                for row_index, token_id in enumerate(item["token_ids"][: max(1, len(item["token_ids"]))]):
                    dispatch_manifest.append(
                        {
                            "round": round_index,
                            "route_item_id": item["route_id"],
                            "route_item_index": int(item["route_item_indices"][min(row_index, len(item["route_item_indices"]) - 1)]),
                            "token_id": int(token_id),
                            "origin_rank": int(item["origin_rank"]),
                            "destination_rank": int(item["destination_rank"]),
                            "expert_id": int(item["expert_id"]),
                            "route_rank": int(item["route_ranks"][min(row_index, len(item["route_ranks"]) - 1)]),
                            "dispatch_row_index": row_index,
                            "microbatch_id": int(plan.microbatch_id),
                        }
                    )
            for item in received_metadata:
                for row_index, token_id in enumerate(item["token_ids"][: max(1, len(item["token_ids"]))]):
                    receive_manifest.append(
                        {
                            "round": round_index,
                            "route_item_id": item["route_id"],
                            "route_item_index": int(item["route_item_indices"][min(row_index, len(item["route_item_indices"]) - 1)]),
                            "token_id": int(token_id),
                            "origin_rank": int(item["origin_rank"]),
                            "destination_rank": int(item["destination_rank"]),
                            "expert_id": int(item["expert_id"]),
                            "route_rank": int(item["route_ranks"][min(row_index, len(item["route_ranks"]) - 1)]),
                            "receive_row_index": row_index,
                            "microbatch_id": int(plan.microbatch_id),
                        }
                    )

        gathered_send_splits = _all_gather_object(planned_send_splits)
        actual_dispatch_rows_matrix = [[int(value) for value in row] for row in gathered_send_splits]
        actual_dispatch_bytes_matrix = [[int(value) * plan.hidden_dim * 2 for value in row] for row in actual_dispatch_rows_matrix]
        if actual_dispatch_rows_matrix != planned_rows_matrix:
            raise RuntimeError(
                f"planned dispatch split matrix does not match actual split matrix for round {round_index}: "
                f"{planned_rows_matrix} vs {actual_dispatch_rows_matrix}"
            )

        _rank_phase_log(protocol.artifact_dir, rank, "destination_compute", round_index, rows=int(recv_payload.shape[0]))
        compute_result, local_compute_time_ms, expert_records = _run_expert_grouped_compute(
            recv_payload,
            received_metadata,
            execution_cache,
        )

        return_start = torch.cuda.Event(enable_timing=True)
        return_end = torch.cuda.Event(enable_timing=True)
        _rank_phase_log(protocol.artifact_dir, rank, "return_payload", round_index)
        return_start.record()
        returned, returned_metadata_tensor, return_rows, planned_return_splits, actual_return_recv_splits, packed_return_metadata_tensor = _return_to_origin(
            rank, world_size, compute_result, recv_metadata_tensor, received_metadata, plan.hidden_dim
        )
        return_end.record()
        torch.cuda.synchronize(device)
        return_time_ms = float(return_start.elapsed_time(return_end))
        _rank_phase_log(
            protocol.artifact_dir,
            rank,
            "return_metadata",
            round_index,
            rows=int(returned_metadata_tensor.shape[0]),
            send_splits=planned_return_splits,
            recv_splits=actual_return_recv_splits,
        )
        if int(returned.shape[0]) != int(returned_metadata_tensor.shape[0]):
            raise RuntimeError("returned payload and metadata row count mismatch")

        gathered_return_splits = _all_gather_object(planned_return_splits)
        actual_return_rows_matrix = [[int(value) for value in row] for row in gathered_return_splits]
        expected_return_rows_matrix = _transpose_matrix(planned_rows_matrix)
        if actual_return_rows_matrix != expected_return_rows_matrix:
            raise RuntimeError(
                f"planned return split matrix does not match actual split matrix for round {round_index}: "
                f"{expected_return_rows_matrix} vs {actual_return_rows_matrix}"
            )

        expected_origin_rows = sum(
            (bucket.payload_rows if not execution_cache.correctness_mode else max(1, bucket.token_count))
            for bucket in local_origin_buckets
        )
        if int(returned.shape[0]) != expected_origin_rows:
            raise RuntimeError("returned rows do not match origin-owned payload rows")
        if execution_cache.correctness_mode:
            packed_return_rows = decode_metadata_tensor_rows(packed_return_metadata_tensor)
            for row_index, row in enumerate(packed_return_rows):
                return_manifest.append(
                    {
                        "round": round_index,
                        "route_item_id": str(row.global_route_item_index),
                        "route_item_index": int(row.global_route_item_index),
                        "token_id": int(row.token_id),
                        "origin_rank": int(row.origin_rank),
                        "destination_rank": int(row.destination_rank),
                        "expert_id": int(row.expert_id),
                        "route_rank": int(row.route_rank),
                        "return_row_index": row_index,
                        "microbatch_id": int(plan.microbatch_id),
                    }
                )
            verified_rows = decode_metadata_tensor_rows(returned_metadata_tensor)
            _rank_phase_log(protocol.artifact_dir, rank, "manifest_verify", round_index, rows=len(verified_rows))
            for row_index, row in enumerate(verified_rows):
                if int(row.origin_rank) != rank:
                    raise RuntimeError("returned metadata row did not return to its original origin rank")
                origin_verified_manifest.append(
                    {
                        "round": round_index,
                        "route_item_id": str(row.global_route_item_index),
                        "route_item_index": int(row.global_route_item_index),
                        "token_id": int(row.token_id),
                        "origin_rank": int(row.origin_rank),
                        "destination_rank": int(row.destination_rank),
                        "expert_id": int(row.expert_id),
                        "route_rank": int(row.route_rank),
                        "verified_row_index": row_index,
                        "microbatch_id": int(plan.microbatch_id),
                    }
                )
        sync_wall_start = time.perf_counter()
        _rank_phase_log(protocol.artifact_dir, rank, "cleanup", round_index)
        dist.barrier()
        total_sync_ms += (time.perf_counter() - sync_wall_start) * 1000.0

        bucket_to_rows = {item["bucket_id"]: int(item["rows"]) for item in packing_metadata}
        bucket_to_bytes = {item["bucket_id"]: int(item["bytes"]) for item in packing_metadata}
        per_bucket = []
        seen = set()
        for bucket_id in round_ids:
            if bucket_id in seen:
                raise RuntimeError(f"duplicate bucket in release round: {bucket_id}")
            seen.add(bucket_id)
            bucket = lookup[bucket_id]
            per_bucket.append(
                {
                    "bucket_id": bucket_id,
                    "route_id": bucket.route_id,
                    "origin_rank": int(bucket.origin_rank),
                    "destination_rank": int(bucket.destination_rank),
                    "destination_id": int(bucket.destination_id),
                    "expert_id": int(bucket.expert_id),
                    "token_count": int(bucket.token_count),
                    "token_ids": list(bucket.token_ids),
                    "payload_rows": int(bucket.payload_rows),
                    "payload_bytes": int(bucket.payload_bytes),
                    "estimated_service_units": float(bucket.estimated_service_units),
                    "dependency_score": float(dependency_score_for_bucket(bucket)),
                    "local_packed_rows": int(bucket_to_rows.get(bucket_id, 0)),
                    "local_packed_bytes": int(bucket_to_bytes.get(bucket_id, 0)),
                }
            )

        rank_projection = []
        for rank_index in range(world_size):
            projected_rows = sum(bucket.payload_rows for bucket in round_buckets if bucket.destination_rank == rank_index)
            projected_tokens = sum(bucket.token_count for bucket in round_buckets if bucket.destination_rank == rank_index)
            projected_service = sum(bucket.estimated_service_units for bucket in round_buckets if bucket.destination_rank == rank_index)
            bucket_sizes = [bucket.payload_rows for bucket in round_buckets if bucket.destination_rank == rank_index]
            rank_projection.append(
                {
                    "rank": rank_index,
                    "projected_send_rows": int(planned_rows_matrix[rank][rank_index]),
                    "projected_recv_rows": int(sum(planned_rows_matrix[source][rank_index] for source in range(world_size))),
                    "projected_send_bytes": int(planned_bytes_matrix[rank][rank_index]),
                    "projected_recv_bytes": int(sum(planned_bytes_matrix[source][rank_index] for source in range(world_size))),
                    "projected_compute_units": float(projected_rows),
                    "projected_service_estimate": float(projected_service),
                    "bucket_count": len(bucket_sizes),
                    "empty": len(bucket_sizes) == 0,
                    "rank_max_bucket_rows": max(bucket_sizes) if bucket_sizes else 0,
                    "rank_total_bucket_rows": sum(bucket_sizes),
                    "rank_total_tokens": int(projected_tokens),
                }
            )
        nonzero = [item["projected_recv_bytes"] for item in rank_projection if item["projected_recv_bytes"] > 0]
        max_recv = max((item["projected_recv_bytes"] for item in rank_projection), default=0)
        bottleneck_ranks = [item["rank"] for item in rank_projection if item["projected_recv_bytes"] == max_recv]

        total_dispatch_ms += dispatch_time_ms
        total_compute_ms += local_compute_time_ms
        total_return_ms += return_time_ms
        total_dispatch_rows_matrix = _sum_matrix(total_dispatch_rows_matrix, actual_dispatch_rows_matrix)
        total_dispatch_bytes_matrix = _sum_matrix(total_dispatch_bytes_matrix, actual_dispatch_bytes_matrix)
        total_return_rows_matrix = _sum_matrix(total_return_rows_matrix, actual_return_rows_matrix)
        total_return_bytes_matrix = _sum_matrix(
            total_return_bytes_matrix,
            [[int(value) * plan.hidden_dim * 2 for value in row] for row in actual_return_rows_matrix],
        )
        round_records.append(
            {
                "release_round": round_index,
                "bucket_ids": list(round_ids),
                "bucket_count": len(round_ids),
                "round_bucket_details": per_bucket,
                "planned_send_splits_rows": [int(value) for value in send_rows.tolist()],
                "actual_send_splits_rows": list(planned_send_splits),
                "metadata_send_splits_rows": list(metadata_send_splits),
                "metadata_recv_splits_rows": list(metadata_recv_splits),
                "planned_return_splits_rows": [int(value) for value in planned_return_splits],
                "actual_return_splits_rows": [int(value) for value in actual_return_recv_splits],
                "planned_dispatch_rows_matrix": planned_rows_matrix,
                "planned_dispatch_bytes_matrix": planned_bytes_matrix,
                "actual_dispatch_rows_matrix": actual_dispatch_rows_matrix,
                "actual_dispatch_bytes_matrix": actual_dispatch_bytes_matrix,
                "planned_return_rows_matrix": expected_return_rows_matrix,
                "actual_return_rows_matrix": actual_return_rows_matrix,
                "rank_projection": rank_projection,
                "pack_time_ms": pack_time_ms,
                "dispatch_bytes": sum(sum(row) for row in actual_dispatch_bytes_matrix),
                "return_bytes": sum(sum(row) for row in actual_return_rows_matrix) * plan.hidden_dim * 2,
                "dispatch_time_ms": dispatch_time_ms,
                "local_compute_time_ms": local_compute_time_ms,
                "return_time_ms": return_time_ms,
                "control_metadata_ms": total_control_metadata_ms,
                "sync_time_ms": total_sync_ms,
                "total_service_time_ms": dispatch_time_ms + local_compute_time_ms + return_time_ms,
                "received_rows": int(recv_rows.sum().item()),
                "returned_rows": int(return_rows.sum().item()),
                "bottleneck_ranks": bottleneck_ranks,
                "rank_completion_spread_proxy_ms": float(max(dispatch_time_ms + local_compute_time_ms + return_time_ms, 0.0)),
                "empty_rank_count": sum(1 for item in rank_projection if item["empty"]),
                "collective_participation_mask": [0 if item["empty"] else 1 for item in rank_projection],
                "max_rank_recv_bytes": int(max_recv),
                "min_nonzero_rank_recv_bytes": int(min(nonzero)) if nonzero else 0,
                "local_origin_bucket_count": len(local_origin_buckets),
                "local_destination_bucket_count": len(local_destination_buckets),
                "remote_dispatch_bytes": _off_diagonal_sum(actual_dispatch_bytes_matrix),
                "remote_return_bytes": _off_diagonal_sum(actual_return_rows_matrix) * plan.hidden_dim * 2,
                "expert_group_records": expert_records,
            }
        )

    local_completion_ms = (time.perf_counter() - microbatch_start_wall) * 1000.0
    gathered_rank_records = _all_gather_object(
        {
            "rank": rank,
            "local_completion_ms": local_completion_ms,
            "dispatch_total_ms": total_dispatch_ms,
            "compute_total_ms": total_compute_ms,
            "return_total_ms": total_return_ms,
            "control_metadata_total_ms": total_control_metadata_ms,
            "sync_total_ms": total_sync_ms,
        }
    )
    global_completion_ms = max(float(item["local_completion_ms"]) for item in gathered_rank_records)
    _rank_phase_log(protocol.artifact_dir, rank, "manifest_gather", -1, rows=len(origin_verified_manifest))
    barrier_wait_ms = max(0.0, global_completion_ms - (total_pack_ms + total_dispatch_ms + total_compute_ms + total_return_ms + total_sync_ms))
    total_bytes = sum(sum(row) for row in total_dispatch_bytes_matrix)
    remote_bytes = _off_diagonal_sum(total_dispatch_bytes_matrix)
    remote_byte_ratio = (remote_bytes / total_bytes) if total_bytes > 0 else 0.0
    communication_semantics_valid = True
    if world_size > 1 and protocol.require_remote_traffic:
        communication_semantics_valid = remote_byte_ratio > 0.0 and _nonzero_off_diagonal_entries(total_dispatch_bytes_matrix) > 0
        if not communication_semantics_valid:
            raise RuntimeError("communication semantics invalid: dispatch matrix is purely diagonal or remote_byte_ratio is zero")
    if world_size == 4 and protocol.require_remote_traffic:
        if _nonzero_off_diagonal_entries(total_dispatch_bytes_matrix) < 6:
            raise RuntimeError("4-GPU correctness requirement failed: fewer than 6 off-diagonal dispatch entries are nonzero")
        for rank_index in range(world_size):
            remote_send = sum(total_dispatch_bytes_matrix[rank_index][j] for j in range(world_size) if j != rank_index)
            remote_recv = sum(total_dispatch_bytes_matrix[i][rank_index] for i in range(world_size) if i != rank_index)
            if remote_send == 0 and remote_recv == 0:
                raise RuntimeError(f"rank {rank_index} has neither remote send nor remote receive")

    rank_record = {
        "rank": rank,
        "rank_pack_time_ms": total_pack_ms,
        "rank_busy_time_ms": total_dispatch_ms + total_compute_ms + total_return_ms,
        "rank_idle_time_ms": 0.0,
        "rank_completion_time_ms": local_completion_ms,
        "received_tokens": received_tokens,
        "sent_tokens": sent_tokens,
        "received_bytes": recv_rows_total * plan.hidden_dim * 2,
        "sent_bytes": sent_rows_total * plan.hidden_dim * 2,
        "rounds_participated": len(round_records),
    }
    gathered_dispatch_manifest = _all_gather_object(dispatch_manifest)
    gathered_receive_manifest = _all_gather_object(receive_manifest)
    gathered_return_manifest = _all_gather_object(return_manifest)
    gathered_origin_verified_manifest = _all_gather_object(origin_verified_manifest)
    return {
        "plan_id": plan.plan_id,
        "strategy": strategy,
        "warmup": warmup,
        "repetition_index": repetition_index,
        "microbatch_id": plan.microbatch_id,
        "global_dispatch_plan": global_plan.to_dict(),
        "origin_sharding": plan.origin_sharding,
        "decision_hash": global_plan.decision_hash,
        "used_python_metadata_gather": used_python_metadata_gather,
        "rounds": round_records,
        "rank_record": rank_record,
        "dispatch_rows_matrix": total_dispatch_rows_matrix,
        "dispatch_bytes_matrix": total_dispatch_bytes_matrix,
        "return_rows_matrix": total_return_rows_matrix,
        "return_bytes_matrix": total_return_bytes_matrix,
        "remote_byte_ratio": remote_byte_ratio,
        "communication_semantics_valid": communication_semantics_valid,
        "global_end_to_end_completion_ms": global_completion_ms,
        "local_end_to_end_completion_ms": local_completion_ms,
        "pack_total_ms": total_pack_ms,
        "dispatch_total_ms": total_dispatch_ms,
        "compute_total_ms": total_compute_ms,
        "return_total_ms": total_return_ms,
        "control_metadata_ms": total_control_metadata_ms,
        "sync_total_ms": total_sync_ms,
        "metric_semantics_version": "poc2.5-final",
        "dispatch_route_manifest": [item for items in gathered_dispatch_manifest for item in items],
        "receive_route_manifest": [item for items in gathered_receive_manifest for item in items],
        "return_route_manifest": [item for items in gathered_return_manifest for item in items],
        "origin_verified_manifest": [item for items in gathered_origin_verified_manifest for item in items],
        "barrier_wait_ms_deprecated": barrier_wait_ms,
        "total_communication_bytes": rank_record["received_bytes"] + rank_record["sent_bytes"],
        "total_gpu_compute_time_ms": total_compute_ms,
        "throughput_tokens_per_s": (sent_tokens / (global_completion_ms / 1000.0)) if global_completion_ms > 0 else 0.0,
        "policy_feature_provenance": _policy_feature_contract(strategy),
    }


def _update_runtime_state(runtime_state: RuntimeState, decision: Any, plan: WorkloadPlan, batch_completion_ms: float) -> None:
    lookup = _bucket_lookup(plan)
    pending: dict[int, float] = {}
    rank_pending: dict[int, float] = {}
    queue_depth: dict[int, int] = {}
    rank_queue: dict[int, int] = {}
    for bucket_id in decision.release_order:
        bucket = lookup[bucket_id]
        pending[bucket.destination_id] = pending.get(bucket.destination_id, 0.0) + float(bucket.estimated_service_units)
        rank_pending[bucket.destination_rank] = rank_pending.get(bucket.destination_rank, 0.0) + float(bucket.estimated_service_units)
        queue_depth[bucket.destination_id] = queue_depth.get(bucket.destination_id, 0) + 1
        rank_queue[bucket.destination_rank] = rank_queue.get(bucket.destination_rank, 0) + 1
        runtime_state.destination_hotness[bucket.destination_id] = min(
            10.0,
            float(runtime_state.destination_hotness.get(bucket.destination_id, 0.0)) * 0.5 + bucket.token_count,
        )
        runtime_state.destination_latency_history.setdefault(bucket.destination_id, []).append(batch_completion_ms / 1000.0)
        runtime_state.destination_latency_history[bucket.destination_id] = runtime_state.destination_latency_history[bucket.destination_id][-6:]
    runtime_state.destination_pending_work = pending
    runtime_state.destination_queue_depth = queue_depth
    runtime_state.rank_backlog_work = rank_pending
    runtime_state.rank_queue_depth = rank_queue
    runtime_state.batch_index += 1


def update_global_runtime_state(
    global_state: GlobalRuntimeState,
    global_plan: GlobalDispatchPlan,
    plan: WorkloadPlan,
    gathered_rank_records: list[dict[str, Any]],
) -> GlobalRuntimeState:
    lookup = _bucket_lookup(plan)
    next_state = GlobalRuntimeState.from_dict(global_state.to_dict())
    pending_dest: dict[int, float] = {}
    rank_pending: dict[int, float] = {}
    queue_depth: dict[int, int] = {}
    rank_queue: dict[int, int] = {}
    expert_pending: dict[int, int] = {}
    rank_compute_queue_rows: dict[int, int] = {}
    for bucket_id in global_plan.release_order:
        bucket = lookup[bucket_id]
        expert_pending[bucket.expert_id] = expert_pending.get(bucket.expert_id, 0) + int(bucket.payload_rows)
        rank_compute_queue_rows[bucket.destination_rank] = rank_compute_queue_rows.get(bucket.destination_rank, 0) + int(bucket.payload_rows)
        next_state.destination_hotness[bucket.destination_id] = min(
            10.0,
            float(next_state.destination_hotness.get(bucket.destination_id, 0.0)) * 0.5 + bucket.token_count,
        )
    next_state.destination_pending_work = pending_dest
    next_state.destination_queue_depth = queue_depth
    next_state.rank_backlog_work = rank_pending
    next_state.rank_queue_depth = rank_queue
    next_state.source_outbound_pending_bytes = {}
    next_state.destination_inbound_pending_bytes = {}
    next_state.flow_pending_bytes = {}
    next_state.return_flow_pending_bytes = {}
    next_state.expert_pending_rows = expert_pending
    next_state.rank_compute_queue_rows = rank_compute_queue_rows
    next_state.global_completion_time_ms = max(float(item["local_completion_ms"]) for item in gathered_rank_records)
    next_state.rank_dispatch_time_ms = {int(item["rank"]): float(item["dispatch_total_ms"]) for item in gathered_rank_records}
    next_state.rank_compute_time_ms = {int(item["rank"]): float(item["compute_total_ms"]) for item in gathered_rank_records}
    next_state.rank_return_time_ms = {int(item["rank"]): float(item["return_total_ms"]) for item in gathered_rank_records}
    next_state.rank_local_completion_time_ms = {int(item["rank"]): float(item["local_completion_ms"]) for item in gathered_rank_records}
    for bucket in lookup.values():
        next_state.destination_latency_history.setdefault(bucket.destination_id, []).append(next_state.global_completion_time_ms / 1000.0)
        next_state.destination_latency_history[bucket.destination_id] = next_state.destination_latency_history[bucket.destination_id][-6:]
    next_state.batch_index += 1
    return next_state


def aggregate_repetition_records(records: list[dict[str, Any]], state_meaningful: bool) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        if record["warmup"]:
            continue
        grouped.setdefault((str(record["strategy"]), int(record["repetition_index"])), []).append(record)

    per_repetition_aggregate_records = []
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for (strategy, repetition_index), items in sorted(grouped.items()):
        aggregate = {
            "strategy": strategy,
            "repetition_index": repetition_index,
            "plan_id": items[0].get("plan_id"),
            "microbatch_count": len(items),
            "end_to_end_batch_completion_ms": float(statistics.fmean(float(item["global_end_to_end_completion_ms"]) for item in items)),
            "dispatch_total_ms": float(statistics.fmean(float(item["dispatch_total_ms"]) for item in items)),
            "compute_total_ms": float(statistics.fmean(float(item["compute_total_ms"]) for item in items)),
            "return_total_ms": float(statistics.fmean(float(item["return_total_ms"]) for item in items)),
            "sync_total_ms": float(statistics.fmean(float(item["sync_total_ms"]) for item in items)),
            "remote_byte_ratio": float(statistics.fmean(float(item["remote_byte_ratio"]) for item in items)),
            "throughput_tokens_per_s": float(statistics.fmean(float(item["throughput_tokens_per_s"]) for item in items)),
            "strategy_order": list(items[0].get("strategy_order", [])),
        }
        per_repetition_aggregate_records.append(aggregate)
        by_strategy.setdefault(strategy, []).append(aggregate)

    summary_by_strategy: dict[str, dict[str, Any]] = {}
    for strategy, items in by_strategy.items():
        completion = [item["end_to_end_batch_completion_ms"] for item in items]
        dispatch = [item["dispatch_total_ms"] for item in items]
        compute = [item["compute_total_ms"] for item in items]
        returned = [item["return_total_ms"] for item in items]
        summary_by_strategy[strategy] = {
            "sample_count": len(items),
            "end_to_end_batch_completion_ms": _stats(completion),
            "dispatch_total_ms": _stats(dispatch),
            "compute_total_ms": _stats(compute),
            "return_total_ms": _stats(returned),
            "sync_total_ms": _stats([item["sync_total_ms"] for item in items]),
            "remote_byte_ratio": _stats([item["remote_byte_ratio"] for item in items]),
            "barrier_wait_ms_deprecated": _stats([item.get("barrier_wait_ms_deprecated", 0.0) for item in items]),
            "throughput_tokens_per_s": _stats([item["throughput_tokens_per_s"] for item in items]),
        }

    paired = {
        "dependency_only_vs_state_only": _paired(by_strategy.get("dependency-only", []), by_strategy.get("state-only", [])),
        "dependency_only_vs_strong_state": _paired(by_strategy.get("dependency-only", []), by_strategy.get("strong-state", [])),
        "dependency_only_vs_lina_inspired": _paired(by_strategy.get("dependency-only", []), by_strategy.get("lina-inspired", [])),
        "dependency_only_vs_shuffled_dependency": _paired(by_strategy.get("dependency-only", []), by_strategy.get("shuffled-dependency", [])),
        "full_vs_state_only": _paired(by_strategy.get("full", []), by_strategy.get("state-only", [])),
        "full_vs_strong_state": _paired(by_strategy.get("full", []), by_strategy.get("strong-state", [])),
        "full_vs_lina_inspired": _paired(by_strategy.get("full", []), by_strategy.get("lina-inspired", [])),
        "full_vs_full_with_shuffled_dependency": _paired(by_strategy.get("full", []), by_strategy.get("full-with-shuffled-dependency", [])),
        "full_vs_fifo": _paired(by_strategy.get("full", []), by_strategy.get("fifo", [])),
        "full_vs_random_order": _paired(by_strategy.get("full", []), by_strategy.get("random-order", [])),
    }
    position_effects = _position_effects(records)
    return {
        "runtime_state_comparison_meaningful": state_meaningful,
        "per_repetition_aggregate_records": per_repetition_aggregate_records,
        "strategies": summary_by_strategy,
        "paired_deltas": paired,
        "position_effects": position_effects,
        "conclusion_status": _conclusion_status(paired, min(len(values) for values in by_strategy.values()) if by_strategy else 0),
        "go_no_go": _go_no_go(paired, state_meaningful),
    }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p50": float(ordered[int(0.5 * (len(ordered) - 1))]),
        "p95": float(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]),
    }


def _paired(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_map = {(item["repetition_index"], item.get("plan_id")): item for item in left}
    right_map = {(item["repetition_index"], item.get("plan_id")): item for item in right}
    paired_keys = sorted(left_map.keys() & right_map.keys())
    paired_count = len(paired_keys)
    if paired_count == 0:
        return {
            "paired_count": 0,
            "win_rate": 0.0,
            "ci95": [0.0, 0.0],
            "bootstrap_ci95": [0.0, 0.0],
            "mean_delta_ms": 0.0,
            "median_delta_ms": 0.0,
            "trimmed_mean_delta_ms": 0.0,
            "leave_one_out_mean_delta_ms": [],
            "per_repetition_delta_ms": [],
            "distribution_asymmetric": False,
            "position_adjusted_mean_delta_ms": 0.0,
        }
    left_position_means = _strategy_position_means(left_map.values())
    right_position_means = _strategy_position_means(right_map.values())
    deltas = []
    adjusted_deltas = []
    per_repetition = []
    for key in paired_keys:
        left_item = left_map[key]
        right_item = right_map[key]
        delta = float(left_item["end_to_end_batch_completion_ms"]) - float(right_item["end_to_end_batch_completion_ms"])
        deltas.append(delta)
        left_position = _strategy_position(left_item)
        right_position = _strategy_position(right_item)
        left_adjusted = float(left_item["end_to_end_batch_completion_ms"]) - left_position_means.get(left_position, 0.0)
        right_adjusted = float(right_item["end_to_end_batch_completion_ms"]) - right_position_means.get(right_position, 0.0)
        adjusted_deltas.append(left_adjusted - right_adjusted)
        per_repetition.append(
            {
                "repetition_index": int(key[0]),
                "plan_id": key[1],
                "delta_ms": delta,
                "left_position": left_position,
                "right_position": right_position,
            }
        )
    mean = float(statistics.fmean(deltas))
    std = float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0
    margin = 1.96 * (std / math.sqrt(max(1, len(deltas))))
    wins = sum(1 for delta in deltas if delta < 0.0)
    bootstrap_ci = _bootstrap_ci(deltas)
    median = float(statistics.median(deltas))
    trimmed = _trimmed_mean(deltas)
    leave_one_out = []
    for index in range(len(deltas)):
        remaining = deltas[:index] + deltas[index + 1 :]
        leave_one_out.append(float(statistics.fmean(remaining)) if remaining else mean)
    distribution_asymmetric = mean < 0.0 and (wins / paired_count) < 0.5
    return {
        "paired_count": paired_count,
        "mean_delta_ms": mean,
        "win_rate": wins / paired_count,
        "ci95": [mean - margin, mean + margin],
        "bootstrap_ci95": bootstrap_ci,
        "median_delta_ms": median,
        "trimmed_mean_delta_ms": trimmed,
        "leave_one_out_mean_delta_ms": leave_one_out,
        "per_repetition_delta_ms": deltas,
        "per_repetition_pairs": per_repetition,
        "distribution_asymmetric": distribution_asymmetric,
        "position_adjusted_mean_delta_ms": float(statistics.fmean(adjusted_deltas)) if adjusted_deltas else mean,
    }


def _trimmed_mean(values: list[float], trim_ratio: float = 0.1) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    trim = int(len(ordered) * trim_ratio)
    trimmed = ordered[trim : len(ordered) - trim] if len(ordered) > 2 * trim else ordered
    return float(statistics.fmean(trimmed))


def _bootstrap_ci(values: list[float], samples: int = 400, seed: int = 42) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(float(statistics.fmean(draw)))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return [lo, hi]


def _position_effects(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    positions: dict[str, dict[int, list[float]]] = {}
    for record in records:
        if record["warmup"]:
            continue
        strategy = str(record["strategy"])
        order = list(record.get("strategy_order", []))
        if strategy not in order:
            continue
        position = order.index(strategy) + 1
        positions.setdefault(strategy, {}).setdefault(position, []).append(float(record["global_end_to_end_completion_ms"]))
    result: dict[str, dict[str, float]] = {}
    for strategy, per_position in positions.items():
        result[strategy] = {
            str(position): float(statistics.fmean(values)) for position, values in sorted(per_position.items())
        }
        ordered_positions = sorted(per_position.items())
        xs = [float(position) for position, _ in ordered_positions]
        ys = [float(statistics.fmean(values)) for _, values in ordered_positions]
        result[strategy]["position_correlation"] = _corr(xs, ys)
    return result


def _conclusion_status(paired: dict[str, Any], repetition_count: int) -> str:
    if repetition_count < 5:
        return "exploratory"
    primary = paired.get("full_vs_strong_state", {})
    ci = primary.get("bootstrap_ci95", [0.0, 0.0])
    win_rate = float(primary.get("win_rate", 0.0))
    if ci[0] < 0.0 and ci[1] < 0.0 and win_rate >= 0.6:
        return "stable_early_signal"
    if ci[0] <= 0.0 <= ci[1]:
        return "exploratory"
    return "no_separable_signal"


def _go_no_go(paired: dict[str, Any], state_meaningful: bool) -> dict[str, Any]:
    if not state_meaningful:
        return {
            "decision": "no_go",
            "label": "no_separable_signal",
            "reason": "runtime_state_comparison_meaningful=false",
        }
    dep_shuffle = paired.get("dependency_only_vs_shuffled_dependency", {})
    full_shuffle = paired.get("full_vs_full_with_shuffled_dependency", {})
    full_strong = paired.get("full_vs_strong_state", {})
    full_lina = paired.get("full_vs_lina_inspired", {})
    if _stable_negative(dep_shuffle) and (_stable_negative(full_strong) or _stable_negative(full_lina)):
        return {
            "decision": "go",
            "label": "stable_early_signal",
            "reason": "dependency beats shuffled control and full beats strong observable baselines",
        }
    if _null_like(dep_shuffle) and _null_like(full_shuffle) and _null_like(full_strong) and _null_like(full_lina):
        return {
            "decision": "no_go",
            "label": "no_separable_signal",
            "reason": "dependency signal is not separable from strong observable state under the current execution model",
        }
    return {
        "decision": "hold",
        "label": "exploratory",
        "reason": "evidence is mixed or confidence intervals still cross zero",
    }


def _stable_negative(item: dict[str, Any]) -> bool:
    ci = item.get("bootstrap_ci95", [0.0, 0.0])
    win_rate = float(item.get("win_rate", 0.0))
    return bool(item.get("paired_count", 0) >= 5 and ci[1] < 0.0 and win_rate >= 0.6)


def _null_like(item: dict[str, Any]) -> bool:
    ci = item.get("bootstrap_ci95", [0.0, 0.0])
    return bool(item.get("paired_count", 0) >= 5 and ci[0] <= 0.0 <= ci[1])


def _strategy_position(record: dict[str, Any]) -> int:
    order = list(record.get("strategy_order", []))
    strategy = str(record.get("strategy"))
    if strategy not in order:
        return 0
    return order.index(strategy) + 1


def _strategy_position_means(records: Any) -> dict[int, float]:
    buckets: dict[int, list[float]] = {}
    for record in records:
        position = _strategy_position(record)
        buckets.setdefault(position, []).append(float(record["end_to_end_batch_completion_ms"]))
    return {position: float(statistics.fmean(values)) for position, values in buckets.items()}


def _corr(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denom_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    denom_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if denom_left == 0.0 or denom_right == 0.0:
        return 0.0
    return numerator / (denom_left * denom_right)


def _build_policy_decision(
    strategy: str,
    scheduler_input: SchedulerInput,
    plan: WorkloadPlan,
    runtime_state: RuntimeState,
    repetition_index: int,
) -> Any:
    buckets = plan.bucket_workloads
    if strategy in {"fifo", "state-only", "dependency-only", "full"}:
        return create_scheduler(strategy).build_decision(scheduler_input)
    if strategy == "strong-state-sort":
        ordered = _order_by_score(
            buckets,
            lambda bucket: strong_state_score_for_bucket(bucket, runtime_state),
        )
        return _decision_from_order(strategy, scheduler_input, ordered)
    if strategy == "strong-state-packer":
        packed = _greedy_minimax_round_pack(
            buckets,
            round_size=scheduler_input.metadata.get("release_round_size", 4),
            runtime_state=runtime_state,
            policy_seed=plan.seed + repetition_index,
            arrival_jitter_mode=str(scheduler_input.metadata.get("arrival_jitter_mode", "none")),
            oracle=False,
        )
        return _decision_from_rounds(strategy, scheduler_input, packed["release_rounds"], packed)
    if strategy == "oracle-minimax-packer":
        packed = _greedy_minimax_round_pack(
            buckets,
            round_size=scheduler_input.metadata.get("release_round_size", 4),
            runtime_state=runtime_state,
            policy_seed=plan.seed + repetition_index,
            arrival_jitter_mode=str(scheduler_input.metadata.get("arrival_jitter_mode", "none")),
            oracle=True,
        )
        return _decision_from_rounds(strategy, scheduler_input, packed["release_rounds"], packed)
    if strategy == "random-order":
        ordered = random_order([bucket.bucket_id for bucket in buckets], plan.seed + repetition_index)
        return _decision_from_order(strategy, scheduler_input, ordered)
    if strategy == "shuffled-dependency":
        ordered = shuffled_dependency_order(buckets, runtime_state, seed=plan.seed + repetition_index, combine_with_state=False)
        return _decision_from_order(strategy, scheduler_input, ordered)
    if strategy == "full-with-shuffled-dependency":
        ordered = shuffled_dependency_order(buckets, runtime_state, seed=plan.seed + repetition_index, combine_with_state=True)
        return _decision_from_order(strategy, scheduler_input, ordered)
    if strategy == "strong-state":
        ordered = _order_by_score(
            buckets,
            lambda bucket: strong_state_score_for_bucket(bucket, runtime_state),
        )
        return _decision_from_order(strategy, scheduler_input, ordered)
    if strategy == "lina-inspired":
        ordered = _order_by_score(
            buckets,
            lambda bucket: lina_inspired_score_for_bucket(bucket, runtime_state),
        )
        return _decision_from_order(strategy, scheduler_input, ordered)
    raise ValueError(f"unsupported strategy: {strategy}")


def _order_by_score(buckets: list[WorkloadBucket], score_fn: Any) -> list[str]:
    scored = [(float(score_fn(bucket)), index, bucket.destination_id, bucket.bucket_id) for index, bucket in enumerate(buckets)]
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in scored]


def _decision_from_order(strategy: str, scheduler_input: SchedulerInput, release_order: list[str]) -> Any:
    base = create_scheduler("fifo").build_decision(
        SchedulerInput(
            strategy="fifo",
            microbatch_id=scheduler_input.microbatch_id,
            layer_id=scheduler_input.layer_id,
            layer_path=scheduler_input.layer_path,
            bucket_records=scheduler_input.bucket_records,
            runtime_state=scheduler_input.runtime_state,
            metadata=scheduler_input.metadata,
        )
    )
    lookup = {item.bucket_id: item for item in base.bucket_decisions}
    ordered = [lookup[bucket_id] for bucket_id in release_order]
    rank_service_order: dict[int, list[str]] = {}
    for item in ordered:
        rank_service_order.setdefault(int(item.destination_rank), []).append(item.bucket_id)
    from .scheduler import SchedulerDecision

    return SchedulerDecision(
        strategy=strategy,
        microbatch_id=scheduler_input.microbatch_id,
        release_order=list(release_order),
        service_order=list(release_order),
        rank_service_order=rank_service_order,
        bucket_decisions=ordered,
    )


def _decision_from_rounds(strategy: str, scheduler_input: SchedulerInput, release_rounds: list[list[str]], packed: dict[str, Any]) -> Any:
    release_order = [bucket_id for round_ids in release_rounds for bucket_id in round_ids]
    decision = _decision_from_order(strategy, scheduler_input, release_order)
    setattr(decision, "release_rounds", [list(round_ids) for round_ids in release_rounds])
    setattr(decision, "round_diagnostics", list(packed.get("round_diagnostics", [])))
    setattr(decision, "round_pressure_score", float(packed.get("round_pressure_score", 0.0)))
    return decision


def environment_snapshot(world_size: int) -> dict[str, Any]:
    commit_hash = None
    dirty = None
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit_hash = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain"], text=True).strip())
    except Exception:
        commit_hash = "unknown"
        dirty = None
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(min(torch.cuda.device_count(), world_size))],
        "hostname": socket.gethostname(),
        "nccl_available": dist.is_nccl_available(),
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_hash": commit_hash,
        "git_dirty": dirty,
        "source_package_version": commit_hash,
    }


def save_artifacts(
    artifact_dir: str | Path,
    *,
    summary: dict[str, Any],
    config: dict[str, Any],
    environment: dict[str, Any],
    protocol: dict[str, Any],
    workload_manifest: dict[str, Any],
    per_repetition: list[dict[str, Any]],
) -> None:
    target = Path(artifact_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (target / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (target / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (target / "workload_manifest.json").write_text(json.dumps(workload_manifest, indent=2), encoding="utf-8")
    (target / "aggregate_statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (target / "per_repetition.jsonl").open("w", encoding="utf-8") as handle:
        for item in per_repetition:
            handle.write(json.dumps(item) + "\n")
    (target / "goal.md").write_text(
        "# POC2 NCCL Dispatch Benchmark\n\n"
        "This benchmark uses real router traces and a controlled expert-like GPU compute harness.\n"
        "Dispatch and return use real GPU-to-GPU NCCL collectives; expert placement is synthetic and controlled.\n"
        "Therefore the result is useful for studying routing-derived scheduling value, but it is not yet a full end-to-end MoE runtime.\n"
        "This round explicitly tests whether dependency-derived information is separable from stronger observable state baselines and shuffled controls.\n",
        encoding="utf-8",
    )


def save_plan_snapshot(artifact_dir: str | Path, plans: list[WorkloadPlan]) -> Path:
    target = Path(artifact_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "plan_snapshot.json"
    path.write_text(json.dumps({"plans": [plan.to_dict() for plan in plans]}, indent=2), encoding="utf-8")
    return path


def load_plan_snapshot(path: str | Path) -> list[WorkloadPlan]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [WorkloadPlan.from_dict(item) for item in payload["plans"]]


def create_workload_plans(protocol: ProtocolConfig) -> tuple[list[WorkloadPlan], dict[str, Any]]:
    if protocol.plan_snapshot_path:
        plans = load_plan_snapshot(protocol.plan_snapshot_path)
        if protocol.audit_plan_id:
            plans = [plan for plan in plans if plan.plan_id == protocol.audit_plan_id]
        manifest = {
            "model_key": plans[0].model_key if plans else protocol.model_key,
            "placement_policy": plans[0].placement_policy if plans else protocol.placement_policy,
            "origin_sharding": plans[0].origin_sharding if plans else protocol.origin_sharding,
            "scenario": protocol.scenario,
            "regime": protocol.regime,
            "world_size": protocol.world_size,
            "hidden_dim": plans[0].hidden_dim if plans else 0,
            "intermediate_dim": plans[0].intermediate_dim if plans else 0,
            "plan_ids": [plan.plan_id for plan in plans],
            "plan_conflict_scores": {plan.plan_id: plan_conflict_score(plan) for plan in plans},
            "plan_snapshot_path": str(protocol.plan_snapshot_path),
        }
        return plans, manifest

    payloads: list[Any] = [None]
    distributed = dist.is_available() and dist.is_initialized()
    current_rank = dist.get_rank() if distributed else 0
    if current_rank == 0:
        random.seed(protocol.seed)
        runner_config = RunnerConfig(
            model_key=protocol.model_key,
            model_path=protocol.model_path,
            output_dir=protocol.output_dir,
            prompts_path=protocol.prompt_path,
            layer=protocol.layer,
            microbatch_size=protocol.microbatch_size,
            microbatch_count=protocol.microbatch_count,
            max_length=protocol.max_length,
            precision=protocol.precision,
            world_size=protocol.world_size,
            top_k=protocol.top_k,
            seed=protocol.seed,
            service_backend="synthetic-matmul",
        )
        model_bundle = _load_model_bundle(runner_config)
        model = model_bundle["model"]
        hidden_dim = max(64, int(getattr(model.config, "hidden_size", 2048) * protocol.hidden_dim_scale))
        intermediate_dim = max(hidden_dim, int(hidden_dim * protocol.intermediate_scale))
        prompts = _load_prompts(protocol.prompt_path, protocol.microbatch_size * protocol.microbatch_count)
        microbatches = [
            prompts[index : index + protocol.microbatch_size]
            for index in range(0, protocol.microbatch_size * protocol.microbatch_count, protocol.microbatch_size)
        ]

        expert_count = int(getattr(model.config, "num_experts", 64))
        placement = build_placement_map(expert_count, protocol.world_size, protocol.placement_policy)
        routed_batches = []
        plans: list[WorkloadPlan] = []
        for microbatch_id, batch_prompts in enumerate(microbatches):
            routing = _collect_microbatch_routing(runner_config, batch_prompts, microbatch_id, model_bundle)
            routing["bucket_records"] = apply_placement(routing["bucket_records"], placement)
            routed_batches.append((microbatch_id, routing))
        global_route_tuples = []
        for microbatch_id, routing in routed_batches:
            for route in list(routing.get("route_items", [])):
                global_route_tuples.append(
                    (
                        int(routing["layer_id"]),
                        int(microbatch_id),
                        int(route["token_pos"]),
                        int(route["token_id"]),
                        int(route["route_rank"]),
                        int(route["expert_id"]),
                        str(route["route_id"]),
                    )
                )
        global_route_tuples = sorted(global_route_tuples)
        global_route_item_index_map = {str(route_tuple): index for index, route_tuple in enumerate(global_route_tuples)}
        for microbatch_id, routing in routed_batches:
            routing["route_item_index_map"] = global_route_item_index_map
            plans.append(
                build_workload_plan(
                    config=protocol,
                    microbatch_id=microbatch_id,
                    routing=routing,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    placement=placement,
                )
            )
        plans = select_plans_for_scenario(plans, protocol.scenario)
        if protocol.audit_plan_id:
            plans = [plan for plan in plans if plan.plan_id == protocol.audit_plan_id]
        manifest = {
            "model_key": protocol.model_key,
            "placement_policy": protocol.placement_policy,
            "origin_sharding": protocol.origin_sharding,
            "scenario": protocol.scenario,
            "regime": protocol.regime,
            "world_size": protocol.world_size,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "plan_ids": [plan.plan_id for plan in plans],
            "plan_conflict_scores": {plan.plan_id: plan_conflict_score(plan) for plan in plans},
        }
        payloads[0] = {
            "plans": [plan.to_dict() for plan in plans],
            "manifest": manifest,
        }

    if distributed:
        dist.broadcast_object_list(payloads, src=0)
    bundle = payloads[0]
    assert bundle is not None
    plans = [
        WorkloadPlan(
            plan_id=item["plan_id"],
            model_key=item["model_key"],
            seed=int(item["seed"]),
            microbatch_id=int(item["microbatch_id"]),
            layer_id=int(item["layer_id"]),
            layer_path=str(item["layer_path"]),
            hidden_dim=int(item["hidden_dim"]),
            intermediate_dim=int(item["intermediate_dim"]),
            placement_policy=str(item["placement_policy"]),
            origin_sharding=str(item.get("origin_sharding", protocol.origin_sharding)),
            placement_mapping=[PlacementEntry(**entry) for entry in item["placement_mapping"]],
            bucket_workloads=[WorkloadBucket(**bucket) for bucket in item["bucket_workloads"]],
            routing_summary=item["routing_summary"],
        )
        for item in bundle["plans"]
    ]
    return plans, bundle["manifest"]
