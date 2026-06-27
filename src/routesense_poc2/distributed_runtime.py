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
STRATEGIES = [
    "fifo",
    "random-order",
    "state-only",
    "strong-state",
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

    def to_scheduler_record(self) -> BucketRecord:
        flow_key = f"{self.origin_rank}->{self.destination_rank}"
        return_key = f"{self.destination_rank}->{self.origin_rank}"
        estimated_dispatch_cost = self.payload_bytes / float(1 << 20)
        estimated_compute_cost = float(self.payload_rows)
        estimated_return_cost = self.payload_bytes / float(1 << 20)
        return BucketRecord(
            bucket_id=self.bucket_id,
            route_item_ids=[self.route_id],
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
    rank_compute_queue_rows: dict[int, int] = field(default_factory=dict)
    expert_pending_rows: dict[int, int] = field(default_factory=dict)
    rank_dispatch_time_ms: dict[int, float] = field(default_factory=dict)
    rank_compute_time_ms: dict[int, float] = field(default_factory=dict)
    rank_return_time_ms: dict[int, float] = field(default_factory=dict)
    rank_local_completion_time_ms: dict[int, float] = field(default_factory=dict)
    global_completion_time_ms: float = 0.0
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalDispatchPlan":
        return cls(**payload)


@dataclass
class ExecutionCache:
    payload_by_bucket_id: dict[str, torch.Tensor]
    metadata_by_bucket_id: dict[str, torch.Tensor]
    expert_weight1: dict[int, torch.Tensor]
    expert_weight2: dict[int, torch.Tensor]
    correctness_mode: bool


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
    normalized = str(mode).strip().lower()
    if world_size <= 1:
        return 0
    if normalized == "round-robin":
        return int(token_id) % world_size
    if normalized == "contiguous":
        if token_count <= 0:
            return 0
        shard_size = max(1, math.ceil(token_count / world_size))
        return min(world_size - 1, int(token_id) // shard_size)
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
    grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
    bucket_workloads: list[WorkloadBucket] = []
    bucket_record_map = {bucket.destination_id: bucket for bucket in routing["bucket_records"]}
    for route in route_items:
        expert_id = int(route["expert_id"])
        token_id = int(route["token_id"])
        destination_rank = int(placement_map.get(expert_id, expert_id % config.world_size))
        origin_rank = assign_origin_rank(token_id, token_count, config.world_size, config.origin_sharding)
        key = (origin_rank, destination_rank, expert_id)
        token_pos = int(route["token_pos"])
        current = grouped.setdefault(
            key,
            {
                "token_ids": [],
                "token_positions": [],
                "route_ids": [],
                "route_ranks": [],
            },
        )
        current["token_ids"].append(token_id)
        current["token_positions"].append(token_pos)
        current["route_ids"].append(str(route["route_id"]))
        current["route_ranks"].append(int(route["route_rank"]))

    ordered_groups = sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0], item[0][2], min(item[1]["token_positions"])))
    for index, ((origin_rank, destination_rank, expert_id), item) in enumerate(ordered_groups):
        base_bucket = bucket_record_map[expert_id]
        token_ids_in_bucket = sorted(item["token_ids"])
        token_positions = sorted(item["token_positions"])
        token_count_in_bucket = len(token_ids_in_bucket)
        payload_rows = max(1, int(round(token_count_in_bucket * max(1.0, config.compute_scale))))
        payload_bytes = payload_rows * hidden_dim * 2
        route_suffix = "_".join(item["route_ids"][:2])
        bucket_workloads.append(
            WorkloadBucket(
                bucket_id=f"mb{microbatch_id}_o{origin_rank}_d{destination_rank}_e{expert_id}_{index}",
                route_id=route_suffix,
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
            )
        )

    token_origin_summary: dict[int, list[int]] = {}
    for token_id in token_ids:
        rank = assign_origin_rank(int(token_id), token_count, config.world_size, config.origin_sharding)
        token_origin_summary.setdefault(rank, []).append(int(token_id))

    routing_summary = dict(routing["routing_summary"])
    routing_summary["origin_sharding"] = config.origin_sharding
    routing_summary["token_to_origin_rank"] = {
        str(token_id): assign_origin_rank(int(token_id), token_count, config.world_size, config.origin_sharding) for token_id in token_ids
    }
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


def resolve_strategy_orders(mode: str, repetitions: int, seed: int) -> list[list[str]]:
    if mode == "fixed":
        return [list(STRATEGIES) for _ in range(repetitions)]
    if mode == "randomized":
        return randomized_orders(STRATEGIES, repetitions, seed)
    if mode == "latin-square":
        return latin_square_orders(STRATEGIES, repetitions)
    if mode == "counterbalanced":
        return counterbalanced_orders(STRATEGIES, repetitions)
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
    send_counts = torch.zeros(world_size, dtype=torch.int64, device=device)
    metadata: list[dict[str, Any]] = []
    packing_metadata: list[dict[str, Any]] = []
    tensors = []
    for bucket in round_buckets:
        if bucket.origin_rank != rank:
            continue
        rows = bucket.payload_rows
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + bucket.origin_rank * 1009 + bucket.destination_rank * 97 + sum(bucket.token_ids) * 13)
        tensor = torch.randn((rows, hidden_dim), generator=generator, dtype=torch.float32).pin_memory()
        tensors.append((bucket.destination_rank, tensor, bucket))
        send_counts[bucket.destination_rank] += rows
        metadata.append(
            {
                "bucket_id": bucket.bucket_id,
                "route_id": bucket.route_id,
                "origin_rank": bucket.origin_rank,
                "destination_rank": bucket.destination_rank,
                "rows": rows,
                "destination_id": bucket.destination_id,
                "expert_id": bucket.expert_id,
                "estimated_service_units": bucket.estimated_service_units,
                "token_count": bucket.token_count,
                "token_ids": list(bucket.token_ids),
                "token_positions": list(bucket.token_positions),
                "payload_bytes": bucket.payload_bytes,
            }
        )
        packing_metadata.append(
            {
                "bucket_id": bucket.bucket_id,
                "route_id": bucket.route_id,
                "origin_rank": bucket.origin_rank,
                "destination_rank": bucket.destination_rank,
                "rows": rows,
                "bytes": rows * hidden_dim * 2,
                "token_count": bucket.token_count,
                "token_ids": list(bucket.token_ids),
            }
        )

    total_rows = int(sum(item[2].payload_rows for item in tensors))
    if total_rows == 0:
        payload = torch.zeros((0, hidden_dim), device=device, dtype=torch.float16)
    else:
        payload = torch.empty((total_rows, hidden_dim), device=device, dtype=torch.float16)
        cursor = {target_rank: 0 for target_rank in range(world_size)}
        ordered = []
        for target_rank in range(world_size):
            ordered.extend([item for item in tensors if item[0] == target_rank])
        row_offset = 0
        for _, cpu_tensor, bucket in ordered:
            rows = bucket.payload_rows
            payload[row_offset : row_offset + rows].copy_(cpu_tensor.to(device=device, dtype=torch.float16, non_blocking=False))
            row_offset += rows
    return payload, send_counts, metadata, packing_metadata


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
    origin_metadata: list[dict[str, Any]],
    hidden_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    return_counts = torch.zeros(world_size, dtype=torch.int64, device=result_payload.device)
    for item in origin_metadata:
        return_counts[int(item["origin_rank"])] += int(item["rows"])
    send_splits = return_counts.tolist()
    recv_return_rows = _exchange_sizes(return_counts)
    recv_splits = recv_return_rows.tolist()
    recv_total = int(sum(recv_splits))
    recv_back = torch.empty((recv_total, hidden_dim), device=result_payload.device, dtype=result_payload.dtype)
    dist.all_to_all_single(
        recv_back.reshape(-1),
        result_payload.reshape(-1),
        output_split_sizes=[int(value) * hidden_dim for value in recv_splits],
        input_split_sizes=[int(value) * hidden_dim for value in send_splits],
    )
    return recv_back, recv_return_rows, [int(v) for v in send_splits], [int(v) for v in recv_splits]


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
        metadata[:, 0] = torch.tensor([int(bucket.token_ids[min(index, len(bucket.token_ids) - 1)]) for index in range(rows)], device=device, dtype=torch.int32)
        metadata[:, 1] = torch.arange(rows, device=device, dtype=torch.int32)
        metadata[:, 2] = int(bucket.origin_rank)
        metadata[:, 3] = int(bucket.destination_rank)
        metadata[:, 4] = int(bucket.expert_id)
        metadata[:, 5] = int(sum(ord(ch) for ch in bucket.route_id) % (1 << 30))
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
    round_buckets: list[WorkloadBucket],
    destination_rank: int,
    *,
    correctness_mode: bool,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for bucket in round_buckets:
        if bucket.destination_rank != destination_rank:
            continue
        rows = bucket.payload_rows if not correctness_mode else max(1, bucket.token_count)
        items.append(
            {
                "bucket_id": bucket.bucket_id,
                "route_id": bucket.route_id,
                "origin_rank": bucket.origin_rank,
                "destination_rank": bucket.destination_rank,
                "rows": rows,
                "destination_id": bucket.destination_id,
                "expert_id": bucket.expert_id,
                "estimated_service_units": bucket.estimated_service_units,
                "token_count": bucket.token_count,
                "token_ids": list(bucket.token_ids),
                "token_positions": list(bucket.token_positions),
                "payload_bytes": rows * bucket.hidden_dim * 2,
            }
        )
    return items


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
    bucket_records = []
    for bucket in plan.bucket_workloads:
        record = bucket.to_scheduler_record()
        flow_key = f"{bucket.origin_rank}->{bucket.destination_rank}"
        return_key = f"{bucket.destination_rank}->{bucket.origin_rank}"
        record.source_outbound_pending_bytes = float(global_state.source_outbound_pending_bytes.get(bucket.origin_rank, 0.0))
        record.destination_inbound_pending_bytes = float(global_state.destination_inbound_pending_bytes.get(bucket.destination_rank, 0.0))
        record.flow_pending_bytes = float(global_state.flow_pending_bytes.get(flow_key, 0.0))
        record.return_flow_pending_bytes = float(global_state.return_flow_pending_bytes.get(return_key, 0.0))
        record.rank_compute_queue_rows = float(global_state.rank_compute_queue_rows.get(bucket.destination_rank, 0.0))
        record.expert_pending_rows = float(global_state.expert_pending_rows.get(bucket.expert_id, 0.0))
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
        metadata={"plan_id": plan.plan_id},
    )
    decision = _build_policy_decision(strategy, scheduler_input, plan, scheduler_state, repetition_index)
    release_rounds = _calc_release_rounds(decision.release_order, protocol.release_round_size)
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
    plan_payload = {
        "workload_plan_id": plan.plan_id,
        "policy_name": strategy,
        "policy_seed": plan.seed + repetition_index,
        "microbatch_id": plan.microbatch_id,
        "release_order": list(decision.release_order),
        "release_rounds": [list(item) for item in release_rounds],
        "total_dispatch_matrix": total_dispatch_matrix,
        "per_round_dispatch_matrices": per_round_dispatch_matrices,
        "scheduler_state_snapshot": global_state.to_dict(),
        "scheduler_decision": decision.to_dict(),
    }
    decision_hash = _hash_payload(plan_payload)
    return GlobalDispatchPlan(
        plan_id=f"{plan.plan_id}_{strategy}_rep{repetition_index}",
        workload_plan_id=plan.plan_id,
        policy_name=strategy,
        policy_seed=plan.seed + repetition_index,
        microbatch_id=plan.microbatch_id,
        release_order=list(decision.release_order),
        release_rounds=[list(item) for item in release_rounds],
        total_dispatch_matrix=total_dispatch_matrix,
        per_round_dispatch_matrices=per_round_dispatch_matrices,
        decision_hash=decision_hash,
        scheduler_state_snapshot=global_state.to_dict(),
        scheduler_decision=decision.to_dict(),
    )


def broadcast_global_dispatch_plan(global_plan: GlobalDispatchPlan | None) -> GlobalDispatchPlan:
    payloads: list[Any] = [global_plan.to_dict() if global_plan is not None else None]
    dist.broadcast_object_list(payloads, src=0)
    if payloads[0] is None:
        raise RuntimeError("global dispatch plan broadcast failed")
    plan = GlobalDispatchPlan.from_dict(payloads[0])
    local_hash = _hash_payload(
        {
            "workload_plan_id": plan.workload_plan_id,
            "policy_name": plan.policy_name,
            "policy_seed": plan.policy_seed,
            "microbatch_id": plan.microbatch_id,
            "release_order": plan.release_order,
            "release_rounds": plan.release_rounds,
            "total_dispatch_matrix": plan.total_dispatch_matrix,
            "per_round_dispatch_matrices": plan.per_round_dispatch_matrices,
            "scheduler_state_snapshot": plan.scheduler_state_snapshot,
            "scheduler_decision": plan.scheduler_decision,
        }
    )
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

    for round_index, round_ids in enumerate(release_rounds):
        round_buckets = [lookup[bucket_id] for bucket_id in round_ids]
        local_origin_buckets = [bucket for bucket in round_buckets if bucket.origin_rank == rank]
        local_destination_buckets = [bucket for bucket in round_buckets if bucket.destination_rank == rank]
        planned_rows_matrix, planned_bytes_matrix = _planned_round_matrix(
            round_buckets,
            world_size,
            plan.hidden_dim,
            correctness_mode=execution_cache.correctness_mode,
        )
        pack_start = torch.cuda.Event(enable_timing=True)
        pack_end = torch.cuda.Event(enable_timing=True)
        pack_start.record()
        send_rows = torch.zeros(world_size, dtype=torch.int64, device=device)
        metadata: list[dict[str, Any]] = []
        packing_metadata: list[dict[str, Any]] = []
        ordered_tensors: list[torch.Tensor] = []
        for bucket in local_origin_buckets:
            rows = bucket.payload_rows if not execution_cache.correctness_mode else max(1, bucket.token_count)
            tensor = execution_cache.payload_by_bucket_id[bucket.bucket_id]
            ordered_tensors.append(tensor)
            send_rows[bucket.destination_rank] += rows
            metadata.append(
                {
                    "bucket_id": bucket.bucket_id,
                    "route_id": bucket.route_id,
                    "origin_rank": bucket.origin_rank,
                    "destination_rank": bucket.destination_rank,
                    "rows": rows,
                    "destination_id": bucket.destination_id,
                    "expert_id": bucket.expert_id,
                    "estimated_service_units": bucket.estimated_service_units,
                    "token_count": bucket.token_count,
                    "token_ids": list(bucket.token_ids),
                    "token_positions": list(bucket.token_positions),
                    "payload_bytes": bucket.payload_bytes,
                }
            )
            packing_metadata.append(
                {
                    "bucket_id": bucket.bucket_id,
                    "route_id": bucket.route_id,
                    "origin_rank": bucket.origin_rank,
                    "destination_rank": bucket.destination_rank,
                    "rows": rows,
                    "bytes": rows * plan.hidden_dim * 2,
                    "token_count": bucket.token_count,
                    "token_ids": list(bucket.token_ids),
                }
            )
        if ordered_tensors:
            payload = torch.cat(
                [tensor for target_rank in range(world_size) for tensor, bucket in [
                    (execution_cache.payload_by_bucket_id[b.bucket_id], b) for b in local_origin_buckets if b.destination_rank == target_rank
                ]],
                dim=0,
            )
        else:
            payload = torch.zeros((0, plan.hidden_dim), device=device, dtype=torch.float16)
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
        dispatch_end.record()
        torch.cuda.synchronize(device)
        dispatch_time_ms = float(dispatch_start.elapsed_time(dispatch_end))
        recv_rows_total += int(recv_rows.sum().item())
        received_tokens += sum(bucket.token_count for bucket in local_destination_buckets)
        control_metadata_start = time.perf_counter()
        if execution_cache.correctness_mode:
            used_python_metadata_gather = True
            all_source_metadata = _all_gather_object(metadata)
            received_metadata = _received_metadata_for_rank(all_source_metadata, rank)
        else:
            received_metadata = _planned_received_metadata_for_rank(
                round_buckets,
                rank,
                correctness_mode=execution_cache.correctness_mode,
            )
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
                            "token_id": int(token_id),
                            "origin_rank": int(item["origin_rank"]),
                            "destination_rank": int(item["destination_rank"]),
                            "expert_id": int(item["expert_id"]),
                            "dispatch_row_index": row_index,
                        }
                    )
            for item in received_metadata:
                for row_index, token_id in enumerate(item["token_ids"][: max(1, len(item["token_ids"]))]):
                    receive_manifest.append(
                        {
                            "round": round_index,
                            "route_item_id": item["route_id"],
                            "token_id": int(token_id),
                            "origin_rank": int(item["origin_rank"]),
                            "destination_rank": int(item["destination_rank"]),
                            "expert_id": int(item["expert_id"]),
                            "receive_row_index": row_index,
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

        compute_result, local_compute_time_ms, expert_records = _run_expert_grouped_compute(
            recv_payload,
            received_metadata,
            execution_cache,
        )

        return_start = torch.cuda.Event(enable_timing=True)
        return_end = torch.cuda.Event(enable_timing=True)
        return_start.record()
        returned, return_rows, planned_return_splits, actual_return_recv_splits = _return_to_origin(
            rank, world_size, compute_result, received_metadata, plan.hidden_dim
        )
        return_end.record()
        torch.cuda.synchronize(device)
        return_time_ms = float(return_start.elapsed_time(return_end))

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
            for item in received_metadata:
                for row_index, token_id in enumerate(item["token_ids"][: max(1, len(item["token_ids"]))]):
                    return_manifest.append(
                        {
                            "round": round_index,
                            "route_item_id": item["route_id"],
                            "token_id": int(token_id),
                            "origin_rank": int(item["origin_rank"]),
                            "destination_rank": int(item["destination_rank"]),
                            "expert_id": int(item["expert_id"]),
                            "return_row_index": row_index,
                        }
                    )
            for bucket in local_origin_buckets:
                for row_index, token_id in enumerate(bucket.token_ids[: max(1, len(bucket.token_ids))]):
                    origin_verified_manifest.append(
                        {
                            "round": round_index,
                            "route_item_id": bucket.route_id,
                            "token_id": int(token_id),
                            "origin_rank": int(bucket.origin_rank),
                            "destination_rank": int(bucket.destination_rank),
                            "expert_id": int(bucket.expert_id),
                            "verified_row_index": row_index,
                        }
                    )
        sync_wall_start = time.perf_counter()
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
        "rank_idle_time_ms": barrier_wait_ms,
        "rank_completion_time_ms": local_completion_ms,
        "received_tokens": received_tokens,
        "sent_tokens": sent_tokens,
        "received_bytes": recv_rows_total * plan.hidden_dim * 2,
        "sent_bytes": sent_rows_total * plan.hidden_dim * 2,
        "rounds_participated": len(round_records),
    }
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
        "dispatch_route_manifest": dispatch_manifest,
        "receive_route_manifest": receive_manifest,
        "return_route_manifest": return_manifest,
        "origin_verified_manifest": origin_verified_manifest,
        "barrier_wait_ms": barrier_wait_ms,
        "total_communication_bytes": rank_record["received_bytes"] + rank_record["sent_bytes"],
        "total_gpu_compute_time_ms": total_compute_ms,
        "throughput_tokens_per_s": (sent_tokens / (global_completion_ms / 1000.0)) if global_completion_ms > 0 else 0.0,
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
    source_outbound: dict[int, int] = {}
    destination_inbound: dict[int, int] = {}
    flow_pending: dict[str, int] = {}
    expert_pending: dict[int, int] = {}
    rank_compute_queue_rows: dict[int, int] = {}
    for bucket_id in global_plan.release_order:
        bucket = lookup[bucket_id]
        pending_dest[bucket.destination_id] = pending_dest.get(bucket.destination_id, 0.0) + float(bucket.estimated_service_units)
        rank_pending[bucket.destination_rank] = rank_pending.get(bucket.destination_rank, 0.0) + float(bucket.estimated_service_units)
        queue_depth[bucket.destination_id] = queue_depth.get(bucket.destination_id, 0) + 1
        rank_queue[bucket.destination_rank] = rank_queue.get(bucket.destination_rank, 0) + 1
        source_outbound[bucket.origin_rank] = source_outbound.get(bucket.origin_rank, 0) + int(bucket.payload_bytes)
        destination_inbound[bucket.destination_rank] = destination_inbound.get(bucket.destination_rank, 0) + int(bucket.payload_bytes)
        flow_pending[_make_flow_key(bucket.origin_rank, bucket.destination_rank)] = (
            flow_pending.get(_make_flow_key(bucket.origin_rank, bucket.destination_rank), 0) + int(bucket.payload_bytes)
        )
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
    next_state.source_outbound_pending_bytes = source_outbound
    next_state.destination_inbound_pending_bytes = destination_inbound
    next_state.flow_pending_bytes = flow_pending
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
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["warmup"]:
            continue
        by_strategy.setdefault(record["strategy"], []).append(record)

    summary_by_strategy: dict[str, dict[str, Any]] = {}
    for strategy, items in by_strategy.items():
        completion = [item["end_to_end_batch_completion_ms"] for item in items]
        dispatch = [item["dispatch_total_ms"] for item in items]
        compute = [item["compute_total_ms"] for item in items]
        returned = [item["return_total_ms"] for item in items]
        barrier = [item["barrier_wait_ms"] for item in items]
        summary_by_strategy[strategy] = {
            "sample_count": len(items),
            "end_to_end_batch_completion_ms": _stats(completion),
            "dispatch_total_ms": _stats(dispatch),
            "compute_total_ms": _stats(compute),
            "return_total_ms": _stats(returned),
            "barrier_wait_ms": _stats(barrier),
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
    left_map = {(item["repetition_index"], item["microbatch_id"], item.get("plan_id")): item for item in left}
    right_map = {(item["repetition_index"], item["microbatch_id"], item.get("plan_id")): item for item in right}
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
                "microbatch_id": int(key[1]),
                "plan_id": key[2],
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
        positions.setdefault(strategy, {}).setdefault(position, []).append(float(record["end_to_end_batch_completion_ms"]))
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


def environment_snapshot(world_size: int) -> dict[str, Any]:
    commit_hash = None
    dirty = None
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
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
    if dist.get_rank() == 0:
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
        plans: list[WorkloadPlan] = []
        for microbatch_id, batch_prompts in enumerate(microbatches):
            routing = _collect_microbatch_routing(runner_config, batch_prompts, microbatch_id, model_bundle)
            routing["bucket_records"] = apply_placement(routing["bucket_records"], placement)
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
