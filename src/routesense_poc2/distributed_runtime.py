from __future__ import annotations

import json
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


@dataclass
class PlacementEntry:
    expert_id: int
    destination_rank: int


@dataclass
class WorkloadBucket:
    bucket_id: str
    destination_id: int
    destination_rank: int
    token_count: int
    payload_rows: int
    hidden_dim: int
    intermediate_dim: int
    estimated_service_units: float
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
        return BucketRecord(
            bucket_id=self.bucket_id,
            destination_id=self.destination_id,
            destination_rank=self.destination_rank,
            arrival_index=0,
            token_count=self.token_count,
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
            placement_mapping=[PlacementEntry(**entry) for entry in payload["placement_mapping"]],
            bucket_workloads=[WorkloadBucket(**bucket) for bucket in payload["bucket_workloads"]],
            routing_summary=dict(payload["routing_summary"]),
        )


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
    rank_pressure = min(1.0, pending / max(1.0, bucket.estimated_service_units * 4.0))
    queue_norm = min(1.0, queue / 8.0)
    latency_norm = min(1.0, latency / 0.2) if latency > 0 else 0.0
    projected_bytes = min(1.0, (bucket.payload_rows * bucket.hidden_dim * 2) / float(1 << 20) / 8.0)
    return (
        0.22 * bucket.size_norm
        + 0.18 * bucket.route_share
        + 0.18 * projected_bytes
        + 0.16 * bucket.density_over_mean
        + 0.10 * bucket.inverse_size_rank_norm
        - 0.10 * rank_pressure
        - 0.04 * queue_norm
        - 0.02 * latency_norm
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
    bucket_workloads: list[WorkloadBucket] = []
    for index, bucket in enumerate(routing["bucket_records"]):
        destination_rank = int(placement_map.get(bucket.destination_id, bucket.destination_id % config.world_size))
        payload_rows = max(1, int(round(bucket.token_count * max(1.0, config.compute_scale))))
        bucket_workloads.append(
            WorkloadBucket(
                bucket_id=bucket.bucket_id,
                destination_id=bucket.destination_id,
                destination_rank=destination_rank,
                token_count=bucket.token_count,
                payload_rows=payload_rows,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                estimated_service_units=bucket.estimated_service_units,
                source_count=bucket.source_count,
                source_coverage=bucket.source_coverage,
                coactive_peer_degree=bucket.coactive_peer_degree,
                coactive_event_density=bucket.coactive_event_density,
                position_spread=bucket.position_spread,
                bridge_score=bucket.bridge_score,
                route_share=bucket.route_share,
                density_over_mean=bucket.density_over_mean,
                size_norm=bucket.size_norm,
                inverse_size_rank_norm=1.0 - (index / max(1, len(routing["bucket_records"]) - 1))
                if routing["bucket_records"]
                else 0.0,
                is_hot_bucket=bucket.is_hot_bucket,
            )
        )

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
        placement_mapping=placement,
        bucket_workloads=bucket_workloads,
        routing_summary=routing["routing_summary"],
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


def resolve_strategy_orders(mode: str, repetitions: int, seed: int) -> list[list[str]]:
    if mode == "fixed":
        return [list(STRATEGIES) for _ in range(repetitions)]
    if mode == "randomized":
        return randomized_orders(STRATEGIES, repetitions, seed)
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
        rows = bucket.payload_rows
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + bucket.destination_id * 97 + bucket.token_count * 13)
        tensor = torch.randn((rows, hidden_dim), generator=generator, dtype=torch.float32).pin_memory()
        tensors.append((bucket.destination_rank, tensor, bucket))
        send_counts[bucket.destination_rank] += rows
        metadata.append(
            {
                "bucket_id": bucket.bucket_id,
                "origin_rank": rank,
                "destination_rank": bucket.destination_rank,
                "rows": rows,
                "destination_id": bucket.destination_id,
                "estimated_service_units": bucket.estimated_service_units,
                "token_count": bucket.token_count,
            }
        )
        packing_metadata.append(
            {
                "bucket_id": bucket.bucket_id,
                "destination_rank": bucket.destination_rank,
                "rows": rows,
                "bytes": rows * hidden_dim * 2,
                "token_count": bucket.token_count,
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return recv, recv_rows


def _run_local_compute(
    payload: torch.Tensor,
    hidden_dim: int,
    intermediate_dim: int,
) -> tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if payload.numel() == 0:
        result = payload
    else:
        weight1 = torch.randn((intermediate_dim, hidden_dim), device=payload.device, dtype=torch.float16)
        weight2 = torch.randn((hidden_dim, intermediate_dim), device=payload.device, dtype=torch.float16)
        hidden = torch.nn.functional.linear(payload, weight1)
        hidden = torch.nn.functional.gelu(hidden)
        result = torch.nn.functional.linear(hidden, weight2)
    end.record()
    torch.cuda.synchronize(payload.device)
    return result, float(start.elapsed_time(end))


def _return_to_origin(
    rank: int,
    world_size: int,
    result_payload: torch.Tensor,
    recv_rows: torch.Tensor,
    origin_metadata: list[dict[str, Any]],
    hidden_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return recv_back, recv_return_rows


def _calc_release_rounds(release_order: list[str], round_size: int) -> list[list[str]]:
    return release_rounds_for_order(release_order, round_size)


def _bucket_lookup(plan: WorkloadPlan) -> dict[str, WorkloadBucket]:
    return {bucket.bucket_id: bucket for bucket in plan.bucket_workloads}


def run_policy_on_plan(
    *,
    protocol: ProtocolConfig,
    plan: WorkloadPlan,
    strategy: str,
    runtime_state: RuntimeState,
    repetition_index: int,
    warmup: bool,
    local_rank: int,
) -> dict[str, Any]:
    device = torch.device(f"cuda:{local_rank}")
    bucket_records = [bucket.to_scheduler_record() for bucket in plan.bucket_workloads]
    scheduler_input = SchedulerInput(
        strategy=strategy,
        microbatch_id=plan.microbatch_id,
        layer_id=plan.layer_id,
        layer_path=plan.layer_path,
        bucket_records=bucket_records,
        runtime_state=runtime_state,
        metadata={"plan_id": plan.plan_id},
    )
    decision_start = time.perf_counter()
    decision = _build_policy_decision(strategy, scheduler_input, plan, runtime_state, repetition_index)
    scheduler_decision_time_ms = (time.perf_counter() - decision_start) * 1000.0
    release_rounds = _calc_release_rounds(decision.release_order, protocol.release_round_size)
    lookup = _bucket_lookup(plan)

    microbatch_start = time.perf_counter()
    round_records = []
    sent_rows_total = 0
    recv_rows_total = 0
    total_dispatch_ms = 0.0
    total_compute_ms = 0.0
    total_return_ms = 0.0
    received_tokens = 0
    sent_tokens = 0

    for round_index, round_ids in enumerate(release_rounds):
        round_buckets = [lookup[bucket_id] for bucket_id in round_ids if lookup[bucket_id].destination_rank == dist.get_rank()]
        payload, send_rows, metadata, packing_metadata = _prepare_round_payload(
            dist.get_rank(), dist.get_world_size(), round_buckets, plan.hidden_dim, plan.seed + repetition_index, device
        )
        sent_rows_total += int(send_rows.sum().item())
        sent_tokens += sum(bucket.token_count for bucket in round_buckets)

        dispatch_start = torch.cuda.Event(enable_timing=True)
        dispatch_end = torch.cuda.Event(enable_timing=True)
        dispatch_start.record()
        recv_payload, recv_rows = _all_to_all_variable(payload, send_rows, plan.hidden_dim)
        dispatch_end.record()
        torch.cuda.synchronize(device)
        dispatch_time_ms = float(dispatch_start.elapsed_time(dispatch_end))
        recv_rows_total += int(recv_rows.sum().item())
        received_tokens += int(recv_rows.sum().item())

        compute_result, local_compute_time_ms = _run_local_compute(recv_payload, plan.hidden_dim, plan.intermediate_dim)

        return_start = torch.cuda.Event(enable_timing=True)
        return_end = torch.cuda.Event(enable_timing=True)
        return_start.record()
        returned, return_rows = _return_to_origin(dist.get_rank(), dist.get_world_size(), compute_result, recv_rows, metadata, plan.hidden_dim)
        return_end.record()
        torch.cuda.synchronize(device)
        return_time_ms = float(return_start.elapsed_time(return_end))

        planned_send_splits = [int(value) for value in send_rows.tolist()]
        actual_send_splits = list(planned_send_splits)
        planned_return_splits = [int(item) for item in return_rows.tolist()]
        actual_return_splits = list(planned_return_splits)
        if planned_send_splits != actual_send_splits:
            raise RuntimeError("planned send splits do not match actual send splits")
        if planned_return_splits != actual_return_splits:
            raise RuntimeError("planned return splits do not match actual return splits")

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
                    "destination_rank": int(bucket.destination_rank),
                    "destination_id": int(bucket.destination_id),
                    "token_count": int(bucket.token_count),
                    "payload_rows": int(bucket.payload_rows),
                    "payload_bytes": int(bucket.payload_rows * plan.hidden_dim * 2),
                    "estimated_service_units": float(bucket.estimated_service_units),
                    "dependency_score": float(dependency_score_for_bucket(bucket)),
                    "state_score": float(state_score_for_bucket(bucket, runtime_state)),
                    "strong_state_score": float(strong_state_score_for_bucket(bucket, runtime_state)),
                    "lina_inspired_score": float(lina_inspired_score_for_bucket(bucket, runtime_state)),
                    "local_packed_rows": int(bucket_to_rows.get(bucket_id, 0)),
                    "local_packed_bytes": int(bucket_to_bytes.get(bucket_id, 0)),
                }
            )

        rank_projection = []
        for rank_index in range(dist.get_world_size()):
            projected_rows = sum(bucket.payload_rows for bucket in [lookup[bucket_id] for bucket_id in round_ids] if bucket.destination_rank == rank_index)
            projected_tokens = sum(bucket.token_count for bucket in [lookup[bucket_id] for bucket_id in round_ids] if bucket.destination_rank == rank_index)
            projected_service = sum(bucket.estimated_service_units for bucket in [lookup[bucket_id] for bucket_id in round_ids] if bucket.destination_rank == rank_index)
            bucket_sizes = [bucket.payload_rows for bucket in [lookup[bucket_id] for bucket_id in round_ids] if bucket.destination_rank == rank_index]
            rank_projection.append(
                {
                    "rank": rank_index,
                    "projected_send_rows": int(send_rows[rank_index].item()),
                    "projected_recv_rows": int(recv_rows[rank_index].item()) if rank_index < recv_rows.numel() else 0,
                    "projected_send_bytes": int(send_rows[rank_index].item()) * plan.hidden_dim * 2,
                    "projected_recv_bytes": int(recv_rows[rank_index].item()) * plan.hidden_dim * 2 if rank_index < recv_rows.numel() else 0,
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
        round_records.append(
            {
                "release_round": round_index,
                "bucket_ids": list(round_ids),
                "bucket_count": len(round_ids),
                "round_bucket_details": per_bucket,
                "planned_send_splits_rows": planned_send_splits,
                "actual_send_splits_rows": actual_send_splits,
                "planned_return_splits_rows": planned_return_splits,
                "actual_return_splits_rows": actual_return_splits,
                "rank_projection": rank_projection,
                "dispatch_bytes": int(send_rows.sum().item()) * plan.hidden_dim * 2,
                "return_bytes": int(return_rows.sum().item()) * plan.hidden_dim * 2,
                "dispatch_time_ms": dispatch_time_ms,
                "local_compute_time_ms": local_compute_time_ms,
                "return_time_ms": return_time_ms,
                "total_service_time_ms": dispatch_time_ms + local_compute_time_ms + return_time_ms,
                "received_rows": int(recv_rows.sum().item()),
                "returned_rows": int(return_rows.sum().item()),
                "bottleneck_ranks": bottleneck_ranks,
                "rank_completion_spread_proxy_ms": float(max(dispatch_time_ms + local_compute_time_ms + return_time_ms, 0.0)),
                "empty_rank_count": sum(1 for item in rank_projection if item["empty"]),
                "collective_participation_mask": [0 if item["empty"] else 1 for item in rank_projection],
                "max_rank_recv_bytes": int(max_recv),
                "min_nonzero_rank_recv_bytes": int(min(nonzero)) if nonzero else 0,
            }
        )
        _ = returned

    microbatch_end = time.perf_counter()
    batch_completion_ms = (microbatch_end - microbatch_start) * 1000.0
    barrier_wait_ms = max(0.0, batch_completion_ms - total_dispatch_ms - total_compute_ms - total_return_ms)
    rank_record = {
        "rank": dist.get_rank(),
        "rank_busy_time_ms": total_dispatch_ms + total_compute_ms + total_return_ms,
        "rank_idle_time_ms": barrier_wait_ms,
        "rank_completion_time_ms": batch_completion_ms,
        "received_tokens": received_tokens,
        "sent_tokens": sent_tokens,
        "received_bytes": recv_rows_total * plan.hidden_dim * 2,
        "sent_bytes": sent_rows_total * plan.hidden_dim * 2,
        "rounds_participated": len(round_records),
    }
    _update_runtime_state(runtime_state, decision, plan, batch_completion_ms)
    return {
        "plan_id": plan.plan_id,
        "strategy": strategy,
        "warmup": warmup,
        "repetition_index": repetition_index,
        "microbatch_id": plan.microbatch_id,
        "scheduler_decision": decision.to_dict(),
        "scheduler_decision_time_ms": scheduler_decision_time_ms,
        "rounds": round_records,
        "rank_record": rank_record,
        "end_to_end_batch_completion_ms": batch_completion_ms,
        "dispatch_total_ms": total_dispatch_ms,
        "compute_total_ms": total_compute_ms,
        "return_total_ms": total_return_ms,
        "barrier_wait_ms": barrier_wait_ms,
        "total_communication_bytes": rank_record["received_bytes"] + rank_record["sent_bytes"],
        "total_gpu_compute_time_ms": total_compute_ms,
        "throughput_tokens_per_s": (sent_tokens / (batch_completion_ms / 1000.0)) if batch_completion_ms > 0 else 0.0,
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
            placement_mapping=[PlacementEntry(**entry) for entry in item["placement_mapping"]],
            bucket_workloads=[WorkloadBucket(**bucket) for bucket in item["bucket_workloads"]],
            routing_summary=item["routing_summary"],
        )
        for item in bundle["plans"]
    ]
    return plans, bundle["manifest"]
