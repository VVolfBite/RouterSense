from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BucketRecord:
    bucket_id: str
    destination_id: int
    destination_rank: int
    arrival_index: int
    token_count: int
    route_item_ids: list[str] = field(default_factory=list)
    origin_rank: int = 0
    expert_id: int = 0
    payload_rows: int = 0
    payload_bytes: int = 0
    token_positions: list[int] = field(default_factory=list)
    source_count: int = 0
    active_destinations: list[int] = field(default_factory=list)
    coactive_destinations: list[int] = field(default_factory=list)
    source_coverage: float = 0.0
    coactive_peer_degree: float = 0.0
    coactive_event_density: float = 0.0
    position_spread: float = 0.0
    bridge_score: float = 0.0
    route_share: float = 0.0
    density_over_mean: float = 0.0
    size_norm: float = 0.0
    inverse_size_rank_norm: float = 0.0
    is_hot_bucket: bool = False
    estimated_service_units: float = 0.0
    source_outbound_pending_bytes: float = 0.0
    destination_inbound_pending_bytes: float = 0.0
    flow_pending_bytes: float = 0.0
    return_flow_pending_bytes: float = 0.0
    rank_compute_queue_rows: float = 0.0
    expert_pending_rows: float = 0.0
    estimated_dispatch_cost: float = 0.0
    estimated_compute_cost: float = 0.0
    estimated_return_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeState:
    destination_pending_work: dict[int, float] = field(default_factory=dict)
    destination_queue_depth: dict[int, int] = field(default_factory=dict)
    rank_backlog_work: dict[int, float] = field(default_factory=dict)
    rank_queue_depth: dict[int, int] = field(default_factory=dict)
    destination_hotness: dict[int, float] = field(default_factory=dict)
    destination_latency_history: dict[int, list[float]] = field(default_factory=dict)
    source_outbound_pending_bytes: dict[int, float] = field(default_factory=dict)
    destination_inbound_pending_bytes: dict[int, float] = field(default_factory=dict)
    flow_pending_bytes: dict[str, float] = field(default_factory=dict)
    return_flow_pending_bytes: dict[str, float] = field(default_factory=dict)
    rank_compute_queue_rows: dict[int, float] = field(default_factory=dict)
    expert_pending_rows: dict[int, float] = field(default_factory=dict)
    rank_dispatch_time_ms: dict[int, float] = field(default_factory=dict)
    rank_compute_time_ms: dict[int, float] = field(default_factory=dict)
    rank_return_time_ms: dict[int, float] = field(default_factory=dict)
    batch_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulerInput:
    strategy: str
    microbatch_id: int
    layer_id: int | None
    layer_path: str | None
    bucket_records: list[BucketRecord]
    runtime_state: RuntimeState
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BucketDecision:
    bucket_id: str
    destination_id: int
    destination_rank: int
    arrival_index: int
    state_only_score: float
    dependency_only_score: float
    full_score: float
    selected_score: float
    state_features: dict[str, float]
    dependency_features: dict[str, float]
    runtime_features: dict[str, float]
    expert_features: dict[str, float]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulerDecision:
    strategy: str
    microbatch_id: int
    release_order: list[str]
    service_order: list[str]
    rank_service_order: dict[int, list[str]]
    bucket_decisions: list[BucketDecision]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "microbatch_id": self.microbatch_id,
            "release_order": list(self.release_order),
            "service_order": list(self.service_order),
            "rank_service_order": {str(key): value for key, value in self.rank_service_order.items()},
            "bucket_decisions": [item.to_dict() for item in self.bucket_decisions],
        }


class BaseScheduler:
    strategy = "base"

    def score_bucket(self, bucket: BucketRecord, state: RuntimeState) -> tuple[float, float, float, dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        state_score, state_features = _state_score(bucket, state)
        dependency_score, dependency_features = _dependency_score(bucket)
        runtime_features = _runtime_features(bucket, state)
        expert_features = _expert_features(bucket, state)
        full_score = 0.5 * state_score + 0.5 * dependency_score
        return state_score, dependency_score, full_score, state_features, dependency_features, runtime_features, expert_features

    def select_score(self, state_score: float, dependency_score: float, full_score: float, bucket: BucketRecord) -> float:
        raise NotImplementedError

    def build_decision(self, scheduler_input: SchedulerInput) -> SchedulerDecision:
        decisions: list[BucketDecision] = []
        for bucket in scheduler_input.bucket_records:
            state_score, dependency_score, full_score, state_features, dependency_features, runtime_features, expert_features = self.score_bucket(
                bucket, scheduler_input.runtime_state
            )
            selected_score = self.select_score(state_score, dependency_score, full_score, bucket)
            decisions.append(
                BucketDecision(
                    bucket_id=bucket.bucket_id,
                    destination_id=bucket.destination_id,
                    destination_rank=bucket.destination_rank,
                    arrival_index=bucket.arrival_index,
                    state_only_score=state_score,
                    dependency_only_score=dependency_score,
                    full_score=full_score,
                    selected_score=selected_score,
                    state_features=state_features,
                    dependency_features=dependency_features,
                    runtime_features=runtime_features,
                    expert_features=expert_features,
                    explanation=_build_explanation(
                        self.strategy,
                        bucket,
                        state_score,
                        dependency_score,
                        full_score,
                        runtime_features,
                    ),
                )
            )

        ordered = sorted(
            decisions,
            key=lambda item: (-float(item.selected_score), int(item.arrival_index), int(item.destination_id)),
        )
        release_order = [item.bucket_id for item in ordered]
        rank_service_order: dict[int, list[str]] = {}
        for item in ordered:
            rank_service_order.setdefault(int(item.destination_rank), []).append(item.bucket_id)
        return SchedulerDecision(
            strategy=self.strategy,
            microbatch_id=scheduler_input.microbatch_id,
            release_order=release_order,
            service_order=list(release_order),
            rank_service_order=rank_service_order,
            bucket_decisions=ordered,
        )


class FIFOScheduler(BaseScheduler):
    strategy = "fifo"

    def select_score(self, state_score: float, dependency_score: float, full_score: float, bucket: BucketRecord) -> float:
        return float(-bucket.arrival_index)


class StateOnlyScheduler(BaseScheduler):
    strategy = "state-only"

    def select_score(self, state_score: float, dependency_score: float, full_score: float, bucket: BucketRecord) -> float:
        return float(state_score)


class DependencyOnlyScheduler(BaseScheduler):
    strategy = "dependency-only"

    def select_score(self, state_score: float, dependency_score: float, full_score: float, bucket: BucketRecord) -> float:
        return float(dependency_score)


class FullScheduler(BaseScheduler):
    strategy = "full"

    def select_score(self, state_score: float, dependency_score: float, full_score: float, bucket: BucketRecord) -> float:
        return float(full_score)


def create_scheduler(strategy: str) -> BaseScheduler:
    normalized = str(strategy).strip().lower()
    if normalized == "fifo":
        return FIFOScheduler()
    if normalized == "state-only":
        return StateOnlyScheduler()
    if normalized == "dependency-only":
        return DependencyOnlyScheduler()
    if normalized == "full":
        return FullScheduler()
    raise ValueError(f"unsupported scheduler strategy: {strategy}")


def _state_score(bucket: BucketRecord, state: RuntimeState) -> tuple[float, dict[str, float]]:
    destination_pending = float(state.destination_pending_work.get(bucket.destination_id, 0.0))
    destination_queue = float(state.destination_queue_depth.get(bucket.destination_id, 0))
    rank_backlog = float(state.rank_backlog_work.get(bucket.destination_rank, 0.0))
    rank_queue = float(state.rank_queue_depth.get(bucket.destination_rank, 0))
    hotness = float(state.destination_hotness.get(bucket.destination_id, 0.0))
    latency_history = state.destination_latency_history.get(bucket.destination_id, [])
    mean_latency = sum(latency_history) / len(latency_history) if latency_history else 0.0

    pending_norm = min(1.0, destination_pending / max(1.0, bucket.estimated_service_units * 4.0))
    queue_norm = min(1.0, destination_queue / 8.0)
    rank_backlog_norm = min(1.0, rank_backlog / max(1.0, bucket.estimated_service_units * 6.0))
    rank_queue_norm = min(1.0, rank_queue / 8.0)
    hotness_norm = min(1.0, hotness / 6.0)
    latency_norm = min(1.0, mean_latency / 0.2) if mean_latency > 0 else 0.0

    score = (
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
    return score, {
        "bucket_size_norm": bucket.size_norm,
        "inverse_size_rank_norm": bucket.inverse_size_rank_norm,
        "selected_route_share": bucket.route_share,
        "density_over_mean_norm": bucket.density_over_mean,
        "is_hot_bucket": 1.0 if bucket.is_hot_bucket else 0.0,
        "destination_pending_work_norm": pending_norm,
        "destination_queue_depth_norm": queue_norm,
        "rank_backlog_work_norm": rank_backlog_norm,
        "rank_queue_depth_norm": rank_queue_norm,
        "destination_hotness_norm": hotness_norm,
        "destination_mean_latency_norm": latency_norm,
    }


def _dependency_score(bucket: BucketRecord) -> tuple[float, dict[str, float]]:
    score = (
        0.25 * bucket.source_coverage
        + 0.30 * bucket.coactive_peer_degree
        + 0.15 * bucket.coactive_event_density
        + 0.15 * bucket.position_spread
        + 0.15 * bucket.bridge_score
    )
    return score, {
        "source_coverage": bucket.source_coverage,
        "coactive_peer_degree": bucket.coactive_peer_degree,
        "coactive_event_density": bucket.coactive_event_density,
        "position_spread": bucket.position_spread,
        "bridge_score": bucket.bridge_score,
    }


def _runtime_features(bucket: BucketRecord, state: RuntimeState) -> dict[str, float]:
    flow_key = f"{bucket.origin_rank}->{bucket.destination_rank}"
    return_key = f"{bucket.destination_rank}->{bucket.origin_rank}"
    return {
        "destination_pending_work": float(state.destination_pending_work.get(bucket.destination_id, 0.0)),
        "destination_queue_depth": float(state.destination_queue_depth.get(bucket.destination_id, 0)),
        "rank_backlog_work": float(state.rank_backlog_work.get(bucket.destination_rank, 0.0)),
        "rank_queue_depth": float(state.rank_queue_depth.get(bucket.destination_rank, 0)),
        "destination_hotness": float(state.destination_hotness.get(bucket.destination_id, 0.0)),
        "source_outbound_pending_bytes": float(state.source_outbound_pending_bytes.get(bucket.origin_rank, 0.0)),
        "destination_inbound_pending_bytes": float(state.destination_inbound_pending_bytes.get(bucket.destination_rank, 0.0)),
        "flow_pending_bytes": float(state.flow_pending_bytes.get(flow_key, 0.0)),
        "return_flow_pending_bytes": float(state.return_flow_pending_bytes.get(return_key, 0.0)),
        "rank_compute_queue_rows": float(state.rank_compute_queue_rows.get(bucket.destination_rank, 0.0)),
        "expert_pending_rows": float(state.expert_pending_rows.get(bucket.expert_id, 0.0)),
        "rank_dispatch_time_ms": float(state.rank_dispatch_time_ms.get(bucket.origin_rank, 0.0)),
        "rank_compute_time_ms": float(state.rank_compute_time_ms.get(bucket.destination_rank, 0.0)),
        "rank_return_time_ms": float(state.rank_return_time_ms.get(bucket.destination_rank, 0.0)),
        "estimated_service_units": float(bucket.estimated_service_units),
    }


def _expert_features(bucket: BucketRecord, state: RuntimeState) -> dict[str, float]:
    return {
        "payload_rows": float(bucket.payload_rows),
        "payload_bytes": float(bucket.payload_bytes),
        "estimated_dispatch_cost": float(bucket.estimated_dispatch_cost),
        "estimated_compute_cost": float(bucket.estimated_compute_cost),
        "estimated_return_cost": float(bucket.estimated_return_cost),
        "expert_pending_rows": float(state.expert_pending_rows.get(bucket.expert_id, 0.0)),
    }


def _build_explanation(
    strategy: str,
    bucket: BucketRecord,
    state_score: float,
    dependency_score: float,
    full_score: float,
    runtime_features: dict[str, float],
) -> str:
    if strategy == "fifo":
        return f"arrival={bucket.arrival_index}"
    if strategy == "state-only":
        return (
            f"size_norm={bucket.size_norm:.3f} pending={runtime_features['destination_pending_work']:.3f} "
            f"rank_backlog={runtime_features['rank_backlog_work']:.3f} score={state_score:.3f}"
        )
    if strategy == "dependency-only":
        return (
            f"coverage={bucket.source_coverage:.3f} bridge={bucket.bridge_score:.3f} "
            f"peer_degree={bucket.coactive_peer_degree:.3f} score={dependency_score:.3f}"
        )
    return (
        f"state={state_score:.3f} dependency={dependency_score:.3f} full={full_score:.3f} "
        f"pending={runtime_features['destination_pending_work']:.3f}"
    )
