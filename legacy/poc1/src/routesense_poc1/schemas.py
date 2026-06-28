from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunConfig:
    model_id: str = "allenai/OLMoE-1B-7B-0125"
    precision: str = "fp16"
    device_map: str = "balanced"
    max_memory: dict[str, str] | None = None
    text: str = "The history of science is a story of"
    layer: str = "auto"
    rank: int = 0
    output_dir: str = "outputs"
    seed: int = 42
    mock: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class EnvironmentInfo:
    cuda_available: bool = False
    device: str = "cpu"
    torch_version: str = "unknown"
    transformers_version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvironmentInfo":
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "EnvironmentInfo":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class MoELayerInfo:
    layer_path: str
    module_class: str
    forward_signature: str
    has_gate: bool
    has_experts: bool
    module_name: str = ""
    candidate_reason: str = ""
    candidate_score: float = 0.0
    child_module_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoELayerInfo":
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MoELayerInfo":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class RouterTrace:
    layer_path: str
    layer_id: int
    token_pos: int
    target_token_id: int
    target_token_text: str
    next_token_id: int | None = None
    next_token_text: str | None = None
    router_logits: list[float] = field(default_factory=list)
    router_probabilities: list[float] = field(default_factory=list)
    selected_expert_ids: list[int] = field(default_factory=list)
    effective_gate_weights: list[float] = field(default_factory=list)
    topk_ranks: list[int] = field(default_factory=list)
    top1_top2_gap: float | None = None
    routing_entropy: float | None = None
    sample_id: str | None = None
    microbatch_id: int | None = None
    sequence_length: int | None = None
    all_expert_count: int | None = None
    selected_destination_ids: list[int] = field(default_factory=list)
    expert_to_destination_map: dict[str, int] = field(default_factory=dict)
    layer_name: str | None = None
    run_id: str | None = None
    mode: str = "mock"
    all_token_selected_expert_ids: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouterTrace":
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "RouterTrace":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class RouteSnapshot:
    layer_path: str
    token_pos: int
    expert_ids: list[int]
    weights: list[float]
    output_shape: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteSnapshot":
        if "output_shape" in data and data["output_shape"] is not None:
            data = dict(data)
            data["output_shape"] = tuple(data["output_shape"])
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "RouteSnapshot":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class AblationResult:
    model_id: str
    layer_path: str
    layer_id: int
    token_pos: int
    ablated_rank: int
    ablated_expert_id: int
    baseline_nll: float
    ablated_nll: float
    delta_nll: float
    weights_before: list[float]
    weights_after: list[float]
    all_checks_passed: bool
    router_trace_hash: str | None = None
    routing_entropy: float | None = None
    top1_top2_gap: float | None = None
    selected_k: int | None = None
    all_selected_expert_ids: list[int] = field(default_factory=list)
    gate_weight_of_ablated_route: float | None = None
    delta_nll_normalized: float | None = None
    mode: str = "mock"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AblationResult":
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "AblationResult":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass
class BucketSummary:
    destination_id: int
    expert_ids_in_bucket: list[int] = field(default_factory=list)
    token_count_in_bucket: int = 0
    token_positions: list[int] = field(default_factory=list)
    selected_route_count: int = 0
    bucket_rank_by_size: int | None = None
    estimated_bytes: int | None = None
    fanin_proxy: int | None = None
    fanout_proxy: int | None = None
    is_hot_bucket: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchRoutingSummary:
    sample_id: str | None = None
    microbatch_id: int | None = None
    layer_id: int | None = None
    layer_path: str | None = None
    sequence_length: int | None = None
    active_destinations: list[int] = field(default_factory=list)
    destination_count: int = 0
    total_selected_routes: int = 0
    source_count: int = 1
    notes: str | None = None
    buckets: list[BucketSummary] = field(default_factory=list)
    source_destination_matrix: list[list[int]] = field(default_factory=list)
    destination_participation_list: list[int] = field(default_factory=list)
    co_active_destinations: list[list[int]] = field(default_factory=list)
    dependency_graph_edges: list[list[int]] = field(default_factory=list)
    barrier_relevant_destinations: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bucket_summaries"] = data["buckets"]
        return data


@dataclass
class CorrectnessCheckResult:
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorrectnessCheckResult":
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CorrectnessCheckResult":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
