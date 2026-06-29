from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    model_id: str = "allenai/OLMoE-1B-7B-0125"
    precision: str = "fp16"
    device_map: str = "balanced"
    max_memory: dict[str | int, str] | None = None
    output_dir: str = "outputs"
    run_id: str | None = None
    seed: int = 42
    verbose: bool = False
    resume: bool = False
    text: str = "The history of science is a story of"
    dataset_id: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "test"
    num_windows: int = 8
    context_length: int = 64
    num_moe_layers: int = 3
    enable_calibration: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        return cls(**data)


@dataclass
class Window:
    document_id: int
    window_id: int
    input_ids: list[int]
    target_pos: int
    ground_truth_token_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MoELayerSpec:
    layer_index: int
    module_path: str
    gate_module_path: str
    experts_module_path: str
    num_experts: int
    top_k: int
    norm_topk_prob: bool
    module_class: str = ""
    gate_class: str = ""
    experts_class: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingContext:
    run_id: str
    document_id: int
    window_id: int
    layer_id: int
    layer_path: str
    token_pos: int
    expert_id: int
    topk_rank: int
    router_logit: float
    router_probability: float
    effective_gate_weight: float
    top1_top2_gap: float
    routing_entropy: float
    baseline_nll: float | None = None
    ablated_nll: float | None = None
    delta_nll: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AblationRecord:
    run_id: str
    document_id: int
    window_id: int
    layer_id: int
    layer_path: str
    token_pos: int
    expert_id: int
    topk_rank: int
    router_logit: float
    router_probability: float
    effective_gate_weight: float
    top1_top2_gap: float
    routing_entropy: float
    baseline_nll: float
    ablated_nll: float
    delta_nll: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentInfo:
    python_version: str
    torch_version: str
    transformers_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_names: list[str] = field(default_factory=list)
    gpu_memory_mb: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Manifest:
    run_id: str
    model_id: str
    dataset_id: str
    dataset_config: str
    dataset_split: str
    config: dict[str, Any]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json_dataclass(payload: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
