from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OLMoEAdapterConfig:
    model_class: str
    num_experts: int
    top_k: int
