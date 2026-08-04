from __future__ import annotations

from collections.abc import Callable

from .schemas import AblationRecord


FEATURE_EXTRACTORS: dict[str, Callable[[AblationRecord], float]] = {
    "effective_gate_weight": lambda record: float(record.effective_gate_weight),
    "router_probability": lambda record: float(record.router_probability),
    "router_logit": lambda record: float(record.router_logit),
    "abs_router_logit": lambda record: abs(float(record.router_logit)),
    "topk_rank": lambda record: float(record.topk_rank),
    "top1_top2_gap": lambda record: float(record.top1_top2_gap),
    "routing_entropy": lambda record: float(record.routing_entropy),
    "layer_id": lambda record: float(record.layer_id),
    "expert_id": lambda record: float(record.expert_id),
}


FEATURE_ORIENTATION: dict[str, bool] = {
    "effective_gate_weight": False,
    "router_probability": False,
    "router_logit": False,
    "abs_router_logit": False,
    "topk_rank": True,
    "top1_top2_gap": False,
    "routing_entropy": True,
    "layer_id": False,
    "expert_id": False,
}
