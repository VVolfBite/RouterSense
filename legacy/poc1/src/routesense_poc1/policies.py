from __future__ import annotations

import random
from collections.abc import Sequence
from math import fabs

from .schemas import AblationRecord


SINGLE_FACTOR_STRATEGIES = {
    "router_probability_min",
    "effective_gate_weight_min",
    "router_logit_min",
    "topk_rank_max",
    "top1_top2_gap_min",
    "routing_entropy_max",
    "abs_router_logit_min",
}


def select_deferrable_expert(
    records: Sequence[AblationRecord],
    strategy: str,
    calibrator=None,
    seed: int = 42,
) -> int | None:
    if not records:
        raise ValueError("records must not be empty")
    if strategy == "full":
        return None
    if strategy == "random":
        return random.Random(seed).choice(list(records)).expert_id
    if strategy == "raw_routing":
        return min(records, key=lambda record: (record.effective_gate_weight, record.topk_rank)).expert_id
    if strategy == "router_probability_min":
        return min(records, key=lambda record: (record.router_probability, record.topk_rank)).expert_id
    if strategy == "effective_gate_weight_min":
        return min(records, key=lambda record: (record.effective_gate_weight, record.topk_rank)).expert_id
    if strategy == "router_logit_min":
        return min(records, key=lambda record: (record.router_logit, record.topk_rank)).expert_id
    if strategy == "topk_rank_max":
        return max(records, key=lambda record: (record.topk_rank, -record.effective_gate_weight)).expert_id
    if strategy == "top1_top2_gap_min":
        return min(records, key=lambda record: (record.top1_top2_gap, record.topk_rank)).expert_id
    if strategy == "routing_entropy_max":
        return max(records, key=lambda record: (record.routing_entropy, -record.topk_rank)).expert_id
    if strategy == "abs_router_logit_min":
        return min(records, key=lambda record: (fabs(record.router_logit), record.topk_rank)).expert_id
    if strategy == "oracle":
        return min(records, key=lambda record: (record.delta_nll, record.topk_rank)).expert_id
    if strategy == "calibrated":
        if calibrator is None:
            raise ValueError("calibrator is required for calibrated strategy")
        if hasattr(calibrator, "model") and hasattr(calibrator, "feature_names"):
            predictions = calibrator.model.predict(_records_to_features(records, calibrator.feature_names))
        else:
            predictions = calibrator.predict(_records_to_features(records))
        best_index = min(range(len(records)), key=lambda index: float(predictions[index]))
        return records[best_index].expert_id
    raise ValueError(f"unsupported strategy: {strategy}")


def _records_to_features(
    records: Sequence[AblationRecord],
    feature_names: Sequence[str] | None = None,
) -> list[list[float]]:
    names = list(feature_names or [
        "effective_gate_weight",
        "router_probability",
        "topk_rank",
        "top1_top2_gap",
        "routing_entropy",
        "layer_id",
        "expert_id",
    ])
    feature_map = {
        "effective_gate_weight": lambda record: record.effective_gate_weight,
        "router_probability": lambda record: record.router_probability,
        "topk_rank": lambda record: float(record.topk_rank),
        "top1_top2_gap": lambda record: record.top1_top2_gap,
        "routing_entropy": lambda record: record.routing_entropy,
        "layer_id": lambda record: float(record.layer_id),
        "expert_id": lambda record: float(record.expert_id),
        "router_logit": lambda record: record.router_logit,
        "abs_router_logit": lambda record: fabs(record.router_logit),
    }
    return [
        [float(feature_map[name](record)) for name in names]
        for record in records
    ]
