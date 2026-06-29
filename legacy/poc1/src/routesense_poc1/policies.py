from __future__ import annotations

import random
from collections.abc import Sequence

from .schemas import AblationRecord


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
    if strategy == "oracle":
        return min(records, key=lambda record: (record.delta_nll, record.topk_rank)).expert_id
    if strategy == "calibrated":
        if calibrator is None:
            raise ValueError("calibrator is required for calibrated strategy")
        predictions = calibrator.predict(_records_to_features(records))
        best_index = min(range(len(records)), key=lambda index: float(predictions[index]))
        return records[best_index].expert_id
    raise ValueError(f"unsupported strategy: {strategy}")


def _records_to_features(records: Sequence[AblationRecord]) -> list[list[float]]:
    return [
        [
            record.effective_gate_weight,
            record.router_probability,
            float(record.topk_rank),
            record.top1_top2_gap,
            record.routing_entropy,
            float(record.layer_id),
            float(record.expert_id),
        ]
        for record in records
    ]

