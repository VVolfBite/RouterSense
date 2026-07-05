"""Priority scoring helpers for multiphase scheduling."""

from __future__ import annotations

from typing import Any

from .flow_model import RUNTIME_LOOKAHEAD_MODE, ResidualFlowState


def ready_flow_candidates(
    *,
    flows: list[ResidualFlowState],
    residual: dict[str, float],
    ready_since: dict[str, float],
    current_time: float,
    release_time: dict[tuple[int, int], float],
    inbound_remaining: dict[tuple[int, int], float],
    downstream_load: dict[tuple[int, int], float],
    age_scale: float,
    residual_weight: float,
    barrier_weight: float,
    age_weight: float,
    prediction_weight: float,
    mode: str,
    prediction_confidence: float,
    base_score_lookup: dict[str, float] | None = None,
    base_priority_weight: float = 0.0,
) -> list[dict[str, Any]]:
    max_residual = max((value for value in residual.values() if value > 0.0), default=1.0)
    ready: list[dict[str, Any]] = []
    for flow in flows:
        remaining = residual[flow.flow_id]
        if remaining <= 0.0:
            continue
        release = release_time[(flow.phase, flow.src_gpu)]
        if current_time + 1e-9 < release:
            continue
        ready_since.setdefault(flow.flow_id, current_time)
        barrier_urgency = downstream_load.get((flow.phase + 1, flow.dst_gpu), 0.0) / max(
            inbound_remaining.get((flow.phase, flow.dst_gpu), remaining),
            1.0,
        )
        age = (current_time - ready_since[flow.flow_id]) / max(age_scale, 1.0)
        prediction_bonus = 0.0
        if mode == RUNTIME_LOOKAHEAD_MODE and flow.phase == 1:
            prediction_bonus = prediction_confidence * downstream_load.get((2, flow.dst_gpu), 0.0) / max(max_residual, 1.0)
        base_priority = 0.0 if base_score_lookup is None else float(base_score_lookup.get(flow.flow_id, 0.0))
        score = (
            base_priority_weight * base_priority
            + residual_weight * (remaining / max_residual)
            + barrier_weight * barrier_urgency
            + age_weight * age
            + prediction_weight * prediction_bonus
        )
        ready.append(
            {
                "flow_id": flow.flow_id,
                "phase": flow.phase,
                "src_gpu": flow.src_gpu,
                "dst_gpu": flow.dst_gpu,
                "residual": remaining,
                "release_time": release,
                "barrier_urgency": barrier_urgency,
                "age": age,
                "prediction_bonus": prediction_bonus,
                "base_priority": base_priority,
                "score": score,
            }
        )
    return ready
