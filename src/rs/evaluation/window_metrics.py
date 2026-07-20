"""Derivation helpers for formal RouterSense performance metrics."""
from __future__ import annotations

import math
from typing import Any, Sequence

from rs.core.contracts.performance import (
    OfflineWindowMetrics,
    PerformanceMetricRecord,
    validate_performance_metrics_payload,
)


def improvement_pct(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None or float(baseline) == 0.0:
        return None
    return 100.0 * (float(baseline) - float(value)) / float(baseline)


def weighted_quantile(values: Sequence[tuple[float, int]], q: float) -> float | None:
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must be within [0, 1]")
    ordered = sorted((float(value), int(weight)) for value, weight in values if int(weight) > 0)
    if not ordered:
        return None
    total = sum(weight for _, weight in ordered)
    threshold = max(1, int(math.ceil(float(q) * total)))
    cursor = 0
    for value, weight in ordered:
        cursor += weight
        if cursor >= threshold:
            return value
    return ordered[-1][0]


def derive_window_metrics(
    plan: Any,
    *,
    planning_ms: float,
    bind_ms: float,
    target_entry_overhead_ms: float | None = None,
) -> OfflineWindowMetrics:
    """Derive the formal offline metrics from a compact/materialized plan."""
    materialized = plan.materialize() if hasattr(plan, "materialize") else plan
    p1: list[tuple[float, int]] = []
    p0: list[tuple[float, int]] = []
    for wave in materialized.waves:
        end_time = float(getattr(wave, "end_time", 0.0))
        for flow in wave.flows:
            phase = str(flow.phase)
            rows = int(getattr(flow, "row_count", getattr(flow, "byte_count", 0)))
            if phase == "p1_return":
                p1.append((end_time, rows))
            elif phase == "p0_dispatch":
                p0.append((end_time, rows))
    entry = float(bind_ms if target_entry_overhead_ms is None else target_entry_overhead_ms)
    result = OfflineWindowMetrics(
        communication_makespan=float(materialized.makespan),
        current_layer_completion=float(materialized.phase_completion[1]),
        first_token_time=min((value for value, _ in p1), default=None),
        tail_latency_p95=weighted_quantile(p1, 0.95),
        tail_latency_p99=weighted_quantile(p1, 0.99),
        tail_latency_max=max((value for value, _ in p1), default=None),
        first_dispatch_arrival=min((value for value, _ in p0), default=None),
        planning_ms=float(planning_ms),
        bind_ms=float(bind_ms),
        target_entry_overhead_ms=entry,
        total_control_ms=float(planning_ms) + float(bind_ms),
        wave_count=int(len(materialized.waves)),
        p1_remote_token_count=int(sum(weight for _, weight in p1)),
    )
    result.validate()
    return result


__all__ = [
    "derive_window_metrics",
    "improvement_pct",
    "validate_performance_metrics_payload",
    "weighted_quantile",
]
