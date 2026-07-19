from __future__ import annotations

"""Public scheduling boundary for Future-P012 prepared-order binding.

Runtime imports this module rather than the private migrated kernel.  The
returned objects are immutable Python contracts and contain no runtime/store
state.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Mapping

import numpy as np

from ._kernel.event_core import bind_prepared_order
from ._kernel.future import _p01_prediction_gate_metrics
from ._kernel.plan import tuple_to_compact_plan


@dataclass(frozen=True)
class PreparedBoundFlow:
    flow_id: str
    phase: str
    src_rank: int
    dst_rank: int
    row_count: int
    release_state: str
    executable: bool


@dataclass(frozen=True)
class PreparedBoundWave:
    wave_id: int
    flows: tuple[PreparedBoundFlow, ...]
    estimated_duration: float


@dataclass(frozen=True)
class PreparedOrderBindDecision:
    accepted: bool
    exact: bool
    waves: tuple[PreparedBoundWave, ...] = ()
    metrics: Mapping[str, object] = field(default_factory=dict)
    elapsed_us: float = 0.0


def _square(value: object, *, name: str, world_size: int, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (world_size, world_size):
        raise ValueError(f"{name} shape {array.shape} != ({world_size}, {world_size})")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    result = np.ascontiguousarray(array.astype(dtype, copy=False))
    return result


def _template(payload: Mapping[str, object]) -> tuple:
    world = int(payload["world_size"])
    phase = np.ascontiguousarray(np.asarray(payload["phase"], dtype=np.int8))
    destination = np.ascontiguousarray(np.asarray(payload["destination"], dtype=np.int16))
    predicted_size = np.ascontiguousarray(np.asarray(payload["predicted_size"], dtype=np.int32))
    if phase.ndim != 2 or phase.shape[1] != world:
        raise ValueError("future prepared phase array has invalid shape")
    if destination.shape != phase.shape or predicted_size.shape != phase.shape:
        raise ValueError("future prepared arrays must share one shape")
    if (predicted_size < 0).any():
        raise ValueError("future prepared predicted_size must be non-negative")
    waves = int(phase.shape[0])
    zeros = np.zeros(waves, dtype=np.float64)
    return (
        0.0,
        waves,
        phase,
        destination,
        predicted_size,
        np.zeros(waves, dtype=np.int32),
        zeros.copy(),
        zeros.copy(),
        zeros.copy(),
        np.zeros(3, dtype=np.float64),
        np.full(world, -1.0, dtype=np.float64),
        np.full(world, -1.0, dtype=np.float64),
        1,
    )


def bind_prepared_order_payload(
    *,
    payload: Mapping[str, object],
    predicted_p0_rows: object,
    actual_p0_rows: object,
    planner_id: str,
    request_digest: str,
) -> PreparedOrderBindDecision:
    """Gate and bind actual P0/P1 without reading P2 truth or running matching."""
    started = time.perf_counter_ns()
    if str(payload.get("semantic_version")) != "future_prepared_order_payload_v1":
        raise ValueError("unsupported Future prepared-order payload version")
    world = int(payload["world_size"])
    predicted_p0 = _square(predicted_p0_rows, name="predicted_p0", world_size=world, dtype=np.int32)
    actual_p0 = _square(actual_p0_rows, name="actual_p0", world_size=world, dtype=np.int32)
    if (predicted_p0 < 0).any() or (actual_p0 < 0).any():
        raise ValueError("P0 rows must be non-negative")
    np.fill_diagonal(predicted_p0, 0)
    np.fill_diagonal(actual_p0, 0)
    actual_p1 = np.ascontiguousarray(actual_p0.T)
    gate = _p01_prediction_gate_metrics(predicted_p0, actual_p0, actual_p1)
    accepted = bool(
        gate["p0"]["relative_l1"] <= float(payload["max_p0_relative_l1"])
        and gate["p1"]["relative_l1"] <= float(payload["max_p1_relative_l1"])
        and gate["p0"]["support_recall"] >= float(payload["min_p0_support_recall"])
        and gate["p0"]["support_precision"] >= float(payload["min_p0_support_precision"])
        and gate["p1"]["support_recall"] >= float(payload["min_p0_support_recall"])
        and gate["p1"]["support_precision"] >= float(payload["min_p0_support_precision"])
    )
    base_metrics: dict[str, Any] = {
        "future_gate_accepted": accepted,
        "future_gate_metrics": gate,
        "prepared_order_digest": str(payload.get("prepared_order_digest", "")),
        "online_matching_solver_calls": 0,
        "online_candidate_selection": False,
    }
    if not accepted:
        return PreparedOrderBindDecision(
            accepted=False,
            exact=False,
            metrics={**base_metrics, "reason": "future_p01_gate_rejected"},
            elapsed_us=(time.perf_counter_ns() - started) / 1000.0,
        )
    exact = bool(np.array_equal(predicted_p0, actual_p0))
    if exact:
        return PreparedOrderBindDecision(
            accepted=True,
            exact=True,
            waves=(),
            metrics={**base_metrics, "bind_strategy": "prepared_order_exact_reuse"},
            elapsed_us=(time.perf_counter_ns() - started) / 1000.0,
        )
    template = _template(payload)
    slope = _square(payload["edge_slope"], name="edge_slope", world_size=world, dtype=np.float64)
    intercept = _square(payload["edge_intercept"], name="edge_intercept", world_size=world, dtype=np.float64)
    zero_p2 = np.zeros_like(actual_p0)
    bound = bind_prepared_order(
        (actual_p0, actual_p1, zero_p2),
        template,
        edge_slope=slope,
        edge_intercept=intercept,
        expert_compute_delay=float(payload["expert_compute_delay"]),
        wave_launch_b=float(payload["wave_launch_b"]),
        max_waves=int(payload["max_waves"]),
    )
    compact = tuple_to_compact_plan(
        bound[:13],
        planner_id=str(planner_id),
        planner_family="future_prepared_p01",
        branch="future_prepared_reconcile",
        request_digest=str(request_digest),
        forecast=False,
        metadata=base_metrics,
        trusted_arrays=True,
    )
    materialized = compact.materialize()
    waves = tuple(
        PreparedBoundWave(
            wave_id=int(wave.wave_id),
            flows=tuple(
                PreparedBoundFlow(
                    flow_id=str(flow.segment_id),
                    phase=str(flow.phase),
                    src_rank=int(flow.src_rank),
                    dst_rank=int(flow.dst_rank),
                    row_count=int(flow.row_count),
                    release_state=str(flow.release_state),
                    executable=bool(flow.executable),
                )
                for flow in wave.flows
            ),
            estimated_duration=float(wave.estimated_duration),
        )
        for wave in materialized.waves
    )
    resized_edges = int(np.count_nonzero((predicted_p0 > 0) & (actual_p0 > 0) & (predicted_p0 != actual_p0)))
    metrics = {
        **base_metrics,
        "bind_strategy": "prepared_order_stable_filter",
        "template_rows_served": int(bound[13]),
        "template_waves_used": int(bound[14]),
        "projected_template_edges": int(bound[15]),
        "topology_tail_rows": int(bound[16]),
        "topology_tail_waves": int(bound[17]),
        "template_support_coverage": float(bound[18]),
        "resized_edges": resized_edges,
        "kernel_plan_digest": materialized.semantic_digest(),
    }
    return PreparedOrderBindDecision(
        accepted=True,
        exact=exact,
        waves=waves,
        metrics=metrics,
        elapsed_us=(time.perf_counter_ns() - started) / 1000.0,
    )


__all__ = [
    "PreparedBoundFlow",
    "PreparedBoundWave",
    "PreparedOrderBindDecision",
    "bind_prepared_order_payload",
]
