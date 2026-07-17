"""Generic release-critical dual scoring for multiphase flow scheduling.

The scorer is model-agnostic. It consumes only residual flow volumes, endpoint
capacities, and the phase-release DAG. It approximates the dual prices of a
smooth maximum over endpoint-workload lower bounds and barrier-to-tail critical
paths. No expert identity, layer type, or model-specific feature is used.
"""

from __future__ import annotations

import math
from typing import Any

from .flow_model import ResidualFlowState


def _softmax_prices(
    values: dict[Any, float],
    temperature: float,
) -> dict[Any, float]:
    if not values:
        return {}

    scale = max(max(values.values()), 1.0)
    tau = max(float(temperature), 1e-6)
    logits = {key: float(value) / scale / tau for key, value in values.items()}
    maximum = max(logits.values())
    exponentials = {
        key: math.exp(max(-60.0, min(60.0, logit - maximum)))
        for key, logit in logits.items()
    }
    total = sum(exponentials.values()) or 1.0
    return {key: value / total for key, value in exponentials.items()}


def release_critical_dual_candidates(
    *,
    flows: list[ResidualFlowState],
    residual: dict[str, float],
    ready_since: dict[str, float],
    current_time: float,
    release_time: dict[tuple[int, int], float],
    inbound_remaining: dict[tuple[int, int], float],
    age_scale: float,
    residual_weight: float,
    barrier_dual_weight: float,
    endpoint_dual_weight: float,
    unlock_weight: float,
    age_weight: float,
    dual_temperature: float,
    transitive_tail_weight: float,
    destination_hotspot_weight: float,
    size_bias_power: float,
    future_matrix: list[list[int]] | None = None,
    future_confidence: float = 1.0,
    base_score_lookup: dict[str, float] | None = None,
    base_priority_weight: float = 0.0,
) -> list[dict[str, Any]]:
    """Build ready candidates priced by release-DAG and endpoint criticality.

    ``future_matrix`` is advisory geometry only. It may alter critical-tail
    prices, but it never creates executable flows or served bytes.
    """

    edge = {
        (flow.phase, flow.src_gpu, flow.dst_gpu): float(
            residual.get(flow.flow_id, 0.0)
        )
        for flow in flows
        if residual.get(flow.flow_id, 0.0) > 1e-9
    }

    if future_matrix is not None:
        confidence = max(0.0, min(1.0, float(future_confidence)))
        for src, row in enumerate(future_matrix):
            for dst, value in enumerate(row):
                key = (2, src, dst)
                if src == dst or float(value) <= 0.0 or key in edge:
                    continue
                edge[key] = confidence * float(value)

    outgoing: dict[tuple[int, int], float] = {}
    incoming: dict[tuple[int, int], float] = {}
    send_total: dict[int, float] = {}
    recv_total: dict[int, float] = {}
    phase_rows: dict[tuple[int, int], list[tuple[int, float]]] = {}

    for (phase, src, dst), volume in edge.items():
        outgoing[(phase, src)] = outgoing.get((phase, src), 0.0) + volume
        incoming[(phase, dst)] = incoming.get((phase, dst), 0.0) + volume
        send_total[src] = send_total.get(src, 0.0) + volume
        recv_total[dst] = recv_total.get(dst, 0.0) + volume
        phase_rows.setdefault((phase, src), []).append((dst, volume))

    ranks = {flow.src_gpu for flow in flows} | {flow.dst_gpu for flow in flows}

    # A P2 source tail contains its serial send work and optional exposure to
    # congested destination ports. Both terms depend only on traffic geometry.
    max_p2_inbound = max(
        [1.0, *[value for (phase, _), value in incoming.items() if phase == 2]]
    )
    p2_tail: dict[int, float] = {}
    for rank in ranks:
        row_volume = outgoing.get((2, rank), 0.0)
        hotspot_exposure = 0.0
        if row_volume > 0.0:
            hotspot_exposure = sum(
                volume * (incoming.get((2, dst), 0.0) / max_p2_inbound)
                for dst, volume in phase_rows.get((2, rank), ())
            )
        p2_tail[rank] = row_volume + (
            float(destination_hotspot_weight) * hotspot_exposure
        )

    # Completing a P0 destination barrier releases one P1 source row. Its tail
    # is that row's work plus the largest P2 tail reachable through its P1
    # destinations. This is a generic release-DAG property.
    p1_tail: dict[int, float] = {}
    for rank in ranks:
        row_volume = outgoing.get((1, rank), 0.0)
        downstream = [
            p2_tail.get(dst, 0.0)
            for dst, volume in phase_rows.get((1, rank), ())
            if volume > 1e-9
        ]
        p1_tail[rank] = row_volume + float(transitive_tail_weight) * (
            max(downstream) if downstream else 0.0
        )

    path_values: dict[tuple[int, int], float] = {}
    for rank in ranks:
        path_values[(0, rank)] = incoming.get((0, rank), 0.0) + p1_tail.get(
            rank, 0.0
        )
        path_values[(1, rank)] = incoming.get((1, rank), 0.0) + p2_tail.get(
            rank, 0.0
        )

    barrier_price = _softmax_prices(path_values, dual_temperature)
    send_price = _softmax_prices(send_total, dual_temperature)
    recv_price = _softmax_prices(recv_total, dual_temperature)
    max_residual = max([1.0, *residual.values()])
    max_tail = max([1.0, *p1_tail.values(), *p2_tail.values()])

    ready: list[dict[str, Any]] = []
    for flow in flows:
        remaining = float(residual.get(flow.flow_id, 0.0))
        source_release = float(release_time[(flow.phase, flow.src_gpu)])
        if remaining <= 1e-9 or current_time + 1e-9 < source_release:
            continue

        ready_since.setdefault(flow.flow_id, current_time)
        size_normalized = remaining / max_residual
        size_bias = max(size_normalized, 1e-9) ** max(
            0.0, float(size_bias_power)
        )
        endpoint_price = send_price.get(flow.src_gpu, 0.0) + recv_price.get(
            flow.dst_gpu, 0.0
        )
        barrier_dual = (
            barrier_price.get((flow.phase, flow.dst_gpu), 0.0)
            if flow.phase < 2
            else 0.0
        )
        inbound = max(
            float(inbound_remaining.get((flow.phase, flow.dst_gpu), remaining)),
            1.0,
        )
        unlock_fraction = min(1.0, remaining / inbound)
        if flow.phase == 0:
            transitive_tail = p1_tail.get(flow.dst_gpu, 0.0)
        elif flow.phase == 1:
            transitive_tail = p2_tail.get(flow.dst_gpu, 0.0)
        else:
            transitive_tail = 0.0
        unlock_value = unlock_fraction * (transitive_tail / max_tail)
        age = (current_time - ready_since[flow.flow_id]) / max(age_scale, 1.0)
        base_priority = (
            0.0
            if base_score_lookup is None
            else float(base_score_lookup.get(flow.flow_id, 0.0))
        )

        residual_component = float(residual_weight) * size_normalized
        barrier_component = float(barrier_dual_weight) * barrier_dual
        endpoint_component = float(endpoint_dual_weight) * endpoint_price
        release_component = float(unlock_weight) * unlock_value
        age_component = float(age_weight) * age
        base_component = float(base_priority_weight) * base_priority
        unscaled_score = (
            residual_component
            + endpoint_component
            + barrier_component
            + release_component
            + age_component
            + base_component
        )
        score = unscaled_score * (0.5 + 0.5 * size_bias)

        ready.append(
            {
                "flow_id": flow.flow_id,
                "phase": flow.phase,
                "src_gpu": flow.src_gpu,
                "dst_gpu": flow.dst_gpu,
                "residual": remaining,
                "release_time": source_release,
                "barrier_urgency": barrier_dual,
                "age": age,
                "prediction_bonus": 0.0,
                "base_priority": base_priority,
                "score": score,
                "residual_component": residual_component,
                "barrier_component": barrier_component,
                "endpoint_pressure_component": endpoint_component,
                "release_gain_component": release_component,
                "critical_path_value": path_values.get(
                    (flow.phase, flow.dst_gpu), 0.0
                ),
                "barrier_dual_price": barrier_dual,
                "send_dual_price": send_price.get(flow.src_gpu, 0.0),
                "recv_dual_price": recv_price.get(flow.dst_gpu, 0.0),
                "unlock_fraction": unlock_fraction,
                "transitive_tail": transitive_tail,
                "size_bias": size_bias,
            }
        )

    return ready


__all__ = ["release_critical_dual_candidates"]
