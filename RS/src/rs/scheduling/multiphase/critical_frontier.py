"""RouterSense Critical Frontier candidate scoring.

RSCF augments the shared RSBC score with release-DAG criticality and endpoint
bottleneck dual prices. Every active feature is derived from residual traffic,
port constraints, and phase-release dependencies. An optional reciprocal-flow
term is retained only as an ablation hook and is disabled in the registered
RSCF v4 kernel.
"""

from __future__ import annotations

from typing import Any

from .critical_dual import release_critical_dual_candidates
from .flow_model import ResidualFlowState
from .scoring import ready_flow_candidates


def critical_frontier_candidates(
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
    release_gain_weight: float,
    endpoint_pressure_weight: float,
    critical_path_weight: float,
    transitive_unlock_weight: float,
    endpoint_dual_weight: float,
    duplex_pair_weight: float,
    dual_temperature: float,
    transitive_tail_weight: float,
    destination_hotspot_weight: float,
    size_bias_power: float,
    mode: str = "execution_window",
    prediction_confidence: float = 0.0,
    future_matrix: list[list[int]] | None = None,
    base_score_lookup: dict[str, float] | None = None,
    base_priority_weight: float = 0.0,
) -> list[dict[str, Any]]:
    """Return ready flows scored by the RSCF shared kernel."""

    common_arguments = {
        "flows": flows,
        "residual": residual,
        "ready_since": ready_since,
        "current_time": current_time,
        "release_time": release_time,
        "inbound_remaining": inbound_remaining,
        "age_scale": age_scale,
        "base_score_lookup": base_score_lookup,
        "base_priority_weight": base_priority_weight,
    }

    base_candidates = ready_flow_candidates(
        **common_arguments,
        downstream_load=downstream_load,
        residual_weight=residual_weight,
        barrier_weight=barrier_weight,
        age_weight=age_weight,
        prediction_weight=prediction_weight,
        endpoint_pressure_weight=endpoint_pressure_weight,
        release_gain_weight=release_gain_weight,
        mode=mode,
        prediction_confidence=prediction_confidence,
    )
    dual_candidates = release_critical_dual_candidates(
        **common_arguments,
        residual_weight=0.0,
        barrier_dual_weight=1.0,
        endpoint_dual_weight=0.0,
        unlock_weight=1.0,
        age_weight=0.0,
        dual_temperature=dual_temperature,
        transitive_tail_weight=transitive_tail_weight,
        destination_hotspot_weight=destination_hotspot_weight,
        size_bias_power=size_bias_power,
        future_matrix=future_matrix,
        future_confidence=prediction_confidence,
    )
    dual_by_flow = {
        str(candidate["flow_id"]): candidate for candidate in dual_candidates
    }
    max_tail = max(
        [
            1.0,
            *[
                float(candidate.get("transitive_tail", 0.0))
                for candidate in dual_candidates
            ],
        ]
    )

    reciprocal_residual: dict[tuple[int, int], float] = {}
    for candidate in base_candidates:
        key = (int(candidate["src_gpu"]), int(candidate["dst_gpu"]))
        reciprocal_residual[key] = max(
            reciprocal_residual.get(key, 0.0),
            float(candidate["residual"]),
        )
    max_ready_residual = max(
        [1.0, *[float(candidate["residual"]) for candidate in base_candidates]]
    )

    output: list[dict[str, Any]] = []
    for candidate in base_candidates:
        dual = dual_by_flow.get(str(candidate["flow_id"]), {})
        reciprocal = reciprocal_residual.get(
            (int(candidate["dst_gpu"]), int(candidate["src_gpu"])),
            0.0,
        )
        pairability = min(float(candidate["residual"]), reciprocal) / (
            max_ready_residual
        )
        critical_path_dual = float(dual.get("barrier_dual_price", 0.0))
        endpoint_dual = float(dual.get("send_dual_price", 0.0)) + float(
            dual.get("recv_dual_price", 0.0)
        )
        transitive_unlock = (
            float(dual.get("unlock_fraction", 0.0))
            * float(dual.get("transitive_tail", 0.0))
            / max_tail
        )
        frontier_component = (
            float(critical_path_weight) * critical_path_dual
            + float(transitive_unlock_weight) * transitive_unlock
            + float(endpoint_dual_weight) * endpoint_dual
            + float(duplex_pair_weight) * pairability
        )

        enriched = dict(candidate)
        enriched.update(
            {
                "critical_path_dual": critical_path_dual,
                "transitive_unlock": transitive_unlock,
                "endpoint_dual": endpoint_dual,
                "duplex_pairability": pairability,
                "critical_frontier_component": frontier_component,
                "score": float(candidate["score"]) + frontier_component,
            }
        )
        output.append(enriched)

    return output


__all__ = ["critical_frontier_candidates"]
