"""Shadow pending-window planning for online multiphase execution.

This module does not mutate the frozen phase-local executor. It converts the
currently available runtime window state into a deterministic shadow view of a
future `multiphase_pending_window` runtime capability.
"""

from __future__ import annotations

import time
from typing import Any

from rs.scheduling.contracts import FlowDemand
from rs.scheduling.multiphase.routersense_lookahead import RouterSenseMultiphaseLookaheadPolicy
from rs.scheduling.validation import stable_hash

from .window_state import OnlineWindowState, build_shadow_problem


def build_pending_window_shadow(
    *,
    state: OnlineWindowState,
    p0_weight: float,
    p1_reservation_weight: float,
    p2_hint_weight: float,
) -> dict[str, Any]:
    problem = build_shadow_problem(
        state=state,
        p0_weight=p0_weight,
        p1_reservation_weight=p1_reservation_weight,
        p2_hint_weight=p2_hint_weight,
    )
    policy = RouterSenseMultiphaseLookaheadPolicy(
        information_mode=problem.options.information_mode,
        p0_weight=p0_weight,
        p1_reservation_weight=p1_reservation_weight,
        p2_hint_weight=p2_hint_weight,
    )
    logical_plan = policy.build_logical_plan(problem)
    ready_ids = {str(flow.flow_id) for flow in problem.flow_window.ready_flows if bool(flow.is_executable)}
    blocked_ids = {str(flow.flow_id) for flow in problem.flow_window.blocked_flows}
    forecast_ids = {str(flow.flow_id) for flow in problem.flow_window.forecast_pressure}
    ready_edges = {_edge_key(flow) for flow in problem.flow_window.ready_flows if bool(flow.is_executable)}
    blocked_edges = {_edge_key(flow) for flow in problem.flow_window.blocked_flows}
    forecast_edges = {_edge_key(flow) for flow in problem.flow_window.forecast_pressure}
    executable_prefix: list[dict[str, Any]] = []
    first_executable_wave: dict[str, Any] | None = None
    for wave in logical_plan.waves:
        selected_ids = tuple(str(flow.flow_id) for flow in wave.flows)
        selected_edges = tuple(
            {
                "flow_id": str(flow.flow_id),
                "phase": str(flow.phase),
                "src_rank": int(flow.src_rank),
                "dst_rank": int(flow.dst_rank),
                "byte_count": int(flow.byte_count),
            }
            for flow in wave.flows
        )
        blocked_selected = tuple(edge["flow_id"] for edge in selected_edges if _edge_key_from_dict(edge) in blocked_edges)
        forecast_selected = tuple(edge["flow_id"] for edge in selected_edges if _edge_key_from_dict(edge) in forecast_edges)
        ready_selected = tuple(edge["flow_id"] for edge in selected_edges if _edge_key_from_dict(edge) in ready_edges)
        wave_payload = {
            "wave_id": int(wave.wave_id),
            "selected_flow_ids": list(selected_ids),
            "selected_edges": list(selected_edges),
            "ready_selected_flow_ids": list(ready_selected),
            "blocked_selected_flow_ids": list(blocked_selected),
            "forecast_selected_flow_ids": list(forecast_selected),
            "duration": float(wave.duration),
            "fully_executable_now": len(blocked_selected) == 0 and len(forecast_selected) == 0,
        }
        executable_prefix.append(wave_payload)
        if first_executable_wave is None and wave_payload["fully_executable_now"]:
            first_executable_wave = wave_payload
    return {
        "ts_us": int(time.time() * 1e6),
        "window_key": state.window_key,
        "layer_name": state.layer_name,
        "layer_id": state.layer_id,
        "execution_capability_required": "multiphase_pending_window",
        "prepared_plan_bound": state.prepared_plan_binding is not None,
        "prepared_window_key": None if state.prepared_plan_binding is None else state.prepared_plan_binding.window_key,
        "source_layer_name": None if state.prepared_plan_binding is None else state.prepared_plan_binding.source_layer_name,
        "release_state": state.release_state.to_dict(),
        "information_mode": problem.options.information_mode,
        "prediction_confidence": float(problem.options.prediction_confidence),
        "ready_flow_count": len(problem.flow_window.ready_flows),
        "blocked_flow_count": len(problem.flow_window.blocked_flows),
        "forecast_flow_count": len(problem.flow_window.forecast_pressure),
        "ready_flow_ids": sorted(ready_ids),
        "blocked_flow_ids": sorted(blocked_ids),
        "forecast_flow_ids": sorted(forecast_ids),
        "ready_flows": [flow.to_dict() for flow in problem.flow_window.ready_flows],
        "blocked_flows": [flow.to_dict() for flow in problem.flow_window.blocked_flows],
        "forecast_flows": [flow.to_dict() for flow in problem.flow_window.forecast_pressure],
        "p0_dispatch_matrix": [list(row) for row in problem.p0_dispatch_matrix],
        "p1_return_matrix": [list(row) for row in problem.p1_return_matrix],
        "p2_next_dispatch_forecast_matrix": [list(row) for row in problem.p2_next_dispatch_forecast_matrix],
        "shadow_policy_name": logical_plan.policy_name,
        "shadow_plan_hash": stable_hash(logical_plan.to_dict()),
        "shadow_wave_count": len(logical_plan.waves),
        "shadow_makespan": float(logical_plan.diagnostics.get("makespan", 0.0)),
        "first_executable_wave": first_executable_wave,
        "executable_prefix": executable_prefix,
        "contains_blocked_future_waves": any(len(item["blocked_selected_flow_ids"]) > 0 for item in executable_prefix),
        "contains_forecast_only_waves": any(len(item["forecast_selected_flow_ids"]) > 0 for item in executable_prefix),
    }


def executable_now(snapshot: dict[str, Any]) -> list[str]:
    first_wave = snapshot.get("first_executable_wave")
    if not isinstance(first_wave, dict):
        return []
    return list(first_wave.get("ready_selected_flow_ids", []) or [])


def classify_flow(flow: FlowDemand) -> str:
    if str(flow.phase) == "p2_next_dispatch_forecast":
        return "forecast_only"
    if str(flow.release_state) == "blocked":
        return "blocked"
    if bool(flow.is_executable):
        return "ready"
    return "unknown"


def _edge_key(flow: FlowDemand) -> tuple[str, int, int]:
    return (str(flow.phase), int(flow.src_rank), int(flow.dst_rank))


def _edge_key_from_dict(flow: dict[str, Any]) -> tuple[str, int, int]:
    return (str(flow.get("phase", "")), int(flow.get("src_rank", -1)), int(flow.get("dst_rank", -1)))


__all__ = ["build_pending_window_shadow", "classify_flow", "executable_now"]
