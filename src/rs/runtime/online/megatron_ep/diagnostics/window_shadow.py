"""Diagnostic shadow plan built with the formal orthogonal planner."""
from __future__ import annotations

import time
from typing import Any

from rs.core.contracts import PlanningConstraints, PlanningIdentity, PlanningTopology, PlanningWeights
from rs.planning import PlannerRegistry
from rs.planning.request_builder import build_window_planning_request
from rs.runtime.online.megatron_ep.state.window_runtime_state import OnlineWindowState


def _zero(n: int):
    return tuple(tuple(0 for _ in range(n)) for _ in range(n))


def _observation_matrix(observation, *, n: int):
    if observation is None:
        return _zero(n)
    rows = [[0 for _ in range(n)] for _ in range(n)]
    src = int(observation.global_rank)
    if 0 <= src < n:
        for dst, value in enumerate(tuple(int(v) for v in observation.per_peer_bytes)[:n]):
            if dst != src:
                rows[src][dst] = int(value)
    return tuple(tuple(row) for row in rows)


def _prepared_matrix(state: OnlineWindowState, *, phase: str, n: int):
    prepared = state.prepared_plan
    if prepared is None:
        return _zero(n)
    if phase == "p2_next_dispatch_forecast":
        value = tuple(tuple(int(v) for v in row) for row in getattr(prepared, "forecast_matrix", ()) or ())
        if len(value) == n:
            return value
    rows = [[0 for _ in range(n)] for _ in range(n)]
    for wave in prepared.logical_plan.waves:
        for flow in wave.flows:
            if str(flow.phase) != phase:
                continue
            src, dst = int(flow.src_rank), int(flow.dst_rank)
            if 0 <= src < n and 0 <= dst < n and src != dst:
                rows[src][dst] += int(flow.byte_count)
    return tuple(tuple(row) for row in rows)


def _nonzero(matrix) -> bool:
    return any(any(int(v) > 0 for v in row) for row in matrix)


def build_window_shadow(*, state: OnlineWindowState, p0_weight: float, p1_reservation_weight: float, p2_hint_weight: float) -> dict[str, Any]:
    n = len(state.ep_group_ranks)
    p0_actual = _observation_matrix(state.p0_observation, n=n)
    p1_actual = _observation_matrix(state.p1_observation, n=n)
    p0 = p0_actual if _nonzero(p0_actual) else _prepared_matrix(state, phase="p0_dispatch", n=n)
    p1 = p1_actual if _nonzero(p1_actual) else _prepared_matrix(state, phase="p1_return", n=n)
    p2 = _prepared_matrix(state, phase="p2_next_dispatch_forecast", n=n)
    information_mode = "p0_p1_p2" if _nonzero(p2) else "p0_p1" if _nonzero(p1) else "p0_only"
    request = build_window_planning_request(
        identity=PlanningIdentity(
            request_id=f"shadow:{state.window_key}",
            run_id="runtime-shadow",
            window_id=str(state.window_key),
            source_layer_id=str(state.layer_id),
            target_layer_id=str(state.layer_id),
        ),
        p0_dispatch_rows=p0,
        p1_return_rows=p1,
        p2_hint_rows=p2,
        predictor_id="prepared_window" if _nonzero(p2) else "zero",
        confidence=1.0 if _nonzero(p2) else 0.0,
        topology=PlanningTopology(world_size=n, full_duplex=True),
        constraints=PlanningConstraints(bucket_rows=0, max_waves=256, expert_compute_delay=0.0, phase_release_model="rank_local"),
        weights=PlanningWeights(p0_weight=float(p0_weight), p1_weight=float(p1_reservation_weight), p2_weight=float(p2_hint_weight)),
        information_mode=information_mode,
        planning_track="runtime_lookahead",
        p2_semantics="advisory_hint",
    )
    plan = PlannerRegistry.create("current:p012:joint:event:rscf", usage="runtime").plan(request)
    completed = set(int(r) for r in state.release_state.p0_dispatch_completed_ranks)
    executable_prefix = []
    first_executable_wave = None
    for wave in plan.waves:
        edges=[]; blocked=[]; forecast=[]; ready=[]
        for flow in wave.flows:
            item={"flow_id":flow.flow_id,"phase":flow.phase,"src_rank":flow.src_rank,"dst_rank":flow.dst_rank,"row_count":flow.row_count}
            edges.append(item)
            if str(flow.phase).lower() in {"p2","p2_next_dispatch","p2_next_dispatch_forecast"}:
                forecast.append(flow.flow_id)
            elif str(flow.phase).lower() in {"p1","p1_return"} and int(flow.src_rank) not in completed:
                blocked.append(flow.flow_id)
            else:
                ready.append(flow.flow_id)
        payload={
            "wave_id":int(wave.wave_id),"selected_flow_ids":[f.flow_id for f in wave.flows],"selected_edges":edges,
            "ready_selected_flow_ids":ready,"blocked_selected_flow_ids":blocked,"forecast_selected_flow_ids":forecast,
            "duration":float(wave.estimated_duration),"fully_executable_now":not blocked and not forecast,
        }
        executable_prefix.append(payload)
        if first_executable_wave is None and payload["fully_executable_now"]:
            first_executable_wave=payload
    return {
        "ts_us":int(time.time()*1e6),"window_key":state.window_key,"layer_name":state.layer_name,"layer_id":state.layer_id,
        "execution_capability_required":"joint_window_async_p2p","prepared_plan_bound":state.prepared_plan_binding is not None,
        "information_mode":information_mode,"shadow_policy_name":plan.planner_id,"shadow_plan_hash":plan.semantic_digest(),
        "shadow_wave_count":len(plan.waves),"shadow_makespan":float(sum(w.estimated_duration for w in plan.waves)),
        "p0_dispatch_matrix":[list(r) for r in p0],"p1_return_matrix":[list(r) for r in p1],"p2_next_dispatch_forecast_matrix":[list(r) for r in p2],
        "first_executable_wave":first_executable_wave,"executable_prefix":executable_prefix,
        "contains_blocked_future_waves":any(x["blocked_selected_flow_ids"] for x in executable_prefix),
        "contains_forecast_only_waves":any(x["forecast_selected_flow_ids"] for x in executable_prefix),
    }


def executable_now(snapshot: dict[str, Any]) -> list[str]:
    first=snapshot.get("first_executable_wave")
    return [] if not isinstance(first,dict) else list(first.get("ready_selected_flow_ids",[]) or [])


def classify_flow(flow) -> str:
    if str(flow.phase).lower().startswith("p2"): return "forecast_only"
    if str(flow.release_state)=="blocked": return "blocked"
    return "ready" if bool(flow.is_executable) else "unknown"

__all__=["build_window_shadow","classify_flow","executable_now"]
