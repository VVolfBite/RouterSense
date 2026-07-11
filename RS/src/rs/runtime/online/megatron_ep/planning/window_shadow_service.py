from __future__ import annotations

from typing import Any

from rs.runtime.online.megatron_ep.pending_window import (
    build_pending_window_shadow,
    build_window_state,
    record_release_event,
)


def build_window_state_record(
    *,
    layer_name: str,
    ep_group_ranks: tuple[int, ...],
    local_rank: int,
    p0_observation: Any | None,
    p1_observation: Any | None,
    prepared_plan: Any | None,
    prepared_plan_binding: Any | None,
    release_state: Any,
) -> tuple[Any, dict[str, Any]]:
    state = build_window_state(
        layer_name=layer_name,
        ep_group_ranks=ep_group_ranks,
        local_rank=local_rank,
        p0_observation=p0_observation,
        p1_observation=p1_observation,
        prepared_plan=prepared_plan,
        prepared_plan_binding=prepared_plan_binding,
        release_state=release_state,
    )
    return state, state.to_record()


def advance_window_release(
    *,
    state: Any,
    event: str,
    rank: int,
    layer_name: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    next_state, release_record = record_release_event(
        state=state,
        event=event,
        rank=rank,
        layer_name=layer_name,
    )
    return next_state, release_record, next_state.to_record()


def maybe_build_window_shadow(
    *,
    enabled: bool,
    state: Any,
    p0_weight: float,
    p1_reservation_weight: float,
    p2_hint_weight: float,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    shadow = build_pending_window_shadow(
        state=state,
        p0_weight=float(p0_weight),
        p1_reservation_weight=float(p1_reservation_weight),
        p2_hint_weight=float(p2_hint_weight),
    )
    first_wave = shadow.get("first_executable_wave") or {}
    shadow.setdefault("shadow_first_wave_flow_ids", list(first_wave.get("selected_flow_ids", []) or []))
    shadow.setdefault("shadow_first_wave_edges", list(first_wave.get("selected_edges", []) or []))
    return shadow


__all__ = [
    "advance_window_release",
    "build_window_state_record",
    "maybe_build_window_shadow",
]
