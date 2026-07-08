"""Backward-compatible wrapper for pending-window shadow scheduling."""

from __future__ import annotations

from typing import Any

from .multiphase_pending_window import build_pending_window_shadow
from .window_state import OnlineWindowState


def build_joint_shadow_snapshot(
    *,
    state: OnlineWindowState,
    p0_weight: float,
    p1_reservation_weight: float,
    p2_hint_weight: float,
) -> dict[str, Any]:
    snapshot = build_pending_window_shadow(
        state=state,
        p0_weight=p0_weight,
        p1_reservation_weight=p1_reservation_weight,
        p2_hint_weight=p2_hint_weight,
    )
    first_wave = snapshot.get("first_executable_wave") or {}
    snapshot.setdefault("shadow_first_wave_flow_ids", list(first_wave.get("selected_flow_ids", []) or []))
    snapshot.setdefault(
        "shadow_first_wave_edges",
        list(first_wave.get("selected_edges", []) or []),
    )
    return snapshot


__all__ = ["build_joint_shadow_snapshot"]
