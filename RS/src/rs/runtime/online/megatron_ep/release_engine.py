"""Release-state updates for online window shadow scheduling."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from .window_state import OnlineWindowState


def record_release_event(
    *,
    state: OnlineWindowState,
    event: str,
    rank: int,
    layer_name: str,
) -> tuple[OnlineWindowState, dict[str, Any]]:
    release_state = state.release_state
    if event == "p0_dispatch_completed":
        updated = replace(
            release_state,
            p0_dispatch_completed_ranks=_append_unique(release_state.p0_dispatch_completed_ranks, rank),
        )
    elif event == "p1_return_materialized":
        updated = replace(
            release_state,
            p1_return_materialized_ranks=_append_unique(release_state.p1_return_materialized_ranks, rank),
        )
    elif event == "p1_return_completed":
        updated = replace(
            release_state,
            p1_return_completed_ranks=_append_unique(release_state.p1_return_completed_ranks, rank),
        )
    else:  # pragma: no cover
        raise ValueError(f"unsupported release event {event!r}")
    next_state = replace(state, release_state=updated)
    record = {
        "ts_us": int(time.time() * 1e6),
        "window_key": state.window_key,
        "layer_name": layer_name,
        "layer_id": state.layer_id,
        "event": event,
        "rank": int(rank),
        "release_state": updated.to_dict(),
    }
    return next_state, record


def _append_unique(values: tuple[int, ...], value: int) -> tuple[int, ...]:
    return values if int(value) in values else (*values, int(value))


__all__ = ["record_release_event"]
