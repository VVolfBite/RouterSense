"""Legacy compatibility bridge for release-state updates."""

from __future__ import annotations

from rs.runtime.online.megatron_ep.planning.window_release_service import record_release_event

__all__ = ["record_release_event"]
