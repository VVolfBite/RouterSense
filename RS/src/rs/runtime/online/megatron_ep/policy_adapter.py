"""Policy registry facade for the formal online runtime path."""

from __future__ import annotations

from rs.scheduling.policy.registry import resolve_phase_policy, supported_phase_policies

__all__ = ["resolve_phase_policy", "supported_phase_policies"]
