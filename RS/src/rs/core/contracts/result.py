"""Result-envelope contracts for RouteSense experiments and runtime outputs."""

from __future__ import annotations

from rs.contracts.result import (
    LEGACY_TRACE_REPLAY_PIPELINE,
    OFFLINE_PIPELINE,
    ONLINE_PIPELINE,
    RunIdentity,
    build_result_envelope,
)

__all__ = [
    "LEGACY_TRACE_REPLAY_PIPELINE",
    "OFFLINE_PIPELINE",
    "ONLINE_PIPELINE",
    "RunIdentity",
    "build_result_envelope",
]
