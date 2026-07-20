"""Compatibility boundary between formal planners and phase materialization.

Only deployable phase-local baselines are resolved through the scheduling
registry. ``prepared_priority`` is an internal materializer for an already
selected Current/Future logical plan; it is deliberately absent from the
public algorithm catalog.
"""
from __future__ import annotations

from rs.scheduling.runtime_bridge.prepared_priority import PreparedPriorityPhasePolicy
from rs.scheduling.catalog import ResolvedAlgorithmId, resolve_algorithm_id
from rs.scheduling.registry import resolve_phase_policy as _resolve_formal_phase_policy
from rs.scheduling.registry import supported_phase_policies

_INTERNAL_MATERIALIZER_ID = "prepared_priority"


def resolve_phase_policy(
    policy_name: str,
    *,
    bucket_rows: int,
    p0_weight: float = 1.0,
    p1_reservation_weight: float = 1.0,
    p2_hint_weight: float = 1.0,
    **kwargs,
):
    name = str(policy_name or "").strip()
    if name == _INTERNAL_MATERIALIZER_ID:
        return PreparedPriorityPhasePolicy(
            bucket_rows=int(bucket_rows),
            p0_weight=float(p0_weight),
            p1_reservation_weight=float(p1_reservation_weight),
            p2_hint_weight=float(p2_hint_weight),
        )
    return _resolve_formal_phase_policy(policy_name=name, bucket_rows=int(bucket_rows), **kwargs)


def supported_runtime_materializers() -> tuple[str, ...]:
    return (_INTERNAL_MATERIALIZER_ID,)


__all__ = [
    "ResolvedAlgorithmId",
    "resolve_algorithm_id",
    "resolve_phase_policy",
    "supported_phase_policies",
    "supported_runtime_materializers",
]
