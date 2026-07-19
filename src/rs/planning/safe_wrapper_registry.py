"""Deprecated compatibility shim. Use :mod:`rs.planning.asset_registry`."""
from .api import PlannerSpec
from .asset_registry import (
    SAFE_PLANNER_ID,
    create_safe_planner as create_registered_safe,
    resolves_safe_planner,
    safe_planner_specs,
)

def merge_safe_registry_specs(existing_specs: tuple[PlannerSpec, ...]) -> tuple[PlannerSpec, ...]:
    existing = tuple(existing_specs)
    occupied = {
        name for spec in existing
        for name in (str(spec.planner_id), *(str(a) for a in spec.historical_aliases))
    }
    spec = safe_planner_specs()[0]
    return existing if ({spec.planner_id, *spec.historical_aliases} & occupied) else (*existing, spec)

__all__ = [
    "SAFE_PLANNER_ID", "create_registered_safe", "merge_safe_registry_specs",
    "resolves_safe_planner", "safe_planner_specs",
]
