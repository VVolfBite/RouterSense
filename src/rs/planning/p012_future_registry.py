"""Deprecated compatibility shim. Use :mod:`rs.planning.asset_registry`."""
from .p012_future import create_p012_planner, merge_p012_registry_specs, p012_planner_specs

def resolves_p012_planner(planner_id: str) -> bool:
    name = str(planner_id)
    return any(name == spec.planner_id or name in spec.historical_aliases for spec in p012_planner_specs())

def create_registered_p012(planner_id: str, config=None):
    return create_p012_planner(planner_id, config)

__all__ = ["create_registered_p012", "p012_planner_specs", "merge_p012_registry_specs", "resolves_p012_planner"]
