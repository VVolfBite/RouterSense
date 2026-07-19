from __future__ import annotations

"""Single formal extension authority for RouterSense deployable planners.

This module is the only extension surface patched into ``rs.planning.registry``.
The algorithm implementations remain in their focused modules, but P012,
Future-P012 and Safe(Joint, Local) no longer patch the canonical registry through
separate branches.
"""

from collections.abc import Callable, Mapping
from typing import Any

from .api import PlannerSpec
from .p012_future import create_p012_planner, merge_p012_registry_specs, p012_planner_specs
from .safe_wrapper import SafePlannerConfig, SafePlannerWrapper

SAFE_PLANNER_ID = "safe_pair"
_SAFE_ALIASES = (
    "safe_pair_selector",
    "safe-planner",
    # Compatibility only. New configs must use safe_pair.
    "safe-U",
    "runtime_safe_u",
    "barrier_criticality_runtime_safe",
    "RS_safe_barrier_criticality",
)

_DEFAULT_RSCF_SAFE_ALIASES = {
    "safe_pair_selector",
    "runtime_safe_u",
    "barrier_criticality_runtime_safe",
    "RS_safe_barrier_criticality",
    "safe-U",
}


def safe_planner_specs() -> tuple[PlannerSpec, ...]:
    return (
        PlannerSpec(
            planner_id=SAFE_PLANNER_ID,
            planner_family="joint",
            deployable=True,
            reference_only=False,
            requires_prediction=False,
            exact=False,
            historical_aliases=_SAFE_ALIASES,
        ),
    )


def planner_specs() -> tuple[PlannerSpec, ...]:
    return (*p012_planner_specs(), *safe_planner_specs())


def merge_planner_specs(existing_specs: tuple[PlannerSpec, ...]) -> tuple[PlannerSpec, ...]:
    """Merge extensions without creating a second ID/alias authority."""
    rows = merge_p012_registry_specs(tuple(existing_specs))
    occupied = {
        name
        for spec in rows
        for name in (str(spec.planner_id), *(str(alias) for alias in spec.historical_aliases))
    }
    safe = safe_planner_specs()[0]
    if not ({safe.planner_id, *safe.historical_aliases} & occupied):
        rows = (*rows, safe)
    return tuple(rows)


def resolves_safe_planner(planner_id: str) -> bool:
    return str(planner_id) in {SAFE_PLANNER_ID, *_SAFE_ALIASES}


def resolves_planner(planner_id: str) -> bool:
    name = str(planner_id)
    return resolves_safe_planner(name) or any(
        name == spec.planner_id or name in spec.historical_aliases
        for spec in p012_planner_specs()
    )


def _mapping(value: object | None, *, name: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def create_safe_planner(
    config: Mapping[str, object] | None,
    *,
    child_factory: Callable[[str, Mapping[str, object] | None], object],
) -> SafePlannerWrapper:
    values = dict(config or {})
    joint_id = str(values.pop("joint_planner_id", ""))
    local_id = str(values.pop("local_planner_id", ""))
    if not joint_id or not local_id:
        raise ValueError("safe_pair requires joint_planner_id and local_planner_id")
    if resolves_safe_planner(joint_id) or resolves_safe_planner(local_id):
        raise ValueError("nested safe_pair planners are forbidden")

    child_config = _mapping(values.pop("child_config", None), name="child_config")
    joint_config = _mapping(values.pop("joint_config", None), name="joint_config")
    local_config = _mapping(values.pop("local_config", None), name="local_config")
    if child_config is not None and (joint_config is not None or local_config is not None):
        raise ValueError("use child_config or equal joint/local configs, not both")
    if child_config is None:
        if (joint_config is None) != (local_config is None):
            raise ValueError("joint_config and local_config must be provided together")
        if joint_config is not None and dict(joint_config) != dict(local_config or {}):
            raise ValueError("paired planners must share identical child configuration")
        child_config = joint_config

    accepted = set(SafePlannerConfig.__dataclass_fields__)
    safe_values = {key: value for key, value in values.items() if key in accepted}
    wrapper_id = str(values.get("wrapper_id", SAFE_PLANNER_ID))
    return SafePlannerWrapper(
        joint_planner=child_factory(joint_id, child_config),
        local_planner=child_factory(local_id, child_config),
        config=SafePlannerConfig(**safe_values),
        wrapper_id=wrapper_id,
    )


def create_planner(
    planner_id: str,
    config: Mapping[str, object] | None,
    *,
    child_factory: Callable[[str, Mapping[str, object] | None], object],
) -> Any:
    if resolves_safe_planner(planner_id):
        values = dict(config or {})
        if str(planner_id) in _DEFAULT_RSCF_SAFE_ALIASES:
            values.setdefault("joint_planner_id", "rscf_joint")
            values.setdefault("local_planner_id", "rscf_local")
            values.setdefault("wrapper_id", str(planner_id))
        return create_safe_planner(values, child_factory=child_factory)
    return create_p012_planner(planner_id, config)


__all__ = [
    "SAFE_PLANNER_ID",
    "create_planner",
    "create_safe_planner",
    "merge_planner_specs",
    "planner_specs",
    "resolves_planner",
    "resolves_safe_planner",
    "safe_planner_specs",
]
