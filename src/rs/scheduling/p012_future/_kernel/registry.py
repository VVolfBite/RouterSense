from __future__ import annotations

from typing import Any

from .axes import CORES, ENGINES, SCOPES, PlannerAxes, axes_from_legacy
from .families import build_scoped_planner


def build_planner(
    branch: str | None = None,
    family: str | None = None,
    *,
    scope: str | None = None,
    engine: str | None = None,
    axes: PlannerAxes | None = None,
    **kwargs: Any,
):
    """Build a P012 kernel planner from explicit axes or a legacy branch.

    Legacy compatibility:
      ``local`` -> Local/Event
      ``event`` -> Joint/Event
      ``global`` -> Joint/Global
    """
    if axes is not None:
        if family is not None and str(family).lower() != axes.core:
            raise ValueError("family disagrees with axes.core")
        scope = axes.scope
        engine = axes.engine
        family = axes.core
    elif scope is None or engine is None:
        if branch is None or family is None:
            raise ValueError("provide axes, explicit scope/engine/family, or legacy branch/family")
        legacy_axes = axes_from_legacy(prefix="p012", branch=branch, core=family)
        scope = legacy_axes.scope
        engine = legacy_axes.engine
        family = legacy_axes.core
    if family is None:
        raise ValueError("family is required")
    return build_scoped_planner(scope=str(scope), engine=str(engine), family=str(family), **kwargs)


def planner_matrix() -> tuple[str, ...]:
    return tuple(
        f"current:p012:{scope}:{engine}:{core}"
        for scope in SCOPES
        for engine in ENGINES
        for core in CORES
    )


__all__ = ["build_planner", "planner_matrix"]
