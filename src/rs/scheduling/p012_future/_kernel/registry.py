from __future__ import annotations

from typing import Any

from .axes import CORES, ENGINES, SCOPES, PlannerAxes
from .families import build_scoped_planner


def build_planner(
    *,
    scope: str | None = None,
    engine: str | None = None,
    family: str | None = None,
    axes: PlannerAxes | None = None,
    **kwargs: Any,
):
    """Build a formal P012 kernel planner from explicit orthogonal axes."""
    if axes is not None:
        if family is not None and str(family).lower() != axes.core:
            raise ValueError("family disagrees with axes.core")
        scope = axes.scope
        engine = axes.engine
        family = axes.core
    if scope is None or engine is None or family is None:
        raise ValueError("scope, engine and family are required")
    return build_scoped_planner(scope=str(scope), engine=str(engine), family=str(family), **kwargs)


def planner_matrix() -> tuple[str, ...]:
    return tuple(
        f"current:p012:{scope}:{engine}:{core}"
        for scope in SCOPES
        for engine in ENGINES
        for core in CORES
    )


__all__ = ["build_planner", "planner_matrix"]
