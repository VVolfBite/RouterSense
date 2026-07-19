from __future__ import annotations

"""Stateless planner-factory wrapper for target-layer online planning.

The existing ``TargetLayerPlannerService`` already accepts ``planner_factory``.
This wrapper injects the frozen topology/cost configuration without owning a
queue, store, token, deadline, or execution path.
"""

from collections.abc import Callable, Mapping
from typing import Any, Protocol


class PlannerConfigProvider(Protocol):
    def __call__(self, planner_id: str) -> Mapping[str, object] | None: ...


def make_target_p012_planner_factory(
    *,
    config_provider: PlannerConfigProvider,
    base_factory: Callable[[str, Any | None], Any] | None = None,
) -> Callable[[str, Any | None], Any]:
    """Return the factory signature consumed by ``TargetLayerPlannerService``."""

    def factory(planner_id: str, config: Any | None = None):
        from rs.planning.registry import PlannerRegistry

        delegate = base_factory or PlannerRegistry.create
        supplied = config_provider(str(planner_id))
        if supplied is None:
            return delegate(str(planner_id), config)
        merged: dict[str, object] = {}
        if isinstance(config, Mapping):
            merged.update(dict(config))
        elif config is not None:
            merged.update(vars(config))
        merged.update(dict(supplied))
        return delegate(str(planner_id), merged)

    return factory


__all__ = ["PlannerConfigProvider", "make_target_p012_planner_factory"]
