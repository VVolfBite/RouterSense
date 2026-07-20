"""Compatibility facade for split lifecycle mixins."""

from __future__ import annotations

from .planning_state import LifecyclePlanningStateMixin
from .planning_joint import LifecycleJointPlanningMixin
from .planning_async import LifecycleAsyncPlanningMixin


class LifecyclePlanningMixin(LifecycleJointPlanningMixin, LifecycleAsyncPlanningMixin, LifecyclePlanningStateMixin):
    """Composite mixin retaining the historical lifecycle import boundary."""

    pass


__all__ = ["LifecyclePlanningMixin"]
