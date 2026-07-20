"""Compatibility facade for split lifecycle mixins."""

from __future__ import annotations

from .hooks_state import LifecycleHookStateMixin
from .hooks_dispatch import LifecycleDispatchHooksMixin
from .hooks_combine import LifecycleCombineHooksMixin


class LifecycleHooksMixin(LifecycleDispatchHooksMixin, LifecycleCombineHooksMixin, LifecycleHookStateMixin):
    """Composite mixin retaining the historical lifecycle import boundary."""

    pass


__all__ = ["LifecycleHooksMixin"]
