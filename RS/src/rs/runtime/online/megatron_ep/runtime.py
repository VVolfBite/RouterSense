"""Formal runtime facade exports for Megatron EP."""

from __future__ import annotations

from integrations.megatron_ep.native_runtime import attach_dispatch_facade

from ._facade import (
    RouterSenseDispatcherFacade,
    SelectedLayerStop,
    UnsupportedSchedulerMode,
    resolve_configured_policy,
)
from ._lifecycle import RouterSenseInjectionRuntime
from ._observation import PolicyRuntimeRecord

__all__ = [
    "RouterSenseDispatcherFacade",
    "PolicyRuntimeRecord",
    "RouterSenseInjectionRuntime",
    "SelectedLayerStop",
    "UnsupportedSchedulerMode",
    "attach_dispatch_facade",
    "resolve_configured_policy",
]
