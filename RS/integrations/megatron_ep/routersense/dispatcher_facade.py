from __future__ import annotations

from rs.runtime.online.megatron_ep._facade import (
    RouterSenseDispatcherFacade,
    SelectedLayerStop,
    UnsupportedSchedulerMode,
    resolve_configured_policy,
)
from rs.runtime.online.megatron_ep._lifecycle import RouterSenseInjectionRuntime
from rs.runtime.online.megatron_ep._observation import PolicyRuntimeRecord


def attach_dispatch_facade(*args, **kwargs):
    from integrations.megatron_ep.native_runtime import attach_dispatch_facade as _attach_dispatch_facade

    return _attach_dispatch_facade(*args, **kwargs)


__all__ = [
    "PolicyRuntimeRecord",
    "RouterSenseDispatcherFacade",
    "RouterSenseInjectionRuntime",
    "SelectedLayerStop",
    "UnsupportedSchedulerMode",
    "attach_dispatch_facade",
    "resolve_configured_policy",
]
