"""Megatron EP 在线运行时的公共 API。

这个文件主要给外部暴露：
- RouterSenseDispatcherFacade
- SelectedLayerStop / UnsupportedSchedulerMode
- 少量 shadow policy 选择入口
它不负责生命周期编排，更多是供 host 和实验脚本调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rs.runtime.online.megatron_ep.config import SPECIAL_SCHEDULER_MODES, resolve_configured_policy, resolve_online_policy_config
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.observation import PolicyRuntimeRecord
from rs.runtime.online.megatron_ep.public_types import SelectedLayerStop, UnsupportedSchedulerMode


@dataclass
class RouterSenseDispatcherFacade:
    native_dispatcher: Callable[..., Any]
    facade_mode: str = "no_op_native_passthrough"
    scheduler_mode: str = "disabled"
    future_hint_mode: str = "none"
    control_mode: str = "default_continue"

    def dispatch(self, *args: Any, **kwargs: Any) -> Any:
        return self.native_dispatcher(*args, **kwargs)

    @classmethod
    def from_config(
        cls,
        *,
        native_dispatcher: Callable[..., Any],
        config: RouterSenseInjectionConfig,
    ) -> "RouterSenseDispatcherFacade":
        scheduler_mode = str(config.scheduler_mode)
        if scheduler_mode not in SPECIAL_SCHEDULER_MODES:
            resolve_online_policy_config(config)
        if config.future_hint_mode != "none":
            raise UnsupportedSchedulerMode(
                f"Unsupported future_hint_mode={config.future_hint_mode!r}; only 'none' is implemented"
            )
        allowed_control_modes = {"default_continue", "sync_before_phase"}
        if config.scheduler_mode == "disabled" and str(config.policy or "disabled") in {"", "disabled"}:
            allowed_control_modes.add("none")
        if config.control_mode not in allowed_control_modes:
            raise UnsupportedSchedulerMode(
                f"Unsupported control_mode={config.control_mode!r}; only {sorted(allowed_control_modes)!r} are implemented"
            )
        return cls(
            native_dispatcher=native_dispatcher,
            scheduler_mode=config.scheduler_mode,
            future_hint_mode=config.future_hint_mode,
            control_mode=config.control_mode,
        )
__all__ = [
    "PolicyRuntimeRecord",
    "RouterSenseDispatcherFacade",
    "RouterSenseInjectionRuntime",
    "SelectedLayerStop",
    "UnsupportedSchedulerMode",
    "resolve_configured_policy",
]


def __getattr__(name: str):
    if name == "RouterSenseInjectionRuntime":
        from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime

        return RouterSenseInjectionRuntime
    raise AttributeError(name)
