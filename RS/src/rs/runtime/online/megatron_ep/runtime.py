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

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.control.shadow_policy.joint_shadow import JointShadowP0P1Policy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_order import NativeOrderPolicy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_passthrough_identity import NativePassthroughIdentityPolicy
from rs.runtime.online.megatron_ep.observation import PolicyRuntimeRecord
from rs.planning.runtime_compat import ResolvedAlgorithmId, resolve_algorithm_id, resolve_phase_policy


class UnsupportedSchedulerMode(ValueError):
    pass


class SelectedLayerStop(RuntimeError):
    pass


_SPECIAL_SCHEDULER_MODES = {
    "disabled",
    "native_order",
    "joint_shadow_p0p1",
    "native_passthrough_identity",
}


def resolve_online_policy_config(config: RouterSenseInjectionConfig) -> ResolvedAlgorithmId | None:
    requested_name = str(config.policy or "").strip()
    if not requested_name:
        scheduler_mode = str(config.scheduler_mode or "").strip()
        if scheduler_mode in _SPECIAL_SCHEDULER_MODES:
            return None
        requested_name = scheduler_mode
    if not requested_name or requested_name == "disabled":
        return None
    try:
        resolved = resolve_algorithm_id(requested_name)
    except ValueError as exc:
        raise UnsupportedSchedulerMode(f"Unsupported scheduler_mode={requested_name!r}") from exc
    if not resolved.spec.online_eligible:
        raise UnsupportedSchedulerMode(
            f"Algorithm {requested_name!r} resolves to {resolved.canonical_name!r}, which is not online-eligible"
        )
    return resolved


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
        if scheduler_mode not in _SPECIAL_SCHEDULER_MODES:
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


def resolve_configured_policy(config: RouterSenseInjectionConfig):
    resolved = resolve_online_policy_config(config)
    if resolved is not None:
        return resolve_phase_policy(
            policy_name=str(resolved.builder_key),
            bucket_rows=config.bucket_rows,
            p0_weight=config.p0_weight,
            p1_reservation_weight=config.p1_reservation_weight,
            p2_hint_weight=config.p2_hint_weight,
            p2_hint_artifact=config.p2_hint_artifact,
        )
    if config.scheduler_mode == "native_passthrough_identity":
        return NativePassthroughIdentityPolicy()
    if config.scheduler_mode == "native_order":
        return NativeOrderPolicy()
    if config.scheduler_mode == "joint_shadow_p0p1":
        return JointShadowP0P1Policy()
    raise UnsupportedSchedulerMode(f"Unsupported scheduler_mode={config.scheduler_mode!r}")


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
