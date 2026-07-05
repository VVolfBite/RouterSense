from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.control.shadow_policy.joint_shadow import JointShadowP0P1Policy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_order import NativeOrderPolicy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_passthrough_identity import NativePassthroughIdentityPolicy
from rs.scheduling.registry import resolve_phase_policy, supported_phase_policies


class UnsupportedSchedulerMode(ValueError):
    pass


class SelectedLayerStop(RuntimeError):
    pass


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
        supported_scheduler_modes = {
            "disabled",
            "native_order",
            "joint_shadow_p0p1",
            "native_passthrough_identity",
            *supported_phase_policies(),
        }
        if config.scheduler_mode not in supported_scheduler_modes:
            raise UnsupportedSchedulerMode(
                "Unsupported scheduler_mode="
                f"{config.scheduler_mode!r}; only {sorted(supported_scheduler_modes)!r} are implemented"
            )
        if config.future_hint_mode != "none":
            raise UnsupportedSchedulerMode(
                f"Unsupported future_hint_mode={config.future_hint_mode!r}; only 'none' is implemented"
            )
        if config.control_mode not in {"default_continue", "sync_before_phase"}:
            raise UnsupportedSchedulerMode(
                f"Unsupported control_mode={config.control_mode!r}; only 'default_continue' and 'sync_before_phase' are implemented"
            )
        return cls(
            native_dispatcher=native_dispatcher,
            scheduler_mode=config.scheduler_mode,
            future_hint_mode=config.future_hint_mode,
            control_mode=config.control_mode,
        )


def resolve_configured_policy(config: RouterSenseInjectionConfig):
    phase_policy_name = str(config.policy) if config.policy else ""
    if not phase_policy_name and config.scheduler_mode in set(supported_phase_policies()):
        phase_policy_name = str(config.scheduler_mode)
    if phase_policy_name:
        return resolve_phase_policy(
            policy_name=phase_policy_name,
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
    "RouterSenseDispatcherFacade",
    "SelectedLayerStop",
    "UnsupportedSchedulerMode",
    "resolve_configured_policy",
]
