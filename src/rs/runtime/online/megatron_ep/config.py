from __future__ import annotations

from rs.planning.runtime_compat import ResolvedAlgorithmId, resolve_algorithm_id, resolve_phase_policy
from rs.runtime.online.megatron_ep.contracts import RouterSenseInjectionConfig
from rs.runtime.online.megatron_ep.control.shadow_policy.joint_shadow import JointShadowP0P1Policy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_order import NativeOrderPolicy
from rs.runtime.online.megatron_ep.control.shadow_policy.native_passthrough_identity import NativePassthroughIdentityPolicy
from rs.runtime.online.megatron_ep.public_types import UnsupportedSchedulerMode


SPECIAL_SCHEDULER_MODES = {
    "disabled",
    "native_order",
    "joint_shadow_p0p1",
    "native_passthrough_identity",
}


def resolve_online_policy_config(config: RouterSenseInjectionConfig) -> ResolvedAlgorithmId | None:
    requested_name = str(config.policy or "").strip()
    if not requested_name:
        scheduler_mode = str(config.scheduler_mode or "").strip()
        if scheduler_mode in SPECIAL_SCHEDULER_MODES:
            return None
        requested_name = scheduler_mode
    if not requested_name or requested_name == "disabled":
        return None
    if requested_name == "prepared_priority":
        if not str(config.planner_id or "").strip():
            raise UnsupportedSchedulerMode("prepared_priority requires an explicit planner_id")
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
