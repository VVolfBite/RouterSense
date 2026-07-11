from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PublicPolicyDescriptor:
    canonical_name: str
    internal_policy_name: str
    aliases: tuple[str, ...]
    deployable_common_core: bool
    reference_only: bool
    not_online_deployable: bool
    family: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PUBLIC_POLICIES: tuple[PublicPolicyDescriptor, ...] = (
    PublicPolicyDescriptor(
        canonical_name="phase_barrier_fifo",
        internal_policy_name="phase_barrier_fifo",
        aliases=(),
        deployable_common_core=True,
        reference_only=False,
        not_online_deployable=False,
        family="deployable_bucket",
    ),
    PublicPolicyDescriptor(
        canonical_name="greedy_ready_set",
        internal_policy_name="greedy_ready_set",
        aliases=(),
        deployable_common_core=True,
        reference_only=False,
        not_online_deployable=False,
        family="deployable_bucket",
    ),
    PublicPolicyDescriptor(
        canonical_name="islip_round_robin",
        internal_policy_name="islip_round_robin",
        aliases=(),
        deployable_common_core=True,
        reference_only=False,
        not_online_deployable=False,
        family="deployable_bucket",
    ),
    PublicPolicyDescriptor(
        canonical_name="birkhoff_bucket_phase_local",
        internal_policy_name="birkhoff_phase_local",
        aliases=("birkhoff_phase_local", "B_birkhoff", "B_birkhoff_wave"),
        deployable_common_core=True,
        reference_only=False,
        not_online_deployable=False,
        family="deployable_bucket",
        notes="formal deployable Birkhoff bucket baseline",
    ),
    PublicPolicyDescriptor(
        canonical_name="paired_b_barrier_criticality",
        internal_policy_name="B_barrier_criticality_matching",
        aliases=(),
        deployable_common_core=True,
        reference_only=False,
        not_online_deployable=False,
        family="deployable_bucket",
    ),
    PublicPolicyDescriptor(
        canonical_name="joint_u_barrier_criticality",
        internal_policy_name="U_barrier_criticality_global_matching",
        aliases=(),
        deployable_common_core=True,
        reference_only=False,
        not_online_deployable=False,
        family="deployable_bucket",
    ),
    PublicPolicyDescriptor(
        canonical_name="runtime_safe_u",
        internal_policy_name="RS_safe_barrier_criticality",
        aliases=("safe-U",),
        deployable_common_core=True,
        reference_only=False,
        not_online_deployable=False,
        family="deployable_bucket",
        notes="runtime-safe-U, not posthoc best-of-U-and-B",
    ),
    PublicPolicyDescriptor(
        canonical_name="birkhoff_fluid_reference",
        internal_policy_name="birkhoff_von_neumann_fluid",
        aliases=("birkhoff_von_neumann_fluid",),
        deployable_common_core=False,
        reference_only=True,
        not_online_deployable=True,
        family="reference_only",
    ),
    PublicPolicyDescriptor(
        canonical_name="exact_small_instance_oracle",
        internal_policy_name="exact_small_instance_reference",
        aliases=("O_local", "O_joint"),
        deployable_common_core=False,
        reference_only=True,
        not_online_deployable=True,
        family="reference_only",
    ),
    PublicPolicyDescriptor(
        canonical_name="posthoc_best_of_u_and_b",
        internal_policy_name="posthoc_best_of_u_and_b",
        aliases=("posthoc_best_of_U_and_B",),
        deployable_common_core=False,
        reference_only=True,
        not_online_deployable=True,
        family="reference_only",
        notes="reference only upper bound for safe selection",
    ),
)


def resolve_public_policy_name(name: str) -> PublicPolicyDescriptor:
    normalized = str(name)
    for descriptor in PUBLIC_POLICIES:
        if normalized == descriptor.canonical_name or normalized == descriptor.internal_policy_name or normalized in descriptor.aliases:
            return descriptor
    raise ValueError(f"unknown public policy name {name!r}")


def deployable_policies() -> tuple[PublicPolicyDescriptor, ...]:
    return tuple(item for item in PUBLIC_POLICIES if not item.reference_only)


def reference_policies() -> tuple[PublicPolicyDescriptor, ...]:
    return tuple(item for item in PUBLIC_POLICIES if item.reference_only)


__all__ = [
    "PUBLIC_POLICIES",
    "PublicPolicyDescriptor",
    "deployable_policies",
    "reference_policies",
    "resolve_public_policy_name",
]
