from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .catalog import (
    AlgorithmSpec,
    deployable_algorithm_specs,
    reference_algorithm_specs,
    resolve_algorithm_id,
)


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
        return {
            "canonical_name": self.canonical_name,
            "internal_policy_name": self.internal_policy_name,
            "aliases": self.aliases,
            "deployable_common_core": self.deployable_common_core,
            "reference_only": self.reference_only,
            "not_online_deployable": self.not_online_deployable,
            "family": self.family,
            "notes": self.notes,
        }


def _descriptor(spec: AlgorithmSpec) -> PublicPolicyDescriptor:
    return PublicPolicyDescriptor(
        canonical_name=spec.canonical_id,
        internal_policy_name=spec.builder_key,
        aliases=tuple(spec.aliases + spec.deprecated_aliases),
        deployable_common_core=bool(spec.deployable),
        reference_only=bool(spec.reference_only),
        not_online_deployable=not bool(spec.online_eligible),
        family=str(spec.family),
        notes=str(spec.notes),
    )


PUBLIC_POLICIES: tuple[PublicPolicyDescriptor, ...] = tuple(
    _descriptor(spec) for spec in (*deployable_algorithm_specs(), *reference_algorithm_specs())
)


def resolve_public_policy_name(name: str) -> PublicPolicyDescriptor:
    return _descriptor(resolve_algorithm_id(name).spec)


def deployable_policies() -> tuple[PublicPolicyDescriptor, ...]:
    return tuple(_descriptor(item) for item in deployable_algorithm_specs())


def reference_policies() -> tuple[PublicPolicyDescriptor, ...]:
    return tuple(_descriptor(item) for item in reference_algorithm_specs())


__all__ = [
    "PUBLIC_POLICIES",
    "PublicPolicyDescriptor",
    "deployable_policies",
    "reference_policies",
    "resolve_public_policy_name",
]
