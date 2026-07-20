"""Canonical formal algorithm catalog.

Only deployable phase-local controls, paper/reference baselines and exact
references live here. RouterSense P01/P012/P0123 planners are registered by the
orthogonal planning registry and are intentionally absent from this catalog.
Retired algorithm aliases are rejected rather than silently translated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AlgorithmSpec:
    canonical_id: str
    display_name: str
    family: str
    scheduling_scope: str
    execution_model: str
    task_granularity: str
    deployable: bool
    reference_only: bool
    online_eligible: bool
    offline_eligible: bool
    supports_p2_hint: bool
    supports_safe_wrapper: bool
    aliases: tuple[str, ...] = ()
    deprecated_aliases: tuple[str, ...] = ()
    builder_key: str = ""
    phase_local_eligible: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedAlgorithmId:
    requested_name: str
    canonical_name: str
    builder_key: str
    is_alias: bool
    is_deprecated: bool
    warning: str | None
    spec: AlgorithmSpec


ALGORITHM_SPECS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        canonical_id="fifo_bucket",
        display_name="FIFO Bucket",
        family="deployable_baseline",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="fifo_bucket",
        phase_local_eligible=True,
    ),
    AlgorithmSpec(
        canonical_id="greedy_bucket",
        display_name="Greedy Ready-Set Bucket",
        family="deployable_baseline",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="greedy_bucket",
        phase_local_eligible=True,
    ),
    AlgorithmSpec(
        canonical_id="birkhoff_bucket_phase_local",
        display_name="Birkhoff Bucket Phase-Local",
        family="deployable_baseline",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="birkhoff_bucket_phase_local",
        phase_local_eligible=True,
    ),
    AlgorithmSpec(
        canonical_id="gmwd_style_reference",
        display_name="GMWD-Style Residual Max-Weight Reference",
        family="paper_reference_baseline",
        scheduling_scope="phase_local_reference",
        execution_model="offline_reference_fluid",
        task_granularity="residual_edge_quantum",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="gmwd_style_reference",
        notes="Residual maximum-weight decomposition core; photonic and compute cost models are omitted.",
    ),
    AlgorithmSpec(
        canonical_id="islip_reference",
        display_name="iSLIP Round-Robin Reference",
        family="paper_reference_baseline",
        scheduling_scope="phase_local_reference",
        execution_model="offline_reference_bucket",
        task_granularity="canonical_bucket",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="islip_reference",
        notes="Literature/control baseline under RouterSense fixed-placement semantics.",
    ),
    AlgorithmSpec(
        canonical_id="fast_stage_reference",
        display_name="FAST Two-Tier Style Reference",
        family="paper_reference_baseline",
        scheduling_scope="phase_local_reference",
        execution_model="offline_reference_bucket",
        task_granularity="canonical_bucket",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="fast_stage_reference",
        notes="Two-tier server-stage core under fixed endpoints; explicit rebalance and redistribution are omitted.",
    ),
    AlgorithmSpec(
        canonical_id="aurora_order_reference",
        display_name="Aurora Fixed-Placement Conflict-Avoiding Style",
        family="paper_reference_baseline",
        scheduling_scope="phase_local_reference",
        execution_model="offline_reference_bucket",
        task_granularity="canonical_bucket",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="aurora_order_reference",
        notes="Exclusive+Homogeneous transmission-order core; placement, colocation and heterogeneity are absent.",
    ),
    AlgorithmSpec(
        canonical_id="power_of_two_reference",
        display_name="Power-of-Two Choices Reference",
        family="diagnostic_reference",
        scheduling_scope="phase_local_reference",
        execution_model="offline_reference_bucket",
        task_granularity="canonical_bucket",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="power_of_two_reference",
    ),
    AlgorithmSpec(
        canonical_id="reverse_bucket_reference",
        display_name="Reverse-Bucket Diagnostic Reference",
        family="diagnostic_reference",
        scheduling_scope="phase_local_reference",
        execution_model="offline_reference_bucket",
        task_granularity="canonical_bucket",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="reverse_bucket_reference",
    ),
    AlgorithmSpec(
        canonical_id="birkhoff_fluid_reference",
        display_name="Birkhoff Fluid Reference",
        family="paper_reference_baseline",
        scheduling_scope="phase_local_reference",
        execution_model="fluid_reference",
        task_granularity="fluid",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="birkhoff_fluid_reference",
    ),
    AlgorithmSpec(
        canonical_id="oracle_local_exact",
        display_name="Oracle Local Exact Bucket-Wave",
        family="oracle",
        scheduling_scope="phase_local_optimal",
        execution_model="exact_reference",
        task_granularity="canonical_remote_edge_bucket",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        builder_key="oracle_local_exact",
    ),
    AlgorithmSpec(
        canonical_id="oracle_joint_exact",
        display_name="Oracle Joint Exact Bucket-Wave",
        family="oracle",
        scheduling_scope="joint_optimal",
        execution_model="exact_reference",
        task_granularity="canonical_remote_edge_bucket",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=False,
        builder_key="oracle_joint_exact",
    ),
)

_BY_CANONICAL = {item.canonical_id: item for item in ALGORITHM_SPECS}


def algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return ALGORITHM_SPECS


def deployable_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return tuple(spec for spec in ALGORITHM_SPECS if spec.deployable)


def reference_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return tuple(spec for spec in ALGORITHM_SPECS if spec.reference_only)


def phase_local_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return tuple(spec for spec in ALGORITHM_SPECS if spec.phase_local_eligible)


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    try:
        return _BY_CANONICAL[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown formal algorithm {name!r}") from exc


def resolve_algorithm_id(requested_name: str) -> ResolvedAlgorithmId:
    normalized = str(requested_name)
    spec = get_algorithm_spec(normalized)
    return ResolvedAlgorithmId(
        requested_name=normalized,
        canonical_name=spec.canonical_id,
        builder_key=spec.builder_key,
        is_alias=False,
        is_deprecated=False,
        warning=None,
        spec=spec,
    )


__all__ = [
    "ALGORITHM_SPECS",
    "AlgorithmSpec",
    "ResolvedAlgorithmId",
    "algorithm_specs",
    "deployable_algorithm_specs",
    "get_algorithm_spec",
    "phase_local_algorithm_specs",
    "reference_algorithm_specs",
    "resolve_algorithm_id",
]
