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
    aliases: tuple[str, ...]
    deprecated_aliases: tuple[str, ...]
    builder_key: str
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
        family="phase_local_baseline",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        aliases=("phase_barrier_fifo",),
        deprecated_aliases=("bucketed_fifo",),
        builder_key="phase_barrier_fifo",
        phase_local_eligible=True,
    ),
    AlgorithmSpec(
        canonical_id="greedy_bucket",
        display_name="Greedy Bucket",
        family="phase_local_baseline",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        aliases=("greedy_ready_set",),
        deprecated_aliases=(),
        builder_key="greedy_ready_set",
        phase_local_eligible=True,
    ),
    AlgorithmSpec(
        canonical_id="islip_bucket",
        display_name="iSLIP Bucket",
        family="phase_local_baseline",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        aliases=("islip_round_robin",),
        deprecated_aliases=(),
        builder_key="islip_round_robin",
        phase_local_eligible=True,
    ),
    AlgorithmSpec(
        canonical_id="birkhoff_bucket_phase_local",
        display_name="Birkhoff Bucket Phase-Local",
        family="phase_local_baseline",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        aliases=("birkhoff_phase_local",),
        deprecated_aliases=(),
        builder_key="birkhoff_phase_local",
        phase_local_eligible=True,
    ),
    AlgorithmSpec(
        canonical_id="greedy_control_local",
        display_name="Greedy Control Local",
        family="paired_family",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=True,
        aliases=('Local(greedy_control)', 'Local(gated_greedy)'),
        deprecated_aliases=('gated_greedy_local', 'B_gated_greedy_maximal'),
        builder_key="greedy_control_local",
        notes="Generic low-cost control; no named-paper reproduction. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="greedy_control_joint",
        display_name="Greedy Control Joint",
        family="paired_family",
        scheduling_scope="joint",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=True,
        aliases=('Joint(greedy_control)', 'Joint(gated_greedy)'),
        deprecated_aliases=('gated_greedy_joint', 'U_gated_greedy_maximal'),
        builder_key="greedy_control_joint",
        notes="Generic low-cost control; no named-paper reproduction. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="gmwd_local",
        display_name="GMWD-style Local",
        family="paired_family",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=True,
        aliases=('Local(gmwd)', 'Local(gated_maxweight)'),
        deprecated_aliases=('gated_maxweight_local', 'B_gated_maxweight_matching'),
        builder_key="gmwd_local",
        notes="Greedy Max-Weight Decomposition core on the residual MoE matrix. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="gmwd_joint",
        display_name="GMWD-style Joint",
        family="paired_family",
        scheduling_scope="joint",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=True,
        aliases=('Joint(gmwd)', 'Joint(gated_maxweight)'),
        deprecated_aliases=('gated_maxweight_joint', 'U_gated_maxweight_matching'),
        builder_key="gmwd_joint",
        notes="Greedy Max-Weight Decomposition core on the residual MoE matrix. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="rsbc_local",
        display_name="RouterSense Barrier Criticality Local",
        family="paired_family",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=True,
        aliases=('Local(rsbc)', 'Local(barrier_criticality)'),
        deprecated_aliases=('barrier_criticality_core_independent', 'B_barrier_criticality_core_independent'),
        builder_key="rsbc_local",
        notes="RouterSense original barrier-criticality and release-gain family. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="rsbc_joint",
        display_name="RouterSense Barrier Criticality Joint",
        family="paired_family",
        scheduling_scope="joint",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=True,
        aliases=('Joint(rsbc)', 'Joint(barrier_criticality)'),
        deprecated_aliases=('barrier_criticality_joint', 'U_barrier_criticality_global_matching'),
        builder_key="rsbc_joint",
        notes="RouterSense original barrier-criticality and release-gain family. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="rscf_local",
        display_name="RouterSense Critical Frontier Local",
        family="paired_family",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=True,
        aliases=("Local(rscf)", "Local(critical_frontier)"),
        deprecated_aliases=(),
        builder_key="rscf_local",
        notes=(
            "RouterSense model-agnostic critical-frontier family. Local and Joint "
            "share one immutable kernel; only visibility and phase coupling differ."
        ),
    ),
    AlgorithmSpec(
        canonical_id="rscf_joint",
        display_name="RouterSense Critical Frontier Joint",
        family="paired_family",
        scheduling_scope="joint",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=True,
        aliases=("Joint(rscf)", "Joint(critical_frontier)"),
        deprecated_aliases=(),
        builder_key="rscf_joint",
        notes=(
            "RouterSense model-agnostic critical-frontier family. Local and Joint "
            "share one immutable kernel; only visibility and phase coupling differ."
        ),
    ),
    AlgorithmSpec(
        canonical_id="rscf_runtime_safe",
        display_name="RouterSense Critical Frontier Runtime Safe",
        family="runtime_safe",
        scheduling_scope="joint_safe_wrapper",
        execution_model="offline_guarded_candidate",
        task_granularity="canonical_bucket",
        deployable=False,
        reference_only=False,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=True,
        aliases=("Safe(rscf)",),
        deprecated_aliases=("RS_safe_rscf",),
        builder_key="RS_safe_rscf",
        notes=(
            "Engineering guard evaluated after the primary RSCF heuristic. It "
            "runs Joint(rscf) and Local(rscf) under the same information and "
            "selects the lower audited makespan; it is not the primary algorithmic contribution."
        ),
    ),
    AlgorithmSpec(
        canonical_id="fast_stage_local",
        display_name="FAST-Stage (single-tier) Local",
        family="paired_family",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=True,
        aliases=('Local(fast_stage)', 'Local(birkhoff_ranked)', 'Local(birkhoff)'),
        deprecated_aliases=('birkhoff_ranked_local',),
        builder_key="fast_stage_local",
        notes="FAST-inspired one-to-one stage ordering; not a full FAST reproduction. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="fast_stage_joint",
        display_name="FAST-Stage (single-tier) Joint",
        family="paired_family",
        scheduling_scope="joint",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=True,
        aliases=('Joint(fast_stage)', 'Joint(birkhoff_ranked)', 'Joint(birkhoff)'),
        deprecated_aliases=('birkhoff_ranked_joint',),
        builder_key="fast_stage_joint",
        notes="FAST-inspired one-to-one stage ordering; not a full FAST reproduction. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="aurora_order_local",
        display_name="Aurora-Order (fixed placement) Local",
        family="paired_family",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=True,
        aliases=('Local(aurora_order)', 'Local(aurora)'),
        deprecated_aliases=('aurora_order_fixed',),
        builder_key="aurora_order_local",
        notes="Aurora-inspired transmission ordering under fixed placement; placement optimization is absent. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="aurora_order_joint",
        display_name="Aurora-Order (fixed placement) Joint",
        family="paired_family",
        scheduling_scope="joint",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=True,
        aliases=('Joint(aurora_order)', 'Joint(aurora)'),
        deprecated_aliases=(),
        builder_key="aurora_order_joint",
        notes="Aurora-inspired transmission ordering under fixed placement; placement optimization is absent. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="adaptive_price_local",
        display_name="Adaptive Barrier Price Local",
        family="paired_family",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=True,
        aliases=('Local(adaptive_price)',),
        deprecated_aliases=('B_barrier_price_adaptive_matching',),
        builder_key="adaptive_price_local",
        notes="Strict experimental price family, excluded from the primary paper set. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="adaptive_price_joint",
        display_name="Adaptive Barrier Price Joint",
        family="paired_family",
        scheduling_scope="joint",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=True,
        aliases=('Joint(adaptive_price)',),
        deprecated_aliases=('U_barrier_price_adaptive_matching',),
        builder_key="adaptive_price_joint",
        notes="Strict experimental price family, excluded from the primary paper set. Local and Joint share one immutable kernel; only visibility and phase coupling differ.",
    ),
    AlgorithmSpec(
        canonical_id="barrier_criticality_phase_local",
        display_name="Barrier Criticality Phase-Local",
        family="phase_local_b",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=True,
        aliases=("Local(barrier_criticality_phase_local)",),
        deprecated_aliases=("B_barrier_criticality_matching", "routersense_p0p1_reservation"),
        builder_key="B_barrier_criticality_matching",
    ),
    AlgorithmSpec(
        canonical_id="barrier_criticality_core_independent",
        display_name="Barrier Criticality Core-Independent",
        family="phase_local_b_core",
        scheduling_scope="phase_local",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=True,
        aliases=("routersense_b_core_independent", "Local(barrier_criticality)"),
        deprecated_aliases=("B_barrier_criticality_core_independent",),
        builder_key="B_barrier_criticality_core_independent",
    ),
    AlgorithmSpec(
        canonical_id="barrier_criticality_joint",
        display_name="Barrier Criticality Joint",
        family="joint_u",
        scheduling_scope="joint",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=True,
        aliases=("Joint(barrier_criticality)",),
        deprecated_aliases=("U_barrier_criticality_global_matching", "routersense_p0p1p2_hint"),
        builder_key="U_barrier_criticality_global_matching",
    ),
    AlgorithmSpec(
        canonical_id="barrier_criticality_runtime_safe",
        display_name="Barrier Criticality Runtime Safe",
        family="runtime_safe",
        scheduling_scope="joint_safe_wrapper",
        execution_model="deployable_bucket",
        task_granularity="canonical_bucket",
        deployable=True,
        reference_only=False,
        online_eligible=True,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=True,
        aliases=("runtime_safe_u",),
        deprecated_aliases=("RS_safe_barrier_criticality", "safe-U"),
        builder_key="RS_safe_barrier_criticality",
    ),
    AlgorithmSpec(
        canonical_id="barrier_criticality_posthoc_best",
        display_name="Barrier Criticality Posthoc Best",
        family="reference",
        scheduling_scope="joint_reference",
        execution_model="posthoc_reference",
        task_granularity="canonical_bucket",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=False,
        aliases=("posthoc_best_of_u_and_b",),
        deprecated_aliases=("posthoc_best_of_U_and_B",),
        builder_key="posthoc_best_of_u_and_b",
    ),
    AlgorithmSpec(
        canonical_id="birkhoff_fluid_reference",
        display_name="Birkhoff Fluid Reference",
        family="reference",
        scheduling_scope="phase_local_reference",
        execution_model="fluid_reference",
        task_granularity="fluid",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        aliases=("birkhoff_von_neumann_fluid",),
        deprecated_aliases=("B_birkhoff", "B_birkhoff_wave"),
        builder_key="birkhoff_von_neumann_fluid",
    ),
    AlgorithmSpec(
        canonical_id="oracle_local_cp_sat",
        display_name="Oracle Local CP-SAT",
        family="oracle",
        scheduling_scope="phase_local_optimal",
        execution_model="exact_reference",
        task_granularity="solver_instance",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=False,
        supports_safe_wrapper=False,
        aliases=("O_local_phase_oracle",),
        deprecated_aliases=("O_local",),
        builder_key="O_local_phase_oracle",
    ),
    AlgorithmSpec(
        canonical_id="oracle_joint_cp_sat",
        display_name="Oracle Joint CP-SAT",
        family="oracle",
        scheduling_scope="joint_optimal",
        execution_model="exact_reference",
        task_granularity="solver_instance",
        deployable=False,
        reference_only=True,
        online_eligible=False,
        offline_eligible=True,
        supports_p2_hint=True,
        supports_safe_wrapper=False,
        aliases=("O_joint_cp_sat_oracle", "exact_small_instance_reference"),
        deprecated_aliases=("O_joint", "exact_small_instance_oracle"),
        builder_key="exact_small_instance_reference",
        notes="Current exact small-instance solver backing the joint oracle reference.",
    ),
)


_BY_CANONICAL = {item.canonical_id: item for item in ALGORITHM_SPECS}
_BY_BUILDER = {item.builder_key: item for item in ALGORITHM_SPECS}
_BY_NAME: dict[str, tuple[AlgorithmSpec, bool]] = {}
for spec in ALGORITHM_SPECS:
    _BY_NAME[spec.canonical_id] = (spec, False)
    _BY_NAME[spec.builder_key] = (spec, False)
    for name in spec.aliases:
        _BY_NAME[name] = (spec, False)
    for name in spec.deprecated_aliases:
        _BY_NAME[name] = (spec, True)


def algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return ALGORITHM_SPECS


def deployable_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return tuple(spec for spec in ALGORITHM_SPECS if spec.deployable)


def reference_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return tuple(spec for spec in ALGORITHM_SPECS if spec.reference_only)


def phase_local_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    return tuple(spec for spec in ALGORITHM_SPECS if spec.phase_local_eligible)


def legacy_algorithm_aliases() -> tuple[str, ...]:
    aliases: list[str] = []
    for spec in ALGORITHM_SPECS:
        aliases.extend(spec.aliases)
        aliases.extend(spec.deprecated_aliases)
    return tuple(sorted(dict.fromkeys(aliases)))


def _dynamic_family_canonical_name(name: str) -> str | None:
    # Lazy import avoids making the lightweight catalog eagerly import the
    # scheduler implementation stack.
    try:
        from rs.scheduling.families import canonical_family_policy_id, parse_scoped_family_policy

        parsed = parse_scoped_family_policy(name)
    except (ImportError, ValueError):
        return None
    if parsed is None:
        return None
    family_id, scope = parsed
    return canonical_family_policy_id(family_id, scope)


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    normalized = str(name)
    if normalized in _BY_CANONICAL:
        return _BY_CANONICAL[normalized]
    if normalized in _BY_BUILDER:
        return _BY_BUILDER[normalized]
    if normalized in _BY_NAME:
        return _BY_NAME[normalized][0]
    dynamic = _dynamic_family_canonical_name(normalized)
    if dynamic is not None and dynamic in _BY_NAME:
        return _BY_NAME[dynamic][0]
    raise ValueError(f"unknown algorithm {name!r}")


def resolve_algorithm_id(requested_name: str) -> ResolvedAlgorithmId:
    normalized = str(requested_name)
    lookup_name = normalized
    if lookup_name not in _BY_NAME:
        dynamic = _dynamic_family_canonical_name(lookup_name)
        if dynamic is None or dynamic not in _BY_NAME:
            raise ValueError(f"unknown algorithm {requested_name!r}")
        lookup_name = dynamic
    spec, deprecated = _BY_NAME[lookup_name]
    is_alias = normalized != spec.canonical_id
    warning = None
    if deprecated:
        warning = f"deprecated algorithm alias {normalized!r}; use {spec.canonical_id!r}"
    elif is_alias:
        warning = f"legacy algorithm alias {normalized!r}; canonical name is {spec.canonical_id!r}"
    return ResolvedAlgorithmId(
        requested_name=normalized,
        canonical_name=spec.canonical_id,
        builder_key=spec.builder_key,
        is_alias=is_alias,
        is_deprecated=deprecated,
        warning=warning,
        spec=spec,
    )


__all__ = [
    "ALGORITHM_SPECS",
    "AlgorithmSpec",
    "ResolvedAlgorithmId",
    "algorithm_specs",
    "deployable_algorithm_specs",
    "get_algorithm_spec",
    "legacy_algorithm_aliases",
    "phase_local_algorithm_specs",
    "reference_algorithm_specs",
    "resolve_algorithm_id",
]
