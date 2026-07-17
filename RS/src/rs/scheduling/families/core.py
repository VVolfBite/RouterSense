"""Literature-grounded kernels for controlled Local/Joint comparisons.

A family describes *how a wave is selected*. Literature/control families retain
their native base score and may add the same model-agnostic RouterSense
critical-frontier lift. ``Local(f)`` and ``Joint(f)`` share the same immutable kernel and differ only in the information/ready-set
scope exposed by :mod:`rs.scheduling.families.scoped`.

Names are deliberately conservative.  A paper name is used only when the
implementation contains the defining scheduling mechanism.  Missing mechanisms
are recorded explicitly so evaluation artifacts cannot silently overclaim a
full reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any


class FamilyScope(str, Enum):
    LOCAL = "local"
    JOINT = "joint"


@dataclass(frozen=True)
class LiteratureLineage:
    paper_label: str
    citation_key: str
    mapping_level: str
    defining_mechanisms: tuple[str, ...]
    implemented_mechanisms: tuple[str, ...]
    missing_mechanisms: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FamilyKernelSpec:
    family_id: str
    display_name: str
    literature: LiteratureLineage
    primary_for_paper: bool
    exact_matching: bool
    atomic: bool
    residual_weight: float
    barrier_weight: float
    age_weight: float
    prediction_weight: float
    endpoint_pressure_weight: float = 0.0
    release_gain_weight: float = 0.0
    adaptive_prices: bool = False
    price_step: float = 0.0
    price_decay: float = 0.0
    price_clip: float = 0.0
    iteration_budget: int = 1
    service_model: str = "fluid_wave"
    base_priority_model: str = "none"
    base_priority_weight: float = 0.0
    scoring_model: str = "weighted_components"
    critical_path_weight: float = 0.0
    transitive_unlock_weight: float = 0.0
    endpoint_dual_weight: float = 0.0
    duplex_pair_weight: float = 0.0
    dual_temperature: float = 0.2
    transitive_tail_weight: float = 1.0
    destination_hotspot_weight: float = 0.0
    size_bias_power: float = 0.0
    kernel_version: str = "v2"
    task_contract_digest: str = "canonical_bucket_tasks_v1"
    bucket_contract_digest: str = "dynamic_or_fixed_bucket_v1"
    cost_contract_digest: str = "phase_aware_wire_cost_v1"
    service_model_id: str = "rank_phase_release_batch_v1"
    solver_budget_digest: str = "max_waves_v1"

    @property
    def matching_core_id(self) -> str:
        return f"family::{self.family_id}::{self.kernel_version}"

    def with_weight_overrides(
        self,
        *,
        residual_weight: float | None = None,
        barrier_weight: float | None = None,
        age_weight: float | None = None,
        prediction_weight: float | None = None,
        endpoint_pressure_weight: float | None = None,
        release_gain_weight: float | None = None,
    ) -> "FamilyKernelSpec":
        return replace(
            self,
            residual_weight=self.residual_weight if residual_weight is None else float(residual_weight),
            barrier_weight=self.barrier_weight if barrier_weight is None else float(barrier_weight),
            age_weight=self.age_weight if age_weight is None else float(age_weight),
            prediction_weight=self.prediction_weight if prediction_weight is None else float(prediction_weight),
            endpoint_pressure_weight=(
                self.endpoint_pressure_weight
                if endpoint_pressure_weight is None
                else float(endpoint_pressure_weight)
            ),
            release_gain_weight=(
                self.release_gain_weight if release_gain_weight is None else float(release_gain_weight)
            ),
        )

    def contract(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "matching_core_id": self.matching_core_id,
            "task_contract_digest": self.task_contract_digest,
            "bucket_contract_digest": self.bucket_contract_digest,
            "cost_contract_digest": self.cost_contract_digest,
            "service_model_id": self.service_model_id,
            "solver_budget_digest": self.solver_budget_digest,
            "literature": self.literature.to_dict(),
            "kernel_parameters": {
                key: value
                for key, value in asdict(self).items()
                if key
                in {
                    "exact_matching",
                    "atomic",
                    "residual_weight",
                    "barrier_weight",
                    "age_weight",
                    "prediction_weight",
                    "endpoint_pressure_weight",
                    "release_gain_weight",
                    "adaptive_prices",
                    "price_step",
                    "price_decay",
                    "price_clip",
                    "iteration_budget",
                    "service_model",
                    "base_priority_model",
                    "base_priority_weight",
                    "scoring_model",
                    "critical_path_weight",
                    "transitive_unlock_weight",
                    "endpoint_dual_weight",
                    "duplex_pair_weight",
                    "dual_temperature",
                    "transitive_tail_weight",
                    "destination_hotspot_weight",
                    "size_bias_power",
                    "kernel_version",
                }
            },
        }


FAMILY_KERNEL_SPECS: dict[str, FamilyKernelSpec] = {
    # Low-complexity control.  It is intentionally not presented as a paper
    # reproduction; it isolates the value of the joint ready set.
    "greedy_control": FamilyKernelSpec(
        family_id="greedy_control",
        display_name="Greedy Control",
        literature=LiteratureLineage(
            paper_label="Greedy control",
            citation_key="control.greedy",
            mapping_level="control",
            defining_mechanisms=("greedy maximal matching",),
            implemented_mechanisms=("greedy maximal matching", "residual-volume ordering"),
            note=("Generic control rather than a named-system reproduction. The native "
                  "greedy score is retained and augmented by the shared, model-agnostic "
                  "RouterSense P2 critical-frontier lift."),
        ),
        primary_for_paper=True,
        exact_matching=False,
        atomic=False,
        residual_weight=1.0,
        barrier_weight=0.0,
        age_weight=0.05,
        prediction_weight=0.0,
        scoring_model="critical_frontier",
        critical_path_weight=0.20,
        transitive_unlock_weight=3.00,
        endpoint_dual_weight=0.50,
        transitive_tail_weight=0.50,
        destination_hotspot_weight=0.20,
        kernel_version="v3-p2lift",
    ),
    # Greedy Max-Weight Decomposition (GMWD): repeatedly solve a maximum-weight
    # matching on the original residual MoE matrix and subtract the minimum
    # residual quantum from selected edges.  The multiphase wrapper changes only
    # which released residual edges are visible.
    "gmwd": FamilyKernelSpec(
        family_id="gmwd",
        display_name="GMWD-style",
        literature=LiteratureLineage(
            paper_label="Greedy Max-Weight Decomposition (GMWD)",
            citation_key="amponsah_addanki_2026_gmwd",
            mapping_level="style",
            defining_mechanisms=(
                "operate directly on the residual traffic matrix",
                "maximum-weight matching per decomposition round",
                "subtract selected residual service until demand is zero",
            ),
            implemented_mechanisms=(
                "operate directly on released residual MoE flows",
                "maximum-weight bipartite matching per wave",
                "minimum selected residual as the service quantum",
            ),
            missing_mechanisms=(
                "paper-specific photonic reconfiguration and expert-compute cost model",
            ),
            note=("The residual max-weight decomposition core is retained. RouterSense "
                  "adds the same model-agnostic P2 critical-frontier lift used across "
                  "scoped families; results must therefore be labeled GMWD-CF or "
                  "GMWD-style + RouterSense lift, not as an unmodified reproduction."),
        ),
        primary_for_paper=True,
        exact_matching=True,
        atomic=False,
        residual_weight=1.0,
        barrier_weight=0.0,
        age_weight=0.0,
        prediction_weight=0.0,
        scoring_model="critical_frontier",
        critical_path_weight=0.20,
        transitive_unlock_weight=3.00,
        endpoint_dual_weight=0.50,
        transitive_tail_weight=0.50,
        destination_hotspot_weight=0.20,
        kernel_version="v3-p2lift",
    ),
    # RouterSense original: barrier urgency plus a normalized release-gain term.
    # The release-gain term is computed by the common kernel for both scopes;
    # Local simply cannot observe downstream phases that were not exposed.
    "rsbc": FamilyKernelSpec(
        family_id="rsbc",
        display_name="RouterSense Barrier Criticality (RSBC)",
        literature=LiteratureLineage(
            paper_label="RouterSense Barrier Criticality (RSBC)",
            citation_key="routersense.rsbc",
            mapping_level="original",
            defining_mechanisms=(
                "release-aware global ready set",
                "barrier criticality",
                "downstream release-gain scoring",
                "maximum-weight matching",
            ),
            implemented_mechanisms=(
                "release-aware ready set",
                "barrier criticality",
                "normalized downstream release-gain scoring",
                "maximum-weight bipartite matching",
            ),
            note=("RouterSense original family. RSBC retains its barrier/release-gain "
                  "components and adds the shared model-agnostic P2 critical-frontier lift."),
        ),
        primary_for_paper=True,
        exact_matching=True,
        atomic=False,
        residual_weight=0.75,
        barrier_weight=2.00,
        age_weight=0.15,
        prediction_weight=0.35,
        release_gain_weight=1.50,
        scoring_model="critical_frontier",
        critical_path_weight=0.35,
        transitive_unlock_weight=3.00,
        endpoint_dual_weight=1.00,
        transitive_tail_weight=0.25,
        destination_hotspot_weight=0.15,
        kernel_version="v5-p2lift",
    ),
    # RouterSense Critical Frontier (RSCF): a model-agnostic extension of RSBC
    # that prices the transitive P0->P1->P2 release DAG and endpoint bottlenecks.
    # It consumes traffic geometry and
    # release constraints only; no expert identity or model-specific feature is
    # part of the kernel.
    "rscf": FamilyKernelSpec(
        family_id="rscf",
        display_name="RouterSense Critical Frontier (RSCF)",
        literature=LiteratureLineage(
            paper_label="RouterSense Critical Frontier (RSCF)",
            citation_key="routersense.rscf",
            mapping_level="original",
            defining_mechanisms=(
                "release-aware global ready set",
                "transitive P0-P1-P2 critical-frontier pricing",
                "release-aware endpoint bottleneck dual pricing",
                "maximum-weight matching",
            ),
            implemented_mechanisms=(
                "traffic-only transitive release-DAG criticality",
                "smooth critical-frontier and endpoint dual prices",
                "maximum-weight bipartite matching",
            ),
            note=(
                "RouterSense original, model-agnostic family. The kernel uses only "
                "residual traffic, endpoint loads, and release dependencies."
            ),
        ),
        primary_for_paper=True,
        exact_matching=True,
        atomic=False,
        residual_weight=0.15,
        barrier_weight=0.50,
        age_weight=0.30,
        prediction_weight=0.0,
        release_gain_weight=2.50,
        scoring_model="critical_frontier",
        critical_path_weight=0.25,
        transitive_unlock_weight=2.50,
        endpoint_dual_weight=1.00,
        duplex_pair_weight=0.00,
        dual_temperature=0.20,
        transitive_tail_weight=0.25,
        destination_hotspot_weight=0.10,
        size_bias_power=0.00,
        kernel_version="v4",
    ),
    # FAST contains intra-server rebalancing plus balanced one-to-one scale-out
    # stages.  Our current single-tier model implements only the stage-ordering
    # core, so the honest paper label is FAST-Stage, not FAST.
    "fast_stage": FamilyKernelSpec(
        family_id="fast_stage",
        display_name="FAST-Stage (single-tier)",
        literature=LiteratureLineage(
            paper_label="FAST stage-ordering core",
            citation_key="lei_et_al_nsdi26_fast",
            mapping_level="inspired",
            defining_mechanisms=(
                "intra-server traffic rebalancing",
                "balanced one-to-one scale-out transfers",
                "two-tier topology awareness",
            ),
            implemented_mechanisms=(
                "one-to-one matching stages",
                "BvN-derived stage priority",
                "residual service until completion",
            ),
            missing_mechanisms=(
                "intra-server rebalancing",
                "server/NIC hierarchy",
                "two-tier scale-out topology model",
            ),
            note=("Must be reported as FAST-Stage-CF (or FAST-inspired stage ordering "
                  "+ RouterSense critical-frontier lift), never as a full FAST reproduction."),
        ),
        primary_for_paper=True,
        exact_matching=True,
        atomic=False,
        residual_weight=0.25,
        barrier_weight=0.0,
        age_weight=0.05,
        prediction_weight=0.0,
        base_priority_model="birkhoff_round_rank",
        base_priority_weight=1.0,
        scoring_model="critical_frontier",
        critical_path_weight=0.25,
        transitive_unlock_weight=2.50,
        endpoint_dual_weight=1.00,
        transitive_tail_weight=0.25,
        destination_hotspot_weight=0.10,
        kernel_version="v3-p2lift",
    ),
    # Aurora jointly optimizes placement and transmission ordering.  The current
    # fixed-placement scheduler implements only the pressure-aware ordering part.
    "aurora_order": FamilyKernelSpec(
        family_id="aurora_order",
        display_name="Aurora-Order (fixed placement)",
        literature=LiteratureLineage(
            paper_label="Aurora transmission ordering",
            citation_key="li_et_al_2024_aurora",
            mapping_level="inspired",
            defining_mechanisms=(
                "token transmission ordering",
                "expert/model placement optimization",
                "cluster-setting-aware optimization",
            ),
            implemented_mechanisms=(
                "source/destination pressure-aware transmission ordering",
                "fixed-placement one-to-one wave construction",
            ),
            missing_mechanisms=(
                "expert/model placement optimization",
                "heterogeneous-cluster optimization cases",
            ),
            note="Ordering-only adaptation under fixed placement.",
        ),
        primary_for_paper=False,
        exact_matching=False,
        atomic=True,
        residual_weight=0.25,
        barrier_weight=0.0,
        age_weight=0.05,
        prediction_weight=0.0,
        endpoint_pressure_weight=1.0,
        service_model="atomic_wave",
    ),
    # Retained as a strict experimental family, but not part of the primary
    # literature-grounded paper set.
    "adaptive_price": FamilyKernelSpec(
        family_id="adaptive_price",
        display_name="Adaptive Barrier Price (experimental)",
        literature=LiteratureLineage(
            paper_label="Adaptive dual-price control",
            citation_key="routersense.experimental_price",
            mapping_level="experimental",
            defining_mechanisms=("adaptive congestion prices",),
            implemented_mechanisms=("shared adaptive destination-price update",),
            note="Exploratory family; excluded from the primary paper family set.",
        ),
        primary_for_paper=False,
        exact_matching=True,
        atomic=False,
        residual_weight=0.75,
        barrier_weight=1.50,
        age_weight=0.15,
        prediction_weight=0.35,
        release_gain_weight=1.0,
        adaptive_prices=True,
        price_step=0.2,
        price_decay=0.1,
        price_clip=8.0,
        iteration_budget=2,
        scoring_model="critical_frontier",
        critical_path_weight=0.20,
        transitive_unlock_weight=2.00,
        endpoint_dual_weight=0.50,
        transitive_tail_weight=0.50,
        destination_hotspot_weight=0.10,
        kernel_version="v3-p2lift",
    ),
}

PRIMARY_FAMILY_IDS = tuple(
    family_id for family_id, spec in FAMILY_KERNEL_SPECS.items() if spec.primary_for_paper
)
EXPERIMENTAL_FAMILY_IDS = tuple(
    family_id for family_id, spec in FAMILY_KERNEL_SPECS.items() if not spec.primary_for_paper
)
# Backward-compatible public constant: strict primary families used by paper
# evaluation.  Experimental strict pairs are inventoried separately.
STRICT_FAMILY_IDS = PRIMARY_FAMILY_IDS

LEGACY_UNPAIRED_FAMILIES = {
    "lagrangian": {
        "local_policy": "B_lagrangian_phase_local",
        "joint_policy": "U_lagrangian",
        "reason": "historical implementations do not share one solver/update core",
    },
    "ibbr": {
        "local_policy": "B_birkhoff",
        "joint_policy": "U_ibbr",
        "reason": "joint side adds iterative repair that is absent from the local side",
    },
}


FAMILY_ID_ALIASES: dict[str, str] = {
    # New literature-grounded names.
    "greedy": "greedy_control",
    "greedy_control": "greedy_control",
    "gmwd": "gmwd",
    "rsbc": "rsbc",
    "rscf": "rscf",
    "critical_frontier": "rscf",
    "router_sense_critical_frontier": "rscf",
    "fast": "fast_stage",
    "fast_style": "fast_stage",
    "fast_stage": "fast_stage",
    "aurora": "aurora_order",
    "aurora_order": "aurora_order",
    # Historical family names.
    "gated_greedy": "greedy_control",
    "gated_greedy_maximal": "greedy_control",
    "gated_maxweight": "gmwd",
    "maxweight": "gmwd",
    "gated_maxweight_matching": "gmwd",
    "barrier_criticality": "rsbc",
    "barrier_criticality_global_matching": "rsbc",
    "barrier_criticality_matching": "rsbc",
    "birkhoff_ranked": "fast_stage",
    "birkhoff": "fast_stage",
    "birkhoff_wave": "fast_stage",
    "barrier_price": "adaptive_price",
    "barrier_price_adaptive_matching": "adaptive_price",
}


def normalize_family_id(family_id: str) -> str:
    normalized = str(family_id).replace("-", "_").strip().lower()
    return FAMILY_ID_ALIASES.get(normalized, normalized)


def get_family_kernel_spec(family_id: str) -> FamilyKernelSpec:
    normalized = normalize_family_id(family_id)
    try:
        return FAMILY_KERNEL_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown strict scheduling family {family_id!r}") from exc


def family_inventory() -> dict[str, Any]:
    def row(family_id: str, spec: FamilyKernelSpec, *, status: str) -> dict[str, Any]:
        return {
            "family_id": family_id,
            "display_name": spec.display_name,
            "local_policy_id": f"{family_id}_local",
            "joint_policy_id": f"{family_id}_joint",
            "local_expression": f"Local({family_id})",
            "joint_expression": f"Joint({family_id})",
            "common_core": spec.contract(),
            "literature": spec.literature.to_dict(),
            "status": status,
        }

    return {
        "schema_version": "scheduling_family_inventory.v2",
        "primary_strict_families": [
            row(family_id, FAMILY_KERNEL_SPECS[family_id], status="STRICT_SAME_CORE_READY")
            for family_id in PRIMARY_FAMILY_IDS
        ],
        # Compatibility field consumed by existing readers.
        "strict_families": [
            row(family_id, FAMILY_KERNEL_SPECS[family_id], status="STRICT_SAME_CORE_READY")
            for family_id in PRIMARY_FAMILY_IDS
        ],
        "experimental_strict_families": [
            row(family_id, FAMILY_KERNEL_SPECS[family_id], status="STRICT_EXPERIMENTAL")
            for family_id in EXPERIMENTAL_FAMILY_IDS
        ],
        "legacy_unpaired_families": [
            {"family_id": family_id, **legacy_row, "status": "LEGACY_NOT_STRICT"}
            for family_id, legacy_row in LEGACY_UNPAIRED_FAMILIES.items()
        ],
    }


__all__ = [
    "EXPERIMENTAL_FAMILY_IDS",
    "FAMILY_ID_ALIASES",
    "FAMILY_KERNEL_SPECS",
    "FamilyKernelSpec",
    "FamilyScope",
    "LEGACY_UNPAIRED_FAMILIES",
    "LiteratureLineage",
    "PRIMARY_FAMILY_IDS",
    "STRICT_FAMILY_IDS",
    "family_inventory",
    "get_family_kernel_spec",
    "normalize_family_id",
]
