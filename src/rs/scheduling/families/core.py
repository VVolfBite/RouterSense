"""Canonical RouterSense scheduling-core definitions.

Only the three formal cores used by the orthogonal P01/P012/P0123 runtime live
here. Literature baselines are isolated under :mod:`rs.reference.baselines`.
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
    kernel_version: str = "v1"
    task_contract_digest: str = "canonical_bucket_tasks_v1"
    bucket_contract_digest: str = "dynamic_or_fixed_bucket_v1"
    cost_contract_digest: str = "phase_aware_wire_cost_v1"
    service_model_id: str = "rank_phase_release_batch_v1"
    solver_budget_digest: str = "max_waves_v1"
    p012_weights: tuple[float, float, float, float] | None = None

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
            endpoint_pressure_weight=(self.endpoint_pressure_weight if endpoint_pressure_weight is None else float(endpoint_pressure_weight)),
            release_gain_weight=(self.release_gain_weight if release_gain_weight is None else float(release_gain_weight)),
        )

    def p012_runtime_weights(self) -> tuple[float, float, float, float]:
        if self.p012_weights is None or len(self.p012_weights) != 4:
            raise ValueError(f"family {self.family_id!r} has no valid P012 runtime weights")
        return tuple(float(value) for value in self.p012_weights)

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
                if key not in {"family_id", "display_name", "literature", "primary_for_paper"}
            },
        }


FAMILY_KERNEL_SPECS: dict[str, FamilyKernelSpec] = {
    "gmwd": FamilyKernelSpec(
        family_id="gmwd",
        display_name="GMWD-style",
        literature=LiteratureLineage(
            paper_label="Greedy Max-Weight Decomposition (GMWD)",
            citation_key="amponsah_addanki_2026_gmwd",
            mapping_level="style",
            defining_mechanisms=(
                "residual traffic matrix",
                "maximum-weight matching per round",
                "subtract selected residual service",
            ),
            implemented_mechanisms=(
                "released residual MoE flows",
                "maximum-weight bipartite matching per wave",
                "minimum selected residual service quantum",
            ),
            missing_mechanisms=("paper-specific photonic reconfiguration and compute model",),
            note="Reported as GMWD-style rather than a full reproduction.",
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
        p012_weights=(1.0, 0.0, 0.0, 0.35),
    ),
    "rsbc": FamilyKernelSpec(
        family_id="rsbc",
        display_name="RouterSense Barrier Criticality (RSBC)",
        literature=LiteratureLineage(
            paper_label="RouterSense Barrier Criticality (RSBC)",
            citation_key="routersense.rsbc",
            mapping_level="original",
            defining_mechanisms=(
                "release-aware ready set",
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
        p012_weights=(0.45, 0.85, 0.75, 0.45),
    ),
    "rscf": FamilyKernelSpec(
        family_id="rscf",
        display_name="RouterSense Critical Frontier (RSCF)",
        literature=LiteratureLineage(
            paper_label="RouterSense Critical Frontier (RSCF)",
            citation_key="routersense.rscf",
            mapping_level="original",
            defining_mechanisms=(
                "release-aware ready set",
                "transitive P0-P1-P2 critical-frontier pricing",
                "endpoint bottleneck pricing",
                "maximum-weight matching",
            ),
            implemented_mechanisms=(
                "traffic-only transitive release-DAG criticality",
                "critical-frontier and endpoint dual prices",
                "maximum-weight bipartite matching",
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
        dual_temperature=0.20,
        transitive_tail_weight=0.25,
        destination_hotspot_weight=0.10,
        kernel_version="v4",
        p012_weights=(0.25, 0.75, 4.0, 0.50),
    ),
}

PRIMARY_FAMILY_IDS = tuple(FAMILY_KERNEL_SPECS)
STRICT_FAMILY_IDS = PRIMARY_FAMILY_IDS
EXPERIMENTAL_FAMILY_IDS: tuple[str, ...] = ()
LEGACY_UNPAIRED_FAMILIES: dict[str, dict[str, str]] = {}
FAMILY_ID_ALIASES = {family_id: family_id for family_id in FAMILY_KERNEL_SPECS}


def normalize_family_id(family_id: str) -> str:
    normalized = str(family_id).replace("-", "_").strip().lower()
    return FAMILY_ID_ALIASES.get(normalized, normalized)


def get_family_kernel_spec(family_id: str) -> FamilyKernelSpec:
    normalized = normalize_family_id(family_id)
    try:
        return FAMILY_KERNEL_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown formal scheduling core {family_id!r}") from exc


def family_inventory() -> dict[str, Any]:
    return {
        "schema_version": "scheduling_core_inventory.v1",
        "formal_cores": [
            {
                "core": family_id,
                "display_name": spec.display_name,
                "common_core": spec.contract(),
                "status": "FORMAL_READY",
            }
            for family_id, spec in FAMILY_KERNEL_SPECS.items()
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
