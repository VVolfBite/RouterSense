"""Shared algorithm kernels for controlled Local/Joint family comparisons.

The family layer deliberately separates *how a wave is selected* from *which
phase information is visible*.  A local and a joint policy therefore consume
one immutable :class:`FamilyKernelSpec`; only the scope adapter differs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any


class FamilyScope(str, Enum):
    LOCAL = "local"
    JOINT = "joint"


@dataclass(frozen=True)
class FamilyKernelSpec:
    family_id: str
    display_name: str
    exact_matching: bool
    atomic: bool
    residual_weight: float
    barrier_weight: float
    age_weight: float
    prediction_weight: float
    adaptive_prices: bool = False
    price_step: float = 0.0
    price_decay: float = 0.0
    price_clip: float = 0.0
    iteration_budget: int = 1
    service_model: str = "fluid_wave"
    base_priority_model: str = "none"
    kernel_version: str = "v1"
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
    ) -> "FamilyKernelSpec":
        return replace(
            self,
            residual_weight=self.residual_weight if residual_weight is None else float(residual_weight),
            barrier_weight=self.barrier_weight if barrier_weight is None else float(barrier_weight),
            age_weight=self.age_weight if age_weight is None else float(age_weight),
            prediction_weight=self.prediction_weight if prediction_weight is None else float(prediction_weight),
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
                    "adaptive_prices",
                    "price_step",
                    "price_decay",
                    "price_clip",
                    "iteration_budget",
                    "service_model",
                    "base_priority_model",
                    "kernel_version",
                }
            },
        }


FAMILY_KERNEL_SPECS: dict[str, FamilyKernelSpec] = {
    # A deliberately low-complexity control: identical scoring, greedy maximal
    # matching, and only the visible ready set changes.
    "gated_greedy": FamilyKernelSpec(
        family_id="gated_greedy",
        display_name="Gated Greedy",
        exact_matching=False,
        atomic=False,
        residual_weight=1.0,
        barrier_weight=1.0,
        age_weight=0.1,
        prediction_weight=0.25,
    ),
    # Historical Tier-1 max-weight candidate.
    "gated_maxweight": FamilyKernelSpec(
        family_id="gated_maxweight",
        display_name="Gated MaxWeight",
        exact_matching=True,
        atomic=False,
        residual_weight=1.0,
        barrier_weight=1.0,
        age_weight=0.1,
        prediction_weight=0.25,
    ),
    # RouterSense main family.
    "barrier_criticality": FamilyKernelSpec(
        family_id="barrier_criticality",
        display_name="Barrier Criticality",
        exact_matching=True,
        atomic=False,
        residual_weight=0.75,
        barrier_weight=1.75,
        age_weight=0.15,
        prediction_weight=0.35,
    ),
    # Birkhoff/BvN ordering is used only as the shared base-priority kernel;
    # Local/Joint scope is still controlled by the same adapter as other
    # families.  This avoids comparing plain Birkhoff against an unrelated
    # iterative-repair algorithm.
    "birkhoff_ranked": FamilyKernelSpec(
        family_id="birkhoff_ranked",
        display_name="Birkhoff-Ranked Matching",
        exact_matching=True,
        atomic=False,
        residual_weight=0.25,
        barrier_weight=0.0,
        age_weight=0.05,
        prediction_weight=0.0,
        base_priority_model="birkhoff_round_rank",
    ),
    # A dual/price-flavoured candidate that reuses exactly the same adaptive
    # price update in both scopes.  The recovered U_lagrangian remains a legacy
    # exploratory policy because its old B-side did not share the same core.
    "adaptive_price": FamilyKernelSpec(
        family_id="adaptive_price",
        display_name="Adaptive Barrier Price",
        exact_matching=True,
        atomic=False,
        residual_weight=0.75,
        barrier_weight=1.75,
        age_weight=0.15,
        prediction_weight=0.35,
        adaptive_prices=True,
        price_step=0.2,
        price_decay=0.1,
        price_clip=8.0,
        iteration_budget=2,
    ),
}

STRICT_FAMILY_IDS = tuple(FAMILY_KERNEL_SPECS)
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
    "greedy": "gated_greedy",
    "gated_greedy_maximal": "gated_greedy",
    "maxweight": "gated_maxweight",
    "gated_maxweight_matching": "gated_maxweight",
    "barrier_criticality_global_matching": "barrier_criticality",
    "barrier_criticality_matching": "barrier_criticality",
    "birkhoff": "birkhoff_ranked",
    "birkhoff_wave": "birkhoff_ranked",
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
    return {
        "schema_version": "scheduling_family_inventory.v1",
        "strict_families": [
            {
                "family_id": family_id,
                "display_name": spec.display_name,
                "local_policy_id": f"{family_id}_local",
                "joint_policy_id": f"{family_id}_joint",
                "local_expression": f"Local({family_id})",
                "joint_expression": f"Joint({family_id})",
                "common_core": spec.contract(),
                "status": "STRICT_SAME_CORE_READY",
            }
            for family_id, spec in FAMILY_KERNEL_SPECS.items()
        ],
        "legacy_unpaired_families": [
            {"family_id": family_id, **row, "status": "LEGACY_NOT_STRICT"}
            for family_id, row in LEGACY_UNPAIRED_FAMILIES.items()
        ],
    }


__all__ = [
    "FAMILY_ID_ALIASES",
    "FAMILY_KERNEL_SPECS",
    "FamilyKernelSpec",
    "FamilyScope",
    "LEGACY_UNPAIRED_FAMILIES",
    "STRICT_FAMILY_IDS",
    "family_inventory",
    "get_family_kernel_spec",
    "normalize_family_id",
]
