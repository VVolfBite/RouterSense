"""Pair-first scheduling algorithm catalog for RouterSense."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AlgorithmMetadata:
    algorithm_id: str
    display_name: str
    heuristic_family: str
    role: str
    paired_algorithm_id: str | None
    local_oracle_reference_id: str | None
    joint_oracle_reference_id: str | None
    granularity_mode: str
    planning_scope: str
    source: str
    online_eligible: bool
    offline_eligible: bool
    heavy_solver: bool
    deterministic_solver: bool
    oracle_like: bool
    recommended_role: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ALGORITHMS: dict[str, AlgorithmMetadata] = {
    "phase_barrier_fifo": AlgorithmMetadata(
        algorithm_id="phase_barrier_fifo",
        display_name="Phase Barrier FIFO",
        heuristic_family="basic_fifo",
        role="basic_baseline",
        paired_algorithm_id=None,
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="single_phase",
        source="current_mainline",
        online_eligible=True,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="basic_baseline",
    ),
    "greedy_ready_set": AlgorithmMetadata(
        algorithm_id="greedy_ready_set",
        display_name="Greedy Ready Set",
        heuristic_family="basic_greedy",
        role="basic_baseline",
        paired_algorithm_id=None,
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="single_phase",
        source="current_mainline",
        online_eligible=True,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="basic_baseline",
    ),
    "B_birkhoff": AlgorithmMetadata(
        algorithm_id="B_birkhoff",
        display_name="Birkhoff Local Oracle",
        heuristic_family="birkhoff_bvn",
        role="o_local_phase_oracle",
        paired_algorithm_id="U_ibbr",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="single_phase",
        source="current_mainline",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=True,
        recommended_role="phase_local_oracle_reference",
        notes="Phase-local oracle-like deterministic reference under fluid / crossbar makespan semantics.",
    ),
    "B_birkhoff_wave": AlgorithmMetadata(
        algorithm_id="B_birkhoff_wave",
        display_name="Birkhoff Wave Local Oracle",
        heuristic_family="birkhoff_bvn",
        role="b_phase_local",
        paired_algorithm_id="U_ibbr",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="legacy_coarse_wave",
        planning_scope="phase_serial",
        source="current_mainline",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=True,
        recommended_role="phase_local_oracle_reference",
    ),
    "B_barrier_aware_birkhoff": AlgorithmMetadata(
        algorithm_id="B_barrier_aware_birkhoff",
        display_name="Barrier-Aware Birkhoff",
        heuristic_family="birkhoff_bvn",
        role="b_phase_local",
        paired_algorithm_id="U_ibbr",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="phase_serial",
        source="legacy_poc1_pending",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="phase_local_family_ablation",
    ),
    "B_barrier_aware_birkhoff_wave": AlgorithmMetadata(
        algorithm_id="B_barrier_aware_birkhoff_wave",
        display_name="Barrier-Aware Birkhoff Wave",
        heuristic_family="birkhoff_bvn",
        role="b_phase_local",
        paired_algorithm_id="U_ibbr",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="legacy_coarse_wave",
        planning_scope="phase_serial",
        source="legacy_poc1_pending",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="phase_local_family_ablation",
    ),
    "U_ibbr": AlgorithmMetadata(
        algorithm_id="U_ibbr",
        display_name="Iterated Birkhoff Barrier Repair",
        heuristic_family="birkhoff_bvn",
        role="u_routersense_joint",
        paired_algorithm_id="B_birkhoff",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="legacy_atomic_token",
        planning_scope="multiphase_joint",
        source="legacy_poc1_pending",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=False,
        oracle_like=False,
        recommended_role="pending_joint_family_candidate",
    ),
    "B_gated_greedy_maximal": AlgorithmMetadata(
        algorithm_id="B_gated_greedy_maximal",
        display_name="Phase-Local Gated Greedy",
        heuristic_family="gated_greedy",
        role="b_phase_local",
        paired_algorithm_id="U_gated_greedy_maximal",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="phase_serial",
        source="derived_phase_local_from_u",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="paired_b_reference",
    ),
    "U_gated_greedy_maximal": AlgorithmMetadata(
        algorithm_id="U_gated_greedy_maximal",
        display_name="Joint Gated Greedy",
        heuristic_family="gated_greedy",
        role="u_routersense_joint",
        paired_algorithm_id="B_gated_greedy_maximal",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="multiphase_joint",
        source="legacy_poc1_recovered",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="paired_u_candidate",
    ),
    "U_gated_greedy_maximal_atomic": AlgorithmMetadata(
        algorithm_id="U_gated_greedy_maximal_atomic",
        display_name="Joint Gated Greedy Atomic",
        heuristic_family="gated_greedy",
        role="u_routersense_joint",
        paired_algorithm_id="B_gated_greedy_maximal",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="legacy_atomic_token",
        planning_scope="multiphase_joint",
        source="legacy_poc1_recovered",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="legacy_granularity_variant",
    ),
    "B_gated_maxweight_matching": AlgorithmMetadata(
        algorithm_id="B_gated_maxweight_matching",
        display_name="Phase-Local Gated Maxweight",
        heuristic_family="gated_maxweight_matching",
        role="b_phase_local",
        paired_algorithm_id="U_gated_maxweight_matching",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="phase_serial",
        source="derived_phase_local_from_u",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="paired_b_reference",
    ),
    "U_gated_maxweight_matching": AlgorithmMetadata(
        algorithm_id="U_gated_maxweight_matching",
        display_name="Joint Gated Maxweight",
        heuristic_family="gated_maxweight_matching",
        role="u_routersense_joint",
        paired_algorithm_id="B_gated_maxweight_matching",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="multiphase_joint",
        source="current_mainline",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="paired_u_candidate",
    ),
    "U_gated_maxweight_matching_atomic": AlgorithmMetadata(
        algorithm_id="U_gated_maxweight_matching_atomic",
        display_name="Joint Gated Maxweight Atomic",
        heuristic_family="gated_maxweight_matching",
        role="u_routersense_joint",
        paired_algorithm_id="B_gated_maxweight_matching",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="legacy_atomic_token",
        planning_scope="multiphase_joint",
        source="current_mainline",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="legacy_granularity_variant",
    ),
    "B_barrier_criticality_matching": AlgorithmMetadata(
        algorithm_id="B_barrier_criticality_matching",
        display_name="Phase-Local Barrier Criticality Matching",
        heuristic_family="barrier_criticality_matching",
        role="b_phase_local",
        paired_algorithm_id="U_barrier_criticality_global_matching",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="phase_serial",
        source="derived_phase_local_from_u",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="paired_b_reference",
    ),
    "U_barrier_criticality_global_matching": AlgorithmMetadata(
        algorithm_id="U_barrier_criticality_global_matching",
        display_name="Joint Barrier Criticality Matching",
        heuristic_family="barrier_criticality_matching",
        role="u_routersense_joint",
        paired_algorithm_id="B_barrier_criticality_matching",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="multiphase_joint",
        source="current_mainline",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="paired_u_candidate",
    ),
    "U_barrier_criticality_global_matching_atomic": AlgorithmMetadata(
        algorithm_id="U_barrier_criticality_global_matching_atomic",
        display_name="Joint Barrier Criticality Matching Atomic",
        heuristic_family="barrier_criticality_matching",
        role="u_routersense_joint",
        paired_algorithm_id="B_barrier_criticality_matching",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="legacy_atomic_token",
        planning_scope="multiphase_joint",
        source="current_mainline",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="legacy_granularity_variant",
    ),
    "B_barrier_price_adaptive_matching": AlgorithmMetadata(
        algorithm_id="B_barrier_price_adaptive_matching",
        display_name="Phase-Local Adaptive Price Matching",
        heuristic_family="barrier_price_adaptive_matching",
        role="b_phase_local",
        paired_algorithm_id="U_barrier_price_adaptive_matching",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="phase_serial",
        source="legacy_poc1_pending",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="pending_phase_local_family",
    ),
    "U_barrier_price_adaptive_matching": AlgorithmMetadata(
        algorithm_id="U_barrier_price_adaptive_matching",
        display_name="Joint Adaptive Price Matching",
        heuristic_family="barrier_price_adaptive_matching",
        role="u_routersense_joint",
        paired_algorithm_id="B_barrier_price_adaptive_matching",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="multiphase_joint",
        source="legacy_poc1_recovered",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="recoverable_joint_family",
    ),
    "U_barrier_price_adaptive_matching_atomic": AlgorithmMetadata(
        algorithm_id="U_barrier_price_adaptive_matching_atomic",
        display_name="Joint Adaptive Price Matching Atomic",
        heuristic_family="barrier_price_adaptive_matching",
        role="u_routersense_joint",
        paired_algorithm_id="B_barrier_price_adaptive_matching",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="legacy_atomic_token",
        planning_scope="multiphase_joint",
        source="legacy_poc1_recovered",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="legacy_granularity_variant",
    ),
    "B_lagrangian_phase_local": AlgorithmMetadata(
        algorithm_id="B_lagrangian_phase_local",
        display_name="Phase-Local Lagrangian",
        heuristic_family="lagrangian_cross_phase",
        role="b_phase_local",
        paired_algorithm_id="U_lagrangian",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="phase_serial",
        source="legacy_poc1_pending",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=False,
        oracle_like=False,
        recommended_role="pending_phase_local_family",
    ),
    "U_lagrangian": AlgorithmMetadata(
        algorithm_id="U_lagrangian",
        display_name="Joint Lagrangian",
        heuristic_family="lagrangian_cross_phase",
        role="u_routersense_joint",
        paired_algorithm_id="B_lagrangian_phase_local",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="legacy_atomic_token",
        planning_scope="multiphase_joint",
        source="current_mainline",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=False,
        oracle_like=False,
        recommended_role="recoverable_joint_family",
    ),
    "O_local_phase_oracle": AlgorithmMetadata(
        algorithm_id="O_local_phase_oracle",
        display_name="Local Phase Oracle",
        heuristic_family="oracle",
        role="o_local_phase_oracle",
        paired_algorithm_id=None,
        local_oracle_reference_id=None,
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="legacy_fluid_reference",
        planning_scope="single_phase",
        source="current_mainline",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=True,
        recommended_role="oracle_reference",
        notes="Represented in practice by B_birkhoff under phase-local fluid semantics.",
    ),
    "O_joint_cp_sat_oracle": AlgorithmMetadata(
        algorithm_id="O_joint_cp_sat_oracle",
        display_name="Joint CP-SAT Oracle",
        heuristic_family="oracle",
        role="o_joint_oracle",
        paired_algorithm_id=None,
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id=None,
        granularity_mode="legacy_fluid_reference",
        planning_scope="execution_window",
        source="legacy_poc1_recovered",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=True,
        deterministic_solver=True,
        oracle_like=True,
        recommended_role="joint_oracle_reference",
        notes="Closest audited historical implementation is pairwise_oracle in legacy scheduler/oracle.py.",
    ),
    "exact_small_instance_reference": AlgorithmMetadata(
        algorithm_id="exact_small_instance_reference",
        display_name="Exact Small Instance Reference",
        heuristic_family="oracle",
        role="o_joint_oracle",
        paired_algorithm_id=None,
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="not_applicable",
        planning_scope="execution_window",
        source="current_mainline",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=True,
        deterministic_solver=True,
        oracle_like=True,
        recommended_role="small_instance_reference",
    ),
    "routersense_p0p1p2_hint": AlgorithmMetadata(
        algorithm_id="routersense_p0p1p2_hint",
        display_name="RouterSense P0/P1/P2 Hint Adapter",
        heuristic_family="early_runtime_hint_adapter",
        role="online_adapter",
        paired_algorithm_id=None,
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="online_phase_sync_adapter",
        source="online_runtime_adapter",
        online_eligible=True,
        offline_eligible=False,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="legacy_online_adapter",
        notes="Early online adapter only; not the core POC1 U-family.",
    ),
    "routersense_joint_priority_phase_sync": AlgorithmMetadata(
        algorithm_id="routersense_joint_priority_phase_sync",
        display_name="RouterSense Joint Priority Phase Sync",
        heuristic_family="joint_priority_bridge",
        role="online_adapter",
        paired_algorithm_id="U_gated_maxweight_matching",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="online_phase_sync_adapter",
        source="online_runtime_adapter",
        online_eligible=True,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="phase_sync_bridge_candidate",
    ),
}


def list_heuristic_families() -> tuple[str, ...]:
    return tuple(sorted({metadata.heuristic_family for metadata in _ALGORITHMS.values()}))


def list_algorithms(role: str | None = None) -> tuple[dict[str, Any], ...]:
    entries = _ALGORITHMS.values()
    if role is not None:
        entries = [entry for entry in entries if entry.role == role]
    return tuple(entry.to_dict() for entry in sorted(entries, key=lambda item: item.algorithm_id))


def get_algorithm_metadata(algorithm_id: str) -> dict[str, Any]:
    return _ALGORITHMS[algorithm_id].to_dict()


def paired_algorithm_for(algorithm_id: str) -> dict[str, Any] | None:
    paired = _ALGORITHMS[algorithm_id].paired_algorithm_id
    if not paired or paired not in _ALGORITHMS:
        return None
    return _ALGORITHMS[paired].to_dict()


def local_oracle_reference() -> dict[str, Any]:
    return _ALGORITHMS["O_local_phase_oracle"].to_dict()


def joint_oracle_reference() -> dict[str, Any]:
    return _ALGORITHMS["O_joint_cp_sat_oracle"].to_dict()


def is_paired_comparison_ready(heuristic_family: str) -> bool:
    family_entries = [entry for entry in _ALGORITHMS.values() if entry.heuristic_family == heuristic_family]
    roles = {entry.role for entry in family_entries}
    if "b_phase_local" not in roles or "u_routersense_joint" not in roles:
        return False
    if any(entry.source in {"legacy_poc1_pending", "pending"} for entry in family_entries if entry.role in {"b_phase_local", "u_routersense_joint"}):
        return False
    family_ids = {entry.algorithm_id for entry in family_entries}
    for entry in family_entries:
        if entry.role == "u_routersense_joint" and (entry.paired_algorithm_id not in family_ids):
            return False
        if entry.role == "b_phase_local" and (entry.paired_algorithm_id not in family_ids):
            return False
    return True


def is_phase_local_oracle(algorithm_id: str) -> bool:
    metadata = _ALGORITHMS[algorithm_id]
    return metadata.role == "o_local_phase_oracle" or metadata.oracle_like and metadata.heuristic_family == "birkhoff_bvn"


def is_joint_oracle(algorithm_id: str) -> bool:
    return _ALGORITHMS[algorithm_id].role == "o_joint_oracle"


def is_legacy_granularity_variant(algorithm_id: str) -> bool:
    return _ALGORITHMS[algorithm_id].granularity_mode in {
        "legacy_coarse_wave",
        "legacy_atomic_token",
        "legacy_fluid_reference",
    }


__all__ = [
    "AlgorithmMetadata",
    "get_algorithm_metadata",
    "is_joint_oracle",
    "is_legacy_granularity_variant",
    "is_paired_comparison_ready",
    "is_phase_local_oracle",
    "joint_oracle_reference",
    "list_algorithms",
    "list_heuristic_families",
    "local_oracle_reference",
    "paired_algorithm_for",
]
