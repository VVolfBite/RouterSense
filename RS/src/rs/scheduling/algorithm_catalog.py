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
        display_name="Birkhoff Phase-Local Baseline",
        heuristic_family="birkhoff_bvn",
        role="b_phase_local",
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
        recommended_role="strong_phase_local_baseline",
        notes="Birkhoff-family engineering phase-local baseline; oracle-like, but not the formal fluid oracle object.",
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
        source="derived_phase_local_from_u",
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
        source="legacy_poc1_recovered",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="paired_joint_family_candidate",
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
        source="derived_phase_local_from_u",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=True,
        oracle_like=False,
        recommended_role="paired_b_reference",
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
        source="derived_phase_local_from_u",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=False,
        oracle_like=False,
        recommended_role="paired_b_reference",
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
        notes="Formal local oracle reference implemented by birkhoff_von_neumann_fluid under fluid / crossbar semantics.",
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
    "RS_safe_gated_greedy": AlgorithmMetadata(
        algorithm_id="RS_safe_gated_greedy",
        display_name="RouterSense Safe Gated Greedy",
        heuristic_family="gated_greedy",
        role="u_routersense_joint",
        paired_algorithm_id="B_gated_greedy_maximal",
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
        recommended_role="paired_u_safe_candidate",
        notes="Guarded safe joint scheduler that never regresses against paired B under the same information set.",
    ),
    "RS_safe_gated_maxweight": AlgorithmMetadata(
        algorithm_id="RS_safe_gated_maxweight",
        display_name="RouterSense Safe Gated Maxweight",
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
        recommended_role="paired_u_safe_candidate",
        notes="Guarded safe joint scheduler that compares raw U and paired B under the same information set.",
    ),
    "RS_safe_barrier_criticality": AlgorithmMetadata(
        algorithm_id="RS_safe_barrier_criticality",
        display_name="RouterSense Safe Barrier Criticality",
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
        recommended_role="paired_u_safe_candidate",
    ),
    "RS_safe_barrier_price": AlgorithmMetadata(
        algorithm_id="RS_safe_barrier_price",
        display_name="RouterSense Safe Barrier Price",
        heuristic_family="barrier_price_adaptive_matching",
        role="u_routersense_joint",
        paired_algorithm_id="B_barrier_price_adaptive_matching",
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
        recommended_role="paired_u_safe_candidate",
    ),
    "RS_safe_lagrangian": AlgorithmMetadata(
        algorithm_id="RS_safe_lagrangian",
        display_name="RouterSense Safe Lagrangian",
        heuristic_family="lagrangian_cross_phase",
        role="u_routersense_joint",
        paired_algorithm_id="B_lagrangian_phase_local",
        local_oracle_reference_id="O_local_phase_oracle",
        joint_oracle_reference_id="O_joint_cp_sat_oracle",
        granularity_mode="dynamic_bucket_current",
        planning_scope="multiphase_joint",
        source="current_mainline",
        online_eligible=False,
        offline_eligible=True,
        heavy_solver=False,
        deterministic_solver=False,
        oracle_like=False,
        recommended_role="paired_u_safe_candidate",
    ),
    "RS_safe_ibbr": AlgorithmMetadata(
        algorithm_id="RS_safe_ibbr",
        display_name="RouterSense Safe IBBR",
        heuristic_family="birkhoff_bvn",
        role="u_routersense_joint",
        paired_algorithm_id="B_birkhoff",
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
        recommended_role="paired_u_safe_candidate",
    ),
}


# Literature-grounded strict same-core families introduced by the
# Local(f)/Joint(f) scope layer.  Historical metadata remains above so old
# artifacts stay readable.
def _register_strict_family_metadata() -> None:
    rows = (
        ("greedy_control", "greedy_control_local", "greedy_control_joint", "Greedy Control"),
        ("gmwd", "gmwd_local", "gmwd_joint", "GMWD-style"),
        ("rsbc", "rsbc_local", "rsbc_joint", "RouterSense Barrier Criticality"),
        ("rscf", "rscf_local", "rscf_joint", "RouterSense Critical Frontier"),
        ("fast_stage", "fast_stage_local", "fast_stage_joint", "FAST-Stage"),
        ("aurora_order", "aurora_order_local", "aurora_order_joint", "Aurora-Order"),
        ("adaptive_price", "adaptive_price_local", "adaptive_price_joint", "Adaptive Price"),
    )
    for family, local_id, joint_id, display in rows:
        _ALGORITHMS[local_id] = AlgorithmMetadata(
            algorithm_id=local_id,
            display_name=f"{display} Local",
            heuristic_family=family,
            role="b_phase_local_strict",
            paired_algorithm_id=joint_id,
            local_oracle_reference_id="O_local_phase_oracle",
            joint_oracle_reference_id="O_joint_cp_sat_oracle",
            granularity_mode="canonical_bucket",
            planning_scope="phase_local_scope_adapter",
            source="literature_grounded_family_scope_layer",
            online_eligible=True,
            offline_eligible=True,
            heavy_solver=False,
            deterministic_solver=True,
            oracle_like=False,
            recommended_role="strict_family_local",
            notes="Shares one immutable kernel with Joint; only visible information and ready-set scope differ.",
        )
        _ALGORITHMS[joint_id] = AlgorithmMetadata(
            algorithm_id=joint_id,
            display_name=f"{display} Joint",
            heuristic_family=family,
            role="u_joint_strict",
            paired_algorithm_id=local_id,
            local_oracle_reference_id="O_local_phase_oracle",
            joint_oracle_reference_id="O_joint_cp_sat_oracle",
            granularity_mode="canonical_bucket",
            planning_scope="global_release_aware_scope_adapter",
            source="literature_grounded_family_scope_layer",
            online_eligible=True,
            offline_eligible=True,
            heavy_solver=False,
            deterministic_solver=True,
            oracle_like=False,
            recommended_role="strict_family_joint",
            notes="Shares one immutable kernel with Local; only visible information and ready-set scope differ.",
        )


_register_strict_family_metadata()


_PAIR_FAMILIES: tuple[tuple[str, str | None, str | None], ...] = (
    ("greedy_control", "greedy_control_local", "greedy_control_joint"),
    ("gmwd", "gmwd_local", "gmwd_joint"),
    ("rsbc", "rsbc_local", "rsbc_joint"),
    ("rscf", "rscf_local", "rscf_joint"),
    ("fast_stage", "fast_stage_local", "fast_stage_joint"),
    ("aurora_order", "aurora_order_local", "aurora_order_joint"),
    ("adaptive_price", "adaptive_price_local", "adaptive_price_joint"),
    # Historical same-family compatibility rows.  They remain readable but are
    # not part of the strict information-scope claim.
    ("birkhoff_bvn", "B_birkhoff", "U_ibbr"),
    ("gated_greedy", "B_gated_greedy_maximal", "U_gated_greedy_maximal"),
    ("gated_maxweight_matching", "B_gated_maxweight_matching", "U_gated_maxweight_matching"),
    ("barrier_criticality_matching", "B_barrier_criticality_matching", "U_barrier_criticality_global_matching"),
    ("legacy_birkhoff_ibbr", "B_birkhoff", "U_ibbr"),
    ("legacy_lagrangian", "B_lagrangian_phase_local", "U_lagrangian"),
    ("cp_lpt", None, None),
)

_PAIR_PENDING_REASONS: dict[str, str] = {
    "legacy_birkhoff_ibbr": "Legacy U adds iterative repair absent from B; not a strict information-scope control.",
    "legacy_lagrangian": "Historical local and joint implementations do not share one update core.",
    "cp_lpt": "Historical U_cp_lpt exists, but no formal paired B-side is promoted yet.",
}


def list_heuristic_families() -> tuple[str, ...]:
    return tuple(sorted({metadata.heuristic_family for metadata in _ALGORITHMS.values()}))


def list_pair_families() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for family, b_id, u_id in _PAIR_FAMILIES:
        b_meta = _ALGORITHMS.get(b_id) if b_id else None
        u_meta = _ALGORITHMS.get(u_id) if u_id else None
        ready = is_paired_comparison_ready(family)
        status = "ready" if ready else "pending"
        rows.append(
            {
                "heuristic_family": family,
                "B_algorithm": b_id,
                "U_algorithm": u_id,
                "B_source": None if b_meta is None else b_meta.source,
                "U_source": None if u_meta is None else u_meta.source,
                "status": status,
                "paired_comparison_ready": ready,
                "pending_reason": "" if ready else _PAIR_PENDING_REASONS.get(family, "pair not yet promoted"),
            }
        )
    return tuple(rows)


def pair_status_summary() -> dict[str, Any]:
    rows = list_pair_families()
    ready_rows = [row for row in rows if row["paired_comparison_ready"]]
    pending_rows = [row for row in rows if not row["paired_comparison_ready"]]
    return {
        "ready_pair_count": len(ready_rows),
        "pending_pair_count": len(pending_rows),
        "ready_pairs": ready_rows,
        "pending_pairs": pending_rows,
    }


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
    pair_rows = [pair for pair in _PAIR_FAMILIES if pair[0] == heuristic_family]
    if not pair_rows:
        return False
    for _family, b_id, u_id in pair_rows:
        if not b_id or not u_id:
            return False
        if b_id not in _ALGORITHMS or u_id not in _ALGORITHMS:
            return False
        b_meta = _ALGORITHMS[b_id]
        u_meta = _ALGORITHMS[u_id]
        if b_meta.source in {"legacy_poc1_pending", "pending"} or u_meta.source in {"legacy_poc1_pending", "pending"}:
            return False
        if b_meta.role not in {"b_phase_local", "b_phase_local_strict", "o_local_phase_oracle"}:
            return False
        if u_meta.role not in {"u_routersense_joint", "u_joint_strict"}:
            return False
        if b_meta.paired_algorithm_id != u_id or u_meta.paired_algorithm_id != b_id:
            return False
    return True


def is_phase_local_oracle(algorithm_id: str) -> bool:
    metadata = _ALGORITHMS[algorithm_id]
    return metadata.role == "o_local_phase_oracle"


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
    "list_pair_families",
    "list_heuristic_families",
    "local_oracle_reference",
    "pair_status_summary",
    "paired_algorithm_for",
]
