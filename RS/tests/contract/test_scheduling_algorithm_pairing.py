from __future__ import annotations

from rs.scheduling.algorithm_catalog import (
    get_algorithm_metadata,
    is_joint_oracle,
    is_legacy_granularity_variant,
    is_paired_comparison_ready,
    is_phase_local_oracle,
    list_heuristic_families,
    pair_status_summary,
    paired_algorithm_for,
)


def test_main_u_algorithms_have_paired_b_or_explicit_pending_catalog() -> None:
    assert "gated_greedy" in list_heuristic_families()
    assert "gated_maxweight_matching" in list_heuristic_families()
    assert "barrier_criticality_matching" in list_heuristic_families()
    assert paired_algorithm_for("U_gated_maxweight_matching")["algorithm_id"] == "B_gated_maxweight_matching"
    assert paired_algorithm_for("U_barrier_criticality_global_matching")["algorithm_id"] == "B_barrier_criticality_matching"


def test_b_birkhoff_is_phase_local_oracle_like_reference() -> None:
    meta = get_algorithm_metadata("B_birkhoff")
    assert meta["role"] == "o_local_phase_oracle"
    assert meta["oracle_like"] is True
    assert meta["deterministic_solver"] is True
    assert is_phase_local_oracle("B_birkhoff") is True


def test_u_algorithms_are_not_marked_as_oracles() -> None:
    assert get_algorithm_metadata("U_gated_maxweight_matching")["role"] == "u_routersense_joint"
    assert get_algorithm_metadata("U_gated_maxweight_matching")["oracle_like"] is False
    assert get_algorithm_metadata("U_barrier_criticality_global_matching")["role"] == "u_routersense_joint"


def test_joint_oracle_and_legacy_granularity_are_classified_explicitly() -> None:
    assert is_joint_oracle("O_joint_cp_sat_oracle") is True
    assert get_algorithm_metadata("O_joint_cp_sat_oracle")["heavy_solver"] is True
    assert is_legacy_granularity_variant("U_gated_maxweight_matching_atomic") is True
    assert is_legacy_granularity_variant("B_birkhoff_wave") is True


def test_p0p1p2_hint_is_classified_as_online_adapter() -> None:
    meta = get_algorithm_metadata("routersense_p0p1p2_hint")
    assert meta["role"] == "online_adapter"
    assert meta["heuristic_family"] == "early_runtime_hint_adapter"


def test_ready_paired_families_are_explicit() -> None:
    assert is_paired_comparison_ready("birkhoff_bvn") is True
    assert is_paired_comparison_ready("gated_greedy") is True
    assert is_paired_comparison_ready("gated_maxweight_matching") is True
    assert is_paired_comparison_ready("barrier_criticality_matching") is True


def test_pair_status_summary_reports_ready_and_pending() -> None:
    summary = pair_status_summary()
    assert summary["ready_pair_count"] >= 6
    assert summary["pending_pair_count"] >= 1
    assert any(row["heuristic_family"] == "birkhoff_bvn" for row in summary["ready_pairs"])
    assert any(row["heuristic_family"] == "gated_maxweight_matching" for row in summary["ready_pairs"])
    assert any(row["heuristic_family"] == "cp_lpt" for row in summary["pending_pairs"])
