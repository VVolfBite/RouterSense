from __future__ import annotations

from rs.scheduling.algorithm_catalog import paired_algorithm_for


def test_same_core_pairing_catalog_is_explicit() -> None:
    assert paired_algorithm_for("U_barrier_criticality_global_matching")["algorithm_id"] == "B_barrier_criticality_matching"
