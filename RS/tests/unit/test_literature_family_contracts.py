from __future__ import annotations

from rs.scheduling.families import (
    EXPERIMENTAL_FAMILY_IDS,
    FAMILY_KERNEL_SPECS,
    STRICT_FAMILY_IDS,
    family_inventory,
)


def test_primary_families_have_honest_literature_labels() -> None:
    assert STRICT_FAMILY_IDS == (
        "greedy_control",
        "gmwd",
        "rsbc",
        "fast_stage",
    )
    assert EXPERIMENTAL_FAMILY_IDS == ("aurora_order", "adaptive_price")
    for family_id in STRICT_FAMILY_IDS:
        lineage = FAMILY_KERNEL_SPECS[family_id].literature
        assert lineage.paper_label
        assert lineage.mapping_level in {"control", "style", "original", "inspired"}
        assert lineage.implemented_mechanisms


def test_fast_and_aurora_names_do_not_overclaim_full_reproduction() -> None:
    fast = FAMILY_KERNEL_SPECS["fast_stage"].literature
    assert fast.mapping_level == "inspired"
    assert "intra-server rebalancing" in fast.missing_mechanisms
    assert "two-tier scale-out topology model" in fast.missing_mechanisms

    aurora = FAMILY_KERNEL_SPECS["aurora_order"].literature
    assert aurora.mapping_level == "inspired"
    assert "expert/model placement optimization" in aurora.missing_mechanisms


def test_gmwd_core_is_residual_maxweight_only() -> None:
    spec = FAMILY_KERNEL_SPECS["gmwd"]
    assert spec.exact_matching is True
    assert spec.atomic is False
    assert spec.residual_weight == 1.0
    assert spec.barrier_weight == 0.0
    assert spec.prediction_weight == 0.0
    assert spec.release_gain_weight == 0.0


def test_rsbc_records_release_gain_as_shared_kernel_parameter() -> None:
    spec = FAMILY_KERNEL_SPECS["rsbc"]
    assert spec.release_gain_weight > 0.0
    contract = spec.contract()
    assert contract["kernel_parameters"]["release_gain_weight"] == spec.release_gain_weight


def test_inventory_separates_primary_and_experimental_pairs() -> None:
    inventory = family_inventory()
    assert {row["family_id"] for row in inventory["primary_strict_families"]} == set(STRICT_FAMILY_IDS)
    assert {row["family_id"] for row in inventory["experimental_strict_families"]} == {"aurora_order", "adaptive_price"}
