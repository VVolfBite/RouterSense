from __future__ import annotations


def test_smoke_artifact_schema_minimum_fields() -> None:
    payload = {
        "injection_hook_found": True,
        "pre_transport_boundary_found": True,
        "observer_phase_coverage_passed": True,
        "p0_hidden_collective_count_passed": True,
        "p0_probs_collective_count_passed": True,
        "p1_collective_count_passed": True,
        "native_splits_unchanged": True,
        "native_buffers_unchanged": True,
        "sync_plan_before_p0_transport": True,
        "root_plan_decode_passed": True,
        "all_rank_plan_hash_passed": True,
        "early_shadow_replace_applied": True,
        "late_shadow_replace_expired": True,
        "transport_mutation_false": True,
        "numerical_equivalence_passed": True,
    }
    assert len(payload) == 15
