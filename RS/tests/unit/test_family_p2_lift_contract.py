from rs.scheduling.families import get_family_kernel_spec


def test_primary_families_share_model_agnostic_p2_lift():
    for family_id in ("greedy_control", "gmwd", "rsbc", "fast_stage"):
        spec = get_family_kernel_spec(family_id)
        assert spec.scoring_model == "critical_frontier"
        assert spec.transitive_unlock_weight > 0.0
        assert "p2lift" in spec.kernel_version
        contract = spec.contract()
        assert contract["kernel_parameters"]["scoring_model"] == "critical_frontier"
