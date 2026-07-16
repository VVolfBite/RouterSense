from __future__ import annotations

import json
from pathlib import Path

from experiments.paper.adapters.scheduling_adapter import execute_policy, replay_window_from_matrices
from rs.scheduling.algorithm_catalog import list_pair_families
from rs.scheduling.catalog import resolve_algorithm_id
from rs.scheduling.families import LEGACY_UNPAIRED_FAMILIES, STRICT_FAMILY_IDS, family_inventory
from rs.scheduling.registry import resolve_policy


def _window():
    fixture = json.loads(Path("tests/fixtures/offline_replay_smoke/replay_layer_1.json").read_text(encoding="utf-8"))
    matrix = lambda key: tuple(tuple(int(value) for value in row) for row in fixture[key])
    return replay_window_from_matrices(
        fixture_id="family-smoke",
        layer_id=1,
        p0_matrix=matrix("p0_dispatch_matrix"),
        p1_matrix=matrix("p1_return_matrix"),
        p2_matrix=matrix("p2_next_dispatch_matrix"),
    )


def test_scope_expression_and_legacy_aliases_resolve_to_same_family_layer() -> None:
    cases = (
        ("Local(greedy_control)", "greedy_control", "local"),
        ("B_gated_greedy_maximal", "greedy_control", "local"),
        ("Joint(gmwd)", "gmwd", "joint"),
        ("U_gated_maxweight_matching", "gmwd", "joint"),
        ("Local(rsbc)", "rsbc", "local"),
        ("U_barrier_criticality_global_matching", "rsbc", "joint"),
        ("Local(fast_stage)", "fast_stage", "local"),
        ("Joint(aurora_order)", "aurora_order", "joint"),
    )
    for name, family_id, scope in cases:
        policy = resolve_policy(policy_name=name, bucket_rows=1)
        assert policy.family_id == family_id
        assert policy.scope.value == scope


def test_catalog_exposes_canonical_family_ids_and_expressions() -> None:
    assert resolve_algorithm_id("Local(greedy_control)").canonical_name == "greedy_control_local"
    assert resolve_algorithm_id("Joint(gmwd)").canonical_name == "gmwd_joint"
    assert resolve_algorithm_id("Local(rsbc)").canonical_name == "rsbc_local"
    assert resolve_algorithm_id("Joint(fast_stage)").canonical_name == "fast_stage_joint"
    assert resolve_algorithm_id("Local(aurora_order)").canonical_name == "aurora_order_local"
    # Historical scope expressions remain compatibility aliases.
    assert resolve_algorithm_id("Local(gated_greedy)").canonical_name == "greedy_control_local"
    assert resolve_algorithm_id("Joint(gated_maxweight)").canonical_name == "gmwd_joint"


def test_all_strict_families_have_valid_same_core_local_joint_plans() -> None:
    window = _window()
    contract_fields = (
        "matching_core_id",
        "task_contract_digest",
        "bucket_contract_digest",
        "cost_contract_digest",
        "service_model_id",
        "solver_budget_digest",
        "kernel_parameters",
    )
    for family_id in STRICT_FAMILY_IDS:
        local = execute_policy(
            replay_window=window,
            policy_name=f"Local({family_id})",
            hint_type="perfect_trace_hint",
            p2_hint_rows=window.p2_truth_rows,
        )
        joint = execute_policy(
            replay_window=window,
            policy_name=f"Joint({family_id})",
            hint_type="perfect_trace_hint",
            p2_hint_rows=window.p2_truth_rows,
        )
        assert local["audit_valid"], (family_id, local["audit"]["validation_errors"])
        assert joint["audit_valid"], (family_id, joint["audit"]["validation_errors"])
        local_meta = local["plan_metadata"]
        joint_meta = joint["plan_metadata"]
        assert local_meta["family_id"] == joint_meta["family_id"] == family_id
        assert local_meta["family_scope"] == "local"
        assert joint_meta["family_scope"] == "joint"
        assert local_meta["phase_independent"] is True
        assert joint_meta["phase_independent"] is False
        assert local_meta["kernel_call_count"] == 3
        assert joint_meta["kernel_call_count"] == 1
        for field in contract_fields:
            assert local_meta["common_core"][field] == joint_meta["common_core"][field], (family_id, field)


def test_family_inventory_separates_strict_and_legacy_unpaired_candidates() -> None:
    inventory = family_inventory()
    strict_ids = {row["family_id"] for row in inventory["strict_families"]}
    assert strict_ids == set(STRICT_FAMILY_IDS)
    assert set(LEGACY_UNPAIRED_FAMILIES) == {"lagrangian", "ibbr"}
    assert all(row["status"] == "STRICT_SAME_CORE_READY" for row in inventory["strict_families"])
    assert all(row["status"] == "LEGACY_NOT_STRICT" for row in inventory["legacy_unpaired_families"])


def test_pair_catalog_marks_new_strict_families_ready() -> None:
    rows = {row["heuristic_family"]: row for row in list_pair_families()}
    for family_id in STRICT_FAMILY_IDS:
        assert rows[family_id]["paired_comparison_ready"] is True
        assert rows[family_id]["status"] == "ready"
