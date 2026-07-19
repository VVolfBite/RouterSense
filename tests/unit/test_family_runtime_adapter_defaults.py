from __future__ import annotations

import json
from pathlib import Path

from rs.runtime.offline.scheduling_adapter import execute_policy, replay_window_from_matrices
from rs.scheduling.families import FAMILY_KERNEL_SPECS


def _window():
    fixture = json.loads(
        Path("tests/fixtures/offline_replay_smoke/replay_layer_1.json").read_text(
            encoding="utf-8"
        )
    )
    matrix = lambda key: tuple(
        tuple(int(value) for value in row) for row in fixture[key]
    )
    return replay_window_from_matrices(
        fixture_id="family-defaults",
        layer_id=1,
        p0_matrix=matrix("p0_dispatch_matrix"),
        p1_matrix=matrix("p1_return_matrix"),
        p2_matrix=matrix("p2_next_dispatch_matrix"),
    )


def test_formal_runtime_adapter_preserves_registered_family_kernel_defaults() -> None:
    window = _window()
    result = execute_policy(
        replay_window=window,
        policy_name="rsbc_joint",
        hint_type="perfect_trace_hint",
        p2_hint_rows=window.p2_truth_rows,
        confidence=1.0,
    )
    kernel = result["plan_metadata"]["common_core"]["kernel_parameters"]
    spec = FAMILY_KERNEL_SPECS["rsbc"]
    assert kernel["residual_weight"] == spec.residual_weight
    assert kernel["barrier_weight"] == spec.barrier_weight
    assert kernel["age_weight"] == spec.age_weight
    assert kernel["prediction_weight"] == spec.prediction_weight
    assert kernel["release_gain_weight"] == spec.release_gain_weight


def test_rscf_local_and_joint_use_identical_critical_frontier_kernel() -> None:
    window = _window()
    rows = []
    for policy_name in ("rscf_local", "rscf_joint"):
        result = execute_policy(
            replay_window=window,
            policy_name=policy_name,
            hint_type="perfect_trace_hint",
            p2_hint_rows=window.p2_truth_rows,
            confidence=1.0,
        )
        assert result["audit_valid"] is True
        rows.append(result["plan_metadata"]["common_core"])
    for key in (
        "matching_core_id",
        "task_contract_digest",
        "bucket_contract_digest",
        "cost_contract_digest",
        "service_model_id",
        "solver_budget_digest",
        "kernel_parameters",
    ):
        assert rows[0][key] == rows[1][key]
