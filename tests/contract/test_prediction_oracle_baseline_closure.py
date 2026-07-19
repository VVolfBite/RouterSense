from __future__ import annotations

from types import SimpleNamespace

import pytest

from rs.runtime.offline.prediction_oracle_baseline_closure import (
    _build_problem_with_hint_and_truth,
    _generate_exact_instances,
    _solve_exact_oracle,
)
from rs.runtime.online.megatron_ep.lifecycle import RouterSenseInjectionRuntime
from rs.scheduling.reference.exact_small_instance import (
    EXACT_COST_MODEL_ID,
    EXACT_REFERENCE_MODEL_ID,
    EXACT_RELEASE_MODEL_ID,
    EXACT_TASK_MODEL_ID,
)


def test_build_problem_separates_planning_hint_from_execution_truth() -> None:
    p0 = ((0, 2), (1, 0))
    p1 = ((0, 1), (2, 0))
    hint = ((0, 0), (0, 0))
    truth = ((0, 3), (4, 0))
    problem = _build_problem_with_hint_and_truth(
        p0=p0,
        p1=p1,
        planning_hint=hint,
        execution_truth=truth,
        hint_type="zero_hint",
        scheduling_mode="execution_window",
        expert_compute_delay=0.0,
    )
    assert problem.forecast is not None
    assert problem.forecast.matrix == hint
    assert problem.p2_next_dispatch_forecast_matrix == truth
    assert problem.forecast.metadata["planning_hint_matrix"] == [[0, 0], [0, 0]]


def test_exact_oracle_joint_not_worse_than_local_on_generated_instance() -> None:
    instance = _generate_exact_instances(1)[0]
    local = _solve_exact_oracle(instance, mode="local")
    joint = _solve_exact_oracle(instance, mode="joint")
    assert local["solver_status"] == "OPTIMAL"
    assert joint["solver_status"] == "OPTIMAL"
    assert int(joint["objective"]) <= int(local["objective"])


def test_exact_oracle_pair_uses_one_runtime_model_and_only_changes_scope() -> None:
    instance = _generate_exact_instances(4)[-1]
    local = _solve_exact_oracle(instance, mode="local")
    joint = _solve_exact_oracle(instance, mode="joint")
    for result in (local, joint):
        assert result["reference_model"] == EXACT_REFERENCE_MODEL_ID
        assert result["task_model_id"] == EXACT_TASK_MODEL_ID
        assert result["cost_model_id"] == EXACT_COST_MODEL_ID
        assert result["release_model_id"] == EXACT_RELEASE_MODEL_ID
        assert result["scope_only_difference"] is True
    assert local["scope"] == "local"
    assert joint["scope"] == "joint"


def test_exact_oracle_no_longer_requires_optional_cp_sat() -> None:
    instance = _generate_exact_instances(1)[0]
    result = _solve_exact_oracle(instance, mode="local")
    assert result["solver_status"] == "OPTIMAL"
    assert result["certified_optimal"] is True


def test_runtime_predictor_name_validation_is_explicit() -> None:
    runtime = RouterSenseInjectionRuntime.__new__(RouterSenseInjectionRuntime)
    runtime.config = SimpleNamespace(online_p2_predictor="oracle")
    with pytest.raises(ValueError):
        runtime._build_online_predictor()
