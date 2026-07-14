from __future__ import annotations

from rs.core.contracts import (
    EvaluationSpec,
    OfflineWindow,
    PlanWave,
    PlannedFlow,
    PredictionHint,
    PredictionIdentity,
    PredictionResult,
    TrafficProvenance,
    WindowPlan,
)
from rs.offline import OfflineEvaluator, OfflinePlanningRequestBuilder, build_execution_truth, build_evaluation_task_set
from rs.simulation import CommonTaskSetSimulator, SimulationSpec


def _window(*, p2_actual=((0, 4), (1, 0))) -> OfflineWindow:
    return OfflineWindow(
        window_identity="fixture:1->2",
        source_layer="1",
        target_layer="2",
        p0_actual=((0, 2), (3, 0)),
        p1_actual=((0, 3), (2, 0)),
        p2_actual=p2_actual,
        placement_snapshot={"group_size": 2},
        traffic_provenance=TrafficProvenance.REAL_EP_OBSERVED,
        matrix_unit="rows",
        return_model="transpose_dispatch",
        raw_token_count=5,
        used_token_count=5,
        dropped_token_count=0,
        drop_reason=None,
        trace_digest="trace-a",
    )


def _spec(*, track="runtime_lookahead") -> EvaluationSpec:
    return EvaluationSpec(
        track=track,
        world_size=2,
        task_granularity="matrix_cell",
        matrix_unit="rows",
        time_unit="row_cost",
        cost_model_id="offline_common_v1",
        release_model="p1_return",
        return_model="transpose_dispatch",
        full_duplex=True,
        launch_cost=0.0,
        bytes_per_row=1,
        bandwidth=1.0,
        compute_delay=0.0,
        p2_semantics="actual",
        residual_policy="reject",
    )


def _prediction(*, hint_rows=((0, 1), (4, 0)), predictor_id="history") -> PredictionResult:
    return PredictionResult(
        identity=PredictionIdentity(request_id="fixture:1->2", source_layer_id="1", target_layer_id="2"),
        hint=PredictionHint(
            predictor_id=predictor_id,
            hint_type="traffic_matrix",
            target_dispatch_rows=hint_rows,
            confidence=0.75,
            oracle=False,
            source_layer_id="1",
            target_layer_id="2",
        ),
    )


def _plan(*, request_digest: str, p2_rows=((0, 4), (1, 0))) -> WindowPlan:
    return WindowPlan(
        planner_id="test",
        planner_family="joint",
        request_digest=request_digest,
        waves=(
            PlanWave(0, (PlannedFlow("p0_dispatch:0->1", "p0_dispatch", 0, 1, 2, "ready", True),), 2.0),
            PlanWave(1, (PlannedFlow("p0_dispatch:1->0", "p0_dispatch", 1, 0, 3, "ready", True),), 3.0),
            PlanWave(2, (PlannedFlow("p1_return:0->1", "p1_return", 0, 1, 3, "blocked", False),), 3.0),
            PlanWave(3, (PlannedFlow("p1_return:1->0", "p1_return", 1, 0, 2, "blocked", False),), 2.0),
            PlanWave(4, (PlannedFlow("p2_next_dispatch:0->1", "p2_next_dispatch", 0, 1, p2_rows[0][1], "after_p1", False),), float(p2_rows[0][1])),
            PlanWave(5, (PlannedFlow("p2_next_dispatch:1->0", "p2_next_dispatch", 1, 0, p2_rows[1][0], "after_p1", False),), float(p2_rows[1][0])),
        ),
        metadata={"legacy_makespan": 999.0},
    )


def test_single_offline_builder_keeps_truth_out_of_planning_request_digest() -> None:
    builder = OfflinePlanningRequestBuilder(bucket_rows=2)
    request_a = builder.build(_window(p2_actual=((0, 9), (2, 0))), _prediction(hint_rows=((0, 1), (4, 0))), _spec())
    request_b = builder.build(_window(p2_actual=((0, 5), (6, 0))), _prediction(hint_rows=((0, 1), (4, 0))), _spec())
    assert request_a.semantic_digest() == request_b.semantic_digest()
    truth_a = build_execution_truth(_window(p2_actual=((0, 9), (2, 0))), _spec())
    truth_b = build_execution_truth(_window(p2_actual=((0, 5), (6, 0))), _spec())
    assert truth_a.truth_digest != truth_b.truth_digest


def test_hint_change_changes_planning_request_but_not_truth_digest() -> None:
    builder = OfflinePlanningRequestBuilder(bucket_rows=2)
    window = _window(p2_actual=((0, 7), (2, 0)))
    request_a = builder.build(window, _prediction(hint_rows=((0, 1), (4, 0))), _spec())
    request_b = builder.build(window, _prediction(hint_rows=((0, 4), (1, 0))), _spec())
    assert request_a.semantic_digest() != request_b.semantic_digest()
    truth_a = build_execution_truth(window, _spec())
    truth_b = build_execution_truth(window, _spec())
    assert truth_a.truth_digest == truth_b.truth_digest


def test_common_evaluator_rejects_unexpected_or_incomplete_tasks() -> None:
    truth = build_execution_truth(_window(), _spec(track="execution_window"))
    bad_plan = WindowPlan(
        planner_id="test",
        planner_family="local",
        request_digest="digest",
        waves=(
            PlanWave(
                wave_id=0,
                flows=(
                    PlannedFlow("p0_dispatch:0->1", "p0_dispatch", 0, 1, 2, "ready", True),
                    PlannedFlow("p1_return:1->0", "p1_return", 1, 0, 99, "blocked", False),
                ),
                estimated_duration=2.0,
            ),
        ),
        metadata={},
    )
    evaluation = OfflineEvaluator().evaluate(bad_plan, truth, _spec(track="execution_window"))
    assert evaluation.valid is False
    assert evaluation.reason in {"row_count_mismatch:p1_return:1->0", "unexpected_task:p1_return:1->0"}


def test_common_evaluator_ignores_reported_metadata_makespan() -> None:
    truth = build_execution_truth(_window(), _spec(track="execution_window"))
    plan_a = WindowPlan(
        planner_id="test",
        planner_family="local",
        request_digest="digest",
        waves=(
            PlanWave(0, (PlannedFlow("p0_dispatch:0->1", "p0_dispatch", 0, 1, 2, "ready", True),), 2.0),
            PlanWave(1, (PlannedFlow("p0_dispatch:1->0", "p0_dispatch", 1, 0, 3, "ready", True),), 3.0),
            PlanWave(2, (PlannedFlow("p1_return:0->1", "p1_return", 0, 1, 3, "blocked", False),), 3.0),
            PlanWave(3, (PlannedFlow("p1_return:1->0", "p1_return", 1, 0, 2, "blocked", False),), 2.0),
            PlanWave(4, (PlannedFlow("p2_next_dispatch:0->1", "p2_next_dispatch", 0, 1, 4, "after_p1", False),), 4.0),
            PlanWave(5, (PlannedFlow("p2_next_dispatch:1->0", "p2_next_dispatch", 1, 0, 1, "after_p1", False),), 1.0),
        ),
        metadata={"legacy_makespan": 1.0},
    )
    plan_b = WindowPlan(
        planner_id="test",
        planner_family="local",
        request_digest="digest",
        waves=plan_a.waves,
        metadata={"legacy_makespan": 999.0},
    )
    eval_a = OfflineEvaluator().evaluate(plan_a, truth, _spec(track="execution_window"))
    eval_b = OfflineEvaluator().evaluate(plan_b, truth, _spec(track="execution_window"))
    assert eval_a.realized_makespan == eval_b.realized_makespan


def test_task_set_tracks_all_three_phases() -> None:
    task_set = build_evaluation_task_set(_window(), _spec(track="execution_window"))
    assert len(task_set.p0_tasks) == 2
    assert len(task_set.p1_tasks) == 2
    assert len(task_set.p2_tasks) == 2


def test_phase_a_simulator_fails_closed_without_execution_truth_adapter() -> None:
    task_set = build_evaluation_task_set(_window(), _spec(track="execution_window"))
    plan = WindowPlan(planner_id="test", planner_family="baseline", request_digest="digest", waves=(), metadata={})
    result = CommonTaskSetSimulator().simulate(
        task_set,
        plan,
        SimulationSpec(
            service_model="offline_common_v1",
            task_granularity="matrix_cell",
            launch_cost=0.0,
            bandwidth=1.0,
            bytes_per_row=1,
            max_inflight=1,
            release_model="p1_return",
            port_model="full_duplex",
            time_unit="row_cost",
        ),
    )
    assert result.success is False
    assert len(result.unresolved_tasks) == len(task_set.tasks)


def test_compute_delay_changes_realized_makespan() -> None:
    truth = build_execution_truth(_window(), _spec(track="execution_window"))
    plan = _plan(request_digest="digest")
    fast = OfflineEvaluator().evaluate(plan, truth, _spec(track="execution_window"))
    slow = OfflineEvaluator().evaluate(
        plan,
        build_execution_truth(_window(), EvaluationSpec(
            track="execution_window",
            world_size=2,
            task_granularity="matrix_cell",
            matrix_unit="rows",
            time_unit="row_cost",
            cost_model_id="offline_common_v1",
            release_model="p1_return",
            return_model="transpose_dispatch",
            full_duplex=True,
            launch_cost=0.0,
            bytes_per_row=1,
            bandwidth=1.0,
            compute_delay=100.0,
            p2_semantics="actual",
            residual_policy="reject",
        )),
        EvaluationSpec(
            track="execution_window",
            world_size=2,
            task_granularity="matrix_cell",
            matrix_unit="rows",
            time_unit="row_cost",
            cost_model_id="offline_common_v1",
            release_model="p1_return",
            return_model="transpose_dispatch",
            full_duplex=True,
            launch_cost=0.0,
            bytes_per_row=1,
            bandwidth=1.0,
            compute_delay=100.0,
            p2_semantics="actual",
            residual_policy="reject",
        ),
    )
    assert fast.realized_makespan is not None
    assert slow.realized_makespan is not None
    assert slow.realized_makespan > fast.realized_makespan


def test_same_wave_dependency_is_invalid() -> None:
    truth = build_execution_truth(_window(), _spec(track="execution_window"))
    bad_plan = WindowPlan(
        planner_id="test",
        planner_family="joint",
        request_digest="digest",
        waves=(
            PlanWave(
                0,
                (
                    PlannedFlow("p0_dispatch:0->1", "p0_dispatch", 0, 1, 2, "ready", True),
                    PlannedFlow("p1_return:1->0", "p1_return", 1, 0, 2, "blocked", False),
                ),
                2.0,
            ),
        ),
        metadata={},
    )
    evaluation = OfflineEvaluator().evaluate(bad_plan, truth, _spec(track="execution_window"))
    assert evaluation.valid is False
    assert evaluation.reason == "dependency_violation"


def test_flow_order_invariance_within_wave() -> None:
    truth = build_execution_truth(_window(), _spec(track="execution_window"))
    wave_a = PlanWave(
        0,
        (
            PlannedFlow("p0_dispatch:0->1", "p0_dispatch", 0, 1, 2, "ready", True),
            PlannedFlow("p0_dispatch:1->0", "p0_dispatch", 1, 0, 3, "ready", True),
        ),
        3.0,
    )
    wave_b = PlanWave(
        0,
        (
            PlannedFlow("p0_dispatch:1->0", "p0_dispatch", 1, 0, 3, "ready", True),
            PlannedFlow("p0_dispatch:0->1", "p0_dispatch", 0, 1, 2, "ready", True),
        ),
        3.0,
    )
    plan_a = WindowPlan(planner_id="test", planner_family="joint", request_digest="digest", waves=(wave_a,), metadata={})
    plan_b = WindowPlan(planner_id="test", planner_family="joint", request_digest="digest", waves=(wave_b,), metadata={})
    eval_a = OfflineEvaluator().evaluate(plan_a, truth, _spec(track="runtime_lookahead"))
    eval_b = OfflineEvaluator().evaluate(plan_b, truth, _spec(track="runtime_lookahead"))
    assert eval_a.valid == eval_b.valid
    assert eval_a.realized_makespan == eval_b.realized_makespan


def test_builder_rejects_world_size_mismatch() -> None:
    builder = OfflinePlanningRequestBuilder(bucket_rows=2)
    bad_spec = EvaluationSpec(
        track="runtime_lookahead",
        world_size=3,
        task_granularity="matrix_cell",
        matrix_unit="rows",
        time_unit="row_cost",
        cost_model_id="offline_common_v1",
        release_model="p1_return",
        return_model="transpose_dispatch",
        full_duplex=True,
        launch_cost=0.0,
        bytes_per_row=1,
        bandwidth=1.0,
        compute_delay=0.0,
        p2_semantics="actual",
        residual_policy="reject",
    )
    try:
        builder.build(_window(), _prediction(hint_rows=((0, 1), (4, 0))), bad_spec)
    except ValueError as exc:
        assert "world size" in str(exc)
    else:
        raise AssertionError("expected world size mismatch rejection")


def test_task_set_and_truth_reject_forged_digests() -> None:
    spec = _spec(track="execution_window")
    task_set = build_evaluation_task_set(_window(), spec)
    forged = task_set.__class__(
        task_set_digest="forged",
        tasks=task_set.tasks,
        p0_tasks=task_set.p0_tasks,
        p1_tasks=task_set.p1_tasks,
        p2_tasks=task_set.p2_tasks,
        world_size=task_set.world_size,
        task_granularity=task_set.task_granularity,
        coverage_summary=task_set.coverage_summary,
    )
    try:
        forged.validate()
    except ValueError as exc:
        assert "task_set_digest" in str(exc)
    else:
        raise AssertionError("expected task_set digest validation failure")
    truth = build_execution_truth(_window(), spec)
    forged_truth = truth.__class__(
        task_set=truth.task_set,
        actual_matrices=truth.actual_matrices,
        actual_release_dependencies=truth.actual_release_dependencies,
        truth_digest="forged",
        provenance=truth.provenance,
    )
    try:
        forged_truth.validate()
    except ValueError as exc:
        assert "truth_digest" in str(exc)
    else:
        raise AssertionError("expected truth digest validation failure")
