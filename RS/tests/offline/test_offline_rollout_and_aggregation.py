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
from rs.offline import (
    OfflineEvaluator,
    OfflinePlanningRequestBuilder,
    build_evaluation_bundle,
    build_execution_truth,
    build_offline_record,
    evaluate_comparison_eligibility,
    paired_aggregate,
    run_prediction_rollout,
    schedule_quality_metrics,
)
from rs.offline.rollout import PredictionRolloutSpec
from rs.prediction import TrafficPredictionTrainingSample
from rs.simulation import CommonTaskSetSimulator, SimulationSpec


def _window(*, p2=((0, 4), (1, 0))) -> OfflineWindow:
    return OfflineWindow(
        window_identity="fixture:1->2",
        source_layer="1",
        target_layer="2",
        p0_actual=((0, 2), (3, 0)),
        p1_actual=((0, 3), (2, 0)),
        p2_actual=p2,
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


def _spec(*, track="execution_window") -> EvaluationSpec:
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


def _prediction(*, rows, predictor_id="history") -> PredictionResult:
    return PredictionResult(
        identity=PredictionIdentity(request_id="fixture", source_layer_id="1", target_layer_id="2"),
        hint=PredictionHint(
            predictor_id=predictor_id,
            hint_type="traffic_matrix",
            target_dispatch_rows=rows,
            confidence=0.75,
            oracle=predictor_id == "perfect_trace_hint",
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


def test_prediction_rollout_uses_train_only_history_and_marks_cold_start() -> None:
    samples = (
        TrafficPredictionTrainingSample(((0, 1), (2, 0)), ((0, 2), (1, 0)), (), ((0, 2), (1, 0)), "1", "2"),
        TrafficPredictionTrainingSample(((0, 2), (1, 0)), ((0, 1), (2, 0)), (((0, 1), (2, 0)),), ((0, 3), (1, 0)), "2", "3"),
        TrafficPredictionTrainingSample(((0, 3), (1, 0)), ((0, 1), (3, 0)), (((0, 2), (1, 0)),), ((0, 4), (1, 0)), "3", "4"),
    )
    records = run_prediction_rollout(
        samples=samples,
        predictor_id="linear",
        rollout_spec=PredictionRolloutSpec(train_count=1, validation_count=1, test_count=1, history_window=1),
    )
    assert records[0].cold_start_used is True
    assert records[1].fit_sample_count == 1
    assert records[2].fit_sample_count == 1


def test_schedule_quality_and_paired_aggregation_work_on_formal_records() -> None:
    window = _window()
    spec = _spec(track="runtime_lookahead")
    truth = build_execution_truth(window, spec)
    builder = OfflinePlanningRequestBuilder(bucket_rows=2)
    pred_zero = _prediction(rows=((0, 0), (0, 0)), predictor_id="zero")
    pred_hist = _prediction(rows=((0, 1), (4, 0)), predictor_id="history")
    pred_perf = _prediction(rows=window.p2_actual, predictor_id="perfect_trace_hint")
    req_zero = builder.build(window, pred_zero, spec)
    req_hist = builder.build(window, pred_hist, spec)
    req_perf = builder.build(window, pred_perf, spec)
    eval_zero = OfflineEvaluator().evaluate(_plan(request_digest=req_zero.semantic_digest(), p2_rows=((0, 0), (0, 0))), truth, spec)
    eval_hist = OfflineEvaluator().evaluate(_plan(request_digest=req_hist.semantic_digest(), p2_rows=((0, 1), (4, 0))), truth, spec)
    eval_perf = OfflineEvaluator().evaluate(_plan(request_digest=req_perf.semantic_digest(), p2_rows=window.p2_actual), truth, spec)
    rec_zero = build_offline_record(window=window, spec=spec, task_set_digest=truth.task_set.task_set_digest, request=req_zero, prediction=pred_zero, plan=_plan(request_digest=req_zero.semantic_digest(), p2_rows=((0, 0), (0, 0))), execution_truth_digest=truth.truth_digest, evaluation=eval_zero, planner_reported_makespan=1.0, audit_status="valid", coverage_status="complete")
    rec_hist = build_offline_record(window=window, spec=spec, task_set_digest=truth.task_set.task_set_digest, request=req_hist, prediction=pred_hist, plan=_plan(request_digest=req_hist.semantic_digest(), p2_rows=((0, 1), (4, 0))), execution_truth_digest=truth.truth_digest, evaluation=eval_hist, planner_reported_makespan=1.0, audit_status="valid", coverage_status="complete")
    rec_perf = build_offline_record(window=window, spec=spec, task_set_digest=truth.task_set.task_set_digest, request=req_perf, prediction=pred_perf, plan=_plan(request_digest=req_perf.semantic_digest(), p2_rows=window.p2_actual), execution_truth_digest=truth.truth_digest, evaluation=eval_perf, planner_reported_makespan=1.0, audit_status="valid", coverage_status="complete")
    metrics = schedule_quality_metrics(predicted_record=rec_hist, zero_record=rec_zero, perfect_record=rec_perf)
    assert metrics["valid"] is True
    aggregate = paired_aggregate((rec_zero, rec_hist, rec_perf), baseline_predictor_id="zero")
    assert aggregate["sample_count"] == 2
    bundle = build_evaluation_bundle(spec=spec, records=(rec_zero, rec_hist, rec_perf), paired_aggregates=(aggregate,))
    assert bundle.schema_version == "offline_bundle_v1"


def test_fairness_gate_rejects_spec_and_prediction_mismatch() -> None:
    window = _window()
    truth = build_execution_truth(window, _spec())
    builder = OfflinePlanningRequestBuilder(bucket_rows=2)
    left_prediction = _prediction(rows=((0, 1), (4, 0)), predictor_id="history")
    right_prediction = _prediction(rows=((0, 1), (4, 0)), predictor_id="copy_current")
    left_request = builder.build(window, left_prediction, _spec())
    right_request = builder.build(window, right_prediction, _spec())
    rejection = evaluate_comparison_eligibility(
        left_spec=_spec(),
        right_spec=_spec(track="runtime_lookahead"),
        left_truth=truth,
        right_truth=truth,
        left_request=left_request,
        right_request=right_request,
        left_prediction=left_prediction,
        right_prediction=right_prediction,
    )
    assert rejection.eligible is False
    assert rejection.reason == "evaluation_spec_mismatch"


def test_window_plan_simulator_reports_unresolved_for_incomplete_plan() -> None:
    window = _window()
    spec = _spec()
    truth = build_execution_truth(window, spec)
    incomplete = WindowPlan(
        planner_id="test",
        planner_family="baseline",
        request_digest="digest",
        waves=(PlanWave(0, (PlannedFlow("p0_dispatch:0->1", "p0_dispatch", 0, 1, 2, "ready", True),), 2.0),),
        metadata={},
    )
    result = CommonTaskSetSimulator().simulate(
        truth.task_set,
        incomplete,
        SimulationSpec("offline_common_v1", "matrix_cell", 0.0, 1.0, 1, 1, "p1_return", "full_duplex", "row_cost"),
    )
    assert result.success is False
    assert len(result.unresolved_tasks) > 0
