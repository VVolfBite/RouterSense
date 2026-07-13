from __future__ import annotations

import pytest

from rs.core.contracts import (
    PlanWave,
    PlannedFlow,
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
    WindowPlan,
)
from rs.planning import CommonCorePlanEstimator, PlanningCostModel


def test_plan_estimator_uses_common_core_metrics() -> None:
    request = PlanningRequest(
        identity=PlanningIdentity(request_id="req"),
        traffic=PlanningTraffic(p0_dispatch_rows=((0, 1), (2, 0)), p1_return_rows=((0, 2), (1, 0))),
        prediction_hint=PredictionHint(predictor_id="zero", hint_type="traffic_matrix", target_dispatch_rows=((0, 0), (0, 0)), confidence=0.0),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    plan = WindowPlan(
        planner_id="test",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=(
            PlanWave(
                wave_id=0,
                flows=(PlannedFlow(flow_id="f", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=3, release_state="ready", executable=True),),
                estimated_duration=0.0,
            ),
        ),
    )
    score = CommonCorePlanEstimator().estimate(plan, request, PlanningCostModel(row_transfer_cost=2.0, launch_cost=1.0))
    assert score.estimated_makespan == 7.0


def test_plan_estimator_ignores_legacy_makespan_metadata_for_decision_value() -> None:
    request = PlanningRequest(
        identity=PlanningIdentity(request_id="req"),
        traffic=PlanningTraffic(p0_dispatch_rows=((0, 1), (2, 0)), p1_return_rows=((0, 2), (1, 0))),
        prediction_hint=PredictionHint(predictor_id="zero", hint_type="traffic_matrix", target_dispatch_rows=((0, 0), (0, 0)), confidence=0.0),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    base_plan = WindowPlan(
        planner_id="test",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=(
            PlanWave(
                wave_id=0,
                flows=(PlannedFlow(flow_id="f", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=3, release_state="ready", executable=True),),
                estimated_duration=0.0,
            ),
        ),
        metadata={"legacy_makespan": 999.0},
    )
    same_plan_different_meta = WindowPlan(
        planner_id="test",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=base_plan.waves,
        metadata={"legacy_makespan": 1.0},
    )
    estimator = CommonCorePlanEstimator()
    first = estimator.estimate(base_plan, request, PlanningCostModel(row_transfer_cost=2.0, launch_cost=1.0))
    second = estimator.estimate(same_plan_different_meta, request, PlanningCostModel(row_transfer_cost=2.0, launch_cost=1.0))
    assert first.estimated_makespan == second.estimated_makespan == 7.0


def test_plan_estimator_ignores_legacy_wave_duration_for_decision_value() -> None:
    request = PlanningRequest(
        identity=PlanningIdentity(request_id="req"),
        traffic=PlanningTraffic(p0_dispatch_rows=((0, 1), (2, 0)), p1_return_rows=((0, 2), (1, 0))),
        prediction_hint=PredictionHint(predictor_id="zero", hint_type="traffic_matrix", target_dispatch_rows=((0, 0), (0, 0)), confidence=0.0),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    fast = WindowPlan(
        planner_id="test",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=(PlanWave(wave_id=0, flows=(PlannedFlow(flow_id="f", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=3, release_state="ready", executable=True),), estimated_duration=0.0),),
    )
    slow_legacy = WindowPlan(
        planner_id="test",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=(PlanWave(wave_id=0, flows=fast.waves[0].flows, estimated_duration=999.0),),
    )
    estimator = CommonCorePlanEstimator()
    first = estimator.estimate(fast, request, PlanningCostModel(row_transfer_cost=2.0, launch_cost=1.0))
    second = estimator.estimate(slow_legacy, request, PlanningCostModel(row_transfer_cost=2.0, launch_cost=1.0))
    assert first.estimated_makespan == second.estimated_makespan == 7.0


def test_plan_estimator_rejects_digest_mismatch_and_invalid_flow() -> None:
    request = PlanningRequest(
        identity=PlanningIdentity(request_id="req"),
        traffic=PlanningTraffic(p0_dispatch_rows=((0, 1), (2, 0)), p1_return_rows=((0, 2), (1, 0))),
        prediction_hint=PredictionHint(predictor_id="zero", hint_type="traffic_matrix", target_dispatch_rows=((0, 0), (0, 0)), confidence=0.0),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    bad_digest = WindowPlan(
        planner_id="test",
        planner_family="baseline",
        request_digest="other",
        waves=(PlanWave(wave_id=0, flows=(PlannedFlow(flow_id="f", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=3, release_state="ready", executable=True),), estimated_duration=0.0),),
    )
    bad_flow = WindowPlan(
        planner_id="test",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=(PlanWave(wave_id=0, flows=(PlannedFlow(flow_id="f", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=-3, release_state="ready", executable=True),), estimated_duration=0.0),),
    )
    estimator = CommonCorePlanEstimator()
    mismatch = estimator.estimate(bad_digest, request, PlanningCostModel())
    invalid = estimator.estimate(bad_flow, request, PlanningCostModel())
    assert mismatch.valid is False and mismatch.reason == "request_digest_mismatch"
    assert invalid.valid is False


def test_plan_estimator_rejects_same_wave_port_conflict() -> None:
    request = PlanningRequest(
        identity=PlanningIdentity(request_id="req"),
        traffic=PlanningTraffic(p0_dispatch_rows=((0, 1), (2, 0)), p1_return_rows=((0, 2), (1, 0))),
        prediction_hint=PredictionHint(predictor_id="zero", hint_type="traffic_matrix", target_dispatch_rows=((0, 0), (0, 0)), confidence=0.0),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    conflict = WindowPlan(
        planner_id="test",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=(
            PlanWave(
                wave_id=0,
                flows=(
                    PlannedFlow(flow_id="f1", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=1, release_state="ready", executable=True),
                    PlannedFlow(flow_id="f2", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=2, release_state="ready", executable=True),
                ),
                estimated_duration=0.0,
            ),
        ),
    )
    score = CommonCorePlanEstimator().estimate(conflict, request, PlanningCostModel())
    assert score.valid is False


def test_plan_estimator_allows_full_duplex_send_and_receive_same_rank() -> None:
    request = PlanningRequest(
        identity=PlanningIdentity(request_id="req"),
        traffic=PlanningTraffic(p0_dispatch_rows=((0, 1, 0), (0, 0, 1), (1, 0, 0)), p1_return_rows=((0, 0, 1), (1, 0, 0), (0, 1, 0))),
        prediction_hint=PredictionHint(predictor_id="zero", hint_type="traffic_matrix", target_dispatch_rows=((0, 0, 0), (0, 0, 0), (0, 0, 0)), confidence=0.0),
        topology=PlanningTopology(world_size=3),
        constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )
    duplex = WindowPlan(
        planner_id="test",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=(
            PlanWave(
                wave_id=0,
                flows=(
                    PlannedFlow(flow_id="f1", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=1, release_state="ready", executable=True),
                    PlannedFlow(flow_id="f2", phase="p0_dispatch", src_rank=2, dst_rank=0, row_count=1, release_state="ready", executable=True),
                ),
                estimated_duration=0.0,
            ),
        ),
    )
    score = CommonCorePlanEstimator().estimate(duplex, request, PlanningCostModel())
    assert score.valid is True


def test_planning_cost_model_rejects_multi_port_configuration() -> None:
    with pytest.raises(ValueError, match="max_outgoing_per_rank_per_wave == 1"):
        PlanningCostModel(max_outgoing_per_rank_per_wave=2).validate()
    with pytest.raises(ValueError, match="max_incoming_per_rank_per_wave == 1"):
        PlanningCostModel(max_incoming_per_rank_per_wave=2).validate()
