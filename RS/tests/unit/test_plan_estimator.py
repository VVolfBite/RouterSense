from __future__ import annotations

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
