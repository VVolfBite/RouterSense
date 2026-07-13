from __future__ import annotations

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningRequest,
    PlanningTopology,
    PlanningTraffic,
    PlanningWeights,
    PredictionHint,
)
from rs.planning import PlannerRegistry, PlannerSelectionMode, PlannerSelector


def _request() -> PlanningRequest:
    return PlanningRequest(
        identity=PlanningIdentity(request_id="req", source_layer_id="1", target_layer_id="2"),
        traffic=PlanningTraffic(p0_dispatch_rows=((0, 4), (3, 0)), p1_return_rows=((0, 3), (4, 0))),
        prediction_hint=PredictionHint(
            predictor_id="copy_current",
            hint_type="traffic_matrix",
            target_dispatch_rows=((0, 4), (3, 0)),
            confidence=1.0,
            source_layer_id="1",
            target_layer_id="2",
        ),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
    )


def test_selector_local_and_joint_modes_do_not_double_plan() -> None:
    selector = PlannerSelector(
        local_planner=PlannerRegistry.create("fifo_bucket", None),
        joint_planner=PlannerRegistry.create("barrier_criticality_joint", None),
    )
    local = selector.select(_request(), mode=PlannerSelectionMode.LOCAL)
    joint = selector.select(_request(), mode=PlannerSelectionMode.JOINT)
    assert local.local_plan is not None and local.joint_plan is None
    assert joint.joint_plan is not None and joint.local_plan is None


def test_selector_compare_returns_both_scores() -> None:
    selector = PlannerSelector(
        local_planner=PlannerRegistry.create("fifo_bucket", None),
        joint_planner=PlannerRegistry.create("barrier_criticality_joint", None),
    )
    selected = selector.select(_request(), mode=PlannerSelectionMode.COMPARE)
    assert selected.local_score is not None
    assert selected.joint_score is not None


def test_selector_select_prebuilt_does_not_require_replanning() -> None:
    request = _request()
    local_plan = PlannerRegistry.create("fifo_bucket", None).plan(request)
    joint_plan = PlannerRegistry.create("barrier_criticality_joint", None).plan(request)
    selector = PlannerSelector(
        local_planner=PlannerRegistry.create("fifo_bucket", None),
        joint_planner=PlannerRegistry.create("barrier_criticality_joint", None),
    )
    selected = selector.select_prebuilt(
        request=request,
        local_plan=local_plan,
        joint_plan=joint_plan,
        mode=PlannerSelectionMode.COMPARE,
    )
    assert selected.local_plan == local_plan
    assert selected.joint_plan == joint_plan
