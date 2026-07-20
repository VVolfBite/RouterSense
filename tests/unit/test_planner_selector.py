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
from rs.planning import PlannerRegistry, PlanningSelectionError, PlannerSelectionMode, PlannerSelector


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
        joint_planner=PlannerRegistry.create("future:p012:joint:global:rscf", None),
    )
    local = selector.select(_request(), mode=PlannerSelectionMode.LOCAL)
    joint = selector.select(_request(), mode=PlannerSelectionMode.JOINT)
    assert local.local_plan is not None and local.joint_plan is None
    assert joint.joint_plan is not None and joint.local_plan is None


def test_selector_compare_returns_both_scores() -> None:
    selector = PlannerSelector(
        local_planner=PlannerRegistry.create("fifo_bucket", None),
        joint_planner=PlannerRegistry.create("future:p012:joint:global:rscf", None),
    )
    selected = selector.select(_request(), mode=PlannerSelectionMode.COMPARE)
    assert selected.local_score is not None
    assert selected.joint_score is not None


def test_selector_select_prebuilt_does_not_require_replanning() -> None:
    request = _request()
    local_plan = PlannerRegistry.create("fifo_bucket", None).plan(request)
    joint_plan = PlannerRegistry.create("future:p012:joint:global:rscf", None).plan(request)
    selector = PlannerSelector(
        local_planner=PlannerRegistry.create("fifo_bucket", None),
        joint_planner=PlannerRegistry.create("future:p012:joint:global:rscf", None),
    )
    selected = selector.select_prebuilt(
        request=request,
        local_plan=local_plan,
        joint_plan=joint_plan,
        mode=PlannerSelectionMode.COMPARE,
    )
    assert selected.local_plan == local_plan
    assert selected.joint_plan == joint_plan


def test_selector_prefers_valid_plan_when_other_is_invalid() -> None:
    request = _request()
    local_plan = PlannerRegistry.create("fifo_bucket", None).plan(request)
    invalid_joint = WindowPlan(
        planner_id="invalid_joint",
        planner_family="joint",
        request_digest=request.semantic_digest(),
        waves=(
            PlanWave(
                wave_id=0,
                flows=(PlannedFlow(flow_id="bad", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=-1, release_state="ready", executable=True),),
                estimated_duration=999.0,
            ),
        ),
    )
    selector = PlannerSelector(
        local_planner=PlannerRegistry.create("fifo_bucket", None),
        joint_planner=PlannerRegistry.create("future:p012:joint:global:rscf", None),
    )
    selected = selector.select_prebuilt(
        request=request,
        local_plan=local_plan,
        joint_plan=invalid_joint,
        mode=PlannerSelectionMode.COMPARE,
    )
    assert selected.selected_plan == local_plan
    assert selected.joint_score is not None and selected.joint_score.valid is False


def test_selector_raises_when_both_plans_invalid() -> None:
    request = _request()
    invalid_local = WindowPlan(
        planner_id="invalid_local",
        planner_family="local",
        request_digest=request.semantic_digest(),
        waves=(PlanWave(wave_id=0, flows=(PlannedFlow(flow_id="bad-local", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=-1, release_state="ready", executable=True),), estimated_duration=0.0),),
    )
    invalid_joint = WindowPlan(
        planner_id="invalid_joint",
        planner_family="joint",
        request_digest=request.semantic_digest(),
        waves=(PlanWave(wave_id=0, flows=(PlannedFlow(flow_id="bad-joint", phase="p0_dispatch", src_rank=0, dst_rank=1, row_count=-1, release_state="ready", executable=True),), estimated_duration=0.0),),
    )
    selector = PlannerSelector(
        local_planner=PlannerRegistry.create("fifo_bucket", None),
        joint_planner=PlannerRegistry.create("future:p012:joint:global:rscf", None),
    )
    with pytest.raises(PlanningSelectionError, match="no_valid_plan"):
        selector.select_prebuilt(
            request=request,
            local_plan=invalid_local,
            joint_plan=invalid_joint,
            mode=PlannerSelectionMode.COMPARE,
        )
