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
from rs.planning.validation import validate_window_plan_for_request


def _request() -> PlanningRequest:
    return PlanningRequest(
        identity=PlanningIdentity(request_id="req", source_layer_id="1", target_layer_id="2"),
        traffic=PlanningTraffic(
            p0_dispatch_rows=((0, 4), (3, 0)),
            p1_return_rows=((0, 3), (4, 0)),
        ),
        prediction_hint=PredictionHint(
            predictor_id="copy_current_dispatch",
            hint_type="copy_current_dispatch",
            target_dispatch_rows=((0, 2), (5, 0)),
            confidence=1.0,
            source_layer_id="1",
            target_layer_id="2",
        ),
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_p1_p2",
        planning_track="runtime_lookahead",
        p2_semantics="advisory_hint",
    )


def test_window_plan_validator_rejects_rank_out_of_range() -> None:
    request = _request()
    plan = WindowPlan(
        planner_id="fifo_bucket",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=(
            PlanWave(
                wave_id=0,
                flows=(PlannedFlow("f0", "p0_dispatch", 0, 2, 1, "ready", True),),
                estimated_duration=0.0,
            ),
        ),
        metadata={},
    )
    with pytest.raises(ValueError, match="dst_rank_out_of_range"):
        validate_window_plan_for_request(plan, request)


def test_window_plan_validator_rejects_duplicate_wave_id() -> None:
    request = _request()
    plan = WindowPlan(
        planner_id="fifo_bucket",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=(
            PlanWave(0, (PlannedFlow("f0", "p0_dispatch", 0, 1, 1, "ready", True),), 0.0),
            PlanWave(0, (PlannedFlow("f1", "p1_return", 1, 0, 1, "blocked", True),), 0.0),
        ),
        metadata={},
    )
    with pytest.raises(ValueError, match="duplicate_wave_id"):
        validate_window_plan_for_request(plan, request)


def test_window_plan_validator_rejects_missing_executable_coverage() -> None:
    request = _request()
    plan = WindowPlan(
        planner_id="fifo_bucket",
        planner_family="baseline",
        request_digest=request.semantic_digest(),
        waves=(
            PlanWave(
                wave_id=0,
                flows=(
                    PlannedFlow("f0", "p0_dispatch", 0, 1, 4, "ready", True),
                    PlannedFlow("f1", "p1_return", 1, 0, 1, "blocked", True),
                    PlannedFlow("f2", "p2_next_dispatch_forecast", 0, 1, 2, "advisory_only", False),
                    PlannedFlow("f3", "p2_next_dispatch_forecast", 1, 0, 5, "advisory_only", False),
                ),
                estimated_duration=0.0,
            ),
        ),
        metadata={},
    )
    with pytest.raises(ValueError, match="missing_executable_coverage"):
        validate_window_plan_for_request(plan, request)


def test_window_plan_validator_rejects_max_waves_exceeded() -> None:
    request = _request()
    limited_request = PlanningRequest(
        identity=request.identity,
        traffic=request.traffic,
        prediction_hint=request.prediction_hint,
        topology=request.topology,
        constraints=PlanningConstraints(bucket_rows=4, max_waves=1, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=request.weights,
        information_mode=request.information_mode,
        planning_track=request.planning_track,
        p2_semantics=request.p2_semantics,
    )
    plan = WindowPlan(
        planner_id="fifo_bucket",
        planner_family="baseline",
        request_digest=limited_request.semantic_digest(),
        waves=(
            PlanWave(0, (PlannedFlow("f0", "p0_dispatch", 0, 1, 4, "ready", True),), 0.0),
            PlanWave(
                1,
                (
                    PlannedFlow("f1", "p1_return", 0, 1, 3, "blocked", True),
                    PlannedFlow("f2", "p1_return", 1, 0, 4, "blocked", True),
                    PlannedFlow("f3", "p2_next_dispatch_forecast", 0, 1, 2, "advisory_only", False),
                    PlannedFlow("f4", "p2_next_dispatch_forecast", 1, 0, 5, "advisory_only", False),
                ),
                0.0,
            ),
        ),
        metadata={},
    )
    with pytest.raises(ValueError, match="max_waves_exceeded"):
        validate_window_plan_for_request(plan, limited_request)
