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
from rs.planning import PlannerRegistry


def _request() -> PlanningRequest:
    return PlanningRequest(
        identity=PlanningIdentity(request_id="req", source_layer_id="1", target_layer_id="2"),
        traffic=PlanningTraffic(
            p0_dispatch_rows=((0, 4), (3, 0)),
            p1_return_rows=((0, 3), (4, 0)),
        ),
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


def test_planner_registry_exposes_formal_metadata() -> None:
    specs = {spec.planner_id: spec for spec in PlannerRegistry.specs()}
    assert "fifo_bucket" in specs
    assert specs["fifo_bucket"].planner_family == "baseline"


def test_formal_planner_returns_window_plan() -> None:
    planner = PlannerRegistry.create("fifo_bucket", None)
    plan = planner.plan(_request())
    assert plan.planner_id == "fifo_bucket"
    assert plan.request_digest == _request().semantic_digest()
