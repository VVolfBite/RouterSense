from __future__ import annotations

import json

from rs.core.contracts import (
    PlanningConstraints,
    PlanningIdentity,
    PlanningTopology,
    PlanningWeights,
)
from rs.planning.request_builder import build_window_planning_request


def test_optional_identity_none_remains_none() -> None:
    request = build_window_planning_request(
        identity=PlanningIdentity(request_id="r", run_id=None, source_layer_id=None, target_layer_id=None),
        p0_dispatch_rows=((0, 1), (1, 0)),
        p1_return_rows=((0, 1), (1, 0)),
        p2_hint_rows=((0, 0), (0, 0)),
        predictor_id="zero_hint",
        confidence=0.0,
        topology=PlanningTopology(world_size=2),
        constraints=PlanningConstraints(bucket_rows=2, max_waves=4, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(),
        information_mode="p0_only",
        hint_type="zero_hint",
        planning_track="runtime_lookahead",
        p2_semantics="absent",
    )
    assert request.identity.run_id is None
    assert request.identity.source_layer_id is None
    assert request.prediction_hint.source_layer_id is None
    assert '"None"' not in json.dumps(request.to_dict())
