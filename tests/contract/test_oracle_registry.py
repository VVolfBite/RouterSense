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
from rs.scheduling.reference import OracleRegistry
from rs.scheduling.reference.exact_small_instance import EXACT_REFERENCE_MODEL_ID


def _request() -> PlanningRequest:
    return PlanningRequest(
        identity=PlanningIdentity(request_id="r1"),
        topology=PlanningTopology(world_size=2, full_duplex=True),
        traffic=PlanningTraffic(
            p0_dispatch_rows=((0, 4), (0, 0)),
            p1_return_rows=((0, 0), (4, 0)),
        ),
        prediction_hint=PredictionHint(
            predictor_id="zero",
            hint_type="zero_hint",
            confidence=0.0,
            oracle=False,
            target_dispatch_rows=((0, 0), (0, 0)),
        ),
        constraints=PlanningConstraints(
            bucket_rows=4,
            max_waves=4,
            expert_compute_delay=0.0,
            phase_release_model="p1_return",
        ),
        weights=PlanningWeights(),
        information_mode="p0_p1",
        planning_track="execution_window",
        p2_semantics="absent",
    )


def test_oracle_registry_returns_formal_window_plan() -> None:
    result = OracleRegistry.solve("O_local", _request())

    assert result.oracle_id == "O_local"
    assert result.plan is not None
    assert result.plan.planner_id == "O_local"
    assert result.plan.request_digest == _request().semantic_digest()
    assert result.plan.metadata["logical_model"] == EXACT_REFERENCE_MODEL_ID
