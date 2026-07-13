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
    assert specs["barrier_criticality_posthoc_best"].planner_family == "reference_joint"
    assert specs["barrier_criticality_posthoc_best"].reference_only is True
    assert specs["barrier_criticality_posthoc_best"].exact is False
    assert specs["birkhoff_fluid_reference"].planner_family == "reference_local"
    assert specs["oracle_local_cp_sat"].planner_family == "exact_local"
    assert specs["oracle_local_cp_sat"].exact is True
    assert specs["oracle_joint_cp_sat"].planner_family == "exact_joint"


def test_formal_planner_returns_window_plan() -> None:
    planner = PlannerRegistry.create("fifo_bucket", None)
    plan = planner.plan(_request())
    assert plan.planner_id == "fifo_bucket"
    assert plan.request_digest == _request().semantic_digest()


def test_request_semantic_digest_excludes_runtime_identity_but_identity_digest_changes() -> None:
    first = _request()
    second = PlanningRequest(
        identity=PlanningIdentity(request_id="other", run_id="other-run", forward_id="other-forward", window_id="other-window", source_layer_id="99", target_layer_id="100"),
        traffic=first.traffic,
        prediction_hint=first.prediction_hint,
        topology=first.topology,
        constraints=first.constraints,
        weights=first.weights,
        information_mode=first.information_mode,
    )
    assert first.semantic_digest() == second.semantic_digest()
    assert first.identity_digest() != second.identity_digest()


def test_window_plan_semantic_digest_excludes_metadata_but_audit_digest_changes() -> None:
    planner = PlannerRegistry.create("fifo_bucket", None)
    plan = planner.plan(_request())
    same_waves_new_metadata = type(plan)(
        planner_id=plan.planner_id,
        planner_family=plan.planner_family,
        request_digest=plan.request_digest,
        waves=plan.waves,
        metadata={"legacy_makespan": 123.0, "note": "different"},
    )
    assert plan.semantic_digest() == same_waves_new_metadata.semantic_digest()
    assert plan.audit_digest() != same_waves_new_metadata.audit_digest()
