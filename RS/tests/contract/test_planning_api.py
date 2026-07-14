from __future__ import annotations

import pytest
import subprocess
import sys
import os
from pathlib import Path

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


def test_request_semantic_digest_excludes_predictor_provenance() -> None:
    first = _request()
    second = PlanningRequest(
        identity=first.identity,
        traffic=first.traffic,
        prediction_hint=PredictionHint(
            predictor_id="history",
            hint_type="different_hint",
            target_dispatch_rows=first.prediction_hint.target_dispatch_rows,
            confidence=0.25,
            oracle=True,
            source_layer_id="999",
            target_layer_id="1000",
        ),
        topology=first.topology,
        constraints=first.constraints,
        weights=first.weights,
        information_mode=first.information_mode,
    )
    assert first.semantic_digest() == second.semantic_digest()


def test_request_validate_rejects_negative_matrix_and_bad_information_mode() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PlanningRequest(
            identity=PlanningIdentity(request_id="bad"),
            traffic=PlanningTraffic(p0_dispatch_rows=((0, -1), (2, 0)), p1_return_rows=((0, 2), (1, 0))),
            prediction_hint=PredictionHint(
                predictor_id="copy_current",
                hint_type="traffic_matrix",
                target_dispatch_rows=((0, 1), (1, 0)),
                confidence=1.0,
            ),
            topology=PlanningTopology(world_size=2),
            constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
            weights=PlanningWeights(),
            information_mode="p0_p1_p2",
        ).validate()
    with pytest.raises(ValueError, match="information_mode"):
        PlanningRequest(
            identity=PlanningIdentity(request_id="bad-mode"),
            traffic=_request().traffic,
            prediction_hint=_request().prediction_hint,
            topology=_request().topology,
            constraints=_request().constraints,
            weights=_request().weights,
            information_mode="unknown",
        ).validate()


def test_planning_request_digest_is_stable_across_processes() -> None:
    code = """
from rs.core.contracts import PlanningConstraints, PlanningIdentity, PlanningRequest, PlanningTopology, PlanningTraffic, PlanningWeights, PredictionHint
request = PlanningRequest(
    identity=PlanningIdentity(request_id="req", run_id="run-a", source_layer_id="1", target_layer_id="2"),
    traffic=PlanningTraffic(p0_dispatch_rows=((0, 4), (3, 0)), p1_return_rows=((0, 3), (4, 0))),
    prediction_hint=PredictionHint(predictor_id="copy_current", hint_type="traffic_matrix", target_dispatch_rows=((0, 4), (3, 0)), confidence=1.0),
    topology=PlanningTopology(world_size=2),
    constraints=PlanningConstraints(bucket_rows=4, max_waves=8, expert_compute_delay=0.0, phase_release_model="p1_return"),
    weights=PlanningWeights(),
    information_mode="p0_p1_p2",
)
print(request.semantic_digest())
"""
    env = dict(os.environ)
    cwd = Path(__file__).resolve().parents[2]
    pythonpath_entries = [str(cwd / "src"), str(cwd)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["PYTHONHASHSEED"] = "123"
    first = subprocess.run([sys.executable, "-c", code], cwd=str(cwd), env=env, capture_output=True, text=True, check=True)
    env["PYTHONHASHSEED"] = "456"
    second = subprocess.run([sys.executable, "-c", code], cwd=str(cwd), env=env, capture_output=True, text=True, check=True)
    assert first.stdout.strip() == second.stdout.strip()


def test_planning_topology_rejects_multi_port_contract() -> None:
    with pytest.raises(ValueError, match="max_outgoing_per_rank_per_wave == 1"):
        PlanningTopology(world_size=2, max_outgoing_per_rank_per_wave=2).validate()
    with pytest.raises(ValueError, match="max_incoming_per_rank_per_wave == 1"):
        PlanningTopology(world_size=2, max_incoming_per_rank_per_wave=2).validate()
