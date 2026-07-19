from __future__ import annotations

from pathlib import Path

from rs.core.contracts import PlanningConstraints, PlanningIdentity, PlanningTopology, PlanningWeights
from rs.planning import PlannerRegistry
from rs.planning.request_builder import build_window_planning_request
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.runtime.online.megatron_ep.target_planning.contracts import _compat_logical_plan_from_window_plan

from experiments.paper.adapters.scheduling_adapter import replay_window_from_matrices
from experiments.paper.contracts import RecordMetadata
from experiments.paper.scheduling_evaluation import evaluate_scheduling
from experiments.paper.trace_dataset import load_replay_fixture
from rs.runtime.offline.replay_unified import PlanningHint, ReplayEngine, build_execution_truth, build_multiphase_problem, build_planning_problem


def test_execution_window_replay_engine_and_evaluator_schedule_executable_p2() -> None:
    fixture = load_replay_fixture(Path(__file__).resolve().parents[2] / "fixtures" / "offline_replay_smoke" / "replay_layer_1.json")
    replay_window = replay_window_from_matrices(
        fixture_id="bridge",
        layer_id=int(fixture["metadata"]["layer_id"]),
        p0_matrix=fixture["p0_dispatch_matrix"],
        p1_matrix=fixture["p1_return_matrix"],
        p2_matrix=fixture["p2_next_dispatch_matrix"],
    )
    hint = PlanningHint(
        hint_type="perfect_trace_hint",
        p2_hint_rows=replay_window.p2_truth_rows,
        confidence=1.0,
        source_layer=replay_window.layer_id,
        target_layer=replay_window.layer_id + 1,
    )
    planning_problem = build_planning_problem(replay_window=replay_window, planning_hint=hint)
    execution_truth = build_execution_truth(replay_window)
    problem = build_multiphase_problem(
        planning_problem=planning_problem,
        execution_truth=execution_truth,
        scheduling_mode="execution_window",
        expert_compute_delay=0.0,
        max_waves=256,
    )
    request = build_window_planning_request(
        identity=PlanningIdentity(
            request_id="bridge:direct",
            run_id="bridge",
            window_id=str(replay_window.window_id),
            source_layer_id=str(replay_window.layer_id),
            target_layer_id=str(replay_window.layer_id + 1),
        ),
        p0_dispatch_rows=replay_window.p0_truth_rows,
        p1_return_rows=replay_window.p1_truth_rows,
        p2_hint_rows=replay_window.p2_truth_rows,
        predictor_id="perfect_trace_hint",
        confidence=1.0,
        topology=PlanningTopology(world_size=replay_window.group_size),
        constraints=PlanningConstraints(bucket_rows=1, max_waves=256, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(p0_weight=1.0, p1_weight=1.0, p2_weight=1.0),
        information_mode="p0_p1_p2",
        hint_type="perfect_trace_hint",
        oracle=True,
        planning_track="execution_window",
        p2_semantics="executable_actual",
    )
    planner = PlannerRegistry.create("U_barrier_criticality_global_matching", None)
    direct_plan = planner.plan(request)
    direct_audit = replay_and_audit_logical_plan(problem, _compat_logical_plan_from_window_plan(direct_plan))
    replay_result = ReplayEngine(scheduling_mode="execution_window", expert_compute_delay=0.0, bucket_rows=1).execute(
        replay_window=replay_window,
        planning_hint=hint,
        policy_name="U_barrier_criticality_global_matching",
    )
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    evaluator = evaluate_scheduling(
        fixture_dir=Path(__file__).resolve().parents[2] / "fixtures" / "offline_replay_smoke",
        metadata=metadata,
        model_id="m",
        model_revision="rev",
        policy_ids=("U_barrier_criticality_global_matching",),
    )
    bridge = evaluator["execution_window_bridge_summary"]
    assert direct_audit["valid"] is True
    assert replay_result["audit_valid"] is True
    assert replay_result["planning_track"] == "execution_window"
    assert replay_result["p2_semantics"] == "executable_actual"
    assert not any("incomplete inbound barrier volume for phase=2" in item for item in direct_audit.get("validation_errors", ()))
    assert not any("incomplete inbound barrier volume for phase=2" in item for item in replay_result["audit"].get("validation_errors", ()))
    assert bridge["direct_global_scheduler"]["audit_valid"] is True
    assert bridge["replay_engine"]["audit_valid"] is True
