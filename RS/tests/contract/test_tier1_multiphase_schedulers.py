from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rs.scheduling import FlowDemand, LogicalSchedulePlan, LogicalWave, resolve_phase_policy, resolve_policy
from rs.scheduling.multiphase.flow_model import EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE
from rs.scheduling.multiphase.replay import replay_and_audit_schedule
from rs.scheduling.multiphase.tier1 import FLUID_SERVICE_MODEL, TIER1_ALGORITHM_IDS
from rs.scheduling.validation import stable_hash, validate_logical_plan

from experiments.offline.run_tier1_cpu_validation import _build_problem
from experiments.offline.run_tier1_cpu_validation import DEFAULT_WAVE_TIER1_POLICIES


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "tier1"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURE_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def _expected_flows(problem):
    flows = list(problem.flow_window.ready_flows + problem.flow_window.blocked_flows)
    if problem.options.scheduling_mode == EXECUTION_WINDOW_MODE:
        for src_rank, row in enumerate(problem.p2_next_dispatch_forecast_matrix):
            for dst_rank, byte_count in enumerate(row):
                if src_rank != dst_rank and int(byte_count) > 0:
                    flows.append(
                        FlowDemand(
                            flow_id=f"p2_next_dispatch:{src_rank}->{dst_rank}",
                            phase="p2_next_dispatch",
                            src_rank=src_rank,
                            dst_rank=dst_rank,
                            byte_count=int(byte_count),
                            release_state="ready",
                            is_executable=True,
                        )
                    )
    return tuple(flows)


@pytest.mark.parametrize("algorithm_id", TIER1_ALGORITHM_IDS)
def test_tier1_runtime_lookahead_suppresses_real_p2_and_is_deterministic(algorithm_id: str) -> None:
    problem = _build_problem(
        _fixture("unlock_hotspot_4rank"),
        mode=RUNTIME_LOOKAHEAD_MODE,
        p2_source="perfect_trace",
        expert_compute_delay=2.0,
    )
    policy = resolve_policy(policy_name=algorithm_id, bucket_rows=0)
    plans = [policy.build_logical_plan(problem) for _ in range(3)]
    assert stable_hash(plans[0].to_dict()) == stable_hash(plans[1].to_dict()) == stable_hash(plans[2].to_dict())
    assert plans[0].diagnostics["algorithm_id"] == algorithm_id
    assert plans[0].diagnostics["mode"] == RUNTIME_LOOKAHEAD_MODE
    assert plans[0].diagnostics["forecast_available"] is True
    if algorithm_id in {"B_birkhoff", "B_birkhoff_wave", "U_lagrangian"}:
        assert plans[0].diagnostics["future_information_mode"] == "none"
        assert plans[0].diagnostics["forecast_consumed"] is False
        assert plans[0].diagnostics["prediction_used"] is False
        assert plans[0].diagnostics["evaluation_eligible"] is True
    else:
        assert plans[0].diagnostics["future_information_mode"] == "oracle_predicted_runtime_lookahead"
        assert plans[0].diagnostics["forecast_consumed"] is True
        assert plans[0].diagnostics["prediction_used"] is True
        assert plans[0].diagnostics["evaluation_eligible"] is False
    assert plans[0].diagnostics["valid"] is True, plans[0].diagnostics["audit"].get("validation_errors")
    assert all(flow.phase != "p2_next_dispatch" for wave in plans[0].waves for flow in wave.flows)
    validation = validate_logical_plan(plans[0], expected_flows=_expected_flows(problem), mode=RUNTIME_LOOKAHEAD_MODE)
    assert validation["valid"], validation["errors"]


@pytest.mark.parametrize("algorithm_id", TIER1_ALGORITHM_IDS)
def test_tier1_execution_window_schedules_real_p2(algorithm_id: str) -> None:
    problem = _build_problem(
        _fixture("unlock_hotspot_4rank"),
        mode=EXECUTION_WINDOW_MODE,
        p2_source="perfect_trace",
        expert_compute_delay=1.0,
    )
    plan = resolve_policy(policy_name=algorithm_id, bucket_rows=0).build_logical_plan(problem)
    assert plan.diagnostics["future_information_mode"] == "oracle_execution_window"
    assert plan.diagnostics["valid"] is True, plan.diagnostics["audit"].get("validation_errors")
    assert any(flow.phase == "p2_next_dispatch" for wave in plan.waves for flow in wave.flows)
    validation = validate_logical_plan(plan, expected_flows=_expected_flows(problem), mode=EXECUTION_WINDOW_MODE)
    assert validation["valid"], validation["errors"]


def test_tier1_atomic_and_fluid_service_models_are_isolated() -> None:
    problem = _build_problem(
        _fixture("fluid_split_4rank"),
        mode=RUNTIME_LOOKAHEAD_MODE,
        p2_source="copy_current_dispatch",
        expert_compute_delay=0.0,
    )
    atomic = resolve_policy(policy_name="U_gated_maxweight_matching_atomic", bucket_rows=0).build_logical_plan(problem)
    fluid = resolve_policy(policy_name="U_gated_maxweight_matching", bucket_rows=0).build_logical_plan(problem)
    atomic_origins = [flow.dependency_metadata["origin_flow_id"] for wave in atomic.waves for flow in wave.flows]
    fluid_origins = [flow.dependency_metadata["origin_flow_id"] for wave in fluid.waves for flow in wave.flows]
    assert len(atomic_origins) == len(set(atomic_origins))
    assert len(fluid_origins) > len(set(fluid_origins))
    assert atomic.diagnostics["service_model"] != FLUID_SERVICE_MODEL
    assert fluid.diagnostics["service_model"] == FLUID_SERVICE_MODEL
    assert validate_logical_plan(fluid, expected_flows=_expected_flows(problem), mode=RUNTIME_LOOKAHEAD_MODE)["valid"]


def test_tier1_policies_are_offline_only() -> None:
    for algorithm_id in TIER1_ALGORITHM_IDS:
        policy = resolve_policy(policy_name=algorithm_id, bucket_rows=0)
        assert policy.capabilities.supports_offline is True
        assert policy.capabilities.supports_online_phase_local_execution is False
        with pytest.raises(Exception):
            resolve_phase_policy(policy_name=algorithm_id, bucket_rows=0)


def test_tier1_cpu_runner_outputs_service_model_comparison(tmp_path: Path) -> None:
    output_dir = tmp_path / "tier1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.run_tier1_cpu_validation",
            "--fixture",
            str(FIXTURE_ROOT / "unlock_hotspot_4rank.json"),
            "--policy",
            "B_birkhoff,U_gated_maxweight_matching",
            "--mode",
            RUNTIME_LOOKAHEAD_MODE,
            "--p2-source",
            "copy_current_dispatch",
            "--expert-compute-delay",
            "1.0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    comparison = json.loads((output_dir / "comparison_by_service_model.json").read_text(encoding="utf-8"))
    assert [item["algorithm_id"] for item in comparison["atomic_comparison"]] == ["B_birkhoff"]
    assert [item["algorithm_id"] for item in comparison["fluid_comparison"]] == ["U_gated_maxweight_matching"]
    assert comparison["atomic_comparison"][0]["planning_time_ms_measured"] >= 0.0
    assert comparison["atomic_comparison"][0]["planning_time_ms_in_plan_hash"] == 0.0
    assert (output_dir / "policy_plan_B_birkhoff.json").exists()
    assert (output_dir / "diagnostics_U_gated_maxweight_matching.json").exists()


def test_tier1_cpu_runner_default_all_uses_wave_track_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "tier1_default"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.run_tier1_cpu_validation",
            "--fixture",
            str(FIXTURE_ROOT / "unlock_hotspot_4rank.json"),
            "--policy",
            "all",
            "--mode",
            RUNTIME_LOOKAHEAD_MODE,
            "--p2-source",
            "copy_current_dispatch",
            "--expert-compute-delay",
            "1.0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert tuple(manifest["policies"]) == DEFAULT_WAVE_TIER1_POLICIES
    comparison = json.loads((output_dir / "comparison_by_service_model.json").read_text(encoding="utf-8"))
    assert comparison["atomic_comparison"] == []
    assert [item["algorithm_id"] for item in comparison["fluid_comparison"]] == [
        "B_birkhoff_wave",
        "U_gated_maxweight_matching",
        "U_barrier_criticality_global_matching",
    ]
    assert [item["algorithm_id"] for item in comparison["other_service_model_comparison"]] == ["U_lagrangian"]


def test_execution_window_rejects_forecast_p2_sources(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.offline.run_tier1_cpu_validation",
            "--fixture",
            str(FIXTURE_ROOT / "unlock_hotspot_4rank.json"),
            "--policy",
            "B_birkhoff",
            "--mode",
            EXECUTION_WINDOW_MODE,
            "--p2-source",
            "copy_current_dispatch",
            "--output-dir",
            str(tmp_path / "invalid"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "execution_window requires" in proc.stderr


def test_execution_window_p2_role_and_runtime_lookahead_source_modes() -> None:
    execution_problem = _build_problem(_fixture("p2_local_release_witness_4rank"), mode=EXECUTION_WINDOW_MODE, p2_source="actual_trace", expert_compute_delay=1.0)
    execution_plan = resolve_policy(policy_name="U_gated_maxweight_matching", bucket_rows=0).build_logical_plan(execution_problem)
    assert execution_plan.diagnostics["p2_role"] == "executable_actual_traffic"
    assert execution_plan.diagnostics["future_information_mode"] == "oracle_execution_window"
    assert execution_plan.diagnostics["evaluation_eligible"] is False
    assert any(flow.phase == "p2_next_dispatch" for wave in execution_plan.waves for flow in wave.flows)
    assert validate_logical_plan(execution_plan, expected_flows=_expected_flows(execution_problem), mode=EXECUTION_WINDOW_MODE, expert_compute_delay=1.0)["valid"]

    zero_problem = _build_problem(_fixture("p2_local_release_witness_4rank"), mode=RUNTIME_LOOKAHEAD_MODE, p2_source="zero_hint", expert_compute_delay=1.0)
    zero_plan = resolve_policy(policy_name="U_gated_maxweight_matching", bucket_rows=0).build_logical_plan(zero_problem)
    assert zero_plan.diagnostics["p2_role"] == "advisory_forecast_pressure"
    assert zero_plan.diagnostics["future_information_mode"] == "none"
    assert zero_plan.diagnostics["forecast_consumed"] is False
    assert zero_plan.diagnostics["prediction_used"] is False

    copy_problem = _build_problem(_fixture("p2_local_release_witness_4rank"), mode=RUNTIME_LOOKAHEAD_MODE, p2_source="copy_current_dispatch", expert_compute_delay=1.0)
    copy_plan = resolve_policy(policy_name="U_gated_maxweight_matching", bucket_rows=0).build_logical_plan(copy_problem)
    assert copy_plan.diagnostics["future_information_mode"] == "heuristic_runtime_lookahead"
    assert copy_plan.diagnostics["forecast_consumed"] is True
    assert copy_plan.diagnostics["evaluation_eligible"] is True

    perfect_plan = resolve_policy(policy_name="U_gated_maxweight_matching", bucket_rows=0).build_logical_plan(
        _build_problem(_fixture("p2_local_release_witness_4rank"), mode=RUNTIME_LOOKAHEAD_MODE, p2_source="perfect_trace", expert_compute_delay=1.0)
    )
    assert perfect_plan.diagnostics["future_information_mode"] == "oracle_predicted_runtime_lookahead"
    assert perfect_plan.diagnostics["forecast_consumed"] is True
    assert perfect_plan.diagnostics["evaluation_eligible"] is False


@pytest.mark.parametrize("algorithm_id", ("B_birkhoff", "B_birkhoff_wave"))
def test_birkhoff_baselines_do_not_consume_unused_oracle_forecast(algorithm_id: str) -> None:
    problem = _build_problem(_fixture("p2_local_release_witness_4rank"), mode=RUNTIME_LOOKAHEAD_MODE, p2_source="perfect_trace", expert_compute_delay=1.0)
    plan = resolve_policy(policy_name=algorithm_id, bucket_rows=0).build_logical_plan(problem)
    assert plan.diagnostics["forecast_available"] is True
    assert plan.diagnostics["forecast_source"] == "perfect_trace"
    assert plan.diagnostics["forecast_consumed"] is False
    assert plan.diagnostics["prediction_used"] is False
    assert plan.diagnostics["future_information_mode"] == "none"
    assert plan.diagnostics["evaluation_eligible"] is True


def test_p1_and_p2_local_release_witnesses() -> None:
    p1_problem = _build_problem(_fixture("p1_local_release_witness_4rank"), mode=RUNTIME_LOOKAHEAD_MODE, p2_source="zero_hint", expert_compute_delay=3.0)
    p1_plan = resolve_policy(policy_name="U_gated_maxweight_matching", bucket_rows=0).build_logical_plan(p1_problem)
    p1_validation = validate_logical_plan(p1_plan, expected_flows=_expected_flows(p1_problem), mode=RUNTIME_LOOKAHEAD_MODE, expert_compute_delay=3.0)
    assert p1_validation["valid"], p1_validation["errors"]

    p2_problem = _build_problem(_fixture("p2_local_release_witness_4rank"), mode=EXECUTION_WINDOW_MODE, p2_source="actual_trace", expert_compute_delay=2.0)
    p2_plan = resolve_policy(policy_name="U_gated_maxweight_matching", bucket_rows=0).build_logical_plan(p2_problem)
    p2_validation = validate_logical_plan(p2_plan, expected_flows=_expected_flows(p2_problem), mode=EXECUTION_WINDOW_MODE, expert_compute_delay=2.0)
    assert p2_validation["valid"], p2_validation["errors"]

    lookahead_problem = _build_problem(_fixture("p2_local_release_witness_4rank"), mode=RUNTIME_LOOKAHEAD_MODE, p2_source="perfect_trace", expert_compute_delay=2.0)
    lookahead_plan = resolve_policy(policy_name="U_gated_maxweight_matching", bucket_rows=0).build_logical_plan(lookahead_problem)
    assert all(flow.phase != "p2_next_dispatch" for wave in lookahead_plan.waves for flow in wave.flows)


def test_barrier_criticality_witness_changes_selection_for_fluid_and_atomic() -> None:
    problem = _build_problem(_fixture("barrier_criticality_switch_witness_4rank"), mode=RUNTIME_LOOKAHEAD_MODE, p2_source="copy_current_dispatch", expert_compute_delay=2.0)
    gated = resolve_policy(policy_name="U_gated_maxweight_matching", bucket_rows=0).build_logical_plan(problem)
    barrier = resolve_policy(policy_name="U_barrier_criticality_global_matching", bucket_rows=0).build_logical_plan(problem)
    gated_sig = [[(flow.phase, flow.src_rank, flow.dst_rank, flow.byte_count) for flow in wave.flows] for wave in gated.waves]
    barrier_sig = [[(flow.phase, flow.src_rank, flow.dst_rank, flow.byte_count) for flow in wave.flows] for wave in barrier.waves]
    divergence = next(idx for idx, (left, right) in enumerate(zip(gated_sig, barrier_sig, strict=False)) if left != right)
    assert divergence == 2
    assert ("p0_dispatch", 1, 3, 2) in gated_sig[2]
    assert ("p1_return", 1, 3, 2) in barrier_sig[2]

    gated_atomic = resolve_policy(policy_name="U_gated_maxweight_matching_atomic", bucket_rows=0).build_logical_plan(problem)
    barrier_atomic = resolve_policy(policy_name="U_barrier_criticality_global_matching_atomic", bucket_rows=0).build_logical_plan(problem)
    assert gated_atomic.diagnostics["makespan"] != barrier_atomic.diagnostics["makespan"]


def test_invalid_late_p0_release_counterexample_is_rejected_by_both_validators() -> None:
    fixture = _fixture("invalid_late_p0_release_counterexample")
    schedule = fixture["invalid_schedule"]
    replay = replay_and_audit_schedule(
        schedule=schedule,
        dispatch_matrix=fixture["p0_dispatch_matrix"],
        combine_matrix=fixture["p1_return_matrix"],
        next_dispatch_matrix=fixture["p2_next_dispatch_matrix"],
        num_gpus=int(fixture["num_gpus"]),
        expert_compute_delay=0.0,
        mode=RUNTIME_LOOKAHEAD_MODE,
        scheduler_name="invalid_counterexample",
    )
    assert replay["valid"] is False
    assert any("p1 local release violation" in error for error in replay["validation_errors"])

    flows = []
    for entry in schedule:
        phase = ("p0_dispatch", "p1_return", "p2_next_dispatch")[int(entry["phase"])]
        flows.append(
            FlowDemand(
                flow_id=str(entry["chunk_id"]),
                phase=phase,
                src_rank=int(entry["src_gpu"]),
                dst_rank=int(entry["dst_gpu"]),
                byte_count=int(entry["served_volume"]),
                release_state="ready",
                is_executable=True,
                dependency_metadata={
                    "origin_flow_id": str(entry["flow_id"]),
                    "service_model": "atomic_chunk",
                    "start": float(entry["start"]),
                    "end": float(entry["end"]),
                },
            )
        )
    plan = LogicalSchedulePlan(
        policy_name="invalid_counterexample",
        waves=tuple(LogicalWave(wave_id=int(flow.dependency_metadata["start"]), flows=(flow,), duration=1.0) for flow in flows),
        diagnostics={},
    )
    expected = (
        FlowDemand("p0_dispatch:0->1", "p0_dispatch", 0, 1, 1, "ready", True),
        FlowDemand("p1_return:1->0", "p1_return", 1, 0, 1, "blocked", False),
    )
    logical = validate_logical_plan(plan, expected_flows=expected, mode=RUNTIME_LOOKAHEAD_MODE, expert_compute_delay=0.0)
    assert logical["valid"] is False
    assert any("p1 local release violation" in error for error in logical["errors"])


def test_max_wave_limit_fails_closed_with_residual_nonzero() -> None:
    problem = _build_problem(
        _fixture("unlock_hotspot_4rank"),
        mode=RUNTIME_LOOKAHEAD_MODE,
        p2_source="zero_hint",
        expert_compute_delay=2.0,
        max_waves=1,
    )
    plan = resolve_policy(policy_name="U_gated_maxweight_matching", bucket_rows=0).build_logical_plan(problem)
    assert plan.diagnostics["solver_status"] == "max_wave_limit_exceeded"
    assert plan.diagnostics["valid"] is False
    assert plan.diagnostics["audit"]["residual_nonzero"] is True
