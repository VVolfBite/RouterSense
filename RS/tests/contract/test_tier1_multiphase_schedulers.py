from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rs.scheduling import FlowDemand, resolve_phase_policy, resolve_policy
from rs.scheduling.multiphase.flow_model import EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE
from rs.scheduling.multiphase.tier1 import FLUID_SERVICE_MODEL, TIER1_ALGORITHM_IDS
from rs.scheduling.validation import stable_hash, validate_logical_plan

from experiments.offline.run_tier1_cpu_validation import _build_problem


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
    assert plans[0].diagnostics["future_information_mode"] == "oracle_predicted_runtime_lookahead"
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
    assert (output_dir / "policy_plan_B_birkhoff.json").exists()
    assert (output_dir / "diagnostics_U_gated_maxweight_matching.json").exists()
