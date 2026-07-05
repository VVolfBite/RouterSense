from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.offline.run_tier1_cpu_validation import _build_problem
from rs.scheduling import FlowDemand, resolve_policy
from rs.scheduling.multiphase.flow_model import EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE
from rs.scheduling.multiphase.tier1 import TIER1_ALGORITHM_IDS
from rs.scheduling.validation import stable_hash, validate_logical_plan


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "tier1"
GOLDEN_PATH = FIXTURE_ROOT / "historical_golden" / "tier1_semantic_witness.json"


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


def _semantic_signature(plan):
    signature = []
    for entry in plan.diagnostics["raw_schedule"]:
        signature.append(
            {
                "phase": ("p0_dispatch", "p1_return", "p2_next_dispatch")[int(entry["phase"])],
                "src": int(entry["src_gpu"]),
                "dst": int(entry["dst_gpu"]),
                "start": round(float(entry["start"]), 6),
                "end": round(float(entry["end"]), 6),
                "served_volume": round(float(entry.get("served_volume", entry.get("size", 0.0))), 6),
            }
        )
    return signature


def _golden_cases():
    payload = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return payload["golden"]


@pytest.mark.parametrize("case", _golden_cases(), ids=lambda case: f"{case['algorithm_id']}:{case['mode']}")
def test_tier1_recovered_semantics_match_historical_golden(case: dict) -> None:
    assert case["algorithm_id"] in TIER1_ALGORITHM_IDS
    fixture = _fixture("unlock_hotspot_4rank")
    assert case["fixture_digest"] == stable_hash(fixture)
    p2_source = "perfect_trace" if case["mode"] == RUNTIME_LOOKAHEAD_MODE else "actual_trace"
    problem = _build_problem(
        fixture,
        mode=case["mode"],
        p2_source=p2_source,
        expert_compute_delay=2.0,
    )
    plan = resolve_policy(policy_name=case["algorithm_id"], bucket_rows=0).build_logical_plan(problem)
    validation = validate_logical_plan(
        plan,
        expected_flows=_expected_flows(problem),
        mode=case["mode"],
        expert_compute_delay=2.0,
    )
    assert plan.diagnostics["service_model"] == case["service_model"]
    assert round(float(plan.diagnostics["makespan"]), 6) == case["makespan"]
    assert {
        "future_information_mode": plan.diagnostics["future_information_mode"],
        "p2_role": plan.diagnostics["p2_role"],
    } == case["release_signature"]
    assert _semantic_signature(plan) == case["semantic_schedule_signature"]
    assert bool(plan.diagnostics["flow_conservation_verified"]) == case["flow_conservation_verified"]
    assert bool(plan.diagnostics["matching_legality_verified"]) == case["matching_legality_verified"]
    assert validation["valid"], validation["errors"]
