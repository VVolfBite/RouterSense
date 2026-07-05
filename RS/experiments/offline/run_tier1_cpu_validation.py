"""CPU-only validation runner for recovered Tier 1 scheduling candidates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rs.scheduling import FlowDemand, FlowWindow, ForecastPressure, GlobalReadySetOptions, LogicalTopology, MultiPhaseSchedulingProblem, ReleaseConstraint, resolve_policy
from rs.scheduling.multiphase.flow_model import EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE
from rs.scheduling.multiphase.tier1 import TIER1_ALGORITHM_IDS, tier1_inventory
from rs.scheduling.validation import stable_hash, validate_logical_plan


def main() -> None:
    args = _parse_args()
    fixture = _load_fixture(Path(args.fixture))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = TIER1_ALGORITHM_IDS if args.policy == "all" else tuple(item.strip() for item in args.policy.split(",") if item.strip())
    problem = _build_problem(
        fixture,
        mode=args.mode,
        p2_source=args.p2_source,
        expert_compute_delay=float(args.expert_compute_delay),
    )
    started = time.perf_counter()
    _write_json(output_dir / "run_manifest.json", {
        "run_kind": "tier1_cpu_validation",
        "fixture": str(args.fixture),
        "policies": list(policies),
        "mode": args.mode,
        "p2_source": args.p2_source,
        "expert_compute_delay": float(args.expert_compute_delay),
    })
    _write_json(output_dir / "problem_contract.json", problem.to_dict())
    _write_json(output_dir / "tier1_inventory.json", {"tier1_algorithms": list(tier1_inventory())})
    comparison: dict[str, list[dict[str, Any]]] = {
        "atomic_comparison": [],
        "fluid_comparison": [],
        "other_service_model_comparison": [],
    }
    for policy_name in policies:
        policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
        plan_started = time.perf_counter()
        plan = policy.build_logical_plan(problem)
        measured_planning_time_ms = (time.perf_counter() - plan_started) * 1000.0
        expected = _expected_real_flows(problem)
        validation = validate_logical_plan(
            plan,
            expected_flows=expected,
            mode=args.mode,
            expert_compute_delay=float(args.expert_compute_delay),
        )
        diagnostics = dict(plan.diagnostics)
        summary = {
            "algorithm_id": diagnostics.get("algorithm_id", policy_name),
            "service_model": diagnostics.get("service_model", "unknown"),
            "mode": args.mode,
            "future_information_mode": diagnostics.get("future_information_mode", "none"),
            "p2_source": args.p2_source,
            "prediction_used": diagnostics.get("prediction_used", False),
            "evaluation_eligible": diagnostics.get("evaluation_eligible", False),
            "makespan": diagnostics.get("makespan"),
            "wave_count": len(plan.waves),
            "planning_time_ms_measured": measured_planning_time_ms,
            "planning_time_ms_in_plan_hash": diagnostics.get("planning_time_ms", 0.0),
            "valid": bool(diagnostics.get("valid", False)) and bool(validation["valid"]),
            "release_barrier_verified": diagnostics.get("release_barrier_verified", False),
            "flow_conservation_verified": diagnostics.get("flow_conservation_verified", False) and bool(validation["coverage_verified"]),
            "matching_legality_verified": diagnostics.get("matching_legality_verified", False) and bool(validation["matching_constraints_verified"]),
            "plan_hash": stable_hash(plan.to_dict()),
        }
        _write_json(output_dir / f"policy_plan_{policy_name}.json", plan.to_dict())
        _write_json(output_dir / f"audit_{policy_name}.json", diagnostics.get("audit", {}))
        _write_json(
            output_dir / f"diagnostics_{policy_name}.json",
            {
                **diagnostics,
                "planning_time_ms_measured": measured_planning_time_ms,
                "planning_time_ms_in_plan_hash": diagnostics.get("planning_time_ms", 0.0),
                "logical_plan_validation": validation,
                "summary": summary,
            },
        )
        bucket = _comparison_bucket(str(summary["service_model"]))
        comparison[bucket].append(summary)
    comparison["elapsed_ms"] = (time.perf_counter() - started) * 1000.0  # type: ignore[index]
    _write_json(output_dir / "comparison_by_service_model.json", comparison)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--policy", default="all", help="Tier 1 algorithm id, comma list, or all")
    parser.add_argument("--mode", choices=(EXECUTION_WINDOW_MODE, RUNTIME_LOOKAHEAD_MODE), default=RUNTIME_LOOKAHEAD_MODE)
    parser.add_argument("--p2-source", choices=("zero_hint", "copy_current_dispatch", "perfect_trace", "actual_trace"), default="zero_hint")
    parser.add_argument("--expert-compute-delay", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_problem(
    fixture: dict[str, Any],
    *,
    mode: str,
    p2_source: str,
    expert_compute_delay: float,
) -> MultiPhaseSchedulingProblem:
    p0 = _matrix(fixture["p0_dispatch_matrix"])
    p1 = _matrix(fixture["p1_return_matrix"])
    if mode == EXECUTION_WINDOW_MODE and p2_source not in {"actual_trace", "perfect_trace"}:
        raise ValueError("execution_window requires --p2-source actual_trace or perfect_trace")
    if "p2_next_dispatch_matrix" not in fixture and mode == EXECUTION_WINDOW_MODE:
        raise ValueError("execution_window requires fixture p2_next_dispatch_matrix")
    actual_p2 = _matrix(fixture.get("p2_next_dispatch_matrix", fixture.get("p2_next_dispatch_forecast_matrix", _zero_like(p0))))
    if mode == EXECUTION_WINDOW_MODE:
        p2 = actual_p2
        oracle = True
        eligible = False
        source = "actual_trace" if p2_source == "actual_trace" else "perfect_trace"
    elif p2_source == "zero_hint":
        p2 = _zero_like(p0)
        oracle = False
        eligible = True
        source = p2_source
    elif p2_source == "copy_current_dispatch":
        p2 = p0
        oracle = False
        eligible = True
        source = p2_source
    elif p2_source == "perfect_trace":
        p2 = actual_p2
        oracle = True
        eligible = False
        source = p2_source
    else:  # pragma: no cover
        raise ValueError(f"unsupported p2_source {p2_source!r}")
    forecast = ForecastPressure(
        source=source,
        digest=stable_hash({"source": source, "matrix": p2}),
        oracle=oracle,
        evaluation_eligible=eligible,
        matrix_shape=(len(p2), len(p2[0]) if p2 else 0),
        matrix_total_bytes=sum(sum(row) for row in p2),
        matrix=p2,
    )
    prediction_confidence = 1.0 if source != "zero_hint" and sum(sum(row) for row in p2) > 0 else 0.0
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows(p0, phase="p0_dispatch", release_state="ready", executable=True),
            blocked_flows=_flows(p1, phase="p1_return", release_state="blocked", executable=False),
            forecast_pressure=_flows(p2, phase="p2_next_dispatch_forecast", release_state="advisory_only", executable=False),
        ),
        topology=LogicalTopology(num_gpus=int(fixture["num_gpus"])),
        release_model=ReleaseConstraint(phase="p1_return", rank=0, release_after_phase="p0_dispatch", expert_compute_delay=expert_compute_delay),
        forecast=forecast,
        options=GlobalReadySetOptions(
            scheduling_mode=mode,
            information_mode="p0_p1_p2" if source != "zero_hint" else "p0_p1",
            prediction_confidence=prediction_confidence,
        ),
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_forecast_matrix=p2,
    )


def _matrix(value: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(item) for item in row) for row in value)


def _zero_like(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(0 for _ in row) for row in matrix)


def _flows(
    matrix: tuple[tuple[int, ...], ...],
    *,
    phase: str,
    release_state: str,
    executable: bool,
) -> tuple[FlowDemand, ...]:
    flows = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            flows.append(
                FlowDemand(
                    flow_id=f"{phase}:{src_rank}->{dst_rank}",
                    phase=phase,
                    src_rank=src_rank,
                    dst_rank=dst_rank,
                    byte_count=int(byte_count),
                    release_state=release_state,
                    is_executable=executable,
                )
            )
    return tuple(flows)


def _expected_real_flows(problem: MultiPhaseSchedulingProblem) -> tuple[FlowDemand, ...]:
    flows = list(problem.flow_window.ready_flows + problem.flow_window.blocked_flows)
    if problem.options.scheduling_mode == EXECUTION_WINDOW_MODE:
        flows.extend(
            _flows(
                problem.p2_next_dispatch_forecast_matrix,
                phase="p2_next_dispatch",
                release_state="ready",
                executable=True,
            )
        )
    return tuple(flows)


def _comparison_bucket(service_model: str) -> str:
    if "fluid" in service_model:
        return "fluid_comparison"
    if "atomic" in service_model:
        return "atomic_comparison"
    return "other_service_model_comparison"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
