#!/usr/bin/env python3
"""Small-grid CPU tuning for existing raw U / safe U families."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

from experiments.offline.replay_fixture_policy_study import _build_problem
from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.scheduling.multiphase.tier1 import Tier1PolicySpec, _build_u_policy
from rs.scheduling.multiphase.recovered_candidates import RecoveredCandidateSpec, _raw_schedule_to_plan, run_global_matching_scheduler
from rs.scheduling import resolve_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    return parser.parse_args()


def _split(fixtures: list[Path]) -> tuple[list[Path], list[Path]]:
    train = [path for idx, path in enumerate(fixtures) if idx % 2 == 0]
    evals = [path for idx, path in enumerate(fixtures) if idx % 2 == 1]
    return train or fixtures, evals or fixtures


def _score_plan(problem, plan) -> float:
    audit = replay_and_audit_logical_plan(problem, plan)
    return float(audit.get("makespan", plan.diagnostics.get("makespan", 0.0)))


def _tune_barrier_criticality(problem, residual_weight: float, barrier_weight: float, age_weight: float, prediction_weight: float):
    spec = Tier1PolicySpec(
        algorithm_id="U_barrier_criticality_global_matching",
        service_model="fluid_wave",
        exact_matching=True,
        atomic=False,
        residual_weight=residual_weight,
        barrier_weight=barrier_weight,
        age_weight=age_weight,
        prediction_weight=prediction_weight,
    )
    return _build_u_policy(problem, spec)


def _tune_gated_maxweight(problem, residual_weight: float, barrier_weight: float, age_weight: float, prediction_weight: float):
    spec = Tier1PolicySpec(
        algorithm_id="U_gated_maxweight_matching",
        service_model="fluid_wave",
        exact_matching=True,
        atomic=False,
        residual_weight=residual_weight,
        barrier_weight=barrier_weight,
        age_weight=age_weight,
        prediction_weight=prediction_weight,
    )
    return _build_u_policy(problem, spec)


def _tune_gated_greedy(problem, residual_weight: float, barrier_weight: float, age_weight: float, prediction_weight: float):
    result = run_global_matching_scheduler(
        [list(row) for row in problem.p0_dispatch_matrix],
        [list(row) for row in problem.p1_return_matrix],
        [list(row) for row in problem.p2_next_dispatch_forecast_matrix],
        int(problem.topology.num_gpus),
        strategy="U_gated_greedy_maximal",
        mode=problem.options.scheduling_mode,
        prediction_confidence=float(problem.options.prediction_confidence),
        expert_compute_delay=float(problem.release_model.expert_compute_delay),
        exact_matching=False,
        wave_quantum=None,
        max_waves=int(problem.options.max_waves),
        residual_weight=residual_weight,
        barrier_weight=barrier_weight,
        age_weight=age_weight,
        prediction_weight=prediction_weight,
        adaptive_prices=False,
        price_step=0.0,
        price_decay=0.0,
        price_clip=0.0,
        iteration_budget=1,
        atomic=False,
    )
    return _raw_schedule_to_plan(
        problem=problem,
        algorithm_id="U_gated_greedy_maximal",
        service_model="fluid_wave",
        raw_schedule=list(result.get("schedule", [])),
        audit=dict(result.get("audit", {})),
        makespan=float(result.get("makespan", 0.0)),
        planning_time_ms=float(result.get("solve_time_ms", 0.0)),
        solver_status=str(result.get("solver_status", "completed")),
        selection_model="tuned_recovered_gated_greedy",
        extra_diagnostics={"tuned_parameters": {"residual_weight": residual_weight, "barrier_weight": barrier_weight, "age_weight": age_weight, "prediction_weight": prediction_weight}},
    )


def _tune_barrier_price(problem, residual_weight: float, barrier_weight: float, age_weight: float, prediction_weight: float):
    result = run_global_matching_scheduler(
        [list(row) for row in problem.p0_dispatch_matrix],
        [list(row) for row in problem.p1_return_matrix],
        [list(row) for row in problem.p2_next_dispatch_forecast_matrix],
        int(problem.topology.num_gpus),
        strategy="U_barrier_price_adaptive_matching",
        mode=problem.options.scheduling_mode,
        prediction_confidence=float(problem.options.prediction_confidence),
        expert_compute_delay=float(problem.release_model.expert_compute_delay),
        exact_matching=True,
        wave_quantum=None,
        max_waves=int(problem.options.max_waves),
        residual_weight=residual_weight,
        barrier_weight=barrier_weight,
        age_weight=age_weight,
        prediction_weight=prediction_weight,
        adaptive_prices=True,
        price_step=0.2,
        price_decay=0.1,
        price_clip=8.0,
        iteration_budget=2,
        atomic=False,
    )
    return _raw_schedule_to_plan(
        problem=problem,
        algorithm_id="U_barrier_price_adaptive_matching",
        service_model="fluid_wave",
        raw_schedule=list(result.get("schedule", [])),
        audit=dict(result.get("audit", {})),
        makespan=float(result.get("makespan", 0.0)),
        planning_time_ms=float(result.get("solve_time_ms", 0.0)),
        solver_status=str(result.get("solver_status", "completed")),
        selection_model="tuned_recovered_barrier_price",
        extra_diagnostics={"tuned_parameters": {"residual_weight": residual_weight, "barrier_weight": barrier_weight, "age_weight": age_weight, "prediction_weight": prediction_weight}},
    )


BUILDERS = {
    "U_gated_greedy_maximal": _tune_gated_greedy,
    "U_gated_maxweight_matching": _tune_gated_maxweight,
    "U_barrier_criticality_global_matching": _tune_barrier_criticality,
    "U_barrier_price_adaptive_matching": _tune_barrier_price,
}
SAFE_BY_RAW = {
    "U_gated_greedy_maximal": "RS_safe_gated_greedy",
    "U_gated_maxweight_matching": "RS_safe_gated_maxweight",
    "U_barrier_criticality_global_matching": "RS_safe_barrier_criticality",
    "U_barrier_price_adaptive_matching": "RS_safe_barrier_price",
}


def run_u_weight_tuning(
    *,
    fixture_dir: Path,
    policies: Iterable[str] | None = None,
    grid_size: int | None = None,
) -> dict[str, Any]:
    fixtures = sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    train_paths, eval_paths = _split(fixtures)
    grid = [
        {"residual_weight": 1.0, "barrier_weight": 1.0, "age_weight": 0.1, "prediction_weight": 0.25},
        {"residual_weight": 0.75, "barrier_weight": 1.75, "age_weight": 0.15, "prediction_weight": 0.35},
        {"residual_weight": 0.85, "barrier_weight": 1.5, "age_weight": 0.1, "prediction_weight": 0.2},
    ]
    if grid_size is not None:
        grid = grid[: max(1, int(grid_size))]
    selected_policies = tuple(policies or BUILDERS.keys())
    summary: dict[str, Any] = {"policies": {}, "recommendation": {}}
    for raw_policy in selected_policies:
        builder = BUILDERS[raw_policy]
        best_train = None
        for params in grid:
            train_scores = []
            for path in train_paths:
                fixture = json.loads(path.read_text(encoding="utf-8"))
                problem = _build_problem(fixture, mode="runtime_lookahead", p2_source="copy_current_dispatch", expert_compute_delay=0.0)
                plan = builder(problem, **params)
                train_scores.append(_score_plan(problem, plan))
            train_mean = statistics.mean(train_scores)
            if best_train is None or train_mean < best_train["train_mean"]:
                best_train = {"params": params, "train_mean": train_mean}
        eval_scores = []
        safe_eval_scores = []
        fallback_count = 0
        for path in eval_paths:
            fixture = json.loads(path.read_text(encoding="utf-8"))
            problem = _build_problem(fixture, mode="runtime_lookahead", p2_source="copy_current_dispatch", expert_compute_delay=0.0)
            tuned_plan = builder(problem, **best_train["params"])
            eval_scores.append(_score_plan(problem, tuned_plan))
            safe_plan = resolve_policy(policy_name=SAFE_BY_RAW[raw_policy], bucket_rows=0).build_logical_plan(problem)
            safe_eval_scores.append(_score_plan(problem, safe_plan))
            fallback_count += int(bool(safe_plan.diagnostics.get("fallback_to_paired_b", False)))
        eval_mean = statistics.mean(eval_scores)
        safe_eval_mean = statistics.mean(safe_eval_scores)
        summary["policies"][raw_policy] = {
            "best_params_train": best_train["params"],
            "train_mean_makespan": best_train["train_mean"],
            "eval_mean_makespan": eval_mean,
            "safe_eval_mean_makespan": safe_eval_mean,
            "fallback_ratio": fallback_count / max(1, len(eval_paths)),
            "overfit_warning": eval_mean > best_train["train_mean"],
            "family_quality_label": (
                "mainline_candidate"
                if SAFE_BY_RAW[raw_policy] in {"RS_safe_barrier_criticality", "RS_safe_gated_greedy"}
                else "keep_diagnostic_only"
            ),
        }
    summary["recommendation"] = {
        "suggest_apply_default_change": False,
        "reason": "CPU-only tuning generated recommendation artifact only; no default weights are changed automatically.",
    }
    summary["selected_policies"] = list(selected_policies)
    summary["grid_size"] = len(grid)
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_u_weight_tuning(
        fixture_dir=Path(args.fixture_dir),
        policies=args.policies,
        grid_size=args.grid_size,
    )
    Path(args.output_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
