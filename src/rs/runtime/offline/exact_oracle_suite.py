"""Deterministic exact-oracle suite on the runtime bucket/wave model.

This module is the single source of truth for tiny O_local/O_joint experiments.
Both scopes consume the exact same canonical remote-edge buckets, matching-wave
cost model, and replay release semantics.  The only changed variable is scope:
phase-barrier local versus rank-release joint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from rs.runtime.offline.replay_unified import (
    PlanningHint,
    ReplayWindow,
    build_execution_truth,
    build_multiphase_problem,
    build_planning_problem,
)
from rs.scheduling.reference.exact_small_instance import solve_problem_exact_with_scope
from rs.scheduling.traffic_matrix import canonicalize_remote_matrix


Matrix = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ExactInstance:
    instance_id: str
    rank_count: int
    p0: Matrix
    p1: Matrix
    p2: Matrix
    sparsity_regime: str
    skew_regime: str
    correlation_regime: str
    p2_strength_regime: str
    seed: int


def _make_instance_matrix(rank_count: int, edges: list[tuple[int, int]]) -> Matrix:
    matrix = [[0 for _ in range(rank_count)] for _ in range(rank_count)]
    for src, dst in edges:
        if src != dst:
            matrix[src][dst] += 1
    return canonicalize_remote_matrix(tuple(tuple(row) for row in matrix))


def generate_exact_instances(target_count: int) -> list[ExactInstance]:
    """Generate the frozen deterministic tiny-suite used by formal closure."""

    regimes = [
        ("sparse", "balanced", "high", "weak"),
        ("sparse", "hotspot", "medium", "medium"),
        ("medium", "balanced", "low", "medium"),
        ("medium", "heavy_tail", "adversarial", "strong"),
        ("dense", "balanced", "high", "strong"),
        ("dense", "hotspot", "medium", "weak"),
        ("dense", "heavy_tail", "low", "medium"),
        ("sparse", "heavy_tail", "adversarial", "strong"),
    ]
    instances: list[ExactInstance] = []
    seed = 7
    while len(instances) < int(target_count):
        for rank_count in (2, 3, 4):
            for sparsity, skew, correlation, strength in regimes:
                rng = random.Random(seed * 1000 + rank_count * 100 + len(instances))
                edge_budget = 2 if sparsity == "sparse" else 3 if sparsity == "medium" else 4
                all_edges = [(src, dst) for src in range(rank_count) for dst in range(rank_count) if src != dst]
                rng.shuffle(all_edges)
                p0_edges = all_edges[:edge_budget]
                p0 = _make_instance_matrix(rank_count, p0_edges)
                p1 = canonicalize_remote_matrix(
                    tuple(tuple(int(p0[dst][src]) for dst in range(rank_count)) for src in range(rank_count))
                )
                if correlation == "high":
                    p2_edges = list(p0_edges)
                elif correlation == "medium":
                    p2_edges = list(p0_edges[: max(1, len(p0_edges) - 1)])
                    p2_edges += [edge for edge in all_edges if edge not in p2_edges][:1]
                elif correlation == "low":
                    p2_edges = [edge for edge in all_edges if edge not in p0_edges][:edge_budget]
                else:
                    p2_edges = list(reversed([edge for edge in all_edges if edge not in p0_edges][:edge_budget]))
                if skew == "hotspot":
                    pivot = rng.randrange(rank_count)
                    p2_edges = [
                        (src, pivot if pivot != src else (pivot + 1) % rank_count)
                        for src, _dst in p2_edges[:edge_budget]
                    ]
                elif skew == "heavy_tail":
                    pivot = rng.randrange(rank_count)
                    extra = (pivot + 1) % rank_count
                    p2_edges = [(pivot, dst if dst != pivot else extra) for _src, dst in p2_edges[:edge_budget]]
                p2_budget = 1 if strength == "weak" else 2 if strength == "medium" else 4
                p2 = _make_instance_matrix(rank_count, p2_edges[:p2_budget])
                instances.append(
                    ExactInstance(
                        instance_id=f"exact_{len(instances):03d}",
                        rank_count=rank_count,
                        p0=p0,
                        p1=p1,
                        p2=p2,
                        sparsity_regime=sparsity,
                        skew_regime=skew,
                        correlation_regime=correlation,
                        p2_strength_regime=strength,
                        seed=seed,
                    )
                )
                if len(instances) >= int(target_count):
                    break
            if len(instances) >= int(target_count):
                break
        seed += 1
    return instances[: int(target_count)]


def build_exact_problem(instance: ExactInstance, *, expert_compute_delay: float = 0.0):
    replay_window = ReplayWindow(
        fixture_id=str(instance.instance_id),
        window_id=f"{instance.instance_id}:0",
        layer_id=0,
        p0_truth_rows=canonicalize_remote_matrix(instance.p0),
        p1_truth_rows=canonicalize_remote_matrix(instance.p1),
        p2_truth_rows=canonicalize_remote_matrix(instance.p2),
        matrix_unit="rows",
        group_size=int(instance.rank_count),
        payload_row_bytes_by_phase={"p0_dispatch": 1, "p1_return": 1, "p2_next_dispatch": 1},
        metadata={"source": "unified_exact_oracle_suite"},
    )
    hint = PlanningHint(
        hint_type="perfect_trace_hint",
        p2_hint_rows=replay_window.p2_truth_rows,
        confidence=1.0,
        source_layer=0,
        target_layer=1,
    )
    planning_problem = build_planning_problem(replay_window=replay_window, planning_hint=hint)
    return build_multiphase_problem(
        planning_problem=planning_problem,
        execution_truth=build_execution_truth(replay_window),
        scheduling_mode="execution_window",
        expert_compute_delay=float(expert_compute_delay),
        max_waves=256,
        bucket_rows=1,
    )


def solve_exact_instance(
    instance: ExactInstance,
    *,
    scope: str,
    expert_compute_delay: float = 0.0,
    time_limit_ms: int = 30_000,
) -> dict[str, Any]:
    problem = build_exact_problem(instance, expert_compute_delay=expert_compute_delay)
    result = solve_problem_exact_with_scope(
        problem,
        scope=str(scope),
        time_limit_ms=int(time_limit_ms),
    )
    status = str(result.get("solver_status", "unknown"))
    return {
        **dict(result),
        "solver_status_raw": status,
        "solver_status": status.upper(),
        "objective": result.get("objective_logical_makespan"),
        "wall_time_s": float(result.get("solver_runtime_ms_wall", 0.0) or 0.0) / 1000.0,
        "task_count": int(result.get("task_count", 0) or 0),
    }


def run_exact_scope_suite(target_count: int = 32) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for instance in generate_exact_instances(int(target_count)):
        local = solve_exact_instance(instance, scope="local")
        joint = solve_exact_instance(instance, scope="joint")
        o_local = local.get("objective")
        o_joint = joint.get("objective")
        comparable = (
            local.get("solver_status") == "OPTIMAL"
            and joint.get("solver_status") == "OPTIMAL"
            and o_local is not None
            and o_joint is not None
        )
        improvement = None
        if comparable and float(o_local) > 0.0:
            improvement = (float(o_local) - float(o_joint)) / float(o_local)
        rows.append(
            {
                "instance_id": instance.instance_id,
                "rank_count": instance.rank_count,
                "sparsity_regime": instance.sparsity_regime,
                "skew_regime": instance.skew_regime,
                "correlation_regime": instance.correlation_regime,
                "p2_strength_regime": instance.p2_strength_regime,
                "seed": instance.seed,
                "O_local": o_local,
                "O_joint": o_joint,
                "oracle_local_status": local.get("solver_status"),
                "oracle_joint_status": joint.get("solver_status"),
                "joint_improvement_vs_local": improvement,
                "dominance_violation": bool(comparable and float(o_joint) > float(o_local) + 1.0e-9),
                "reference_model": local.get("reference_model"),
                "task_model_id": local.get("task_model_id"),
                "cost_model_id": local.get("cost_model_id"),
                "release_model_id": local.get("release_model_id"),
                "local_runtime_ms": local.get("solver_runtime_ms_wall"),
                "joint_runtime_ms": joint.get("solver_runtime_ms_wall"),
            }
        )
    comparable_rows = [
        row
        for row in rows
        if row["oracle_local_status"] == "OPTIMAL"
        and row["oracle_joint_status"] == "OPTIMAL"
        and row["O_local"] is not None
        and row["O_joint"] is not None
    ]
    gains = [float(row["joint_improvement_vs_local"]) for row in comparable_rows if row["joint_improvement_vs_local"] is not None]
    gains_sorted = sorted(gains)

    def quantile(q: float) -> float | None:
        if not gains_sorted:
            return None
        index = min(len(gains_sorted) - 1, max(0, int(round((len(gains_sorted) - 1) * q))))
        return gains_sorted[index]

    positive = [value for value in gains if value > 1.0e-12]
    return {
        "schema_version": "routersense_unified_exact_oracle_suite.v1",
        "instance_count": len(rows),
        "comparable_count": len(comparable_rows),
        "dominance_violation_count": sum(int(bool(row["dominance_violation"])) for row in comparable_rows),
        "joint_better_count": len(positive),
        "tie_count": sum(int(abs(float(value)) <= 1.0e-12) for value in gains),
        "mean_joint_improvement_vs_local": None if not gains else sum(gains) / len(gains),
        "median_joint_improvement_vs_local": quantile(0.5),
        "p90_joint_improvement_vs_local": quantile(0.9),
        "mean_improvement_on_joint_opportunity_subset": None if not positive else sum(positive) / len(positive),
        "rows": rows,
    }


__all__ = [
    "ExactInstance",
    "build_exact_problem",
    "generate_exact_instances",
    "run_exact_scope_suite",
    "solve_exact_instance",
]
