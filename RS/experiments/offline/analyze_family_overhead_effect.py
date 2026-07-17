"""Controlled same-process effect and planning-overhead analysis.

The script uses perfect P012 information for both Local(f) and Joint(f), so the
only causal difference is phase coupling/ready-set scope.  Planning times are
reported separately from the abstract traffic makespan because converting them
into net runtime benefit requires a measured transport cost model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from experiments.paper.adapters.scheduling_adapter import (
    execute_policy,
    replay_window_from_matrices,
)
from rs.scheduling.families import canonical_family_policy_id
from rs.scheduling.families.core import FamilyScope


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(quantile)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    rows = [float(value) for value in values]
    return {
        "n": len(rows),
        "mean": statistics.mean(rows) if rows else None,
        "median": statistics.median(rows) if rows else None,
        "p90": _percentile(rows, 0.90),
        "p95": _percentile(rows, 0.95),
        "min": min(rows) if rows else None,
        "max": max(rows) if rows else None,
    }


def _select(instances: list[dict[str, Any]], per_vep: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for vep in sorted({int(row["virtual_ep_size"]) for row in instances}):
        rows = sorted(
            (
                row
                for row in instances
                if int(row["virtual_ep_size"]) == vep
                and bool(row.get("p2_available", False))
            ),
            key=lambda row: (
                str(row["sample_id"]),
                int(row["layer_id"]),
                str(row["traffic_instance_id"]),
            ),
        )
        if len(rows) <= per_vep:
            output.extend(rows)
            continue
        indexes = {
            round(index * (len(rows) - 1) / max(per_vep - 1, 1))
            for index in range(per_vep)
        }
        output.extend(rows[index] for index in sorted(indexes))
    return output


def _window(item: dict[str, Any]):
    matrix = lambda key: tuple(
        tuple(int(value) for value in row) for row in item[key]
    )
    return replay_window_from_matrices(
        fixture_id=str(item["traffic_instance_id"]),
        layer_id=int(item["layer_id"]),
        p0_matrix=matrix("P0_dispatch_matrix"),
        p1_matrix=matrix("P1_return_matrix"),
        p2_matrix=matrix("P2_next_layer_dispatch_matrix"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--families",
        nargs="+",
        default=("greedy_control", "gmwd", "rsbc", "rscf", "fast_stage"),
    )
    parser.add_argument("--per-vep", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--max-waves",
        type=int,
        default=4096,
        help=(
            "Explicit offline safety budget. The formal adapter keeps its runtime "
            "default; this analysis opts into a larger cap for dense virtual-EP windows."
        ),
    )
    args = parser.parse_args()

    instances = json.loads(args.instances.read_text(encoding="utf-8"))
    selected = _select(instances, max(1, int(args.per_vep)))
    records: list[dict[str, Any]] = []
    for family_id in args.families:
        policy_ids = {
            "local": canonical_family_policy_id(family_id, FamilyScope.LOCAL),
            "joint": canonical_family_policy_id(family_id, FamilyScope.JOINT),
        }
        for item in selected:
            window = _window(item)
            measured: dict[str, dict[str, Any]] = {}
            for scope, policy_id in policy_ids.items():
                runs: list[dict[str, Any]] = []
                for run_index in range(max(0, args.warmups) + max(1, args.repeats)):
                    result = execute_policy(
                        replay_window=window,
                        policy_name=policy_id,
                        hint_type="perfect_trace_hint",
                        p2_hint_rows=window.p2_truth_rows,
                        confidence=1.0,
                        max_waves=int(args.max_waves),
                    )
                    if not bool(result.get("audit_valid", False)):
                        raise RuntimeError(
                            f"invalid plan for {item['traffic_instance_id']} / {policy_id}"
                        )
                    if run_index >= max(0, args.warmups):
                        runs.append(result)
                digests = {str(row["logical_plan_digest"]) for row in runs}
                measured[scope] = {
                    "objective": statistics.median(float(row["makespan"]) for row in runs),
                    "wall_ms": [float(row["planning_runtime_ms_wall"]) for row in runs],
                    "kernel_ms": [
                        float(row["plan_metadata"].get("kernel_runtime_ms", 0.0))
                        for row in runs
                    ],
                    "wave_count": [float(row["audit"].get("wave_count", 0)) for row in runs],
                    "deterministic": len(digests) == 1,
                }
            local_objective = float(measured["local"]["objective"])
            joint_objective = float(measured["joint"]["objective"])
            records.append(
                {
                    "traffic_instance_id": item["traffic_instance_id"],
                    "sample_id": item["sample_id"],
                    "layer_id": int(item["layer_id"]),
                    "virtual_ep_size": int(item["virtual_ep_size"]),
                    "family_id": family_id,
                    "local_objective": local_objective,
                    "joint_objective": joint_objective,
                    "joint_improvement_pct": (
                        100.0 * (local_objective - joint_objective) / local_objective
                        if local_objective > 0.0
                        else 0.0
                    ),
                    "local_wall_ms": measured["local"]["wall_ms"],
                    "joint_wall_ms": measured["joint"]["wall_ms"],
                    "local_kernel_ms": measured["local"]["kernel_ms"],
                    "joint_kernel_ms": measured["joint"]["kernel_ms"],
                    "local_wave_count": measured["local"]["wave_count"],
                    "joint_wave_count": measured["joint"]["wave_count"],
                    "local_deterministic": measured["local"]["deterministic"],
                    "joint_deterministic": measured["joint"]["deterministic"],
                }
            )

    summary: dict[str, Any] = {}
    for family_id in args.families:
        rows = [row for row in records if row["family_id"] == family_id]
        improvements = [float(row["joint_improvement_pct"]) for row in rows]
        local_wall = [value for row in rows for value in row["local_wall_ms"]]
        joint_wall = [value for row in rows for value in row["joint_wall_ms"]]
        local_kernel = [value for row in rows for value in row["local_kernel_ms"]]
        joint_kernel = [value for row in rows for value in row["joint_kernel_ms"]]
        summary[family_id] = {
            "instance_count": len(rows),
            "effect_joint_improvement_pct": _stats(improvements),
            "win_tie_loss": {
                "win": sum(value > 1e-9 for value in improvements),
                "tie": sum(abs(value) <= 1e-9 for value in improvements),
                "loss": sum(value < -1e-9 for value in improvements),
            },
            "local_planning_wall_ms": _stats(local_wall),
            "joint_planning_wall_ms": _stats(joint_wall),
            "local_kernel_ms": _stats(local_kernel),
            "joint_kernel_ms": _stats(joint_kernel),
            "joint_over_local_wall_ratio": (
                statistics.median(joint_wall) / statistics.median(local_wall)
                if local_wall and statistics.median(local_wall) > 0.0
                else None
            ),
            "joint_over_local_kernel_ratio": (
                statistics.median(joint_kernel) / statistics.median(local_kernel)
                if local_kernel and statistics.median(local_kernel) > 0.0
                else None
            ),
            "all_plans_deterministic": all(
                row["local_deterministic"] and row["joint_deterministic"]
                for row in rows
            ),
        }

    artifact = {
        "schema_version": "family_overhead_effect.v1",
        "information_scope": "p012_perfect",
        "planning_timing": "upfront_for_both_scopes",
        "selected_instance_count": len(selected),
        "per_vep": int(args.per_vep),
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "max_waves": int(args.max_waves),
        "timing_note": (
            "Python CPU planner timing. Abstract traffic makespan and planner milliseconds "
            "must not be subtracted without an independently calibrated transport cost model."
        ),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
