"""Decompose P01 coupling and P2 lookahead value for scheduling families.

The primary structural comparison gives Local(f) and Joint(f) the same perfect
P012 information.  A second dynamic replay reveals each true P2 source row only
when its P1 barrier completes.  This keeps algorithm scope, information scope,
and planning timing explicit rather than mixing them in one label.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from rs.runtime.offline.p2_information_value import simulate_p2_information
from rs.scheduling.families import get_family_kernel_spec
from rs.scheduling.multiphase.flow_model import EXECUTION_WINDOW_MODE
from rs.scheduling.multiphase.scheduler_state import run_global_matching_scheduler
from rs.scheduling.multiphase.tier1 import (
    _base_score_lookup_from_phase_orders,
    _birkhoff_phase_orders,
)


def _zero(size: int) -> list[list[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


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
        "p10": _percentile(rows, 0.10),
        "p90": _percentile(rows, 0.90),
        "min": min(rows) if rows else None,
        "max": max(rows) if rows else None,
    }


def _run_joint(
    p0: list[list[int]],
    p1: list[list[int]],
    p2: list[list[int]],
    family_id: str,
) -> dict[str, Any]:
    spec = get_family_kernel_spec(family_id)
    matrices = [p0, p1, p2]
    base_lookup = None
    if spec.base_priority_model == "birkhoff_round_rank":
        base_lookup = _base_score_lookup_from_phase_orders(
            _birkhoff_phase_orders(matrices)
        )
    return run_global_matching_scheduler(
        p0,
        p1,
        p2,
        len(p0),
        strategy=f"decomposition:{family_id}:joint",
        mode=EXECUTION_WINDOW_MODE,
        prediction_confidence=1.0,
        expert_compute_delay=0.0,
        exact_matching=bool(spec.exact_matching),
        wave_quantum=None,
        max_waves=4096,
        residual_weight=float(spec.residual_weight),
        barrier_weight=float(spec.barrier_weight),
        age_weight=float(spec.age_weight),
        prediction_weight=float(spec.prediction_weight),
        endpoint_pressure_weight=float(spec.endpoint_pressure_weight),
        release_gain_weight=float(spec.release_gain_weight),
        adaptive_prices=bool(spec.adaptive_prices),
        price_step=float(spec.price_step),
        price_decay=float(spec.price_decay),
        price_clip=float(spec.price_clip),
        iteration_budget=int(spec.iteration_budget),
        atomic=bool(spec.atomic),
        prediction_matrix=p2,
        base_score_lookup=base_lookup,
        base_priority_weight=(
            float(spec.base_priority_weight) if base_lookup is not None else 0.0
        ),
        scoring_model=str(spec.scoring_model),
        critical_path_weight=float(spec.critical_path_weight),
        transitive_unlock_weight=float(spec.transitive_unlock_weight),
        endpoint_dual_weight=float(spec.endpoint_dual_weight),
        duplex_pair_weight=float(spec.duplex_pair_weight),
        dual_temperature=float(spec.dual_temperature),
        transitive_tail_weight=float(spec.transitive_tail_weight),
        destination_hotspot_weight=float(spec.destination_hotspot_weight),
        size_bias_power=float(spec.size_bias_power),
    )


def _run_local(
    p0: list[list[int]],
    p1: list[list[int]],
    p2: list[list[int]],
    family_id: str,
    *,
    include_p2: bool,
) -> dict[str, Any]:
    size = len(p0)
    zero = _zero(size)
    matrices = [p0, p1] + ([p2] if include_p2 else [])
    total_makespan = 0.0
    total_runtime_ms = 0.0
    total_waves = 0
    valid = True
    for phase, matrix in enumerate(matrices):
        dispatch, combine, next_dispatch = (
            (matrix, zero, zero)
            if phase == 0
            else (zero, matrix, zero)
            if phase == 1
            else (zero, zero, matrix)
        )
        result = _run_joint(dispatch, combine, next_dispatch, family_id)
        valid = valid and bool(result.get("audit", {}).get("valid", False))
        total_makespan += float(result.get("makespan", 0.0))
        total_runtime_ms += float(result.get("solve_time_ms", 0.0))
        total_waves += int(result.get("wave_count", 0))
    return {
        "makespan": total_makespan,
        "solve_time_ms": total_runtime_ms,
        "wave_count": total_waves,
        "valid": valid,
    }


def _evaluate_one(
    task: tuple[dict[str, Any], tuple[str, ...]],
) -> list[dict[str, Any]]:
    item, family_ids = task
    p0 = item["P0_dispatch_matrix"]
    p1 = item["P1_return_matrix"]
    p2 = item["P2_next_layer_dispatch_matrix"]
    zero = _zero(len(p0))
    output: list[dict[str, Any]] = []
    for family_id in family_ids:
        local_p01 = _run_local(p0, p1, zero, family_id, include_p2=False)
        joint_p01 = _run_joint(p0, p1, zero, family_id)
        local_p012 = _run_local(p0, p1, p2, family_id, include_p2=True)
        perfect_p012 = simulate_p2_information(
            p0_dispatch_matrix=p0,
            p1_return_matrix=p1,
            p2_truth_matrix=p2,
            family_id=family_id,
            information_mode="perfect",
        )
        reactive_p012 = simulate_p2_information(
            p0_dispatch_matrix=p0,
            p1_return_matrix=p1,
            p2_truth_matrix=p2,
            family_id=family_id,
            information_mode="reactive",
        )
        if not (
            local_p01["valid"]
            and bool(joint_p01.get("audit", {}).get("valid", False))
            and local_p012["valid"]
            and perfect_p012.valid
            and reactive_p012.valid
        ):
            raise RuntimeError(
                f"invalid replay for {item['traffic_instance_id']} / {family_id}"
            )

        local_p01_ms = float(local_p01["makespan"])
        joint_p01_ms = float(joint_p01["makespan"])
        local_p012_ms = float(local_p012["makespan"])
        perfect_p012_ms = float(perfect_p012.makespan)
        reactive_p012_ms = float(reactive_p012.makespan)

        local_p2_ms = local_p012_ms - local_p01_ms
        p01_only_total_ms = joint_p01_ms + local_p2_ms
        p01_gain = local_p012_ms - p01_only_total_ms
        p2_interaction_gain = p01_only_total_ms - perfect_p012_ms
        total_gain = local_p012_ms - perfect_p012_ms
        output.append(
            {
                "traffic_instance_id": item["traffic_instance_id"],
                "sample_id": item["sample_id"],
                "layer_id": int(item["layer_id"]),
                "virtual_ep_size": int(item["virtual_ep_size"]),
                "p2_available": bool(item.get("p2_available", False)),
                "family_id": family_id,
                "local_p01_makespan": local_p01_ms,
                "joint_p01_makespan": joint_p01_ms,
                "local_p012_makespan": local_p012_ms,
                "p01_only_total_makespan": p01_only_total_ms,
                "joint_p012_perfect_makespan": perfect_p012_ms,
                "joint_p012_reactive_makespan": reactive_p012_ms,
                "p01_coupling_gain_pct_of_local_p012": (
                    100.0 * p01_gain / local_p012_ms if local_p012_ms > 0.0 else 0.0
                ),
                "p2_and_cross_phase_gain_pct_of_local_p012": (
                    100.0 * p2_interaction_gain / local_p012_ms
                    if local_p012_ms > 0.0
                    else 0.0
                ),
                "total_joint_gain_pct_of_local_p012": (
                    100.0 * total_gain / local_p012_ms
                    if local_p012_ms > 0.0
                    else 0.0
                ),
                "perfect_information_gain_vs_reactive_pct": (
                    100.0 * (reactive_p012_ms - perfect_p012_ms) / reactive_p012_ms
                    if reactive_p012_ms > 0.0
                    else 0.0
                ),
                "p2_share_of_positive_total_gain": (
                    p2_interaction_gain / total_gain if total_gain > 1e-9 else None
                ),
                "local_p012_kernel_ms": float(local_p012["solve_time_ms"]),
                "joint_p012_kernel_ms": float(perfect_p012.planning_time_ms),
                "reactive_p012_kernel_ms": float(reactive_p012.planning_time_ms),
            }
        )
    return output


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    metrics = (
        "p01_coupling_gain_pct_of_local_p012",
        "p2_and_cross_phase_gain_pct_of_local_p012",
        "total_joint_gain_pct_of_local_p012",
        "perfect_information_gain_vs_reactive_pct",
        "p2_share_of_positive_total_gain",
    )
    for family_id in sorted({str(row["family_id"]) for row in records}):
        rows = [row for row in records if row["family_id"] == family_id]
        p2_rows = [row for row in rows if row["p2_available"]]
        summary = {
            "instance_count": len(rows),
            "p2_instance_count": len(p2_rows),
        }
        for metric in metrics:
            source = p2_rows if metric != "p01_coupling_gain_pct_of_local_p012" else rows
            summary[metric] = _stats(
                row[metric] for row in source if row[metric] is not None
            )
        summary["by_vep"] = {}
        for vep in sorted({int(row["virtual_ep_size"]) for row in rows}):
            group = [row for row in p2_rows if int(row["virtual_ep_size"]) == vep]
            summary["by_vep"][str(vep)] = {
                metric: _stats(
                    row[metric] for row in group if row[metric] is not None
                )
                for metric in metrics
            }
        output[family_id] = summary
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=("rsbc", "rscf"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    instances = json.loads(args.instances.read_text(encoding="utf-8"))
    if args.limit > 0:
        instances = instances[: int(args.limit)]
    family_ids = tuple(str(value) for value in args.families)
    tasks = [(item, family_ids) for item in instances]
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        nested = list(executor.map(_evaluate_one, tasks, chunksize=1))
    records = [row for group in nested for row in group]
    artifact = {
        "schema_version": "p2_information_decomposition.v1",
        "information_contract": {
            "local_p012": "perfect P012 upfront, per-phase independent plans",
            "joint_p012_perfect": "perfect P012 upfront, one release-aware global window",
            "joint_p012_reactive": (
                "P0/P1 upfront; each true P2 source row revealed only after its P1 barrier"
            ),
            "p01_only_total": "Joint(P01) followed by independent Local(P2)",
        },
        "input_instance_count": len(instances),
        "families": list(family_ids),
        "summary": _summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))


if __name__ == "__main__":
    main()
