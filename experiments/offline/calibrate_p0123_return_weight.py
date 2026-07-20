"""Development-only calibration for the advisory P3 return weight.

P3 is derived as transpose(P2) and is never executable in this study.  This
entrypoint compares the current P0/P1 plan produced by frozen P012 against
P0123 advisory scoring on development traces only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from rs.core.contracts import PlanningConstraints, PlanningIdentity, PlanningTopology, PlanningWeights
from rs.planning import PlannerRegistry
from rs.planning.request_builder import build_window_planning_request

from experiments.offline.compare_p012_p0123_future import matrix, model_roots, remote_total, signature

Matrix = tuple[tuple[int, ...], ...]


def _request(instance_id: str, p0: Matrix, p1: Matrix, p2: Matrix, *, mode: str, p3_weight: float):
    return build_window_planning_request(
        identity=PlanningIdentity(request_id=instance_id),
        p0_dispatch_rows=p0,
        p1_return_rows=p1,
        p2_hint_rows=p2,
        predictor_id="fate_cross_layer_gate_v1",
        confidence=1.0,
        topology=PlanningTopology(world_size=len(p0)),
        constraints=PlanningConstraints(
            bucket_rows=1,
            max_waves=4096,
            expert_compute_delay=0.0,
            phase_release_model="p1_return",
        ),
        weights=PlanningWeights(p3_return_weight=float(p3_weight)),
        information_mode=mode,
    )


def _makespan(plan) -> float:
    return float(dict(plan.metadata).get("makespan", sum(float(w.estimated_duration) for w in plan.waves)) or 0.0)


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[round(0.95 * (len(ordered) - 1))],
        "max": max(ordered),
    }


def calibrate(*, bundle: Path, planner_id: str, weights: tuple[float, ...]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    planner = PlannerRegistry.create(planner_id, usage="runtime")
    with tempfile.TemporaryDirectory(prefix="rs_p0123_calibration_") as tmp:
        for root in model_roots(bundle, Path(tmp)):
            model = root.name.replace("RouterSense_fate_trace_", "").replace("_single_gpu_20260718", "")
            traffic = json.loads((root / "traffic" / "traffic_instances.json").read_text(encoding="utf-8"))
            hints = {
                row["instance_id"]: row
                for row in (
                    json.loads(line)
                    for line in (root / "fate" / "fate_hints.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            for item in traffic:
                instance_id = str(item["instance_id"])
                hint = hints.get(instance_id)
                if not instance_id.startswith("dev-") or hint is None:
                    continue
                p0 = matrix(item["P0_matrix"])
                p1 = matrix(item["P1_matrix"])
                truth = matrix(item["P2_truth_matrix"])
                if remote_total(truth) == 0:
                    continue
                p2 = matrix(hint["target_dispatch_rows"])
                p012 = planner.plan(_request(f"{model}:{instance_id}:p012", p0, p1, p2, mode="p0_p1_p2", p3_weight=0.0))
                p012_makespan = _makespan(p012)
                for weight in weights:
                    started = time.perf_counter_ns()
                    p0123 = planner.plan(
                        _request(
                            f"{model}:{instance_id}:p0123:{weight}",
                            p0,
                            p1,
                            p2,
                            mode="p0_p1_p2_p3",
                            p3_weight=float(weight),
                        )
                    )
                    planning_us = (time.perf_counter_ns() - started) / 1000.0
                    p0123_makespan = _makespan(p0123)
                    delta = 0.0 if p012_makespan <= 0 else 100.0 * (p0123_makespan - p012_makespan) / p012_makespan
                    records.append(
                        {
                            "model": model,
                            "instance_id": instance_id,
                            "weight": float(weight),
                            "delta_pct": float(delta),
                            "plan_changed": signature(p0123) != signature(p012),
                            "planning_us": float(planning_us),
                        }
                    )
    sweep: list[dict[str, Any]] = []
    for weight in weights:
        rows = [row for row in records if float(row["weight"]) == float(weight)]
        deltas = [float(row["delta_pct"]) for row in rows]
        sweep.append(
            {
                "weight": float(weight),
                "instances": len(rows),
                "wins_ties_losses": [
                    sum(value < -1e-9 for value in deltas),
                    sum(abs(value) <= 1e-9 for value in deltas),
                    sum(value > 1e-9 for value in deltas),
                ],
                "delta_pct": _stats(deltas),
                "plan_changed_rate": 0.0 if not rows else sum(bool(row["plan_changed"]) for row in rows) / len(rows),
                "planning_us": _stats([float(row["planning_us"]) for row in rows]),
            }
        )
    # Lowest development mean regret wins.  Ties favor the smaller weight.
    selected = min(sweep, key=lambda row: (float(row["delta_pct"]["mean"]), float(row["weight"])))
    return {
        "schema_version": "routersense.p0123_return_weight_calibration.v1",
        "data_split": "development_only",
        "validation_used_for_selection": False,
        "planner_id": planner_id,
        "selection_rule": "minimum mean P0123-vs-P012 makespan delta; tie -> lower weight",
        "selected_weight": float(selected["weight"]),
        "sweep": sweep,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--planner", default="current:p0123:joint:event:rscf")
    parser.add_argument("--weights", default="0.01,0.02,0.05,0.1,0.2,0.25,0.5,1.0")
    args = parser.parse_args()
    weights = tuple(float(item.strip()) for item in str(args.weights).split(",") if item.strip())
    if not weights or any(value < 0.0 for value in weights):
        raise ValueError("weights must be a non-empty comma-separated list of non-negative values")
    result = calibrate(bundle=args.bundle, planner_id=str(args.planner), weights=weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_weight": result["selected_weight"], "sweep": result["sweep"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
