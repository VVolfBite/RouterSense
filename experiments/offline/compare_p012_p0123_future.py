"""CPU trace comparison for P012, advisory P0123, and Future-P012."""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from rs.core.contracts import PlanningConstraints, PlanningIdentity, PlanningTopology, PlanningWeights
from rs.planning import PlannerPolicyConfig, PlannerRegistry
from rs.planning.request_builder import build_window_planning_request
from rs.runtime.online.megatron_ep.target_planning.contracts import TwoHorizonPrediction
from rs.runtime.online.megatron_ep.target_planning.planner_service import TargetLayerPlannerMetrics, TargetLayerPlannerService, TargetLayerPlanningRequest
from rs.runtime.online.megatron_ep.target_planning.predictor import TwoHorizonPredictionBundle
from rs.runtime.online.megatron_ep.target_planning.reconcile import reconcile_once
from rs.runtime.online.megatron_ep.target_planning.store import TargetPlanStore
from rs.scheduling.validation import stable_hash

Matrix = tuple[tuple[int, ...], ...]


def matrix(raw: Any) -> Matrix:
    return tuple(tuple(int(v) for v in row) for row in raw)


def transpose(rows: Matrix) -> Matrix:
    return tuple(tuple(int(rows[col][row]) for col in range(len(rows))) for row in range(len(rows)))


def remote_total(rows: Matrix) -> int:
    return sum(int(v) for i, row in enumerate(rows) for j, v in enumerate(row) if i != j)


def signature(plan) -> tuple:
    return tuple((f.phase, f.src_rank, f.dst_rank, f.row_count) for w in plan.waves for f in w.flows)


def makespan(plan) -> float:
    return float(dict(plan.metadata).get("makespan", sum(float(w.estimated_duration) for w in plan.waves)) or 0.0)


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(v) for v in values)
    p95 = ordered[round(0.95 * (len(ordered) - 1))]
    return {"mean": statistics.fmean(ordered), "median": statistics.median(ordered), "p95": p95, "max": max(ordered)}


def request(instance_id: str, p0: Matrix, p1: Matrix, p2: Matrix, mode: str, p3_weight: float):
    return build_window_planning_request(
        identity=PlanningIdentity(request_id=instance_id),
        p0_dispatch_rows=p0,
        p1_return_rows=p1,
        p2_hint_rows=p2,
        predictor_id="fate_cross_layer_gate_v1",
        confidence=1.0,
        topology=PlanningTopology(world_size=len(p0)),
        constraints=PlanningConstraints(bucket_rows=1, max_waves=4096, expert_compute_delay=0.0, phase_release_model="p1_return"),
        weights=PlanningWeights(p3_return_weight=p3_weight),
        information_mode=mode,
    )


class FixedTwoHorizon:
    def __init__(self, h1: Matrix, h2: Matrix, source_layer: str, confidence: float) -> None:
        now = time.perf_counter_ns()
        target = str(int(source_layer) + 1) if source_layer.isdigit() else f"{source_layer}+1"
        target2 = str(int(source_layer) + 2) if source_layer.isdigit() else f"{source_layer}+2"
        self.bundle = TwoHorizonPredictionBundle(
            h1=TwoHorizonPrediction(1, source_layer, target, "rows", h1, stable_hash([list(r) for r in h1]), "fate_cross_layer_gate_v1", confidence, now, 0.0),
            h2=TwoHorizonPrediction(2, target, target2, "rows", h2, stable_hash([list(r) for r in h2]), "copy_h1_bridge_v1", min(confidence, 0.5), now, 0.0),
        )

    def predict_two_horizon(self, **_kwargs):
        return self.bundle


def _discover_model_roots(root: Path) -> list[Path]:
    roots: set[Path] = set()
    if (root / "traffic" / "traffic_instances.json").is_file():
        roots.add(root)
    for traffic_file in root.rglob("traffic/traffic_instances.json"):
        candidate = traffic_file.parent.parent
        if (candidate / "fate" / "fate_hints.jsonl").is_file():
            roots.add(candidate)
    return sorted(roots)


def _extract_zip_roots(archive: Path, target: Path) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(target)
    roots = _discover_model_roots(target)
    for child in sorted(target.rglob("*.zip")):
        nested_target = target / f"__nested_{child.stem}"
        with zipfile.ZipFile(child) as inner:
            inner.extractall(nested_target)
        roots.extend(_discover_model_roots(nested_target))
    return sorted(set(roots))


def model_roots(bundle: Path, temp: Path) -> list[Path]:
    if bundle.is_dir():
        roots = _discover_model_roots(bundle)
        for index, child in enumerate(sorted(bundle.glob("*.zip"))):
            roots.extend(_extract_zip_roots(child, temp / "directory_zips" / f"{index:03d}_{child.stem}"))
        return sorted(set(roots))
    return _extract_zip_roots(bundle, temp / "outer")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"instances": 0}
    deltas = [float(r["p0123_delta_pct"]) for r in rows]
    status: dict[str, int] = {}
    for row in rows:
        key = str(row["future_reconcile_status"])
        status[key] = status.get(key, 0) + 1
    return {
        "instances": len(rows),
        "p0123_vs_p012_wins_ties_losses": [sum(x < -1e-9 for x in deltas), sum(abs(x) <= 1e-9 for x in deltas), sum(x > 1e-9 for x in deltas)],
        "p0123_plan_changed_rate": sum(bool(r["p0123_plan_changed"]) for r in rows) / len(rows),
        "p0123_delta_pct": stats(deltas),
        "p012_build_us": stats([float(r["p012_build_us"]) for r in rows]),
        "p0123_build_us": stats([float(r["p0123_build_us"]) for r in rows]),
        "future_build_us": stats([float(r["future_build_us"]) for r in rows]),
        "future_plan_equivalence_rate": sum(bool(r["future_same_plan_as_on_demand"]) for r in rows) / len(rows),
        "future_reconcile_status": status,
        "future_preserved_edge_ratio": stats([float(r["future_preserved_edge_ratio"]) for r in rows]),
        "future_reconcile_us": stats([float(r["future_reconcile_us"]) for r in rows]),
    }


def evaluate(bundle: Path, core: str, scope: str, engine: str, split_prefix: str, p3_weight: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rs_modes_") as tmp:
        for root in model_roots(bundle, Path(tmp)):
            slug = root.name.replace("RouterSense_fate_trace_", "").replace("_single_gpu_20260718", "")
            traffic = json.loads((root / "traffic" / "traffic_instances.json").read_text(encoding="utf-8"))
            hints = {x["instance_id"]: x for x in (json.loads(line) for line in (root / "fate" / "fate_hints.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
            p012_id = f"current:p012:{scope}:{engine}:{core}"
            p0123_id = f"current:p0123:{scope}:{engine}:{core}"
            future_id = f"future:p012:{scope}:{engine}:{core}"
            paired_local_id = f"future:p012:local:{engine}:{core}"
            p012_planner = PlannerRegistry.create(p012_id, usage="runtime")
            p0123_planner = PlannerRegistry.create(p0123_id, usage="runtime")
            for item in traffic:
                instance_id = str(item["instance_id"])
                hint = hints.get(instance_id)
                if not instance_id.startswith(split_prefix) or hint is None:
                    continue
                p0, p1, truth = matrix(item["P0_matrix"]), matrix(item["P1_matrix"]), matrix(item["P2_truth_matrix"])
                if remote_total(truth) == 0:
                    continue
                p2 = matrix(hint["target_dispatch_rows"])
                t0 = time.perf_counter_ns(); p012_req = request(f"{slug}:{instance_id}:p012", p0, p1, p2, "p0_p1_p2", 0.0); p012 = p012_planner.plan(p012_req); p012_us = (time.perf_counter_ns()-t0)/1000
                t0 = time.perf_counter_ns(); p0123_req = request(f"{slug}:{instance_id}:p0123", p0, p1, p2, "p0_p1_p2_p3", float(p3_weight)); p0123 = p0123_planner.plan(p0123_req); p0123_us = (time.perf_counter_ns()-t0)/1000

                layer = str(item.get("trace_sample_id", instance_id).split(":")[-1])
                service = TargetLayerPlannerService(
                    store=TargetPlanStore(),
                    two_horizon_predictor_factory=lambda _name, h1=p2, source=layer, conf=float(hint.get("confidence", 0.75)): FixedTwoHorizon(h1, h1, source, conf),
                )
                future_req = TargetLayerPlanningRequest(
                    run_id=f"trace:{slug}", forward_epoch=0, microbatch_id=instance_id,
                    source_layer_id=layer, target_layer_id=str(int(layer)+1) if layer.isdigit() else f"{layer}+1",
                    current_p0_rows=p0, previous_p0_rows=None, predictor_name="copy_h1_bridge_v1",
                    policy_id=future_id, joint_planner_id=future_id,
                    local_planner_id=(future_id if scope == "local" else paired_local_id),
                    safe_projection_mode="disabled",
                    group_size=len(p0), bucket_rows=1, policy_options=PlannerPolicyConfig(), topology_digest="vep4",
                    bucket_contract_digest="canonical_bucket_rows", information_mode="p0_p1_p2", planning_track="runtime_lookahead",
                    planning_timing="previous_layer", max_waves=4096,
                )
                metrics = TargetLayerPlannerMetrics()
                try:
                    t0 = time.perf_counter_ns()
                    built = service._build_target_plan(request=future_req, metrics=metrics)  # noqa: SLF001
                    future_us = (time.perf_counter_ns()-t0)/1000
                finally:
                    service.close()
                same = p012_planner.plan(built.planning_request)
                t0 = time.perf_counter_ns(); reconciled = reconcile_once(prepared_plan=built.prepared_plan, actual_p0_rows=truth); reconcile_us = (time.perf_counter_ns()-t0)/1000
                m0, m1 = makespan(p012), makespan(p0123)
                rows.append({
                    "model": slug, "instance_id": instance_id,
                    "p012_makespan": m0, "p0123_makespan": m1,
                    "p0123_delta_pct": 0.0 if m0 <= 0 else 100.0*(m1-m0)/m0,
                    "p0123_plan_changed": signature(p0123) != signature(p012),
                    "p012_build_us": p012_us, "p0123_build_us": p0123_us,
                    "future_build_us": future_us,
                    "future_same_plan_as_on_demand": (
                        built.prepared_plan.window_plan is not None
                        and signature(built.prepared_plan.window_plan) == signature(same)
                    ),
                    "future_reconcile_status": reconciled.status,
                    "future_preserved_edge_ratio": reconciled.preserved_edge_ratio,
                    "future_reconcile_us": reconcile_us,
                })
    by_model = {model: aggregate([r for r in rows if r["model"] == model]) for model in sorted({r["model"] for r in rows})}
    return {
        "schema_version": "routersense.p012_modes.trace_validation.v3",
        "core": str(core),
        "scope": str(scope),
        "engine": str(engine),
        "planner_ids": {
            "p012": f"current:p012:{scope}:{engine}:{core}",
            "p0123": f"current:p0123:{scope}:{engine}:{core}",
            "future": f"future:p012:{scope}:{engine}:{core}",
            "strict_paired_local": f"future:p012:local:{engine}:{core}",
        },
        "split_prefix": split_prefix,
        "p3_return_weight": float(p3_weight),
        "aggregate": aggregate(rows),
        "by_model": by_model,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core", choices=("gmwd", "rsbc", "rscf"), default="rscf")
    parser.add_argument("--scope", choices=("local", "joint"), default="joint")
    parser.add_argument("--engine", choices=("event", "global"), default="global")
    parser.add_argument(
        "--branch", choices=("event", "global"), default=None,
        help="deprecated compatibility alias; implies --scope joint and sets --engine",
    )
    parser.add_argument("--split-prefix", default="val-")
    parser.add_argument("--p3-weight", type=float, default=0.01)
    args = parser.parse_args()
    scope = "joint" if args.branch is not None else args.scope
    engine = args.branch or args.engine
    result = evaluate(args.bundle, args.core, scope, engine, args.split_prefix, args.p3_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
