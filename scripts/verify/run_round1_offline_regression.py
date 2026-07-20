from __future__ import annotations

"""Small deterministic offline regression for the round-1 code convergence.

The probe is intentionally narrow: RSCF, Current-P012, Local/Joint and
Event/Global over the same eight validation windows at EP 4/8/12/16.  It uses
the formal performance metric schema and fails if the aggregate Joint gain
falls below conservative regression thresholds.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

import numpy as np

from rs.core.contracts.performance import (
    MetricBaselineIdentity,
    MetricProvenance,
    OfflineWindowMetrics,
    OptimizationMetrics,
    PerformanceMetricRecord,
    StrategyIdentity,
)
from rs.evaluation.window_metrics import derive_window_metrics, improvement_pct
from rs.scheduling.p012_future._kernel.contracts import (
    AffineLinkCost,
    ForecastPlanningRequest,
    HomogeneousTopology,
    P2RevealRequest,
    PlannerConstraints,
    TrafficHint,
)
from rs.scheduling.p012_future._kernel.families import build_scoped_planner

SEED = 20260720
MODEL_QUOTAS = {"deepseekv2lite": 3, "qwen15moe": 3, "olmoe": 2}
EP_SIZES = (4, 8, 12, 16)
THRESHOLDS = {4: 2.0, 8: 6.0, 12: 9.0, 16: 9.0}


@dataclass(frozen=True)
class WindowKey:
    model: str
    sample_id: str
    layer_id: int

    @property
    def stable_key(self) -> str:
        return f"{self.model}:{self.sample_id}:{self.layer_id}"


@dataclass(frozen=True)
class ModelData:
    model: str
    num_experts: int
    layers: dict[tuple[str, int], tuple[tuple[int, int], ...]]
    sample_layers: dict[str, tuple[int, ...]]


def _stable_owner(sample_id: str, token_position: int, world_size: int) -> int:
    payload = f"{SEED}:{sample_id}:{token_position}:{world_size}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % int(world_size)


def _mapping(num_experts: int, world_size: int) -> np.ndarray:
    base, extra = divmod(int(num_experts), int(world_size))
    out = np.empty(int(num_experts), dtype=np.int32)
    cursor = 0
    for rank in range(int(world_size)):
        count = base + (1 if rank < extra else 0)
        out[cursor : cursor + count] = rank
        cursor += count
    return out


def _matrix(
    assignments: Iterable[tuple[int, int]], *, sample_id: str, num_experts: int, ep: int
) -> np.ndarray:
    placement = _mapping(num_experts, ep)
    out = np.zeros((ep, ep), dtype=np.int32)
    for token_position, expert_id in assignments:
        source = _stable_owner(sample_id, int(token_position), ep)
        destination = int(placement[int(expert_id)])
        if source != destination:
            out[source, destination] += 1
    return out


def _load_model(root: Path) -> ModelData:
    slug = root.name.replace("RouterSense_fate_trace_", "").replace("_single_gpu_20260718", "")
    traffic = json.loads((root / "traffic" / "traffic_instances.json").read_text(encoding="utf-8"))
    if not traffic:
        raise ValueError(f"empty traffic artifact: {root}")
    num_experts = len(traffic[0]["expert_to_rank_mapping"])
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = {}
    with (root / "trace" / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split")) != "validation":
                continue
            key = (str(row["sample_id"]), int(row["layer_id"]))
            grouped.setdefault(key, []).append((int(row["token_position"]), int(row["expert_id"])))
    frozen = {key: tuple(values) for key, values in grouped.items()}
    sample_layers: dict[str, list[int]] = {}
    for sample_id, layer_id in frozen:
        sample_layers.setdefault(sample_id, []).append(layer_id)
    return ModelData(
        model=slug,
        num_experts=num_experts,
        layers=frozen,
        sample_layers={sample: tuple(sorted(layers)) for sample, layers in sample_layers.items()},
    )


def _select_windows(models: dict[str, ModelData]) -> list[WindowKey]:
    selected: list[WindowKey] = []
    for model, quota in MODEL_QUOTAS.items():
        candidates: list[WindowKey] = []
        for sample_id, layers in models[model].sample_layers.items():
            candidates.extend(WindowKey(model, sample_id, layer) for layer in layers[:-1])
        candidates.sort(key=lambda item: hashlib.sha256(item.stable_key.encode("utf-8")).hexdigest())
        if len(candidates) < quota:
            raise ValueError(f"not enough windows for {model}: {len(candidates)} < {quota}")
        selected.extend(candidates[:quota])
    selected.sort(key=lambda item: hashlib.sha256(item.stable_key.encode("utf-8")).hexdigest())
    return selected


def _truth(data: ModelData, key: WindowKey, ep: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layers = data.sample_layers[key.sample_id]
    index = layers.index(key.layer_id)
    next_layer = layers[index + 1]
    p0 = _matrix(
        data.layers[(key.sample_id, key.layer_id)],
        sample_id=key.sample_id,
        num_experts=data.num_experts,
        ep=ep,
    )
    p1 = np.ascontiguousarray(p0.T)
    p2 = _matrix(
        data.layers[(key.sample_id, next_layer)],
        sample_id=key.sample_id,
        num_experts=data.num_experts,
        ep=ep,
    )
    return p0, p1, p2


def _digest_tree(trace_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(trace_root.glob("RouterSense_fate_trace_*_single_gpu_20260718/trace/trace.jsonl")):
        digest.update(path.parent.parent.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _run_plan(
    *, scope: str, engine: str, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
    ep: int, request_id: str, repeats: int,
) -> tuple[Any, float, float]:
    planner = build_scoped_planner(scope=scope, engine=engine, family="rscf")
    hint = TrafficHint(
        predictor_id="perfect",
        target_dispatch_rows=p2,
        confidence=1.0,
        hint_kind="perfect_trace_hint",
        oracle=True,
        matrix_kind="remote_rows",
        metadata={"round1_regression": True},
    )
    request = ForecastPlanningRequest(
        p0_dispatch_rows=p0,
        p1_return_rows=p1,
        prediction_hint=hint,
        topology=HomogeneousTopology.contiguous(ep, ep),
        cost_model=AffineLinkCost(),
        constraints=PlannerConstraints(expert_compute_delay=0.0, max_waves=10000),
        request_id=request_id,
    )
    planning: list[float] = []
    binding: list[float] = []
    bound = None
    for _ in range(max(1, int(repeats))):
        started = time.perf_counter_ns()
        artifact = planner.plan_forecast(request)
        planning.append((time.perf_counter_ns() - started) / 1e6)
        reveal = P2RevealRequest(
            forecast_request_digest=artifact.semantic_digest(),
            p2_truth_rows=p2,
            request_id=request_id,
        )
        started = time.perf_counter_ns()
        bound = planner.bind(artifact, request, reveal)
        binding.append((time.perf_counter_ns() - started) / 1e6)
    assert bound is not None
    materialized = bound.materialize() if hasattr(bound, "materialize") else bound
    if not bool(materialized.valid):
        raise AssertionError(f"invalid plan for {request_id}")
    return bound, float(statistics.median(planning)), float(statistics.median(binding))


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else float(statistics.fmean(clean))


def run(trace_root: Path, output_dir: Path, *, repeats: int = 3) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = sorted(trace_root.glob("RouterSense_fate_trace_*_single_gpu_20260718"))
    models = {data.model: data for data in (_load_model(root) for root in roots)}
    missing = set(MODEL_QUOTAS) - set(models)
    if missing:
        raise ValueError(f"missing model traces: {sorted(missing)}")
    windows = _select_windows(models)
    trace_digest = _digest_tree(trace_root)
    sample_digest = hashlib.sha256("\n".join(key.stable_key for key in windows).encode("utf-8")).hexdigest()

    raw: list[dict[str, Any]] = []
    metric_records: list[dict[str, object]] = []
    for ep in EP_SIZES:
        for key in windows:
            p0, p1, p2 = _truth(models[key.model], key, ep)
            per_engine: dict[str, dict[str, Any]] = {}
            for engine in ("event", "global"):
                for scope in ("local", "joint"):
                    planner_id = f"current:p012:{scope}:{engine}:rscf"
                    bound, planning_ms, bind_ms = _run_plan(
                        scope=scope,
                        engine=engine,
                        p0=p0,
                        p1=p1,
                        p2=p2,
                        ep=ep,
                        request_id=f"round1:{ep}:{key.stable_key}:{scope}:{engine}",
                        repeats=repeats,
                    )
                    metrics = derive_window_metrics(
                        bound,
                        planning_ms=planning_ms,
                        bind_ms=bind_ms,
                        target_entry_overhead_ms=planning_ms + bind_ms,
                    )
                    row = {
                        "ep": ep,
                        "model": key.model,
                        "sample_id": key.sample_id,
                        "layer_id": key.layer_id,
                        "instance_key": key.stable_key,
                        "scope": scope,
                        "engine": engine,
                        "planner_id": planner_id,
                        **metrics.to_dict(),
                    }
                    if scope == "local":
                        per_engine[engine] = row
                    raw.append(row)
            for engine in ("event", "global"):
                baseline = per_engine[engine]
                for row in [item for item in raw[-4:] if item["engine"] == engine]:
                    opt = OptimizationMetrics(
                        communication_optimization_pct=improvement_pct(
                            baseline["communication_makespan"], row["communication_makespan"]
                        ),
                        tail_optimization_pct=improvement_pct(
                            baseline["tail_latency_p99"], row["tail_latency_p99"]
                        ),
                        first_token_optimization_pct=improvement_pct(
                            baseline["first_token_time"], row["first_token_time"]
                        ),
                        scope_communication_gain_pct=(
                            improvement_pct(baseline["communication_makespan"], row["communication_makespan"])
                            if row["scope"] == "joint" else 0.0
                        ),
                        scope_tail_gain_pct=(
                            improvement_pct(baseline["tail_latency_p99"], row["tail_latency_p99"])
                            if row["scope"] == "joint" else 0.0
                        ),
                        scope_first_token_gain_pct=(
                            improvement_pct(baseline["first_token_time"], row["first_token_time"])
                            if row["scope"] == "joint" else 0.0
                        ),
                    )
                    row.update(opt.to_dict())
                    record = PerformanceMetricRecord(
                        strategy=StrategyIdentity(
                            planner_id=str(row["planner_id"]),
                            timing="current",
                            horizon="p012",
                            scope=str(row["scope"]),
                            engine=engine,
                            core="rscf",
                            predictor="perfect",
                            prediction_fidelity="perfect_trace",
                            safety="raw",
                        ),
                        baseline=MetricBaselineIdentity(
                            planner_id=f"current:p012:local:{engine}:rscf",
                            comparison_key=f"same-sample-same-engine:ep{ep}:{key.stable_key}",
                        ),
                        provenance=MetricProvenance(
                            metric_domain="offline_logical",
                            time_unit="logical_time",
                            trace_digest=trace_digest,
                            sample_set_digest=sample_digest,
                            measurement_status="complete",
                            source="scripts/verify/run_round1_offline_regression.py",
                            ep_size=ep,
                            sample_count=1,
                            metadata={"timing_repeats": repeats, "model": key.model},
                        ),
                        metrics=OfflineWindowMetrics.from_dict(row),
                        optimization=opt,
                    )
                    metric_records.append(record.to_dict())

    summary: list[dict[str, Any]] = []
    for ep in EP_SIZES:
        for engine in ("event", "global"):
            rows = [row for row in raw if row["ep"] == ep and row["engine"] == engine and row["scope"] == "joint"]
            summary.append({
                "ep": ep,
                "engine": engine,
                "sample_count": len(rows),
                "communication_optimization_pct_mean": _mean([row["communication_optimization_pct"] for row in rows]),
                "tail_optimization_pct_mean": _mean([row["tail_optimization_pct"] for row in rows]),
                "first_token_optimization_pct_mean": _mean([row["first_token_optimization_pct"] for row in rows]),
                "planning_ms_p95": sorted(float(row["planning_ms"]) for row in rows)[max(0, int(np.ceil(0.95 * len(rows))) - 1)],
                "regression_count": sum(float(row["communication_optimization_pct"]) < -1e-9 for row in rows),
            })

    failures = []
    for ep in EP_SIZES:
        best = max(
            float(row["communication_optimization_pct_mean"])
            for row in summary if row["ep"] == ep
        )
        if best + 1e-9 < THRESHOLDS[ep]:
            failures.append({"ep": ep, "best_gain": best, "threshold": THRESHOLDS[ep]})

    payload = {
        "schema": "routersense.round1_offline_regression.v1",
        "trace_digest": trace_digest,
        "sample_set_digest": sample_digest,
        "selected_windows": [key.__dict__ for key in windows],
        "ep_sizes": list(EP_SIZES),
        "core": "rscf",
        "horizon": "p012",
        "timing": "current",
        "predictor": "perfect",
        "summary": summary,
        "performance_metrics": metric_records,
        "rows": raw,
        "thresholds": THRESHOLDS,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    (output_dir / "round1_offline_regression.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "round1_offline_regression_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    with (output_dir / "round1_offline_regression_rows.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader(); writer.writerows(raw)
    if failures:
        raise SystemExit(f"offline regression failed: {failures}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()
    payload = run(args.trace_root, args.output_dir, repeats=args.timing_repeats)
    print(json.dumps({"status": payload["status"], "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
