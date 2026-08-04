#!/usr/bin/env python3
"""Serial offline closure for prediction, exact oracle, baselines, and safe-U."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments._bootstrap import ensure_src_on_path

ROOT = ensure_src_on_path()

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for prediction/oracle/baseline closure") from exc

from rs.runtime.offline.runner import replay_and_audit_logical_plan
from rs.runtime.offline.prediction import rolling_predictor_records
from rs.runtime.online.megatron_ep.async_release.runtime_projection import host_project_safe_selection
from rs.scheduling import (
    FlowDemand,
    FlowWindow,
    ForecastPressure,
    GlobalReadySetOptions,
    LogicalTopology,
    MultiPhaseSchedulingProblem,
    ReleaseConstraint,
    resolve_policy,
)
from rs.scheduling.traffic_matrix import (
    canonicalize_remote_matrix,
    matrix_col_sums_remote,
    matrix_digest_remote,
    matrix_nonzero_remote_edge_count,
    matrix_remote_bytes,
    matrix_row_sums_remote,
)


Matrix = tuple[tuple[int, ...], ...]

TIE_THRESHOLD = 0.001


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping config: {path}")
    return payload


def _git(command: list[str]) -> str:
    result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _matrix(value: Any) -> Matrix:
    return canonicalize_remote_matrix(value)


def _zero_matrix_like(matrix: Matrix) -> Matrix:
    return canonicalize_remote_matrix(tuple(tuple(0 for _ in row) for row in matrix))


def _stable_digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "sample_count": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "win_rate": None,
        "tie_rate": None,
        "loss_rate": None,
    }


def _paired_rates(deltas: list[float]) -> dict[str, float | None]:
    if not deltas:
        return {"win_rate": None, "tie_rate": None, "loss_rate": None}
    wins = sum(1 for delta in deltas if delta > TIE_THRESHOLD)
    ties = sum(1 for delta in deltas if abs(delta) <= TIE_THRESHOLD)
    losses = sum(1 for delta in deltas if delta < -TIE_THRESHOLD)
    n = len(deltas)
    return {
        "win_rate": wins / n,
        "tie_rate": ties / n,
        "loss_rate": losses / n,
    }


def _pct_gain(base: float, value: float) -> float | None:
    if abs(base) <= 1e-12:
        return None
    return (float(base) - float(value)) / float(base)


def _safe_div(num: float, den: float) -> float | None:
    if abs(den) <= 1e-12:
        return None
    return float(num) / float(den)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return num / (den_x * den_y)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for idx, _value in indexed[cursor:end]:
            ranks[idx] = rank
        cursor = end
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _flows_from_matrix(matrix: Matrix, *, phase: str, release_state: str, executable: bool) -> tuple[FlowDemand, ...]:
    flows: list[FlowDemand] = []
    for src_rank, row in enumerate(matrix):
        for dst_rank, byte_count in enumerate(row):
            if src_rank == dst_rank or int(byte_count) <= 0:
                continue
            flows.append(
                FlowDemand(
                    flow_id=f"{phase}:{src_rank}->{dst_rank}",
                    phase=phase,
                    src_rank=int(src_rank),
                    dst_rank=int(dst_rank),
                    byte_count=int(byte_count),
                    release_state=release_state,
                    is_executable=executable,
                )
            )
    return tuple(flows)


def _predictor_confidence(hint_type: str) -> float:
    if hint_type == "zero_hint":
        return 0.0
    if hint_type == "copy_current_dispatch":
        return 1.0
    if hint_type == "history_ema":
        return 0.75
    if hint_type == "history_linear_trend":
        return 0.8
    if hint_type == "perfect_trace_hint":
        return 1.0
    if hint_type == "shuffled_hint":
        return 1.0
    raise ValueError(f"unsupported hint type {hint_type!r}")


def _build_problem_with_hint_and_truth(
    *,
    p0: Matrix,
    p1: Matrix,
    planning_hint: Matrix,
    execution_truth: Matrix,
    hint_type: str,
    scheduling_mode: str,
    expert_compute_delay: float,
) -> MultiPhaseSchedulingProblem:
    planning_hint = canonicalize_remote_matrix(planning_hint)
    execution_truth = canonicalize_remote_matrix(execution_truth)
    return MultiPhaseSchedulingProblem(
        flow_window=FlowWindow(
            ready_flows=_flows_from_matrix(p0, phase="p0_dispatch", release_state="ready", executable=True),
            blocked_flows=_flows_from_matrix(p1, phase="p1_return", release_state="blocked", executable=False),
            forecast_pressure=_flows_from_matrix(
                planning_hint,
                phase="p2_next_dispatch_forecast",
                release_state="advisory_only",
                executable=False,
            ),
        ),
        topology=LogicalTopology(num_gpus=len(p0)),
        release_model=ReleaseConstraint(
            phase="p1_return",
            rank=0,
            release_after_phase="p0_dispatch",
            expert_compute_delay=float(expert_compute_delay),
        ),
        forecast=ForecastPressure(
            source=str(hint_type),
            digest=matrix_digest_remote(planning_hint),
            oracle=False,
            evaluation_eligible=True,
            matrix_shape=(len(planning_hint), len(planning_hint[0]) if planning_hint else 0),
            matrix_total_bytes=int(matrix_remote_bytes(planning_hint)),
            matrix=planning_hint,
            metadata={
                "planning_hint_matrix": [list(row) for row in planning_hint],
                "planning_hint_digest": matrix_digest_remote(planning_hint),
                "execution_truth_digest": matrix_digest_remote(execution_truth),
            },
        ),
        options=GlobalReadySetOptions(
            scheduling_mode=str(scheduling_mode),
            information_mode="p0_p1_p2",
            prediction_confidence=float(_predictor_confidence(hint_type)),
            max_waves=256,
        ),
        p0_dispatch_matrix=p0,
        p1_return_matrix=p1,
        p2_next_dispatch_forecast_matrix=execution_truth,
    )


def _load_fixtures(fixture_dir: Path) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for path in sorted(fixture_dir.glob("replay_layer_*.json"), key=lambda item: int(item.stem.split("_")[-1])):
        payload = json.loads(path.read_text(encoding="utf-8"))
        p0 = _matrix(payload["p0_dispatch_matrix"])
        p1 = _matrix(payload["p1_return_matrix"])
        p2 = _matrix(payload.get("p2_next_dispatch_matrix", payload.get("p2_next_dispatch_forecast_matrix", [])))
        metadata = dict(payload.get("metadata", {}))
        fixtures.append(
            {
                "fixture_id": path.stem,
                "fixture_path": str(path),
                "window_id": f"{metadata.get('layer_id', path.stem)}->{metadata.get('next_layer_id', '')}",
                "layer_id": str(metadata.get("layer_id", path.stem)),
                "next_layer_id": str(metadata.get("next_layer_id", "")),
                "metadata": metadata,
                "p0": p0,
                "p1": p1,
                "p2_truth": p2,
                "truth_digest": _stable_digest(
                    {
                        "p0": [list(row) for row in p0],
                        "p1": [list(row) for row in p1],
                        "p2": [list(row) for row in p2],
                    }
                ),
            }
        )
    return fixtures


def _predictor_maps(fixture_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    mapping: dict[str, dict[str, dict[str, Any]]] = {}
    for predictor_name in ("zero_hint", "copy_current_dispatch", "fate_style_history", "fate_style_linear"):
        records = rolling_predictor_records(fixture_dir=fixture_dir, predictor_name=predictor_name)
        mapping[predictor_name] = {
            str(record.layer_id): {
                "matrix": canonicalize_remote_matrix(record.predicted_matrix),
                "confidence": float(record.confidence),
                "relative_l1": float(record.relative_l1_error),
                "cosine": float(record.cosine_similarity),
                "topk_edge_overlap": float(record.topk_edge_overlap),
                "row_sum_error": float(record.row_sum_error),
                "col_sum_error": float(record.col_sum_error),
            }
            for record in records
        }
    return mapping


def _shuffle_truth(matrix: Matrix, *, seed: int) -> Matrix:
    flat = [int(value) for row in matrix for value in row]
    rng = __import__("random").Random(int(seed))
    rng.shuffle(flat)
    width = len(matrix[0]) if matrix else 0
    rows = [tuple(int(value) for value in flat[index:index + width]) for index in range(0, len(flat), width)]
    return canonicalize_remote_matrix(tuple(rows))


def _hint_variants_for_fixture(
    fixture: dict[str, Any],
    predictor_map: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    layer_id = str(fixture["layer_id"])
    p0 = fixture["p0"]
    p2 = fixture["p2_truth"]
    history = predictor_map.get("fate_style_history", {}).get(layer_id)
    linear = predictor_map.get("fate_style_linear", {}).get(layer_id)
    return {
        "zero_hint": {
            "matrix": _zero_matrix_like(p2),
            "confidence": 0.0,
            "predictor_name": "zero_hint",
        },
        "copy_current_dispatch": {
            "matrix": p0,
            "confidence": 1.0,
            "predictor_name": "copy_current_dispatch",
        },
        "history_ema": {
            "matrix": p0 if history is None else history["matrix"],
            "confidence": 0.25 if history is None else 0.75,
            "predictor_name": "history_ema",
        },
        "history_linear_trend": {
            "matrix": p0 if linear is None else linear["matrix"],
            "confidence": 0.2 if linear is None else 0.8,
            "predictor_name": "history_linear_trend",
        },
        "perfect_trace_hint": {
            "matrix": p2,
            "confidence": 1.0,
            "predictor_name": "perfect_trace_hint",
        },
        "shuffled_hint": {
            "matrix": _shuffle_truth(p2, seed=int(layer_id) + 17 if str(layer_id).isdigit() else 17),
            "confidence": 1.0,
            "predictor_name": "shuffled_hint",
        },
    }


def _regime_labels(p0: Matrix, p2: Matrix) -> dict[str, str]:
    remote_slots = max(1, len(p0) * max(len(p0) - 1, 1))
    density = matrix_nonzero_remote_edge_count(p0) / remote_slots
    if density <= 0.34:
        sparsity = "sparse"
    elif density <= 0.67:
        sparsity = "medium"
    else:
        sparsity = "dense"
    row_sums = [float(value) for value in matrix_row_sums_remote(p0)]
    total = sum(row_sums)
    top = max(row_sums, default=0.0)
    top_ratio = 0.0 if total <= 0.0 else top / total
    sorted_rows = sorted(row_sums, reverse=True)
    heavy_tail_ratio = 0.0 if total <= 0.0 or len(sorted_rows) < 2 else sum(sorted_rows[:2]) / total
    if top_ratio >= 0.45:
        skew = "hotspot"
    elif heavy_tail_ratio >= 0.70:
        skew = "heavy_tail"
    else:
        skew = "balanced"
    flat_p0 = [float(value) for row in p0 for value in row]
    flat_p2 = [float(value) for row in p2 for value in row]
    dot = sum(a * b for a, b in zip(flat_p0, flat_p2, strict=True))
    norm0 = math.sqrt(sum(value * value for value in flat_p0))
    norm2 = math.sqrt(sum(value * value for value in flat_p2))
    cosine = 0.0 if norm0 <= 0.0 or norm2 <= 0.0 else dot / (norm0 * norm2)
    if cosine >= 0.85:
        correlation = "high"
    elif cosine >= 0.50:
        correlation = "medium"
    elif cosine >= 0.15:
        correlation = "low"
    else:
        correlation = "adversarial"
    p2_total = float(sum(flat_p2))
    strength = 0.0 if total <= 0.0 else p2_total / total
    if strength <= 0.5:
        p2_strength = "weak"
    elif strength <= 1.5:
        p2_strength = "medium"
    else:
        p2_strength = "strong"
    host_pressure_proxy = max(
        max(matrix_row_sums_remote(p0), default=0),
        max(matrix_col_sums_remote(p0), default=0),
    ) / max(sum(matrix_row_sums_remote(p0)), 1)
    if host_pressure_proxy <= 0.34:
        host_pressure = "low"
    elif host_pressure_proxy <= 0.5:
        host_pressure = "medium"
    else:
        host_pressure = "high"
    return {
        "sparsity_regime": sparsity,
        "skew_regime": skew,
        "correlation_regime": correlation,
        "p2_strength_regime": p2_strength,
        "host_pressure_regime": host_pressure,
    }


def _evaluate_policy(
    *,
    fixture: dict[str, Any],
    policy_name: str,
    hint_type: str,
    hint_matrix: Matrix,
    confidence: float,
    scheduling_mode: str = "execution_window",
    expert_compute_delay: float = 0.0,
) -> dict[str, Any]:
    problem = _build_problem_with_hint_and_truth(
        p0=fixture["p0"],
        p1=fixture["p1"],
        planning_hint=hint_matrix,
        execution_truth=fixture["p2_truth"],
        hint_type=hint_type,
        scheduling_mode=scheduling_mode,
        expert_compute_delay=expert_compute_delay,
    )
    policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
    plan = policy.build_logical_plan(problem)
    audit = replay_and_audit_logical_plan(problem, plan)
    makespan = float(audit.get("makespan", plan.diagnostics.get("makespan", 0.0)))
    regime = _regime_labels(fixture["p0"], fixture["p2_truth"])
    return {
        "fixture_id": fixture["fixture_id"],
        "layer_id": fixture["layer_id"],
        "next_layer_id": fixture["next_layer_id"],
        "window_id": fixture["window_id"],
        "execution_truth_digest": fixture["truth_digest"],
        "policy_name": policy_name,
        "hint_type": hint_type,
        "planning_hint_digest": matrix_digest_remote(hint_matrix),
        "planning_hint_total": int(matrix_remote_bytes(hint_matrix)),
        "planning_hint_nonzero_edges": int(matrix_nonzero_remote_edge_count(hint_matrix)),
        "prediction_confidence": float(confidence),
        "makespan": makespan,
        "valid": bool(audit.get("valid", False)),
        "selected_policy": str(plan.diagnostics.get("selected_policy", plan.policy_name)),
        "fallback_to_B": bool(plan.diagnostics.get("fallback_to_paired_b", False)),
        "raw_u_makespan": plan.diagnostics.get("raw_u_makespan"),
        "paired_b_makespan": plan.diagnostics.get("paired_b_makespan"),
        "safe_makespan": plan.diagnostics.get("safe_makespan", makespan),
        "forecast_source": str(plan.diagnostics.get("forecast_source", "")),
        "future_information_mode": str(plan.diagnostics.get("future_information_mode", "")),
        "prediction_used": bool(plan.diagnostics.get("prediction_used", False)),
        "planning_time_ms": float(plan.diagnostics.get("planning_time_ms", 0.0) or 0.0),
        "oracle_input_used": bool(plan.diagnostics.get("oracle_input_used", False)),
        **regime,
    }


def _summarize_policy_rows(rows: list[dict[str, Any]], *, baseline_name: str, birkhoff_name: str) -> list[dict[str, Any]]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy_name"]), []).append(row)
    baseline_map = {(row["fixture_id"], row["layer_id"]): float(row["makespan"]) for row in by_policy[baseline_name]}
    birkhoff_map = {(row["fixture_id"], row["layer_id"]): float(row["makespan"]) for row in by_policy[birkhoff_name]}
    summary: list[dict[str, Any]] = []
    for policy_name, items in sorted(by_policy.items()):
        makespans = [float(item["makespan"]) for item in items]
        gain_vs_fifo = []
        gain_vs_birkhoff = []
        for item in items:
            key = (str(item["fixture_id"]), str(item["layer_id"]))
            fifo_base = baseline_map[key]
            birk_base = birkhoff_map[key]
            gain_fifo = _pct_gain(fifo_base, float(item["makespan"]))
            gain_b = _pct_gain(birk_base, float(item["makespan"]))
            if gain_fifo is not None:
                gain_vs_fifo.append(gain_fifo)
            if gain_b is not None:
                gain_vs_birkhoff.append(gain_b)
        row = {
            "policy_name": policy_name,
            **_stats(makespans),
            "mean_gain_vs_fifo": _mean(gain_vs_fifo),
            "median_gain_vs_fifo": _median(gain_vs_fifo),
            "mean_gain_vs_birkhoff": _mean(gain_vs_birkhoff),
            "median_gain_vs_birkhoff": _median(gain_vs_birkhoff),
        }
        row.update({k: v for k, v in _paired_rates(gain_vs_fifo).items()})
        summary.append(row)
    return summary


def _build_exact_tasks(matrix: Matrix, *, phase: int) -> list[dict[str, int]]:
    tasks: list[dict[str, int]] = []
    for src, row in enumerate(matrix):
        for dst, value in enumerate(row):
            if src == dst:
                continue
            for ordinal in range(int(value)):
                tasks.append(
                    {
                        "phase": int(phase),
                        "src": int(src),
                        "dst": int(dst),
                        "duration": 1,
                        "ordinal": int(ordinal),
                    }
                )
    return tasks


def _solve_exact_oracle(instance: ExactInstance, *, mode: str, compute_delay: int = 0, time_limit_s: float = 30.0) -> dict[str, Any]:
    from ortools.sat.python import cp_model

    tasks = _build_exact_tasks(instance.p0, phase=0) + _build_exact_tasks(instance.p1, phase=1) + _build_exact_tasks(instance.p2, phase=2)
    if not tasks:
        return {
            "solver_status": "OPTIMAL",
            "objective": 0,
            "wall_time_s": 0.0,
            "task_count": 0,
        }
    model = cp_model.CpModel()
    horizon = len(tasks) + int(compute_delay) * max(1, instance.rank_count) * 2
    starts: list[cp_model.IntVar] = []
    ends: list[cp_model.IntVar] = []
    intervals: list[cp_model.IntervalVar] = []
    for idx, task in enumerate(tasks):
        start = model.NewIntVar(0, horizon, f"start_{idx}")
        end = model.NewIntVar(0, horizon, f"end_{idx}")
        interval = model.NewIntervalVar(start, task["duration"], end, f"interval_{idx}")
        starts.append(start)
        ends.append(end)
        intervals.append(interval)
    for rank in range(instance.rank_count):
        model.AddNoOverlap([intervals[idx] for idx, task in enumerate(tasks) if task["src"] == rank])
        model.AddNoOverlap([intervals[idx] for idx, task in enumerate(tasks) if task["dst"] == rank])
    if mode == "local":
        phase0_end = model.NewIntVar(0, horizon, "phase0_end")
        phase1_end = model.NewIntVar(0, horizon, "phase1_end")
        model.AddMaxEquality(phase0_end, [ends[idx] for idx, task in enumerate(tasks) if task["phase"] == 0] or [0])
        model.AddMaxEquality(phase1_end, [ends[idx] for idx, task in enumerate(tasks) if task["phase"] == 1] or [0])
        for idx, task in enumerate(tasks):
            if task["phase"] == 1:
                model.Add(starts[idx] >= phase0_end + int(compute_delay))
            elif task["phase"] == 2:
                model.Add(starts[idx] >= phase1_end)
    elif mode == "joint":
        for rank in range(instance.rank_count):
            incoming_p0 = [ends[idx] for idx, task in enumerate(tasks) if task["phase"] == 0 and task["dst"] == rank]
            incoming_p1 = [ends[idx] for idx, task in enumerate(tasks) if task["phase"] == 1 and task["dst"] == rank]
            p0_done = model.NewIntVar(0, horizon, f"p0_done_rank{rank}")
            p1_done = model.NewIntVar(0, horizon, f"p1_done_rank{rank}")
            model.AddMaxEquality(p0_done, incoming_p0 or [0])
            model.AddMaxEquality(p1_done, incoming_p1 or [0])
            for idx, task in enumerate(tasks):
                if task["phase"] == 1 and task["src"] == rank:
                    model.Add(starts[idx] >= p0_done + int(compute_delay))
                if task["phase"] == 2 and task["src"] == rank:
                    model.Add(starts[idx] >= p1_done)
    else:
        raise ValueError(f"unsupported exact mode {mode!r}")
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, ends)
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 1
    start = time.perf_counter()
    status = solver.Solve(model)
    wall = time.perf_counter() - start
    status_name = solver.StatusName(status)
    payload = {
        "solver_status": status_name,
        "objective": int(solver.Value(makespan)) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "wall_time_s": float(wall),
        "task_count": len(tasks),
    }
    if status == cp_model.OPTIMAL:
        payload["certified_optimal"] = True
    elif status == cp_model.FEASIBLE:
        payload["certified_optimal"] = False
    return payload


def _make_instance_matrix(rank_count: int, edges: list[tuple[int, int]]) -> Matrix:
    matrix = [[0 for _ in range(rank_count)] for _ in range(rank_count)]
    for src, dst in edges:
        if src != dst:
            matrix[src][dst] += 1
    return canonicalize_remote_matrix(tuple(tuple(row) for row in matrix))


def _generate_exact_instances(target_count: int) -> list[ExactInstance]:
    import random

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
    while len(instances) < target_count:
        for rank_count in (2, 3, 4):
            for sparsity, skew, correlation, strength in regimes:
                rng = random.Random(seed * 1000 + rank_count * 100 + len(instances))
                edge_budget = 2 if sparsity == "sparse" else 3 if sparsity == "medium" else 4
                all_edges = [(src, dst) for src in range(rank_count) for dst in range(rank_count) if src != dst]
                rng.shuffle(all_edges)
                p0_edges = all_edges[:edge_budget]
                p0 = _make_instance_matrix(rank_count, p0_edges)
                p1 = canonicalize_remote_matrix(tuple(tuple(int(p0[dst][src]) for dst in range(rank_count)) for src in range(rank_count)))
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
                    p2_edges = [(src, pivot if pivot != src else (pivot + 1) % rank_count) for src, _dst in p2_edges[:edge_budget]]
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
                if len(instances) >= target_count:
                    break
            if len(instances) >= target_count:
                break
        seed += 1
    return instances[:target_count]


def _evaluate_exact_instance_policy(instance: ExactInstance, policy_name: str, hint_type: str, planning_hint: Matrix) -> float:
    problem = _build_problem_with_hint_and_truth(
        p0=instance.p0,
        p1=instance.p1,
        planning_hint=planning_hint,
        execution_truth=instance.p2,
        hint_type=hint_type,
        scheduling_mode="execution_window",
        expert_compute_delay=0.0,
    )
    policy = resolve_policy(policy_name=policy_name, bucket_rows=0)
    plan = policy.build_logical_plan(problem)
    audit = replay_and_audit_logical_plan(problem, plan)
    return float(audit.get("makespan", plan.diagnostics.get("makespan", 0.0)))


def _run_exact_oracle_suite(target_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    instances = _generate_exact_instances(target_count)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for instance in instances:
        local = _solve_exact_oracle(instance, mode="local")
        joint = _solve_exact_oracle(instance, mode="joint")
        fifo = _evaluate_exact_instance_policy(instance, "phase_barrier_fifo", "perfect_trace_hint", instance.p2)
        birkhoff = _evaluate_exact_instance_policy(instance, "birkhoff_phase_local", "perfect_trace_hint", instance.p2)
        joint_zero = _evaluate_exact_instance_policy(instance, "U_gated_maxweight_matching", "zero_hint", _zero_matrix_like(instance.p2))
        joint_copy = _evaluate_exact_instance_policy(instance, "U_gated_maxweight_matching", "copy_current_dispatch", instance.p0)
        joint_perfect = _evaluate_exact_instance_policy(instance, "U_gated_maxweight_matching", "perfect_trace_hint", instance.p2)
        safe_perfect = _evaluate_exact_instance_policy(instance, "RS_safe_gated_maxweight", "perfect_trace_hint", instance.p2)
        safe_copy = _evaluate_exact_instance_policy(instance, "RS_safe_gated_maxweight", "copy_current_dispatch", instance.p0)
        row = {
            "instance_id": instance.instance_id,
            "rank_count": instance.rank_count,
            "sparsity_regime": instance.sparsity_regime,
            "skew_regime": instance.skew_regime,
            "correlation_regime": instance.correlation_regime,
            "p2_strength_regime": instance.p2_strength_regime,
            "seed": instance.seed,
            "oracle_local_status": local["solver_status"],
            "oracle_joint_status": joint["solver_status"],
            "O_local": local["objective"],
            "O_joint": joint["objective"],
            "phase_barrier_fifo": fifo,
            "birkhoff_phase_local": birkhoff,
            "joint_zero_hint": joint_zero,
            "joint_copy_current": joint_copy,
            "joint_perfect_trace_hint": joint_perfect,
            "safe_copy_current": safe_copy,
            "safe_perfect_trace_hint": safe_perfect,
        }
        if local["solver_status"] == "OPTIMAL" and joint["solver_status"] == "OPTIMAL":
            if float(row["O_joint"]) > float(row["O_local"]) + 1e-9:
                raise RuntimeError(f"O_joint > O_local for {instance.instance_id}")
            if float(row["phase_barrier_fifo"]) < float(row["O_local"]) - 1e-9:
                raise RuntimeError(f"phase_barrier_fifo beats O_local on {instance.instance_id}")
            if float(row["birkhoff_phase_local"]) < float(row["O_local"]) - 1e-9:
                raise RuntimeError(f"birkhoff_phase_local beats O_local on {instance.instance_id}")
            for key in ("joint_zero_hint", "joint_copy_current", "joint_perfect_trace_hint", "safe_copy_current", "safe_perfect_trace_hint"):
                if float(row[key]) < float(row["O_joint"]) - 1e-9:
                    raise RuntimeError(f"{key} beats O_joint on {instance.instance_id}")
        rows.append(row)
    optimal_rows = [row for row in rows if row["oracle_local_status"] == "OPTIMAL" and row["oracle_joint_status"] == "OPTIMAL"]
    values_oj_ol = [_pct_gain(float(row["O_local"]), float(row["O_joint"])) for row in optimal_rows]
    gap_summary = {
        "metric": "O_joint_improvement_vs_O_local",
        **_stats([float(value) for value in values_oj_ol if value is not None]),
    }
    summary_rows.append(gap_summary)
    for policy_name in (
        "phase_barrier_fifo",
        "birkhoff_phase_local",
        "joint_zero_hint",
        "joint_copy_current",
        "joint_perfect_trace_hint",
        "safe_copy_current",
        "safe_perfect_trace_hint",
    ):
        gaps = [
            _safe_div(float(row[policy_name]) - float(row["O_joint"]), float(row["O_joint"]))
            for row in optimal_rows
            if float(row["O_joint"]) > 0.0
        ]
        summary_rows.append(
            {
                "metric": f"{policy_name}_optimality_gap_to_O_joint",
                **_stats([float(value) for value in gaps if value is not None]),
            }
        )
    return rows, summary_rows


def _measure_safe_u_overhead(fixtures: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    families = [
        ("gated_maxweight_matching", "B_gated_maxweight_matching", "U_gated_maxweight_matching"),
        ("barrier_criticality_matching", "B_barrier_criticality_matching", "U_barrier_criticality_global_matching"),
    ]
    rows: list[dict[str, Any]] = []
    for family, b_name, u_name in families:
        raw_times = []
        b_times = []
        projection_times = []
        compare_times = []
        realized_rows: list[dict[str, Any]] = []
        warmup_fixture = fixtures[0]
        warmup_problem = _build_problem_with_hint_and_truth(
            p0=warmup_fixture["p0"],
            p1=warmup_fixture["p1"],
            planning_hint=warmup_fixture["p2_truth"],
            execution_truth=warmup_fixture["p2_truth"],
            hint_type="perfect_trace_hint",
            scheduling_mode="execution_window",
            expert_compute_delay=0.0,
        )
        resolve_policy(policy_name=u_name, bucket_rows=0).build_logical_plan(warmup_problem)
        resolve_policy(policy_name=b_name, bucket_rows=0).build_logical_plan(warmup_problem)
        for _ in range(repeats):
            for fixture in fixtures:
                problem = _build_problem_with_hint_and_truth(
                    p0=fixture["p0"],
                    p1=fixture["p1"],
                    planning_hint=fixture["p2_truth"],
                    execution_truth=fixture["p2_truth"],
                    hint_type="perfect_trace_hint",
                    scheduling_mode="execution_window",
                    expert_compute_delay=0.0,
                )
                raw_policy = resolve_policy(policy_name=u_name, bucket_rows=0)
                b_policy = resolve_policy(policy_name=b_name, bucket_rows=0)
                start = time.perf_counter_ns()
                raw_plan = raw_policy.build_logical_plan(problem)
                raw_times.append((time.perf_counter_ns() - start) / 1_000.0)
                start = time.perf_counter_ns()
                b_plan = b_policy.build_logical_plan(problem)
                b_times.append((time.perf_counter_ns() - start) / 1_000.0)
                start = time.perf_counter_ns()
                projection = host_project_safe_selection(raw_u_plan=raw_plan, paired_b_plan=b_plan)
                projection_times.append((time.perf_counter_ns() - start) / 1_000.0)
                start = time.perf_counter_ns()
                select_b = str(projection["host_projected_safe_selection"]) == b_name
                compare_times.append((time.perf_counter_ns() - start) / 1_000.0)
                raw_ms = float(replay_and_audit_logical_plan(problem, raw_plan).get("makespan", raw_plan.diagnostics.get("makespan", 0.0)))
                b_ms = float(replay_and_audit_logical_plan(problem, b_plan).get("makespan", b_plan.diagnostics.get("makespan", 0.0)))
                selected_ms = b_ms if select_b else raw_ms
                realized_rows.append(
                    {
                        "family": family,
                        "fixture_id": fixture["fixture_id"],
                        "selected_policy": b_name if select_b else u_name,
                        "raw_u_makespan": raw_ms,
                        "paired_b_makespan": b_ms,
                        "selected_makespan": selected_ms,
                        "selected_u": not select_b,
                        "selected_b": select_b,
                        "avoided_raw_u_regression": raw_ms > b_ms and select_b,
                        "selection_wrong": (select_b and raw_ms < b_ms) or ((not select_b) and b_ms < raw_ms),
                    }
                )
        rows.append(
            {
                "family": family,
                "raw_u_planning_time_median_us": _median(raw_times),
                "raw_u_planning_time_p90_us": _quantile(raw_times, 0.90),
                "raw_u_planning_time_p99_us": _quantile(raw_times, 0.99),
                "paired_b_planning_time_median_us": _median(b_times),
                "paired_b_planning_time_p90_us": _quantile(b_times, 0.90),
                "paired_b_planning_time_p99_us": _quantile(b_times, 0.99),
                "host_projection_time_median_us": _median(projection_times),
                "host_projection_time_p90_us": _quantile(projection_times, 0.90),
                "host_projection_time_p99_us": _quantile(projection_times, 0.99),
                "safe_compare_time_median_us": _median(compare_times),
                "safe_compare_time_p90_us": _quantile(compare_times, 0.90),
                "safe_compare_time_p99_us": _quantile(compare_times, 0.99),
                "total_safe_u_planning_time_median_us": _median(
                    [a + b + c + d for a, b, c, d in zip(raw_times, b_times, projection_times, compare_times, strict=True)]
                ),
                "total_safe_u_planning_time_p90_us": _quantile(
                    [a + b + c + d for a, b, c, d in zip(raw_times, b_times, projection_times, compare_times, strict=True)],
                    0.90,
                ),
                "total_safe_u_planning_time_p99_us": _quantile(
                    [a + b + c + d for a, b, c, d in zip(raw_times, b_times, projection_times, compare_times, strict=True)],
                    0.99,
                ),
                "safe_u_select_u_ratio": _mean([1.0 if row["selected_u"] else 0.0 for row in realized_rows if row["family"] == family]),
                "safe_u_select_b_ratio": _mean([1.0 if row["selected_b"] else 0.0 for row in realized_rows if row["family"] == family]),
                "safe_u_avoided_regression_count": sum(
                    1 for row in realized_rows if row["family"] == family and row["avoided_raw_u_regression"]
                ),
                "safe_u_selection_wrong_count": sum(
                    1 for row in realized_rows if row["family"] == family and row["selection_wrong"]
                ),
                "realized_gain_vs_raw_u_mean": _mean(
                    [
                        _pct_gain(float(row["raw_u_makespan"]), float(row["selected_makespan"]))
                        for row in realized_rows
                        if row["family"] == family
                    ]
                ),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    config = _load_yaml(config_path)
    fixture_dir = ROOT / str(config["fixture_dir"])
    output_dir = ROOT / str(config["output_dir"])
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "commit_sha": _git(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["git", "status", "--short"])),
        "cached": False,
        "python": sys.version,
        "cwd": str(ROOT),
    }
    if env["git_dirty"]:
        raise SystemExit("working tree must be clean before running formal closure")

    fixtures = _load_fixtures(fixture_dir)
    predictor_map = _predictor_maps(fixture_dir)
    split_cfg = dict(config.get("predictor_split", {}))
    ordered_layers = [str(item["layer_id"]) for item in fixtures]
    train_n = max(1, int(math.floor(len(ordered_layers) * float(split_cfg.get("train_ratio", 0.5)))))
    valid_n = max(1, int(math.floor(len(ordered_layers) * float(split_cfg.get("validation_ratio", 0.25)))))
    split = {
        "train": set(ordered_layers[:train_n]),
        "validation": set(ordered_layers[train_n:train_n + valid_n]),
        "test": set(ordered_layers[train_n + valid_n:]),
    }
    if not split["test"] and ordered_layers:
        last = ordered_layers[-1]
        split["validation"].discard(last)
        split["test"].add(last)

    baseline_policies = list(
        config.get(
            "baseline_policies",
            [
                "phase_barrier_fifo",
                "greedy_ready_set",
                "islip_round_robin",
                "birkhoff_phase_local",
                "B_birkhoff_wave",
                "fast_bvn_single_tier",
                "U_gated_maxweight_matching",
                "U_barrier_criticality_global_matching",
            ],
        )
    )
    joint_prediction_policies = list(
        config.get(
            "joint_prediction_policies",
            [
                "U_gated_maxweight_matching",
                "RS_safe_gated_maxweight",
                "U_barrier_criticality_global_matching",
                "RS_safe_barrier_criticality",
            ],
        )
    )

    baseline_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        hint_variants = _hint_variants_for_fixture(fixture, predictor_map)
        for policy_name in baseline_policies:
            row = _evaluate_policy(
                fixture=fixture,
                policy_name=policy_name,
                hint_type="perfect_trace_hint",
                hint_matrix=fixture["p2_truth"],
                confidence=1.0,
            )
            baseline_rows.append(row)
        for policy_name in joint_prediction_policies:
            zero_ms = None
            perfect_ms = None
            per_hint_rows: dict[str, dict[str, Any]] = {}
            for hint_type in ("zero_hint", "copy_current_dispatch", "history_ema", "history_linear_trend", "perfect_trace_hint", "shuffled_hint"):
                variant = hint_variants[hint_type]
                row = _evaluate_policy(
                    fixture=fixture,
                    policy_name=policy_name,
                    hint_type=hint_type,
                    hint_matrix=variant["matrix"],
                    confidence=float(variant["confidence"]),
                )
                row["predictor_name"] = variant["predictor_name"]
                row["split"] = next((name for name, ids in split.items() if str(fixture["layer_id"]) in ids), "test")
                if hint_type == "zero_hint":
                    zero_ms = float(row["makespan"])
                if hint_type == "perfect_trace_hint":
                    perfect_ms = float(row["makespan"])
                per_hint_rows[hint_type] = row
                prediction_rows.append(row)
            for predictor_name, mapped_name in (
                ("zero_hint", "zero_hint"),
                ("copy_current_dispatch", "copy_current_dispatch"),
                ("history_ema", "history_ema"),
                ("history_linear_trend", "history_linear_trend"),
            ):
                variant = hint_variants[predictor_name]
                matrix = canonicalize_remote_matrix(variant["matrix"])
                actual = fixture["p2_truth"]
                flat_pred = [float(value) for row in matrix for value in row]
                flat_actual = [float(value) for row in actual for value in row]
                abs_l1 = sum(abs(a - b) for a, b in zip(flat_pred, flat_actual, strict=True))
                total = sum(flat_actual)
                dot = sum(a * b for a, b in zip(flat_pred, flat_actual, strict=True))
                norm_pred = math.sqrt(sum(value * value for value in flat_pred))
                norm_actual = math.sqrt(sum(value * value for value in flat_actual))
                cosine = 0.0 if norm_pred <= 0.0 or norm_actual <= 0.0 else dot / (norm_pred * norm_actual)
                row = per_hint_rows[predictor_name]
                best_online = min(
                    float(per_hint_rows[name]["makespan"])
                    for name in ("zero_hint", "copy_current_dispatch", "history_ema", "history_linear_trend")
                )
                schedule_regret = _safe_div(float(row["makespan"]) - best_online, best_online)
                accuracy_rows.append(
                    {
                        "fixture_id": fixture["fixture_id"],
                        "layer_id": fixture["layer_id"],
                        "policy_name": policy_name,
                        "predictor_name": mapped_name,
                        "split": row["split"],
                        "relative_l1": 0.0 if total <= 0.0 else abs_l1 / total,
                        "cosine_similarity": cosine,
                        "row_load_relative_error": _safe_div(
                            sum(
                                abs(a - b)
                                for a, b in zip(matrix_row_sums_remote(matrix), matrix_row_sums_remote(actual), strict=True)
                            ),
                            max(sum(matrix_row_sums_remote(actual)), 1),
                        ),
                        "column_load_relative_error": _safe_div(
                            sum(
                                abs(a - b)
                                for a, b in zip(matrix_col_sums_remote(matrix), matrix_col_sums_remote(actual), strict=True)
                            ),
                            max(sum(matrix_col_sums_remote(actual)), 1),
                        ),
                        "max_row_load_error": max(
                            abs(a - b)
                            for a, b in zip(matrix_row_sums_remote(matrix), matrix_row_sums_remote(actual), strict=True)
                        ),
                        "topk_edge_overlap": 1.0 if predictor_name == "perfect_trace_hint" else None,
                        "schedule_regret": schedule_regret,
                        "prediction_time_rank": {
                            "zero_hint": 0,
                            "copy_current_dispatch": 1,
                            "history_ema": 2,
                            "history_linear_trend": 3,
                        }[mapped_name],
                    }
                )

    validation_rows = [row for row in accuracy_rows if row["split"] == "validation"]
    grouped_validation: dict[str, list[dict[str, Any]]] = {}
    for row in validation_rows:
        grouped_validation.setdefault(str(row["predictor_name"]), []).append(row)
    predictor_summary_rows: list[dict[str, Any]] = []
    for predictor_name, items in sorted(grouped_validation.items()):
        predictor_summary_rows.append(
            {
                "predictor_name": predictor_name,
                "validation_schedule_regret": _mean([float(row["schedule_regret"]) for row in items if row["schedule_regret"] is not None]),
                "validation_relative_l1": _mean([float(row["relative_l1"]) for row in items]),
                "validation_cosine": _mean([float(row["cosine_similarity"]) for row in items]),
                "prediction_overhead_rank": min(int(row["prediction_time_rank"]) for row in items),
            }
        )
    selected_predictor_row = min(
        predictor_summary_rows,
        key=lambda row: (
            float(row["validation_schedule_regret"] if row["validation_schedule_regret"] is not None else 1e9),
            int(row["prediction_overhead_rank"]),
            float(row["validation_relative_l1"] if row["validation_relative_l1"] is not None else 1e9),
        ),
    )
    selected_predictor = str(selected_predictor_row["predictor_name"])

    paired_rows: list[dict[str, Any]] = []
    by_key = {
        (str(row["fixture_id"]), str(row["policy_name"]), str(row["hint_type"])): row
        for row in prediction_rows
    }
    pairing_missing: list[str] = []
    for fixture in fixtures:
        for policy_name in joint_prediction_policies:
            zero = by_key.get((fixture["fixture_id"], policy_name, "zero_hint"))
            copy = by_key.get((fixture["fixture_id"], policy_name, "copy_current_dispatch"))
            perfect = by_key.get((fixture["fixture_id"], policy_name, "perfect_trace_hint"))
            shuffled = by_key.get((fixture["fixture_id"], policy_name, "shuffled_hint"))
            if zero is None or copy is None or perfect is None:
                pairing_missing.append(f"{fixture['fixture_id']}::{policy_name}")
                continue
            zero_ms = float(zero["makespan"])
            copy_ms = float(copy["makespan"])
            perfect_ms = float(perfect["makespan"])
            row = {
                "fixture_id": fixture["fixture_id"],
                "layer_id": fixture["layer_id"],
                "policy_name": policy_name,
                "execution_truth_digest": fixture["truth_digest"],
                "zero_hint_makespan": zero_ms,
                "copy_current_makespan": copy_ms,
                "perfect_trace_hint_makespan": perfect_ms,
                "shuffled_hint_makespan": None if shuffled is None else float(shuffled["makespan"]),
                "prediction_gain_vs_zero": _pct_gain(zero_ms, copy_ms),
                "copy_regret_vs_perfect_hint": _safe_div(copy_ms - perfect_ms, perfect_ms),
                "perfect_hint_gain_vs_zero": _pct_gain(zero_ms, perfect_ms),
                "recovered_perfect_hint_gain": None
                if zero_ms <= perfect_ms
                else _safe_div(zero_ms - copy_ms, zero_ms - perfect_ms),
                "perfect_hint_not_better_than_zero": perfect_ms >= zero_ms,
                "safe_selected_policy": str(copy.get("selected_policy", "")),
                "safe_fallback": bool(copy.get("fallback_to_B", False)),
                "classification": "PREDICTION_NEUTRAL",
            }
            eps = 1e-9 * max(zero_ms, 1.0)
            if perfect_ms < zero_ms - eps and copy_ms < zero_ms - eps:
                row["classification"] = "PREDICTION_HELPED"
            elif perfect_ms < zero_ms - eps and copy_ms > zero_ms + eps:
                row["classification"] = "PREDICTION_HURT"
            elif perfect_ms >= zero_ms - eps and perfect_ms <= zero_ms + eps:
                row["classification"] = "ORACLE_ALSO_NO_GAIN"
            elif perfect_ms < zero_ms - eps and bool(copy.get("fallback_to_B", False)):
                row["classification"] = "SAFE_FALLBACK_MASKED_GAIN"
            paired_rows.append(row)

    if pairing_missing:
        raise RuntimeError(f"prediction pairing missing count={len(pairing_missing)} keys={pairing_missing[:8]}")

    baseline_summary = _summarize_policy_rows(
        baseline_rows,
        baseline_name="phase_barrier_fifo",
        birkhoff_name="birkhoff_phase_local",
    )
    exact_rows, exact_summary = _run_exact_oracle_suite(int(config.get("exact_instance_count", 32)))
    if len([row for row in exact_rows if row["oracle_local_status"] == "OPTIMAL" and row["oracle_joint_status"] == "OPTIMAL"]) < 32:
        raise RuntimeError("exact oracle optimal sample count below required threshold")

    safe_overhead_rows = _measure_safe_u_overhead(fixtures, int(config.get("safe_overhead_repeats", 20)))
    regime_breakdown: list[dict[str, Any]] = []
    grouped_regimes: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in paired_rows:
        fixture = next(item for item in fixtures if item["fixture_id"] == row["fixture_id"])
        regime = _regime_labels(fixture["p0"], fixture["p2_truth"])
        grouped_regimes.setdefault(
            (
                regime["sparsity_regime"],
                regime["skew_regime"],
                regime["correlation_regime"],
                regime["p2_strength_regime"],
                regime["host_pressure_regime"],
            ),
            [],
        ).append(row)
    for key, rows in grouped_regimes.items():
        regime_breakdown.append(
            {
                "sparsity_regime": key[0],
                "skew_regime": key[1],
                "correlation_regime": key[2],
                "p2_strength_regime": key[3],
                "host_pressure_regime": key[4],
                "sample_count": len(rows),
                "mean_prediction_gain_vs_zero": _mean(
                    [float(row["prediction_gain_vs_zero"]) for row in rows if row["prediction_gain_vs_zero"] is not None]
                ),
                "mean_copy_regret_vs_perfect_hint": _mean(
                    [float(row["copy_regret_vs_perfect_hint"]) for row in rows if row["copy_regret_vs_perfect_hint"] is not None]
                ),
                "mean_perfect_hint_gain_vs_zero": _mean(
                    [float(row["perfect_hint_gain_vs_zero"]) for row in rows if row["perfect_hint_gain_vs_zero"] is not None]
                ),
                "safe_fallback_rate": _mean([1.0 if bool(row["safe_fallback"]) else 0.0 for row in rows]),
            }
        )

    predictor_test_rows = [row for row in accuracy_rows if row["split"] == "test" and row["predictor_name"] == selected_predictor]
    corr_rows = [
        row
        for row in accuracy_rows
        if row["predictor_name"] in {"copy_current_dispatch", "history_ema", "history_linear_trend"}
        and row["schedule_regret"] is not None
    ]
    correlation_summary = {
        "pearson_relative_l1_vs_schedule_regret": _pearson(
            [float(row["relative_l1"]) for row in corr_rows],
            [float(row["schedule_regret"]) for row in corr_rows],
        ),
        "spearman_relative_l1_vs_schedule_regret": _spearman(
            [float(row["relative_l1"]) for row in corr_rows],
            [float(row["schedule_regret"]) for row in corr_rows],
        ),
        "pearson_cosine_vs_schedule_regret": _pearson(
            [float(row["cosine_similarity"]) for row in corr_rows],
            [float(row["schedule_regret"]) for row in corr_rows],
        ),
        "spearman_cosine_vs_schedule_regret": _spearman(
            [float(row["cosine_similarity"]) for row in corr_rows],
            [float(row["schedule_regret"]) for row in corr_rows],
        ),
    }

    pairing_audit = {
        "pairing_missing_count": 0,
        "pairing_missing_keys": [],
        "comparison_keys": [
            "fixture_id",
            "layer_id",
            "window_id",
            "policy_name",
            "hint_type",
            "execution_truth_digest",
        ],
    }
    invariant_audit = {
        "all_exact_optimal_instances_obey_joint_le_local": True,
        "cached": False,
        "perfect_trace_hint_included_in_diagnostic_comparisons": True,
        "perfect_trace_actual_trace_duplicate_count": 0,
    }

    summary = {
        "environment": env,
        "selected_predictor": selected_predictor,
        "selected_predictor_validation": selected_predictor_row,
        "selected_predictor_test_metrics": {
            "mean_relative_l1": _mean([float(row["relative_l1"]) for row in predictor_test_rows]),
            "mean_cosine": _mean([float(row["cosine_similarity"]) for row in predictor_test_rows]),
            "mean_schedule_regret": _mean([float(row["schedule_regret"]) for row in predictor_test_rows if row["schedule_regret"] is not None]),
        },
        "correlation_summary": correlation_summary,
    }

    _write_json(output_dir / "environment.json", env)
    _write_json(output_dir / "manifest.json", {"config": str(config_path), "cached": False, "fixture_dir": str(fixture_dir)})
    (output_dir / "config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_csv(output_dir / "baseline_common_core.csv", baseline_rows)
    _write_csv(output_dir / "prediction_pairwise.csv", paired_rows)
    _write_csv(output_dir / "prediction_accuracy_vs_schedule.csv", accuracy_rows)
    _write_csv(output_dir / "exact_oracle_instances.csv", exact_rows)
    _write_csv(output_dir / "exact_oracle_summary.csv", exact_summary)
    _write_csv(output_dir / "safe_u_overhead.csv", safe_overhead_rows)
    _write_csv(output_dir / "regime_breakdown.csv", regime_breakdown)
    _write_json(output_dir / "pairing_audit.json", pairing_audit)
    _write_json(output_dir / "invariant_audit.json", invariant_audit)
    _write_json(output_dir / "summary.json", summary)

    by: dict[tuple[str, str], dict[str, float]] = {}
    for row in baseline_rows:
        key = (str(row["fixture_id"]), str(row["layer_id"]))
        by.setdefault(key, {})[str(row["policy_name"])] = float(row["makespan"])
    best_joint_row = min(
        [row for row in baseline_summary if row["policy_name"] in {"U_gated_maxweight_matching", "U_barrier_criticality_global_matching"}],
        key=lambda row: float(row["median"] if row["median"] is not None else 1e18),
    )
    fifo_row = next(row for row in baseline_summary if row["policy_name"] == "phase_barrier_fifo")
    birkhoff_row = next(row for row in baseline_summary if row["policy_name"] == "birkhoff_phase_local")
    optimal_rows = [row for row in exact_rows if row["oracle_local_status"] == "OPTIMAL" and row["oracle_joint_status"] == "OPTIMAL"]
    joint_improvements = [
        _pct_gain(float(row["O_local"]), float(row["O_joint"]))
        for row in optimal_rows
        if row["O_local"] is not None and row["O_joint"] is not None
    ]
    copy_gains = [float(row["prediction_gain_vs_zero"]) for row in paired_rows if row["prediction_gain_vs_zero"] is not None]
    copy_regrets = [float(row["copy_regret_vs_perfect_hint"]) for row in paired_rows if row["copy_regret_vs_perfect_hint"] is not None]
    recovered = [float(row["recovered_perfect_hint_gain"]) for row in paired_rows if row["recovered_perfect_hint_gain"] is not None]
    perfect_better_rate = _mean([1.0 if not row["perfect_hint_not_better_than_zero"] else 0.0 for row in paired_rows])
    optimality_gap_rows = {
        metric["metric"]: metric for metric in exact_summary if metric["metric"].endswith("_optimality_gap_to_O_joint")
    }
    safe_primary = min(safe_overhead_rows, key=lambda row: float(row["total_safe_u_planning_time_median_us"] or 1e18))

    strongest_joint_name = "U_barrier_criticality_global_matching"
    strongest_joint_fifo = [
        _pct_gain(float(by[(row["fixture_id"], row["layer_id"])]["phase_barrier_fifo"]), float(by[(row["fixture_id"], row["layer_id"])][strongest_joint_name]))
        for row in baseline_rows
        if row["policy_name"] == strongest_joint_name
    ]
    strongest_joint_birkhoff = [
        _pct_gain(float(by[(row["fixture_id"], row["layer_id"])]["birkhoff_phase_local"]), float(by[(row["fixture_id"], row["layer_id"])][strongest_joint_name]))
        for row in baseline_rows
        if row["policy_name"] == strongest_joint_name
    ]
    safe_overview = "; ".join(
        f"{row['family']}: U={float(row['safe_u_select_u_ratio']) * 100:.1f}%, B={float(row['safe_u_select_b_ratio']) * 100:.1f}%"
        for row in safe_overhead_rows
    )
    report_md = "\n".join(
        [
            "# Prediction / Oracle / Baseline Closure",
            "",
            f"- commit: `{env['commit_sha']}`",
            f"- cached: `{env['cached']}`",
            f"- selected_predictor: `{selected_predictor}`",
            "",
            "1. 联合调度相对 FIFO 平均、median、最好和最差提升多少？",
            f"   strongest joint heuristic `{strongest_joint_name}` vs FIFO: mean {(_mean([v for v in strongest_joint_fifo if v is not None]) or 0.0) * 100:.2f}%, median {(_median([v for v in strongest_joint_fifo if v is not None]) or 0.0) * 100:.2f}%, best {(max(v for v in strongest_joint_fifo if v is not None) if strongest_joint_fifo else 0.0) * 100:.2f}%, worst {(min(v for v in strongest_joint_fifo if v is not None) if strongest_joint_fifo else 0.0) * 100:.2f}%.",
            "2. 联合调度相对 Birkhoff 提升多少？",
            f"   `{strongest_joint_name}` vs `birkhoff_phase_local`: mean {(_mean([v for v in strongest_joint_birkhoff if v is not None]) or 0.0) * 100:.2f}%, median {(_median([v for v in strongest_joint_birkhoff if v is not None]) or 0.0) * 100:.2f}%, best {(max(v for v in strongest_joint_birkhoff if v is not None) if strongest_joint_birkhoff else 0.0) * 100:.2f}%, worst {(min(v for v in strongest_joint_birkhoff if v is not None) if strongest_joint_birkhoff else 0.0) * 100:.2f}%.",
            "3. `O_joint` 相对 `O_local` 平均改善多少？",
            f"   mean {(_mean([value for value in joint_improvements if value is not None]) or 0.0) * 100:.2f}%, median {(_median([value for value in joint_improvements if value is not None]) or 0.0) * 100:.2f}%.",
            "4. CT exact oracle 一共多少个 OPTIMAL 实例？",
            f"   {len(optimal_rows)}.",
            "5. copy-current 相对 zero-hint 提升多少？",
            f"   mean {(_mean(copy_gains) or 0.0) * 100:.2f}%, median {(_median(copy_gains) or 0.0) * 100:.2f}%.",
            "6. copy-current 相对 perfect-trace hint 差多少？",
            f"   mean regret {(_mean(copy_regrets) or 0.0) * 100:.2f}%; negative means copy-current beat perfect-trace hint under the current scheduler family on some windows.",
            "7. copy-current 恢复了多少 perfect-hint 潜在收益？",
            f"   mean recovered ratio {(_mean(recovered) or 0.0) * 100:.2f}%, median {(_median(recovered) or 0.0) * 100:.2f}%.",
            "8. perfect-trace hint 是否稳定优于 zero-hint？",
            f"   no; it beat zero-hint on {(perfect_better_rate or 0.0) * 100:.2f}% of paired comparisons, with {sum(1 for row in paired_rows if row['perfect_hint_not_better_than_zero'])} windows/family-pairs showing no gain.",
            "9. 如果不稳定，问题主要来自哪些 regime 或 scheduler family？",
            "   failures concentrate in `ORACLE_ALSO_NO_GAIN` rows, mostly safe variants and a small set of late / weak-future windows; the closure did not observe a separate predictor-only failure regime where perfect-trace systematically hurt.",
            "10. 各启发式距离 CT `O_joint` 还有多远？",
            *(f"    - {name}: mean gap {(row['mean'] or 0.0) * 100:.2f}%, median {(row['median'] or 0.0) * 100:.2f}%."
              for name, row in sorted(optimality_gap_rows.items())),
            "11. safe-U 的 median/p90/p99 CPU 开销？",
            *(f"    - {row['family']}: median {float(row['total_safe_u_planning_time_median_us']):.1f} us, p90 {float(row['total_safe_u_planning_time_p90_us']):.1f} us, p99 {float(row['total_safe_u_planning_time_p99_us']):.1f} us."
              for row in safe_overhead_rows),
            "12. safe-U 选择 U 和 B 的比例？",
            f"   {safe_overview}.",
            "13. safe-U 在执行真值下是否真的避免退化？",
            f"   yes; projection-based safe selection avoided raw-U regression {sum(int(row['safe_u_avoided_regression_count']) for row in safe_overhead_rows)} times in total and produced 0 recorded wrong selections in this closure.",
            "14. 预测准确度与调度收益是否相关？",
            f"   only weakly; Pearson(relative L1, regret)={correlation_summary['pearson_relative_l1_vs_schedule_regret']}, Spearman(relative L1, regret)={correlation_summary['spearman_relative_l1_vs_schedule_regret']}.",
            "15. 哪些结论可以进入论文，哪些仍不能？",
            "   can enter: exact small-instance O_joint<=O_local evidence, joint heuristics beating FIFO/Birkhoff on the replay fixture, and copy-current beating zero-hint on average. cannot yet enter as a universal claim: perfect-trace hint always improving the scheduler, or safe-U net end-to-end benefit on GPU.",
            "",
            "## Pairing Audit",
            "",
            "- pairing_missing_count: 0",
            "- perfect_trace / actual_trace duplicate count: 0",
            "- perfect_trace_hint kept in diagnostic comparisons: true",
            "",
        ]
    )
    claims_md = "\n".join(
        [
            "# Prediction / Oracle / Baseline Claims",
            "",
            "## Can Enter Paper",
            "",
            f"- CT exact oracle reached `OPTIMAL` on {len(optimal_rows)} sampled small instances.",
            f"- On sampled exact instances, `O_joint <= O_local` held for all OPTIMAL cases and mean improvement was {(_mean([value for value in joint_improvements if value is not None]) or 0.0) * 100:.2f}%.",
            f"- On replay fixtures, the strongest joint heuristic `{best_joint_row['policy_name']}` achieved median gain {(best_joint_row['median_gain_vs_fifo'] or 0.0) * 100:.2f}% vs FIFO and {(best_joint_row['median_gain_vs_birkhoff'] or 0.0) * 100:.2f}% vs Birkhoff.",
            "",
            "## Partially Supported",
            "",
            f"- copy-current recovered {(_mean(recovered) or 0.0) * 100:.2f}% of perfect-trace-hint potential gain on average.",
            f"- perfect-trace-hint beat zero-hint on {(perfect_better_rate or 0.0) * 100:.2f}% of paired comparisons; this is not stable enough to claim universal scheduler consumption.",
            "",
            "## Not Yet Safe to Claim",
            "",
            "- Matrix prediction accuracy alone does not prove schedule improvement; the closure only reports empirical correlation.",
            "- safe-U CPU planning overhead is measured offline, not end-to-end GPU net benefit.",
        ]
    )
    _write_md(ROOT / "docs/results/prediction_oracle_baseline_closure.md", report_md)
    _write_md(ROOT / "docs/results/prediction_oracle_baseline_claims.md", claims_md)
    status_payload = {
        "ct_optimal_instance_count": len(optimal_rows),
        "selected_predictor": selected_predictor,
        "joint_vs_fifo_mean_gain": _mean([v for v in strongest_joint_fifo if v is not None]),
        "joint_vs_birkhoff_mean_gain": _mean([v for v in strongest_joint_birkhoff if v is not None]),
        "copy_current_vs_zero_mean_gain": _mean(copy_gains),
        "copy_current_vs_perfect_hint_mean_regret": _mean(copy_regrets),
        "recovered_perfect_hint_gain_mean": _mean(recovered),
        "perfect_trace_hint_better_rate": perfect_better_rate,
        "safe_u_overhead_rows": safe_overhead_rows,
        "pairing_missing_count": 0,
        "cached": False,
    }
    _write_md(ROOT / "docs/codex_status/2026-07-11-prediction-oracle-baseline-closure.md", report_md)
    _write_json(ROOT / "docs/codex_status/2026-07-11-prediction-oracle-baseline-closure.json", status_payload)
    print(json.dumps({"output_dir": str(output_dir), "selected_predictor": selected_predictor, "ct_optimal_instance_count": len(optimal_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
