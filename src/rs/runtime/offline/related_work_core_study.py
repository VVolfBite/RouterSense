"""Reusable logical-time study for related-work scheduling cores.

The study deliberately compares only the scheduling cores under RouterSense's
shared fixed-endpoint, phase-serial logical execution contract.  It does not
claim to reproduce each paper's full deployment, topology, or compute model.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rs.runtime.offline.policy_study import build_replay_problem, expected_replay_flows
from rs.runtime.offline.runner import replay_and_audit_logical_plan, summarize_schedule_tail_metrics
from rs.scheduling import resolve_policy
from rs.scheduling.validation import validate_logical_plan

DEFAULT_POLICIES: tuple[str, ...] = (
    "fifo_bucket",
    "greedy_bucket",
    "birkhoff_bucket_phase_local",
    "gmwd_style_reference",
    "islip_reference",
    "fast_stage_reference",
    "aurora_order_reference",
)


def discover_trace_roots(trace_root: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for root in sorted(Path(trace_root).glob("RouterSense_fate_trace_*_single_gpu_20260718")):
        traffic = root / "traffic" / "traffic_instances.json"
        if not traffic.is_file():
            continue
        model = root.name.removeprefix("RouterSense_fate_trace_").removesuffix("_single_gpu_20260718")
        roots[model] = root
    if not roots:
        raise ValueError(f"no trace roots found under {trace_root}")
    return roots


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(quantile)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    fraction = position - left
    return ordered[left] * (1.0 - fraction) + ordered[right] * fraction


def _fixture(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "num_gpus": int(item["virtual_ep_size"]),
        "p0_dispatch_matrix": item["P0_matrix"],
        "p1_return_matrix": item["P1_matrix"],
        "p2_next_dispatch_matrix": item["P2_truth_matrix"],
    }


def _comparison(
    *,
    policy_name: str,
    baseline_name: str,
    by_instance: Mapping[tuple[str, str], Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    values: list[float] = []
    for policies in by_instance.values():
        policy = policies.get(policy_name)
        baseline = policies.get(baseline_name)
        if policy is None or baseline is None or float(baseline["makespan"]) <= 0.0:
            continue
        values.append(
            100.0
            * (float(baseline["makespan"]) - float(policy["makespan"]))
            / float(baseline["makespan"])
        )
    return {
        "n": len(values),
        "mean_improvement_pct": statistics.mean(values) if values else None,
        "median_improvement_pct": statistics.median(values) if values else None,
        "p05_improvement_pct": _percentile(values, 0.05),
        "p95_improvement_pct": _percentile(values, 0.95),
        "win": sum(value > 1e-9 for value in values),
        "tie": sum(abs(value) <= 1e-9 for value in values),
        "loss": sum(value < -1e-9 for value in values),
    }


def run_related_work_core_study(
    *,
    trace_roots: Mapping[str, Path],
    policy_names: Sequence[str] = DEFAULT_POLICIES,
    max_waves: int = 4096,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for model, root in sorted(trace_roots.items()):
        instances = json.loads((Path(root) / "traffic" / "traffic_instances.json").read_text(encoding="utf-8"))
        for item in instances:
            problem = build_replay_problem(
                fixture=_fixture(item),
                mode="execution_window",
                p2_source="actual_trace",
                expert_compute_delay=0.0,
                max_waves=int(max_waves),
            )
            expected = expected_replay_flows(problem)
            for policy_name in policy_names:
                policy = resolve_policy(policy_name=str(policy_name), bucket_rows=0)
                plan_started = time.perf_counter_ns()
                try:
                    plan = policy.build_logical_plan(problem)
                    planning_us = (time.perf_counter_ns() - plan_started) / 1000.0
                    validation = validate_logical_plan(
                        plan,
                        expected_flows=expected,
                        mode="execution_window",
                        expert_compute_delay=0.0,
                    )
                    audit = replay_and_audit_logical_plan(problem, plan)
                    tail = summarize_schedule_tail_metrics(problem=problem, plan=plan, audit=audit)
                    valid = bool(validation["valid"]) and bool(audit.get("valid", False))
                    error = "" if valid else json.dumps(
                        {"validation": validation.get("errors", []), "audit": audit.get("errors", [])},
                        ensure_ascii=False,
                    )
                    makespan = float(plan.diagnostics.get("makespan", audit.get("makespan", 0.0)))
                    wave_count = int(len(plan.waves))
                except Exception as exc:  # fail closed while preserving the complete matrix
                    planning_us = (time.perf_counter_ns() - plan_started) / 1000.0
                    valid = False
                    error = repr(exc)
                    makespan = math.nan
                    wave_count = -1
                    tail = {}
                rows.append(
                    {
                        "model": str(model),
                        "instance_id": str(item["instance_id"]),
                        "trace_sample_id": str(item.get("trace_sample_id", "")),
                        "virtual_ep_size": int(item["virtual_ep_size"]),
                        "policy_name": str(policy_name),
                        "policy_version": str(getattr(policy, "policy_version", "v1")),
                        "valid": bool(valid),
                        "makespan": makespan,
                        "wave_count": wave_count,
                        "planning_us": float(planning_us),
                        "p95_completion": float(tail.get("p95_flow_completion", 0.0) or 0.0),
                        "p99_completion": float(tail.get("p99_flow_completion", 0.0) or 0.0),
                        "first_p1_token": float(tail.get("first_p1_token_time", 0.0) or 0.0),
                        "error": error,
                    }
                )

    by_instance: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if bool(row["valid"]):
            by_instance[(str(row["model"]), str(row["instance_id"]))][str(row["policy_name"])] = row

    summary: dict[str, Any] = {}
    for policy_name in policy_names:
        policy_rows = [row for row in rows if row["policy_name"] == policy_name]
        valid_rows = [row for row in policy_rows if bool(row["valid"])]
        makespans = [float(row["makespan"]) for row in valid_rows]
        wave_counts = [float(row["wave_count"]) for row in valid_rows]
        planning = [float(row["planning_us"]) for row in valid_rows]
        summary[str(policy_name)] = {
            "policy_version": str(policy_rows[0]["policy_version"]) if policy_rows else "",
            "instance_count": len(policy_rows),
            "valid_count": len(valid_rows),
            "invalid_count": len(policy_rows) - len(valid_rows),
            "makespan_mean": statistics.mean(makespans) if makespans else None,
            "makespan_median": statistics.median(makespans) if makespans else None,
            "makespan_p95": _percentile(makespans, 0.95),
            "wave_count_mean": statistics.mean(wave_counts) if wave_counts else None,
            "wave_count_median": statistics.median(wave_counts) if wave_counts else None,
            "planning_us_median": statistics.median(planning) if planning else None,
            "planning_us_p95": _percentile(planning, 0.95),
            "vs_birkhoff": _comparison(
                policy_name=str(policy_name),
                baseline_name="birkhoff_bucket_phase_local",
                by_instance=by_instance,
            ),
            "vs_greedy": _comparison(
                policy_name=str(policy_name),
                baseline_name="greedy_bucket",
                by_instance=by_instance,
            ),
        }

    return {
        "schema_version": "related_work_core4_quickstudy.v1",
        "scope": {
            "models": sorted(trace_roots),
            "traffic_instances": len(by_instance),
            "mode": "execution_window",
            "p2_source": "actual_trace",
            "expert_compute_delay": 0.0,
            "logical_time_only": True,
            "mapping_level": "style",
        },
        "policies": [str(name) for name in policy_names],
        "summary": summary,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def write_study_artifacts(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "related_work_core4_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact = {key: value for key, value in payload.items() if key != "rows"}
    (output_dir / "related_work_core4_summary.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = list(payload.get("rows", []))
    if rows:
        with (output_dir / "related_work_core4_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary_rows: list[dict[str, Any]] = []
    for policy_name, row in payload.get("summary", {}).items():
        birkhoff = row["vs_birkhoff"]
        greedy = row["vs_greedy"]
        summary_rows.append(
            {
                "policy_name": policy_name,
                "policy_version": row["policy_version"],
                "valid_count": row["valid_count"],
                "invalid_count": row["invalid_count"],
                "makespan_mean": row["makespan_mean"],
                "makespan_median": row["makespan_median"],
                "makespan_p95": row["makespan_p95"],
                "wave_count_mean": row["wave_count_mean"],
                "wave_count_median": row["wave_count_median"],
                "planning_us_median": row["planning_us_median"],
                "planning_us_p95": row["planning_us_p95"],
                "vs_birkhoff_mean_pct": birkhoff["mean_improvement_pct"],
                "vs_birkhoff_median_pct": birkhoff["median_improvement_pct"],
                "vs_birkhoff_wtl": f"{birkhoff['win']}/{birkhoff['tie']}/{birkhoff['loss']}",
                "vs_greedy_mean_pct": greedy["mean_improvement_pct"],
                "vs_greedy_median_pct": greedy["median_improvement_pct"],
                "vs_greedy_wtl": f"{greedy['win']}/{greedy['tie']}/{greedy['loss']}",
            }
        )
    if summary_rows:
        with (output_dir / "related_work_core4_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)


__all__ = [
    "DEFAULT_POLICIES",
    "discover_trace_roots",
    "run_related_work_core_study",
    "write_study_artifacts",
]
