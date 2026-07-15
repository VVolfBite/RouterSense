#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distributed._gpu_runner_common import (
    available_cuda_count,
    build_policy_correctness_config,
    copy_config,
    dump_yaml,
    load_official_config,
    read_json,
    run_subprocess,
    torchrun_policy_command,
    write_json,
    write_runner_result_bundle,
)
from experiments.online.support.runtime_presets import resolve_strategy_runtime
from rs.core.layer_ids import stable_layer_count_map, stable_layer_ids


DEFAULT_STRATEGIES = (
    "native",
    "fifo_async_p2p",
    "greedy_async_p2p",
    "birkhoff_phase_local_sync",
    "birkhoff_phase_local_async_p2p",
    "routersense_b_core_independent_async",
    "routersense_u_core_zero_raw_async",
    "routersense_u_core_predicted_raw_async",
    "routersense_u_core_predicted_safe_async",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4GPU A2 comparison body with one torchrun process per strategy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategies", nargs="*", default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--measure-iters", type=int, default=None)
    parser.add_argument("--selected-layers", default=None)
    parser.add_argument("--profile", default=None, choices=("debug", "execution", "perf", "timeline_light", "attribution_light"))
    parser.add_argument("--preflight-mode", default=None, choices=("full", "compact"))
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--c2-summary-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _fallback(output_dir: Path, *, world_size: int, config: str, strategies: list[str], warmup_iters: int, measure_iters: int, selected_layers: str, profile: str, preflight_mode: str, dry_run: bool) -> dict:
    cuda_count = available_cuda_count()
    payload = {
        "runner": "run_gpu_a2_strategy_compare",
        "config": str(config),
        "strategies": list(strategies),
        "warmup_iters": int(warmup_iters),
        "measure_iters": int(measure_iters),
        "selected_layers": str(selected_layers),
        "profile": str(profile),
        "preflight_mode": str(preflight_mode),
        "world_size": int(world_size),
        "dry_run": bool(dry_run),
        "status": "IMPLEMENTED_GPU_BLOCKED_BY_ENVIRONMENT",
        "fallback_used": True,
        "result_eligible_for_performance_comparison": False,
        "fallback_reason": "gpu_environment_insufficient_world_size",
        "fallback_command": [],
        "fallback_returncode": 247,
        "cuda_device_count": int(cuda_count),
    }
    (output_dir / "fallback_stdout.log").write_text("", encoding="utf-8")
    (output_dir / "fallback_stderr.log").write_text(
        f"cuda_device_count={cuda_count} world_size={world_size} strict_gpu_run_blocked\n",
        encoding="utf-8",
    )
    return payload


def _summary_stats(values: list[float]) -> dict[str, float | list[float]]:
    if not values:
        return {"raw": [], "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    ordered = sorted(float(v) for v in values)
    def _pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = (len(ordered) - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return ordered[lo]
        frac = idx - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac
    return {
        "raw": ordered,
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "p25": float(_pct(0.25)),
        "p75": float(_pct(0.75)),
        "min": float(min(ordered)),
        "max": float(max(ordered)),
        "std": float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
    }


def _safe_gain(by_name: dict[str, dict], left: str, right: str) -> float | None:
    left_row = by_name.get(left) or {}
    right_row = by_name.get(right) or {}
    left_value = (((left_row.get("metrics") or {}).get("total_forward_us") or {}).get("median"))
    right_value = (((right_row.get("metrics") or {}).get("total_forward_us") or {}).get("median"))
    if left_value in (None, 0) or right_value in (None, 0):
        return None
    return float((float(right_value) - float(left_value)) / float(right_value))


def _load_rank_summaries(run_dir: Path) -> list[dict]:
    file_summaries = [
        read_json(path)
        for path in sorted(run_dir.glob("rank*_summary.json"))
        if path.name.startswith("rank") and "_prepared_plan_" not in path.name
    ]
    if file_summaries:
        return file_summaries
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return []
    payload = read_json(summary_path)
    embedded = payload.get("rank_summaries")
    if isinstance(embedded, list):
        return [dict(item) for item in embedded if isinstance(item, dict)]
    details = payload.get("details", {})
    embedded = details.get("rank_summaries") if isinstance(details, dict) else None
    if isinstance(embedded, list):
        return [dict(item) for item in embedded if isinstance(item, dict)]
    return []


def _metric_series(rank_summaries_or_rank0: dict | list[dict], run_dir: Path) -> dict[str, list[float]]:
    del run_dir
    if isinstance(rank_summaries_or_rank0, dict):
        rank_summaries = [rank_summaries_or_rank0]
    else:
        rank_summaries = list(rank_summaries_or_rank0)
    per_rank_records = [
        [row for row in list(summary.get("repeat_records") or []) if not bool(row.get("warmup", False))]
        for summary in rank_summaries
    ]
    repeat_records = per_rank_records[0] if per_rank_records else []
    metrics: dict[str, list[float]] = {
        "total_forward_us": [],
        "dispatch_transport_us": [],
        "return_transport_us": [],
        "communication_makespan_us": [],
        "p0_communication_makespan_us": [],
        "p1_communication_makespan_us": [],
        "rank0_transport_us": [],
        "max_rank_transport_active_us": [],
        "rank_skew_us": [],
        "p2p_enqueue_us": [],
        "p2p_wait_us": [],
        "prediction_us": [],
        "raw_u_build_us": [],
        "paired_b_build_us": [],
        "host_projection_us": [],
        "safe_selection_us": [],
        "plan_agreement_us": [],
        "local_materialization_us": [],
        "preflight_us": [],
        "control_overhead_us": [],
        "scheduling_overhead_us": [],
    }
    for idx, row in enumerate(repeat_records):
        dispatch_transport_us = float(row.get("dispatch_transport_us", 0.0) or 0.0)
        return_transport_us = float(row.get("return_transport_us", 0.0) or 0.0)
        p2p_enqueue_us = float(row.get("batch_submit_us", row.get("submit_us", 0.0)) or 0.0)
        p2p_wait_us = float(row.get("wait_us", 0.0) or 0.0)
        control_overhead_us = sum(
            float(row.get(key, 0.0) or 0.0)
            for key in (
                "prediction_us",
                "raw_u_build_us",
                "paired_b_build_us",
                "host_projection_us",
                "safe_selection_us",
                "plan_agreement_us",
                "local_materialization_us",
                "preflight_us",
            )
        )
        metrics["total_forward_us"].append(float(row.get("global_max_forward_us", 0.0) or 0.0))
        metrics["dispatch_transport_us"].append(dispatch_transport_us)
        metrics["return_transport_us"].append(return_transport_us)
        first_submit = []
        last_complete = []
        p0_first_submit = []
        p0_last_complete = []
        p1_first_submit = []
        p1_last_complete = []
        rank_active = []
        for rank_rows in per_rank_records:
            if idx >= len(rank_rows):
                continue
            rank_row = rank_rows[idx]
            start_ns = int(rank_row.get("first_transport_submit_ns", 0) or 0)
            end_ns = int(rank_row.get("last_transport_complete_ns", 0) or 0)
            p0_start_ns = int(rank_row.get("p0_first_submit_ns", 0) or 0)
            p0_end_ns = int(rank_row.get("p0_last_complete_ns", 0) or 0)
            p1_start_ns = int(rank_row.get("p1_first_submit_ns", 0) or 0)
            p1_end_ns = int(rank_row.get("p1_last_complete_ns", 0) or 0)
            if start_ns > 0 and end_ns >= start_ns:
                first_submit.append(start_ns)
                last_complete.append(end_ns)
                rank_active.append((end_ns - start_ns) / 1000.0)
            if p0_start_ns > 0 and p0_end_ns >= p0_start_ns:
                p0_first_submit.append(p0_start_ns)
                p0_last_complete.append(p0_end_ns)
            if p1_start_ns > 0 and p1_end_ns >= p1_start_ns:
                p1_first_submit.append(p1_start_ns)
                p1_last_complete.append(p1_end_ns)
        metrics["communication_makespan_us"].append(
            float((max(last_complete) - min(first_submit)) / 1000.0) if first_submit and last_complete else 0.0
        )
        metrics["p0_communication_makespan_us"].append(
            float((max(p0_last_complete) - min(p0_first_submit)) / 1000.0) if p0_first_submit and p0_last_complete else 0.0
        )
        metrics["p1_communication_makespan_us"].append(
            float((max(p1_last_complete) - min(p1_first_submit)) / 1000.0) if p1_first_submit and p1_last_complete else 0.0
        )
        metrics["rank0_transport_us"].append(
            float((int(row.get("last_transport_complete_ns", 0) or 0) - int(row.get("first_transport_submit_ns", 0) or 0)) / 1000.0)
            if int(row.get("first_transport_submit_ns", 0) or 0) > 0 and int(row.get("last_transport_complete_ns", 0) or 0) >= int(row.get("first_transport_submit_ns", 0) or 0)
            else dispatch_transport_us + return_transport_us
        )
        metrics["max_rank_transport_active_us"].append(float(max(rank_active)) if rank_active else 0.0)
        metrics["rank_skew_us"].append(float(max(rank_active) - min(rank_active)) if len(rank_active) >= 2 else 0.0)
        metrics["p2p_enqueue_us"].append(p2p_enqueue_us)
        metrics["p2p_wait_us"].append(p2p_wait_us)
        metrics["prediction_us"].append(float(row.get("prediction_us", 0.0) or 0.0))
        metrics["raw_u_build_us"].append(float(row.get("raw_u_build_us", 0.0) or 0.0))
        metrics["paired_b_build_us"].append(float(row.get("paired_b_build_us", 0.0) or 0.0))
        metrics["host_projection_us"].append(float(row.get("host_projection_us", 0.0) or 0.0))
        metrics["safe_selection_us"].append(float(row.get("safe_selection_us", 0.0) or 0.0))
        metrics["plan_agreement_us"].append(float(row.get("plan_agreement_us", 0.0) or 0.0))
        metrics["local_materialization_us"].append(float(row.get("local_materialization_us", 0.0) or 0.0))
        metrics["preflight_us"].append(float(row.get("preflight_us", 0.0) or 0.0))
        metrics["control_overhead_us"].append(control_overhead_us)
        metrics["scheduling_overhead_us"].append(control_overhead_us)
    return metrics


def _load_c2_status(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    payload = read_json(path)
    return str(payload.get("status", "")) == "passed"


def _normalize_safe_projection_mode(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "none", "null"}:
        return "disabled"
    return text


def _expected_count_contract(
    *,
    rank_summaries: list[dict],
    expected_world_size: int | None,
    warmup_iters: int | None,
    measure_iters: int | None,
    selected_layer_ids: list[str] | None,
    prediction_source_layer_ids: list[str] | None,
    strategy: str | None,
) -> dict:
    reasons: list[str] = []
    if expected_world_size is None or warmup_iters is None or measure_iters is None or selected_layer_ids is None:
        return {"available": False, "eligibility_reasons": ["expected_count_contract_missing_inputs"]}
    forward_count = int(warmup_iters) + int(measure_iters)
    selected_layers = stable_layer_ids(selected_layer_ids)
    prediction_layers = stable_layer_ids(prediction_source_layer_ids or [])
    expected_selected_per_rank = int(forward_count * len(selected_layers))
    expected_prediction_per_rank = int(forward_count * len(prediction_layers))
    expected_selected_all = int(expected_selected_per_rank * int(expected_world_size))
    expected_prediction_all = int(expected_prediction_per_rank * int(expected_world_size))

    def _actual_list(field: str) -> list[int | None]:
        values: list[int | None] = []
        for item in rank_summaries:
            values.append(int(item[field]) if field in item else None)
        return values

    selected_p0 = _actual_list("selected_p0_hook_count")
    selected_p1 = _actual_list("selected_p1_hook_count")
    prediction_p0 = _actual_list("prediction_source_p0_hook_count")
    for field, values, expected in (
        ("selected_p0_hook_count", selected_p0, expected_selected_per_rank),
        ("selected_p1_hook_count", selected_p1, expected_selected_per_rank),
        ("prediction_source_p0_hook_count", prediction_p0, expected_prediction_per_rank),
    ):
        for index, actual in enumerate(values):
            rank = int(rank_summaries[index].get("rank", index))
            if actual is None:
                reasons.append(f"{field}:rank={rank}:missing")
            elif int(actual) != int(expected):
                reasons.append(f"{field}:rank={rank}:expected={expected}:actual={actual}")
    selected_p0_sum = None if any(value is None for value in selected_p0) else int(sum(int(value) for value in selected_p0))
    selected_p1_sum = None if any(value is None for value in selected_p1) else int(sum(int(value) for value in selected_p1))
    prediction_p0_sum = None if any(value is None for value in prediction_p0) else int(sum(int(value) for value in prediction_p0))
    if selected_p0_sum is not None and selected_p0_sum != expected_selected_all:
        reasons.append(f"selected_p0_hook_count_all_rank:expected={expected_selected_all}:actual={selected_p0_sum}")
    if selected_p1_sum is not None and selected_p1_sum != expected_selected_all:
        reasons.append(f"selected_p1_hook_count_all_rank:expected={expected_selected_all}:actual={selected_p1_sum}")
    if prediction_p0_sum is not None and prediction_p0_sum != expected_prediction_all:
        reasons.append(f"prediction_source_p0_hook_count_all_rank:expected={expected_prediction_all}:actual={prediction_p0_sum}")
    payload: dict[str, object] = {
        "forward_count": int(forward_count),
        "selected_layer_count": int(len(selected_layers)),
        "prediction_source_layer_count": int(len(prediction_layers)),
        "expected_selected_p0_hook_count_per_rank": expected_selected_per_rank,
        "actual_selected_p0_hook_count_per_rank": selected_p0,
        "selected_p0_hook_count_exact": all(value == expected_selected_per_rank for value in selected_p0),
        "expected_selected_p1_hook_count_per_rank": expected_selected_per_rank,
        "actual_selected_p1_hook_count_per_rank": selected_p1,
        "selected_p1_hook_count_exact": all(value == expected_selected_per_rank for value in selected_p1),
        "expected_selected_p0_hook_count_all_rank": expected_selected_all,
        "actual_selected_p0_hook_count_all_rank": selected_p0_sum,
        "selected_p0_hook_count_all_rank_exact": selected_p0_sum == expected_selected_all,
        "expected_selected_p1_hook_count_all_rank": expected_selected_all,
        "actual_selected_p1_hook_count_all_rank": selected_p1_sum,
        "selected_p1_hook_count_all_rank_exact": selected_p1_sum == expected_selected_all,
        "expected_prediction_source_p0_hook_count_per_rank": expected_prediction_per_rank,
        "actual_prediction_source_p0_hook_count_per_rank": prediction_p0,
        "prediction_source_p0_hook_count_exact": all(value == expected_prediction_per_rank for value in prediction_p0),
        "expected_prediction_source_p0_hook_count_all_rank": expected_prediction_all,
        "actual_prediction_source_p0_hook_count_all_rank": prediction_p0_sum,
        "prediction_source_p0_hook_count_all_rank_exact": prediction_p0_sum == expected_prediction_all,
    }
    if str(strategy) == "routersense_u_core_zero_raw_async":
        raw_per_rank = _actual_list("raw_u_build_count")
        raw_all = None if any(value is None for value in raw_per_rank) else int(sum(int(value) for value in raw_per_rank))
        raw_layer_valid = True
        raw_layer_payload: list[dict[str, object]] = []
        for item in rank_summaries:
            rank = int(item.get("rank", len(raw_layer_payload)))
            by_layer = stable_layer_count_map(dict(item.get("raw_u_build_count_by_layer_per_rank") or item.get("raw_u_build_count_by_layer") or {}))
            for layer_id in selected_layers:
                actual = int(by_layer.get(layer_id, 0))
                valid = actual <= forward_count
                raw_layer_payload.append({"rank": rank, "layer": layer_id, "expected_upper_bound": forward_count, "actual": actual, "valid": valid})
                if not valid:
                    raw_layer_valid = False
                    reasons.append(f"raw_u_build_count_by_layer:rank={rank}:layer={layer_id}:upper={forward_count}:actual={actual}")
        for index, actual in enumerate(raw_per_rank):
            rank = int(rank_summaries[index].get("rank", index))
            if actual is None:
                reasons.append(f"raw_u_build_count:rank={rank}:missing")
            elif int(actual) > expected_selected_per_rank:
                reasons.append(f"raw_u_build_count:rank={rank}:upper={expected_selected_per_rank}:actual={actual}")
        if raw_all is not None and raw_all > expected_selected_all:
            reasons.append(f"raw_u_build_count_all_rank:upper={expected_selected_all}:actual={raw_all}")
        payload.update(
            {
                "expected_raw_u_build_upper_bound_per_rank": expected_selected_per_rank,
                "actual_raw_u_build_count_per_rank": raw_per_rank,
                "raw_u_build_count_per_rank_valid": all(value is not None and int(value) <= expected_selected_per_rank for value in raw_per_rank),
                "expected_raw_u_build_upper_bound_all_rank": expected_selected_all,
                "actual_raw_u_build_count_all_rank": raw_all,
                "raw_u_build_count_all_rank_valid": raw_all is not None and raw_all <= expected_selected_all,
                "raw_u_build_count_by_layer_per_rank_valid": bool(raw_layer_valid),
                "raw_u_build_count_by_layer_per_rank_contract": raw_layer_payload,
            }
        )
    payload["available"] = not reasons
    payload["hotpath_exact_count_eligible"] = not reasons
    payload["eligibility_reasons"] = reasons
    return payload


def aggregate_hotpath_rank_counts(
    rank_summaries: list[dict],
    *,
    expected_world_size: int | None = None,
    warmup_iters: int | None = None,
    measure_iters: int | None = None,
    selected_layer_ids: list[str] | None = None,
    prediction_source_layer_ids: list[str] | None = None,
    strategy: str | None = None,
) -> dict:
    fields = (
        "selected_p0_hook_count",
        "selected_p1_hook_count",
        "prediction_source_p0_hook_count",
        "none_heavy_hook_count",
        "real_p0_execution_count",
        "real_p1_execution_count",
        "shadow_dispatch_execution_count",
        "shadow_combine_execution_count",
        "observation_finalize_dispatch_count",
        "observation_finalize_combine_count",
        "shadow_policy_agreement_count",
        "shadow_plan_build_count",
        "shadow_control_collective_count",
        "raw_u_build_count",
        "paired_b_build_count",
        "predict_count",
    )
    reasons: list[str] = []
    observed_ranks: list[int] = []
    by_rank: dict[int, dict] = {}
    for item in rank_summaries:
        if "rank" not in item:
            reasons.append("rank_summary_missing_rank")
            continue
        rank = int(item["rank"])
        if rank in by_rank:
            reasons.append(f"duplicate_rank:{rank}")
        by_rank[rank] = item
        observed_ranks.append(rank)
    if expected_world_size is not None and len(by_rank) != int(expected_world_size):
        reasons.append(f"rank_count_mismatch:observed={len(by_rank)} expected={int(expected_world_size)}")
    payload: dict[str, object] = {
        "rank_count_observed": int(len(by_rank)),
        "rank_ids_observed": sorted(by_rank),
        "available": not reasons,
        "eligibility_reasons": reasons,
    }
    ordered = [by_rank[rank] for rank in sorted(by_rank)]
    for field in fields:
        per_rank: list[int | None] = []
        missing: list[int] = []
        for rank, item in zip(sorted(by_rank), ordered, strict=True):
            if field in item:
                per_rank.append(int(item.get(field, 0) or 0))
            else:
                per_rank.append(None)
                missing.append(rank)
        base = field
        if missing:
            reasons.append(f"{field}_missing_ranks:{missing}")
            payload[f"{base}_per_rank"] = per_rank
            payload[f"{base}_all_rank_sum"] = None
            continue
        payload[f"{base}_per_rank"] = per_rank
        payload[f"{base}_all_rank_sum"] = int(sum(int(v or 0) for v in per_rank))
    if int(payload.get("none_heavy_hook_count_all_rank_sum") or 0) > 0:
        reasons.append("none_heavy_hook_count_positive")
    exact_contract = _expected_count_contract(
        rank_summaries=ordered,
        expected_world_size=expected_world_size,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        selected_layer_ids=selected_layer_ids,
        prediction_source_layer_ids=prediction_source_layer_ids,
        strategy=strategy,
    )
    payload.update(exact_contract)
    reasons.extend(str(item) for item in exact_contract.get("eligibility_reasons", []) or [])
    if str(strategy) in {"routersense_b_core_independent_async", "routersense_u_core_zero_raw_async"}:
        expected_selected_per_rank = int(exact_contract.get("expected_selected_p0_hook_count_per_rank", 0) or 0)
        for field in ("real_p0_execution_count", "real_p1_execution_count"):
            values = list(payload.get(f"{field}_per_rank", []) or [])
            for index, actual in enumerate(values):
                rank = int(ordered[index].get("rank", index))
                if actual is None:
                    reasons.append(f"{field}:rank={rank}:missing")
                elif int(actual) != expected_selected_per_rank:
                    reasons.append(f"{field}:rank={rank}:expected={expected_selected_per_rank}:actual={actual}")
        for field in (
            "shadow_dispatch_execution_count",
            "shadow_combine_execution_count",
            "shadow_policy_agreement_count",
            "shadow_plan_build_count",
            "shadow_control_collective_count",
        ):
            values = list(payload.get(f"{field}_per_rank", []) or [])
            for index, actual in enumerate(values):
                rank = int(ordered[index].get("rank", index))
                if actual is None:
                    reasons.append(f"{field}:rank={rank}:missing")
                elif int(actual) != 0:
                    reasons.append(f"{field}:rank={rank}:expected=0:actual={actual}")
        for field in ("observation_finalize_dispatch_count", "observation_finalize_combine_count"):
            values = list(payload.get(f"{field}_per_rank", []) or [])
            for index, actual in enumerate(values):
                rank = int(ordered[index].get("rank", index))
                if actual is None:
                    reasons.append(f"{field}:rank={rank}:missing")
                elif int(actual) != expected_selected_per_rank:
                    reasons.append(f"{field}:rank={rank}:expected={expected_selected_per_rank}:actual={actual}")
    payload["available"] = not reasons
    payload["hotpath_eligible"] = not reasons
    payload["eligibility_reasons"] = reasons
    return payload


def _build_strategy_result(*, strategy: str, run_dir: Path, summary_payload: dict, c2_passed: bool) -> dict:
    details = summary_payload.get("details", {}) if isinstance(summary_payload, dict) else {}
    rank_summaries = _load_rank_summaries(run_dir)
    if not rank_summaries:
        return {
            "name": strategy,
            "status": "ineligible",
            "result_eligible_for_performance_comparison": False,
            "summary_status": str(summary_payload.get("status", "")),
            "metrics": {},
            "eligibility_checks": {"rank_summaries_present": False},
        }
    rank0_summary = rank_summaries[0]
    hotpath_counts = aggregate_hotpath_rank_counts(
        rank_summaries,
        expected_world_size=int(details.get("world_size", 0) or 0) or None,
        warmup_iters=int(details.get("warmup_iters", 0) or 0),
        measure_iters=int(details.get("measure_iters", 0) or 0),
        selected_layer_ids=[str(item) for item in (details.get("selected_layer_ids") or [])],
        prediction_source_layer_ids=[str(item) for item in (rank0_summary.get("prediction_source_layer_ids") or [])],
        strategy=str(strategy),
    )
    raw_series = _metric_series(rank_summaries, run_dir)
    metrics = {name: _summary_stats(values) for name, values in raw_series.items()}
    metrics.update(hotpath_counts)
    metrics["safe_selected_policy"] = rank0_summary.get("safe_selected_policy")
    metrics["fallback_count"] = int(rank0_summary.get("phase_sync_fallback_count", 0) or 0)
    metrics["timeout_count"] = int(rank0_summary.get("timeout_count", 0) or 0)
    metrics["all_work_completed"] = bool(rank0_summary.get("all_work_completed", True))
    metrics["async_executor_invocation_count"] = int(rank0_summary.get("async_executor_invocation_count", 0) or 0)
    metrics["batch_isend_irecv_call_count"] = int(rank0_summary.get("batch_isend_irecv_call_count", 0) or 0)
    metrics["real_send_op_count"] = int(rank0_summary.get("real_send_op_count", 0) or 0)
    metrics["real_recv_op_count"] = int(rank0_summary.get("real_recv_op_count", 0) or 0)
    metrics["local_copy_task_count"] = int(rank0_summary.get("local_copy_task_count", 0) or 0)
    metrics["selected_layer_match_count"] = int(rank0_summary.get("selected_layer_match_count", 0) or 0)
    metrics["selected_p0_hook_count"] = int(rank0_summary.get("selected_p0_hook_count", 0) or 0)
    metrics["selected_p1_hook_count"] = int(rank0_summary.get("selected_p1_hook_count", 0) or 0)
    metrics["selected_transport_execution_count"] = int(rank0_summary.get("selected_transport_execution_count", 0) or 0)
    metrics["transport_mutation_count"] = int(rank0_summary.get("transport_mutation_count", 0) or 0)
    metrics["prediction_created"] = bool(rank0_summary.get("prediction_created_stage")) or (
        str(rank0_summary.get("predictor_name", "") or "").strip() not in {"", "none", "zero_hint"}
        and bool(rank0_summary.get("prediction_digest"))
    )
    metrics["prediction_consumed"] = bool(rank0_summary.get("prediction_first_consumed_stage")) or bool(
        rank0_summary.get("prepared_target_logical_plan_digest")
    )
    metrics["target_plan_created"] = bool(rank0_summary.get("prepared_target_logical_plan_digest"))
    metrics["plan_digest_present"] = bool(
        rank0_summary.get("logical_plan_digest")
        or rank0_summary.get("compiled_plan_digest")
        or rank0_summary.get("stored_p1_logical_plan_digest")
        or rank0_summary.get("stored_p1_plan_digest")
    )
    metrics["paired_b_build_count"] = int(rank0_summary.get("paired_b_build_count", 0) or 0)
    metrics["host_projection_count"] = int(rank0_summary.get("host_projection_count", 0) or 0)
    metrics["execution_origin"] = str(rank0_summary.get("execution_origin", ""))
    metrics["c2_passed"] = bool(c2_passed)
    metrics["requested_preflight_mode"] = str(rank0_summary.get("requested_preflight_mode", ""))
    metrics["effective_preflight_mode"] = str(rank0_summary.get("effective_preflight_mode", ""))
    metrics["executor_preflight_mode"] = str(rank0_summary.get("executor_preflight_mode", ""))
    metrics["preflight_mode_match"] = all(bool(item.get("preflight_mode_match", False)) for item in rank_summaries)
    metrics["preflight_collective_count_exact"] = all(
        bool(item.get("preflight_collective_count_exact", False)) for item in rank_summaries
    )
    metrics["expected_preflight_collective_count_per_rank"] = [
        int(item.get("expected_preflight_collective_count", 0) or 0) for item in rank_summaries
    ]
    metrics["actual_preflight_collective_count_per_rank"] = [
        int(item.get("preflight_collective_count", 0) or 0) for item in rank_summaries
    ]
    metrics["preflight_mode_per_rank"] = [
        {
            "rank": int(item.get("rank", -1)),
            "requested": str(item.get("requested_preflight_mode", "")),
            "effective": str(item.get("effective_preflight_mode", "")),
            "executor": str(item.get("executor_preflight_mode", "")),
            "match": bool(item.get("preflight_mode_match", False)),
            "expected_collectives": int(item.get("expected_preflight_collective_count", 0) or 0),
            "actual_collectives": int(item.get("preflight_collective_count", 0) or 0),
            "collective_count_exact": bool(item.get("preflight_collective_count_exact", False)),
        }
        for item in rank_summaries
    ]
    metrics["tokenization_shape_valid_all_ranks"] = all(
        bool(item.get("tokenization_shape_valid", False)) for item in rank_summaries
    )
    expected = resolve_strategy_runtime(strategy_name=str(strategy), runtime_line="async_release" if "async" in str(strategy) else "phase_sync")
    expected_policy_name = str(expected.get("policy", ""))
    observed_policy_name = str(rank0_summary.get("policy_name", ""))
    policy_matches = (
        observed_policy_name in {"", "disabled"}
        if expected_policy_name == ""
        else observed_policy_name == expected_policy_name
    )
    expected_safe_mode = _normalize_safe_projection_mode(expected.get("safe_projection_mode", "disabled"))
    observed_safe_mode = _normalize_safe_projection_mode(rank0_summary.get("safe_projection_mode", "disabled"))
    is_native = str(strategy) in {"native", "disabled"}
    is_phase_sync = str(strategy) == "birkhoff_phase_local_sync"
    is_async_baseline = str(strategy) in {
        "birkhoff_phase_local_async_p2p",
        "fifo_async_p2p",
        "greedy_async_p2p",
        "routersense_b_core_independent_async",
    }
    is_u_zero = str(strategy) == "routersense_u_core_zero_raw_async"
    checks = {
        "summary_status_ready": str(summary_payload.get("status", "")) == "ready",
        "all_rank_summaries_present": len(rank_summaries) > 0,
        "all_ranks_completed": all(bool(item.get("forward_completed") or item.get("forward_partial_stop")) for item in rank_summaries),
        "selected_layer_match": int(metrics["selected_layer_match_count"]) > 0 if not is_native else True,
        "selected_p0_hook": int(metrics["selected_p0_hook_count"]) > 0 if not is_native else True,
        "selected_p1_hook": int(metrics["selected_p1_hook_count"]) > 0 if not is_native else True,
        "selected_transport_execution": int(metrics["selected_transport_execution_count"]) > 0 if not is_native else True,
        "plan_digest_present": bool(metrics["plan_digest_present"]) if not (is_native or is_phase_sync) else True,
        "transport_mutation": int(metrics["transport_mutation_count"]) > 0 if not is_native else True,
        "all_work_completed": (
            all(bool(item.get("all_work_completed", False)) for item in rank_summaries)
            if not is_native
            else all(bool(item.get("forward_completed") or item.get("forward_partial_stop")) for item in rank_summaries)
        ),
        "fallback_zero": int(metrics["fallback_count"]) == 0,
        "timeout_zero": (
            all(int(item.get("timeout_count", 1)) == 0 for item in rank_summaries)
            if not is_native
            else True
        ),
        "policy_matches": bool(policy_matches),
        "execution_mode_matches": str(rank0_summary.get("execution_mode", "")) == str(expected.get("execution_mode", "")),
        "safe_projection_mode_matches": (
            observed_safe_mode == expected_safe_mode
            or (
                expected_safe_mode == "host_select"
                and int(metrics["paired_b_build_count"]) > 0
                and int(metrics["host_projection_count"]) > 0
            )
        ),
        "preflight_mode_matches": (
            all(
                str(item.get("requested_preflight_mode", "")) != ""
                and str(item.get("requested_preflight_mode", "")) == str(item.get("effective_preflight_mode", ""))
                and str(item.get("effective_preflight_mode", "")) == str(item.get("executor_preflight_mode", ""))
                and bool(item.get("preflight_mode_match", False))
                for item in rank_summaries
            )
        ),
        "preflight_collective_count_exact": (
            all(bool(item.get("preflight_collective_count_exact", False)) for item in rank_summaries)
            if not is_native
            else True
        ),
        "tokenization_shape_valid": bool(metrics["tokenization_shape_valid_all_ranks"]),
        "hotpath_rank_aggregate_available": bool(hotpath_counts.get("available", False)) if not is_native else True,
        "c2_passed": bool(c2_passed),
    }
    expected_predictor = str(expected.get("online_p2_predictor", "none"))
    is_predicted_joint = bool(expected_predictor != "none" and "u_core_predicted" in str(strategy))
    is_safe = expected_safe_mode == "host_select"
    if is_native:
        checks["prediction_boundary"] = True
    elif is_phase_sync or is_async_baseline:
        checks["prediction_boundary"] = not bool(metrics["prediction_created"]) and not bool(metrics["prediction_consumed"]) and not bool(metrics["target_plan_created"])
    elif is_u_zero:
        checks["prediction_boundary"] = not bool(metrics["prediction_created"]) and not bool(metrics["prediction_consumed"]) and not bool(metrics["target_plan_created"])
    else:
        checks["prediction_boundary"] = bool(metrics["prediction_created"]) and bool(metrics["prediction_consumed"]) and bool(metrics["target_plan_created"])
    if is_predicted_joint:
        checks["target_plan_terminal"] = str(metrics["execution_origin"]) in {"prepared_exact", "prepared_repaired", "provisional_only", "provisional_then_late_suffix", "prepared_rejected", "prepared_expired"}
    else:
        checks["target_plan_terminal"] = not bool(metrics["target_plan_created"])
    if is_safe:
        checks["safe_variant_recorded"] = bool(metrics["safe_selected_policy"]) or int(metrics["host_projection_count"]) > 0
        checks["safe_build_counts"] = int(metrics["paired_b_build_count"]) > 0 and int(metrics["host_projection_count"]) > 0
    else:
        checks["safe_variant_recorded"] = True
        checks["safe_build_counts"] = True
    eligible = all(bool(value) for value in checks.values())
    return {
        "name": strategy,
        "status": "eligible" if eligible else "ineligible",
        "result_eligible_for_performance_comparison": bool(eligible),
        "summary_status": str(summary_payload.get("status", "")),
        "metrics": metrics,
        "output_checksum": details.get("output_checksum"),
        "eligibility_checks": checks,
    }


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_config(Path(args.config), output_dir)
    base = load_official_config(Path(args.config))
    resolved_strategies = list(args.strategies) if args.strategies is not None else list(base.get("strategies") or DEFAULT_STRATEGIES)
    resolved_warmup = int(args.warmup_iters if args.warmup_iters is not None else (base.get("evaluation", {}) or {}).get("warmup", 3))
    resolved_measure = int(args.measure_iters if args.measure_iters is not None else (base.get("evaluation", {}) or {}).get("repeats", 10))
    resolved_selected_layers = str(args.selected_layers if args.selected_layers is not None else base.get("selected_layers", "all"))
    resolved_profile = str(args.profile if args.profile is not None else base.get("profile", "perf"))
    resolved_preflight_mode = str(args.preflight_mode if args.preflight_mode is not None else base.get("preflight_mode", "compact"))
    resolved_world_size = int(args.world_size if args.world_size is not None else base.get("world_size", (base.get("topology", {}) or {}).get("world_size", 4)))
    c2_summary_path = Path(args.c2_summary_path) if args.c2_summary_path else (output_dir.parent / "c2" / "c2_runner_summary.json")
    c2_passed = _load_c2_status(c2_summary_path)
    payload = {
        "runner": "run_gpu_a2_strategy_compare",
        "config": str(args.config),
        "strategies": list(resolved_strategies),
        "warmup_iters": int(resolved_warmup),
        "measure_iters": int(resolved_measure),
        "selected_layers": str(resolved_selected_layers),
        "profile": str(resolved_profile),
        "preflight_mode": str(resolved_preflight_mode),
        "world_size": int(resolved_world_size),
        "dry_run": bool(args.dry_run),
        "c2_summary_path": str(c2_summary_path),
    }
    if args.dry_run:
        payload["status"] = "dry_run_ready"
        write_json(output_dir / "a2_runner_summary.json", payload)
        write_runner_result_bundle(output_dir, runner_name="run_gpu_a2_strategy_compare", payload=payload, run_kind="GPU_PERFORMANCE")
        print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
        return 0
    if available_cuda_count() < int(resolved_world_size):
        payload = _fallback(
            output_dir,
            world_size=int(resolved_world_size),
            config=str(args.config),
            strategies=list(resolved_strategies),
            warmup_iters=int(resolved_warmup),
            measure_iters=int(resolved_measure),
            selected_layers=str(resolved_selected_layers),
            profile=str(resolved_profile),
            preflight_mode=str(resolved_preflight_mode),
            dry_run=bool(args.dry_run),
        )
        write_json(output_dir / "a2_runner_summary.json", payload)
        write_runner_result_bundle(output_dir, runner_name="run_gpu_a2_strategy_compare", payload=payload, run_kind="GPU_PERFORMANCE")
        print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
        return 0 if int(payload["fallback_returncode"]) == 0 else int(payload["fallback_returncode"])

    generated_dir = output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    strategy_results: list[dict] = []
    for strategy in resolved_strategies:
        strategy_root = output_dir / "per_strategy" / str(strategy)
        run_name = f"a2_{strategy}"
        strategy_config = build_policy_correctness_config(
            base_comparison=base,
            strategy_name=str(strategy),
            run_name=run_name,
            output_root=strategy_root,
            profile=str(resolved_profile),
            selected_layers=str(resolved_selected_layers),
            save_logits=False,
            preflight_mode=str(resolved_preflight_mode),
        )
        config_path = generated_dir / f"{strategy}.yaml"
        dump_yaml(config_path, strategy_config)
        native = str(strategy) in {"native", "disabled"}
        cmd = torchrun_policy_command(
            config_path=config_path,
            run_id=run_name,
            output_dir=strategy_root,
            world_size=int(resolved_world_size),
            native=native,
        )
        proc = run_subprocess(
            cmd,
            extra_env={
                "ROUTERSENSE_WARMUP_ITERS": str(int(resolved_warmup)),
                "ROUTERSENSE_MEASURE_ITERS": str(int(resolved_measure)),
            },
        )
        (output_dir / f"{strategy}_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (output_dir / f"{strategy}_stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            payload.update({"status": f"{strategy}_failed", "failed_strategy": str(strategy), "returncode": int(proc.returncode), "failed_command": cmd})
            write_json(output_dir / "a2_runner_summary.json", payload)
            write_runner_result_bundle(output_dir, runner_name="run_gpu_a2_strategy_compare", payload=payload, run_kind="GPU_PERFORMANCE")
            print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
            return int(proc.returncode)
        run_dir = strategy_root / run_name
        summary_payload = read_json(run_dir / "summary.json")
        strategy_results.append(_build_strategy_result(strategy=str(strategy), run_dir=run_dir, summary_payload=summary_payload, c2_passed=c2_passed))

    by_name = {row["name"]: row for row in strategy_results}
    payload.update(
        {
            "status": "executed",
            "strategies": strategy_results,
            "c2_passed": bool(c2_passed),
            "backend_gain": _safe_gain(by_name, "birkhoff_phase_local_async_p2p", "birkhoff_phase_local_sync"),
            "phase_sync_joint_gain": None,
            "async_raw_joint_gain": _safe_gain(by_name, "routersense_u_core_zero_raw_async", "birkhoff_phase_local_async_p2p"),
            "raw_prediction_gain": _safe_gain(by_name, "routersense_u_core_predicted_raw_async", "routersense_u_core_zero_raw_async"),
            "safe_prediction_gain": None,
            "safe_zero_gain": None,
            "safe_predicted_gain": _safe_gain(by_name, "routersense_u_core_predicted_safe_async", "routersense_u_core_predicted_raw_async"),
            "full_system_gain": _safe_gain(by_name, "routersense_u_core_predicted_safe_async", "native"),
        }
    )
    write_json(output_dir / "a2_runner_summary.json", payload)
    write_runner_result_bundle(output_dir, runner_name="run_gpu_a2_strategy_compare", payload=payload, run_kind="GPU_PERFORMANCE")
    print((output_dir / "a2_runner_summary.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
