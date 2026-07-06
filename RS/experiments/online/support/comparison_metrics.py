from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def communication_makespan_from_timeline(timeline: list[dict[str, Any]]) -> float:
    before = [int(row.get("ts_us", 0)) for row in timeline if row.get("event") == "before_wave"]
    after = [int(row.get("ts_us", 0)) for row in timeline if row.get("event") == "after_wave"]
    if not before or not after:
        return 0.0
    return float(max(after) - min(before))


def plan_timing_from_timeline(timeline: list[dict[str, Any]]) -> dict[str, float]:
    before_by_key: dict[tuple[str, str], int] = {}
    observation_by_key: dict[tuple[str, str], int] = {}
    plan_compute_us = 0.0
    agreement_us = 0.0
    barrier_wait_us = 0.0
    phase_count = 0
    for row in timeline:
        layer = str(row.get("layer", row.get("layer_name", "")))
        phase = str(row.get("phase_name", row.get("phase", "")))
        event = str(row.get("event", ""))
        ts = int(row.get("ts_us", 0))
        key = (layer, phase)
        if event in {"p0_pre_transport_observation_ready", "p1_pre_transport_observation_ready"}:
            obs_phase = "P0" if event.startswith("p0") else "P1"
            observation_by_key[(layer, obs_phase)] = ts
        elif event == "before_phase_plan":
            before_by_key[key] = ts
            obs_ts = observation_by_key.get(key)
            if obs_ts is not None and ts >= obs_ts:
                barrier_wait_us += float(ts - obs_ts)
        elif event == "phase_execution_plan_agreed":
            start = before_by_key.get(key)
            if start is not None and ts >= start:
                agreement_us += float(ts - start)
                plan_compute_us += float(ts - start)
                phase_count += 1
    return {
        "plan_computation_us": plan_compute_us,
        "plan_agreement_us": agreement_us,
        "scheduling_overhead_us": agreement_us,
        "avg_plan_computation_us": plan_compute_us / phase_count if phase_count else 0.0,
        "barrier_wait_us": barrier_wait_us,
        "planned_phase_count": float(phase_count),
    }


def transport_metrics(events: list[dict[str, Any]]) -> dict[str, float]:
    p0 = 0.0
    p1 = 0.0
    active = 0
    local_rows = 0
    for row in events:
        phase = str(row.get("phase", row.get("phase_name", ""))).upper()
        duration = float(row.get("duration_us", row.get("elapsed_us", row.get("time_us", 0.0))) or 0.0)
        row_count = int(row.get("row_count", row.get("rows", 0)) or 0)
        if bool(row.get("is_local", False)) or str(row.get("event_type", "")) == "local_copy":
            local_rows += row_count
        if duration > 0.0 or row_count > 0:
            active += 1
        if phase == "P0":
            p0 += duration
        elif phase == "P1":
            p1 += duration
    return {"p0_makespan_us": p0, "p1_makespan_us": p1, "active_wave_count": float(active), "local_copy_rows": float(local_rows)}


def scheduled_plan_metrics(plans: list[dict[str, Any]]) -> dict[str, float]:
    wave_count = 0
    all_gather = build = broadcast = verify = total = 0.0
    for plan in plans:
        waves = plan.get("waves", []) or []
        wave_count += len(waves)
        metrics = plan.get("metrics", {}) or {}
        all_gather += float(metrics.get("all_gather_time_us", 0.0) or 0.0)
        build += float(metrics.get("build_plan_time_us", 0.0) or 0.0)
        broadcast += float(metrics.get("broadcast_time_us", 0.0) or 0.0)
        verify += float(metrics.get("verify_time_us", 0.0) or 0.0)
        total += float(metrics.get("total_agreement_time_us", 0.0) or 0.0)
    return {
        "total_wave_count": float(wave_count),
        "all_gather_time_us": all_gather,
        "build_plan_time_us": build,
        "broadcast_time_us": broadcast,
        "hash_verify_time_us": verify,
        "plan_metrics_total_agreement_us": total,
    }


def bundle_metrics(bundles: list[dict[str, Any]]) -> dict[str, float]:
    total = 0
    for row in bundles:
        segment = row.get("outgoing_segment", row)
        if bool(segment.get("is_local", False)):
            continue
        total += int(segment.get("byte_count", row.get("byte_count", 0)) or 0)
    return {"communication_bytes_total": float(total)}


def plan_arrival_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    before = sum(1 for row in rows if row.get("arrival_status") == "before_commit")
    inflight = sum(1 for row in rows if row.get("arrival_status") == "in_flight")
    none = sum(1 for row in rows if row.get("arrival_status") == "none")
    with_plan = sum(1 for row in rows if bool(row.get("has_prepared_plan", False)))
    ages = [float(row.get("plan_age_us", 0) or 0) for row in rows if bool(row.get("has_prepared_plan", False))]
    return {
        "plan_arrival_before_commit_count": float(before),
        "plan_arrival_in_flight_count": float(inflight),
        "plan_arrival_none_count": float(none),
        "avg_plan_age_us": float(statistics.fmean(ages)) if ages else 0.0,
        "calibrated_hint_coverage_pct": (100.0 * with_plan / total) if total else 0.0,
    }


def p2_hint_modes_from_phase_contexts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        hint = row.get("p2_hint", {}) or {}
        mode = str(hint.get("hint_mode", "none"))
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def metrics_from_rank_dir(rank_dir: Path, *, rank: int = 0) -> dict[str, Any]:
    timeline = read_jsonl(rank_dir / f"rank{rank}_control_timeline.jsonl")
    plans = read_jsonl(rank_dir / f"rank{rank}_scheduled_phase_plans.jsonl")
    transport = read_jsonl(rank_dir / f"rank{rank}_transport_execution.jsonl")
    bundles = read_jsonl(rank_dir / f"rank{rank}_transport_bundles.jsonl")
    arrivals = read_jsonl(rank_dir / f"rank{rank}_plan_arrival_records.jsonl")
    phase_contexts = read_jsonl(rank_dir / f"rank{rank}_phase_contexts.jsonl")
    summary = read_json(rank_dir / f"rank{rank}_summary.json")
    metrics: dict[str, Any] = {
        "communication_makespan_us": communication_makespan_from_timeline(timeline),
        "remote_dispatch_rows": float(summary.get("remote_dispatch_rows", summary.get("p0_remote_rows", 0)) or 0),
        "remote_combine_rows": float(summary.get("remote_combine_rows", summary.get("p1_remote_rows", 0)) or 0),
        "p2_hint_modes_used": p2_hint_modes_from_phase_contexts(phase_contexts),
    }
    metrics.update(plan_timing_from_timeline(timeline))
    metrics.update(scheduled_plan_metrics(plans))
    metrics.update(transport_metrics(transport))
    metrics.update(bundle_metrics(bundles))
    metrics.update(plan_arrival_metrics(arrivals))
    return metrics


def aggregate_repetitions(repetition_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for metrics in repetition_metrics for key, value in metrics.items() if isinstance(value, (int, float))})
    aggregated = {key: summarize_values([float(metrics.get(key, 0.0)) for metrics in repetition_metrics]) for key in keys}
    hint_counts: dict[str, int] = {}
    for metrics in repetition_metrics:
        for mode, count in (metrics.get("p2_hint_modes_used", {}) or {}).items():
            hint_counts[str(mode)] = hint_counts.get(str(mode), 0) + int(count)
    aggregated["p2_hint_modes_used"] = hint_counts
    return aggregated


def _mean(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, {})
    if isinstance(value, dict):
        return float(value.get("mean", 0.0) or 0.0)
    return float(value or 0.0)


def add_baseline_deltas(strategy_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(strategy_metrics)
    base_comm = _mean(baseline_metrics, "communication_makespan_us")
    comm = _mean(strategy_metrics, "communication_makespan_us")
    overhead = _mean(strategy_metrics, "scheduling_overhead_us")
    forward = _mean(strategy_metrics, "total_forward_us")
    base_forward = _mean(baseline_metrics, "total_forward_us")
    savings = base_comm - comm
    result["net_comm_savings_us"] = summarize_values([savings])
    result["net_benefit_us"] = summarize_values([savings - overhead])
    result["benefit_ratio"] = summarize_values([savings / overhead if overhead > 0 else math.inf if savings > 0 else 0.0])
    result["forward_speedup_pct"] = summarize_values([100.0 * (base_forward - forward) / base_forward if base_forward > 0 else 0.0])
    result["scheduling_fraction_pct"] = summarize_values([100.0 * overhead / forward if forward > 0 else 0.0])
    return result


def build_comparison_report(
    *,
    run_id: str,
    baseline: str,
    strategies: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_entry = next((item for item in strategies if item["name"] == baseline), None)
    baseline_metrics = baseline_entry["metrics"] if baseline_entry is not None else {}
    enriched = []
    for item in strategies:
        metrics = add_baseline_deltas(item["metrics"], baseline_metrics) if baseline_metrics else item["metrics"]
        enriched.append({**item, "metrics": metrics})
    pairwise_vs_baseline = {}
    for item in enriched:
        if item["name"] == baseline:
            continue
        metrics = item["metrics"]
        base_comm = _mean(baseline_metrics, "communication_makespan_us")
        comm = _mean(metrics, "communication_makespan_us")
        base_wave = _mean(baseline_metrics, "total_wave_count")
        wave = _mean(metrics, "total_wave_count")
        pairwise_vs_baseline[item["name"]] = {
            "comm_makespan_delta_pct": 100.0 * (comm - base_comm) / base_comm if base_comm > 0 else 0.0,
            "wave_count_delta_pct": 100.0 * (wave - base_wave) / base_wave if base_wave > 0 else 0.0,
            "net_benefit_us": _mean(metrics, "net_benefit_us"),
            "forward_speedup_pct": _mean(metrics, "forward_speedup_pct"),
        }
    head_to_head = {}
    by_name = {item["name"]: item for item in enriched}
    pairs = [
        ("routersense_p0p1p2_hint", "routersense_p0p1_reservation", "P2 prediction gain over P1-only reservation"),
        ("routersense_p0p1p2_hint", "birkhoff_phase_local", "Cross-phase coordination gain over per-phase local optimal"),
    ]
    for left, right, description in pairs:
        if left in by_name and right in by_name:
            left_comm = _mean(by_name[left]["metrics"], "communication_makespan_us")
            right_comm = _mean(by_name[right]["metrics"], "communication_makespan_us")
            head_to_head[f"{left}_vs_{right}"] = {
                "comm_makespan_delta_pct": 100.0 * (left_comm - right_comm) / right_comm if right_comm > 0 else 0.0,
                "description": description,
            }
    return {
        "run_id": run_id,
        "baseline": baseline,
        "strategies": enriched,
        "pairwise_vs_baseline": pairwise_vs_baseline,
        "pairwise_head_to_head": head_to_head,
    }


def format_metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key, {})
    if not isinstance(value, dict):
        return str(value)
    return f"{float(value.get('mean', 0.0)):.2f} ± {float(value.get('std', 0.0)):.2f}"


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Strategy Comparison Report", "", "## Communication Scheduling Quality", "| Strategy | Wave Count | Comm Makespan (us) | Remote Rows | Local Copy Rows |", "|---|---:|---:|---:|---:|"]
    for item in report.get("strategies", []):
        metrics = item["metrics"]
        lines.append(
            f"| {item['name']} | {format_metric(metrics, 'total_wave_count')} | {format_metric(metrics, 'communication_makespan_us')} | "
            f"{format_metric(metrics, 'remote_dispatch_rows')} | {format_metric(metrics, 'local_copy_rows')} |"
        )
    lines.extend(["", "## Scheduling Overhead", "| Strategy | Plan Compute (us) | Agreement (us) | Total Overhead (us) | Fraction (%) |", "|---|---:|---:|---:|---:|"])
    for item in report.get("strategies", []):
        metrics = item["metrics"]
        lines.append(
            f"| {item['name']} | {format_metric(metrics, 'plan_computation_us')} | {format_metric(metrics, 'plan_agreement_us')} | "
            f"{format_metric(metrics, 'scheduling_overhead_us')} | {format_metric(metrics, 'scheduling_fraction_pct')} |"
        )
    lines.extend(["", "## Net Benefit Analysis", "| Strategy | Comm Savings (us) | Sched Overhead (us) | Net Benefit (us) | Benefit Ratio |", "|---|---:|---:|---:|---:|"])
    for item in report.get("strategies", []):
        metrics = item["metrics"]
        lines.append(
            f"| {item['name']} | {format_metric(metrics, 'net_comm_savings_us')} | {format_metric(metrics, 'scheduling_overhead_us')} | "
            f"{format_metric(metrics, 'net_benefit_us')} | {format_metric(metrics, 'benefit_ratio')} |"
        )
    lines.extend(["", "## Shadow Plan Arrival", "| Strategy | Before Commit | In-Flight | None | Coverage (%) | Avg Age (us) |", "|---|---:|---:|---:|---:|---:|"])
    for item in report.get("strategies", []):
        metrics = item["metrics"]
        lines.append(
            f"| {item['name']} | {format_metric(metrics, 'plan_arrival_before_commit_count')} | {format_metric(metrics, 'plan_arrival_in_flight_count')} | "
            f"{format_metric(metrics, 'plan_arrival_none_count')} | {format_metric(metrics, 'calibrated_hint_coverage_pct')} | {format_metric(metrics, 'avg_plan_age_us')} |"
        )
    return "\n".join(lines) + "\n"
