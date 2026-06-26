from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def build_eval_summary(strategy_runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_strategy: dict[str, dict[str, Any]] = {}
    routing_mode = None
    service_mode = None
    service_backend = None
    backend_realism_level = None
    for run in strategy_runs:
        strategy = str(run["strategy"])
        routing_mode = routing_mode or run.get("routing_mode")
        service_mode = service_mode or run.get("service_mode")
        service_backend = service_backend or run.get("service_backend")
        backend_realism_level = backend_realism_level or run.get("backend_realism_level")
        microbatches = run.get("microbatches", [])
        total_batch_completion = [float(item.get("batch_completion_time", 0.0)) for item in microbatches]
        barrier_proxy = [float(item.get("barrier_proxy", 0.0)) for item in microbatches]
        idle_proxy = [float(item.get("per_rank_idle_proxy_sum", 0.0)) for item in microbatches]
        critical_completion = [float(item.get("critical_bucket_completion_time", 0.0)) for item in microbatches]
        per_strategy[strategy] = {
            "microbatch_count": len(microbatches),
            "mean_total_batch_completion_proxy": _mean(total_batch_completion),
            "mean_barrier_join_proxy": _mean(barrier_proxy),
            "mean_per_rank_idle_proxy_sum": _mean(idle_proxy),
            "mean_critical_bucket_completion_proxy": _mean(critical_completion),
            "mean_release_order_delta_vs_fifo": _mean(
                [float(item.get("release_order_delta_vs_fifo", 0.0)) for item in microbatches]
            ),
        }

    baseline = per_strategy.get("fifo", {})
    comparisons: dict[str, dict[str, Any]] = {}
    for strategy, metrics in per_strategy.items():
        if strategy == "fifo":
            continue
        comparisons[strategy] = {
            "delta_total_batch_completion_proxy_vs_fifo": _delta(
                metrics.get("mean_total_batch_completion_proxy"),
                baseline.get("mean_total_batch_completion_proxy"),
            ),
            "delta_barrier_join_proxy_vs_fifo": _delta(
                metrics.get("mean_barrier_join_proxy"),
                baseline.get("mean_barrier_join_proxy"),
            ),
            "delta_per_rank_idle_proxy_sum_vs_fifo": _delta(
                metrics.get("mean_per_rank_idle_proxy_sum"),
                baseline.get("mean_per_rank_idle_proxy_sum"),
            ),
            "delta_critical_bucket_completion_proxy_vs_fifo": _delta(
                metrics.get("mean_critical_bucket_completion_proxy"),
                baseline.get("mean_critical_bucket_completion_proxy"),
            ),
        }

    return {
        "routing_mode": routing_mode,
        "service_mode": service_mode,
        "service_backend": service_backend,
        "backend_realism_level": backend_realism_level,
        "metric_note": (
            "All reported completion, barrier, idle, and critical-bucket values are proxy metrics. "
            "They measure scheduler behavior under the current synthetic service backend, not final EP runtime latency."
        ),
        "strategies": per_strategy,
        "comparisons_vs_fifo": comparisons,
        "judgment": _judge(per_strategy, comparisons),
    }


def build_sweep_aggregate(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    signal_scores: list[dict[str, Any]] = []
    for item in run_summaries:
        model_key = str(item["model_key"])
        backend = str(item["service_backend"])
        summary = item["summary"]
        strategies = summary.get("strategies", {})
        state = strategies.get("state-only", {})
        dep = strategies.get("dependency-only", {})
        full = strategies.get("full", {})
        fifo = strategies.get("fifo", {})
        comparisons = {
            "dependency_only_vs_state_only": _pairwise_delta(dep, state),
            "full_vs_state_only": _pairwise_delta(full, state),
            "full_vs_fifo": _pairwise_delta(full, fifo),
        }
        run_entry = {
            "model_key": model_key,
            "service_backend": backend,
            "routing_mode": item.get("routing_mode"),
            "service_mode": item.get("service_mode"),
            "service_is_synthetic": item.get("service_is_synthetic", True),
            "backend_realism_level": item.get("backend_realism_level", "synthetic"),
            "output_dir": item.get("output_dir"),
            "strategies": strategies,
            "comparisons": comparisons,
            "short_conclusion": _short_conclusion(comparisons),
        }
        runs.append(run_entry)
        signal_scores.append(
            {
                "model_key": model_key,
                "service_backend": backend,
                "signal_score": _signal_score(comparisons),
                "short_conclusion": run_entry["short_conclusion"],
            }
        )

    best = max(signal_scores, key=lambda item: item["signal_score"]) if signal_scores else None
    by_backend = _group_signal(signal_scores, "service_backend")
    by_model = _group_signal(signal_scores, "model_key")
    backend_judgment = _best_group(by_backend)
    model_judgment = _best_group(by_model)
    strength = _signal_strength(best["signal_score"] if best else 0.0)

    return {
        "goal": "real routing + synthetic service under single-node 4-GPU prototype",
        "run_count": len(runs),
        "runs": runs,
        "mainline_signal_strength": strength,
        "best_backend_for_signal": backend_judgment,
        "best_model_for_signal": model_judgment,
        "recommended_next_step": _recommended_next_step(strength, backend_judgment, model_judgment),
        "cross_model_agreement": _cross_model_agreement(runs),
    }


def load_summary_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_goal(path: str | Path, body: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def export_artifact_bundle(
    artifact_dir: str | Path,
    *,
    summary: dict[str, Any],
    config: dict[str, Any],
    environment: dict[str, Any],
    goal: str,
) -> None:
    target = Path(artifact_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (target / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (target / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (target / "goal.md").write_text(goal, encoding="utf-8")


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _delta(value: Any, baseline: Any) -> float:
    return float(value or 0.0) - float(baseline or 0.0)


def _judge(per_strategy: dict[str, dict[str, Any]], comparisons: dict[str, dict[str, Any]]) -> str:
    dep = comparisons.get("dependency-only")
    full = comparisons.get("full")
    state = comparisons.get("state-only")
    if not dep or not full or not state:
        return "incomplete"
    dep_beats_state = dep["delta_total_batch_completion_proxy_vs_fifo"] < state["delta_total_batch_completion_proxy_vs_fifo"]
    full_beats_state = full["delta_total_batch_completion_proxy_vs_fifo"] < state["delta_total_batch_completion_proxy_vs_fifo"]
    if dep_beats_state or full_beats_state:
        return "dependency-aware scheduling shows an early proxy advantage over state-only in this run"
    return "dependency-aware scheduling has not yet shown a proxy advantage over state-only in this run"


def _pairwise_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        "delta_mean_total_batch_completion_proxy": float(left.get("mean_total_batch_completion_proxy", 0.0))
        - float(right.get("mean_total_batch_completion_proxy", 0.0)),
        "delta_mean_barrier_join_proxy": float(left.get("mean_barrier_join_proxy", 0.0))
        - float(right.get("mean_barrier_join_proxy", 0.0)),
        "delta_mean_per_rank_idle_proxy_sum": float(left.get("mean_per_rank_idle_proxy_sum", 0.0))
        - float(right.get("mean_per_rank_idle_proxy_sum", 0.0)),
        "delta_mean_critical_bucket_completion_proxy": float(left.get("mean_critical_bucket_completion_proxy", 0.0))
        - float(right.get("mean_critical_bucket_completion_proxy", 0.0)),
        "delta_mean_release_order_delta_vs_fifo": float(left.get("mean_release_order_delta_vs_fifo", 0.0))
        - float(right.get("mean_release_order_delta_vs_fifo", 0.0)),
    }


def _short_conclusion(comparisons: dict[str, Any]) -> str:
    dep = comparisons["dependency_only_vs_state_only"]["delta_mean_total_batch_completion_proxy"]
    full = comparisons["full_vs_state_only"]["delta_mean_total_batch_completion_proxy"]
    if dep < 0.0 or full < 0.0:
        return "dependency-aware shows early proxy advantage"
    return "dependency-aware does not show early proxy advantage"


def _signal_score(comparisons: dict[str, Any]) -> float:
    dep = -float(comparisons["dependency_only_vs_state_only"]["delta_mean_total_batch_completion_proxy"])
    full = -float(comparisons["full_vs_state_only"]["delta_mean_total_batch_completion_proxy"])
    barrier = -min(
        float(comparisons["dependency_only_vs_state_only"]["delta_mean_barrier_join_proxy"]),
        float(comparisons["full_vs_state_only"]["delta_mean_barrier_join_proxy"]),
    )
    return 0.45 * max(dep, 0.0) + 0.45 * max(full, 0.0) + 0.10 * max(barrier, 0.0)


def _group_signal(items: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in items:
        grouped.setdefault(str(item[key]), []).append(float(item["signal_score"]))
    return {name: float(statistics.fmean(values)) for name, values in grouped.items()} if grouped else {}


def _best_group(grouped: dict[str, float]) -> str | None:
    if not grouped:
        return None
    return max(grouped, key=grouped.get)


def _signal_strength(score: float) -> str:
    if score >= 0.01:
        return "strong"
    if score > 0.0:
        return "moderate"
    return "weak"


def _recommended_next_step(strength: str, best_backend: str | None, best_model: str | None) -> str:
    if strength == "strong":
        return f"continue mainline with {best_model or 'best model'} on {best_backend or 'best backend'} and push service path closer to real expert compute"
    if strength == "moderate":
        return f"expand microbatch count on {best_model or 'best model'} and re-check signal stability under {best_backend or 'best backend'}"
    return "signal is weak; re-check backend sensitivity and confirm service cost model before pushing runtime complexity"


def _cross_model_agreement(runs: list[dict[str, Any]]) -> str:
    by_model: dict[str, list[str]] = {}
    for run in runs:
        by_model.setdefault(str(run["model_key"]), []).append(str(run["short_conclusion"]))
    normalized = {model: max(set(values), key=values.count) for model, values in by_model.items()} if by_model else {}
    if len(set(normalized.values())) <= 1:
        return "qwen agrees with olmoe" if normalized else "insufficient"
    return "qwen does not agree with olmoe"
