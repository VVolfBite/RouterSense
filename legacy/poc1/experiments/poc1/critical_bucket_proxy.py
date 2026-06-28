#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROXY_SUMMARY = ROOT / "outputs" / "poc1_proxy_compare" / "proxy_comparison_summary.json"
OUTPUT_PATH = ROOT / "outputs" / "poc1_proxy_compare" / "critical_bucket_proxy_summary.json"
GOAL_PATH = ROOT / "outputs" / "poc1_proxy_compare" / "goal.md"


CRITICAL_DEFINITIONS = {
    "v1": {
        "description": "Structure-heavy pseudo critical bucket from the previous round.",
        "formula": (
            "0.35*bucket_size_norm + 0.15*selected_route_share + 0.20*bridge_score "
            "+ 0.15*source_coverage + 0.10*position_spread + 0.05*coactive_peer_degree"
        ),
        "weights": {
            "bucket_size_norm": 0.35,
            "selected_route_share": 0.15,
            "bridge_score": 0.20,
            "source_coverage": 0.15,
            "position_spread": 0.10,
            "coactive_peer_degree": 0.05,
        },
    },
    "v2": {
        "description": (
            "More independent pseudo critical bucket. It emphasizes load/concentration and token coverage, "
            "then uses only small structural tie-breakers to reduce overlap with dependency-only scoring."
        ),
        "formula": (
            "0.45*bucket_size_norm + 0.20*selected_route_share + 0.15*density_over_mean_norm "
            "+ 0.10*source_coverage + 0.05*position_spread + 0.05*bridge_score"
        ),
        "weights": {
            "bucket_size_norm": 0.45,
            "selected_route_share": 0.20,
            "density_over_mean_norm": 0.15,
            "source_coverage": 0.10,
            "position_spread": 0.05,
            "bridge_score": 0.05,
        },
    },
}


def main() -> int:
    proxy_summary = _load_json(PROXY_SUMMARY)
    all_results = {}
    for version, definition in CRITICAL_DEFINITIONS.items():
        sample_results = [_analyze_sample(sample, definition["weights"]) for sample in proxy_summary.get("samples", [])]
        all_results[version] = {
            "definition": {
                "name": f"critical_bucket_score_{version}",
                "description": definition["description"],
                "formula": definition["formula"],
                "weights": definition["weights"],
            },
            "sample_count": len(sample_results),
            "samples": sample_results,
            "aggregate": _aggregate(sample_results),
            "per_layer": _per_layer(sample_results),
            "per_category": _per_category(sample_results),
            "grouped_results": _grouped_results(sample_results),
        }

    payload = {
        "mode": "critical_bucket_proxy_v2",
        "purpose": "Compare state/dependency/full rankings against v1 and more independent v2 pseudo critical bucket definitions.",
        "input_proxy_summary": str(PROXY_SUMMARY),
        "definition_comparison": {
            "what_changed": (
                "v2 reduces direct dependency-style features by lowering bridge/position weights and adding "
                "density_over_mean_norm plus stronger bucket load/share terms."
            ),
            "recommended_definition_for_next_decision": "v2",
        },
        "definitions": all_results,
        "comparison_summary": _compare_definitions(all_results),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _append_goal()
    print(f"wrote {OUTPUT_PATH}")
    return 0


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _analyze_sample(sample: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    buckets = []
    for bucket in sample.get("bucket_scores", []):
        state_features = bucket.get("state_features", {})
        dep_features = bucket.get("dependency_features", {})
        feature_values = {
            "bucket_size_norm": float(state_features.get("bucket_size_norm", 0.0)),
            "selected_route_share": float(state_features.get("selected_route_share", 0.0)),
            "density_over_mean_norm": float(state_features.get("density_over_mean_norm", 0.0)),
            "bridge_score": float(dep_features.get("bridge_score", 0.0)),
            "source_coverage": float(dep_features.get("source_coverage", 0.0)),
            "position_spread": float(dep_features.get("position_spread", 0.0)),
            "coactive_peer_degree": float(dep_features.get("coactive_peer_degree", 0.0)),
        }
        critical_score = sum(weights.get(name, 0.0) * value for name, value in feature_values.items())
        buckets.append(
            {
                "destination_id": int(bucket["destination_id"]),
                "critical_bucket_score": critical_score,
                "state_only_score": bucket["state_only_score"],
                "dependency_only_score": bucket["dependency_only_score"],
                "full_score": bucket["full_score"],
            }
        )

    critical_ranked = _rank(buckets, "critical_bucket_score")
    state_ranked = _rank(buckets, "state_only_score")
    dep_ranked = _rank(buckets, "dependency_only_score")
    full_ranked = _rank(buckets, "full_score")
    critical_top1 = _top_ids(critical_ranked, 1)
    critical_top3 = _top_ids(critical_ranked, 3)
    state_top3 = _top_ids(state_ranked, 3)
    dep_top3 = _top_ids(dep_ranked, 3)
    full_top3 = _top_ids(full_ranked, 3)

    return {
        "sample_id": sample["sample_id"],
        "base_sample_id": sample.get("base_sample_id"),
        "layer_key": sample.get("layer_key"),
        "prompt_category": sample.get("prompt_category"),
        "max_to_avg_bucket_ratio": sample.get("max_to_avg_bucket_ratio"),
        "active_destination_count": sample.get("active_destination_count"),
        "hot_bucket_count": sample.get("hot_bucket_count"),
        "mean_bridge_score": sample.get("mean_bridge_score"),
        "top3_bucket_mass": sample.get("top3_bucket_mass"),
        "critical_top1_destination": critical_top1[0] if critical_top1 else None,
        "critical_top3_destinations": critical_top3,
        "state_top1_destination": _top_ids(state_ranked, 1)[0] if state_ranked else None,
        "dependency_top1_destination": _top_ids(dep_ranked, 1)[0] if dep_ranked else None,
        "full_top1_destination": _top_ids(full_ranked, 1)[0] if full_ranked else None,
        "state_top1_hit": _top1_hit(state_ranked, critical_top1),
        "dependency_top1_hit": _top1_hit(dep_ranked, critical_top1),
        "full_top1_hit": _top1_hit(full_ranked, critical_top1),
        "state_top3_hit": _topk_hit(state_top3, critical_top3),
        "dependency_top3_hit": _topk_hit(dep_top3, critical_top3),
        "full_top3_hit": _topk_hit(full_top3, critical_top3),
        "state_rank_of_critical_top1": _rank_of(state_ranked, critical_top1[0] if critical_top1 else None),
        "dependency_rank_of_critical_top1": _rank_of(dep_ranked, critical_top1[0] if critical_top1 else None),
        "full_rank_of_critical_top1": _rank_of(full_ranked, critical_top1[0] if critical_top1 else None),
        "state_critical_spearman": _spearman(buckets, "state_only_score", "critical_bucket_score"),
        "dependency_critical_spearman": _spearman(buckets, "dependency_only_score", "critical_bucket_score"),
        "full_critical_spearman": _spearman(buckets, "full_score", "critical_bucket_score"),
    }


def _rank(items: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (-float(item[score_key]), int(item["destination_id"])))


def _top_ids(items: list[dict[str, Any]], k: int) -> list[int]:
    return [int(item["destination_id"]) for item in items[:k]]


def _top1_hit(ranked: list[dict[str, Any]], critical_top1: list[int]) -> bool:
    return bool(ranked and critical_top1 and int(ranked[0]["destination_id"]) == critical_top1[0])


def _topk_hit(proxy_topk: list[int], critical_topk: list[int]) -> bool:
    return bool(set(proxy_topk) & set(critical_topk))


def _rank_of(ranked: list[dict[str, Any]], destination_id: int | None) -> int | None:
    if destination_id is None:
        return None
    for index, item in enumerate(ranked, start=1):
        if int(item["destination_id"]) == destination_id:
            return index
    return None


def _spearman(items: list[dict[str, Any]], left_key: str, right_key: str) -> float | None:
    if len(items) < 2:
        return None
    left_ranks = _rank_positions(items, left_key)
    right_ranks = _rank_positions(items, right_key)
    n = len(items)
    diff_sq = sum((left_ranks[item["destination_id"]] - right_ranks[item["destination_id"]]) ** 2 for item in items)
    return 1.0 - (6.0 * diff_sq) / (n * (n * n - 1))


def _rank_positions(items: list[dict[str, Any]], score_key: str) -> dict[int, int]:
    return {int(item["destination_id"]): index for index, item in enumerate(_rank(items, score_key), start=1)}


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    aggregate = _metrics(samples)
    aggregate["best_top1_proxy"] = _best_proxy(aggregate, "top1_hit_rate")
    aggregate["best_top3_proxy"] = _best_proxy(aggregate, "top3_hit_rate")
    aggregate["best_rank_proxy"] = _best_proxy(aggregate, "mean_rank_of_critical_top1", lower_is_better=True)
    aggregate["best_spearman_proxy"] = _best_proxy(aggregate, "mean_critical_spearman")
    aggregate["short_conclusion"] = _conclusion(aggregate)
    return aggregate


def _metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(samples),
        "state_top1_hit_rate": _hit_rate(samples, "state_top1_hit"),
        "dependency_top1_hit_rate": _hit_rate(samples, "dependency_top1_hit"),
        "full_top1_hit_rate": _hit_rate(samples, "full_top1_hit"),
        "state_top3_hit_rate": _hit_rate(samples, "state_top3_hit"),
        "dependency_top3_hit_rate": _hit_rate(samples, "dependency_top3_hit"),
        "full_top3_hit_rate": _hit_rate(samples, "full_top3_hit"),
        "state_mean_rank_of_critical_top1": _mean(samples, "state_rank_of_critical_top1"),
        "dependency_mean_rank_of_critical_top1": _mean(samples, "dependency_rank_of_critical_top1"),
        "full_mean_rank_of_critical_top1": _mean(samples, "full_rank_of_critical_top1"),
        "state_mean_critical_spearman": _mean(samples, "state_critical_spearman"),
        "dependency_mean_critical_spearman": _mean(samples, "dependency_critical_spearman"),
        "full_mean_critical_spearman": _mean(samples, "full_critical_spearman"),
    }


def _hit_rate(samples: list[dict[str, Any]], key: str) -> float:
    return sum(1 for sample in samples if sample[key]) / len(samples) if samples else 0.0


def _mean(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [sample[key] for sample in samples if sample[key] is not None]
    return statistics.fmean(values) if values else None


def _best_proxy(metrics: dict[str, Any], suffix: str, lower_is_better: bool = False) -> str:
    candidates = {
        "state": metrics.get(f"state_{suffix}"),
        "dependency": metrics.get(f"dependency_{suffix}"),
        "full": metrics.get(f"full_{suffix}"),
    }
    candidates = {key: value for key, value in candidates.items() if value is not None}
    if not candidates:
        return "none"
    return min(candidates, key=candidates.get) if lower_is_better else max(candidates, key=candidates.get)


def _per_layer(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(layer): _aggregate([sample for sample in samples if sample.get("layer_key") == layer])
        for layer in sorted({sample.get("layer_key") for sample in samples})
    }


def _per_category(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        str(category): _aggregate([sample for sample in samples if sample.get("prompt_category") == category])
        for category in sorted({sample.get("prompt_category") for sample in samples})
    }


def _grouped_results(samples: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "max_to_avg_bucket_ratio": _group_by_median(samples, "max_to_avg_bucket_ratio"),
        "top3_bucket_mass": _group_by_median(samples, "top3_bucket_mass"),
        "mean_bridge_score": _group_by_median(samples, "mean_bridge_score"),
        "active_destination_count": _group_by_median(samples, "active_destination_count"),
        "hot_bucket_count": _group_by_median(samples, "hot_bucket_count"),
    }
    groups["strongest_grouping_dimension"] = max(
        groups, key=lambda key: abs(groups[key]["full_minus_state_top1_hit_delta_higher_minus_lower"])
    )
    return groups


def _group_by_median(samples: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = sorted(float(sample.get(metric) or 0.0) for sample in samples)
    if not values:
        return {"metric": metric}
    midpoint = len(values) // 2
    threshold = values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2.0
    higher = [sample for sample in samples if float(sample.get(metric) or 0.0) >= threshold]
    lower = [sample for sample in samples if float(sample.get(metric) or 0.0) < threshold]
    higher_metrics = _metrics(higher)
    lower_metrics = _metrics(lower)
    return {
        "metric": metric,
        "median_threshold": threshold,
        "higher": higher_metrics,
        "lower": lower_metrics,
        "full_minus_state_top1_hit_delta_higher_minus_lower": (
            higher_metrics["full_top1_hit_rate"] - higher_metrics["state_top1_hit_rate"]
        )
        - (lower_metrics["full_top1_hit_rate"] - lower_metrics["state_top1_hit_rate"]),
        "dependency_minus_state_top1_hit_delta_higher_minus_lower": (
            higher_metrics["dependency_top1_hit_rate"] - higher_metrics["state_top1_hit_rate"]
        )
        - (lower_metrics["dependency_top1_hit_rate"] - lower_metrics["state_top1_hit_rate"]),
    }


def _compare_definitions(results: dict[str, Any]) -> dict[str, Any]:
    comparison = {}
    for version, payload in results.items():
        aggregate = payload["aggregate"]
        comparison[version] = {
            "state_top1_hit_rate": aggregate["state_top1_hit_rate"],
            "dependency_top1_hit_rate": aggregate["dependency_top1_hit_rate"],
            "full_top1_hit_rate": aggregate["full_top1_hit_rate"],
            "dependency_minus_state_top1_hit_rate": aggregate["dependency_top1_hit_rate"]
            - aggregate["state_top1_hit_rate"],
            "full_minus_state_top1_hit_rate": aggregate["full_top1_hit_rate"] - aggregate["state_top1_hit_rate"],
            "best_top1_proxy": aggregate["best_top1_proxy"],
        }
    comparison["definition_with_larger_dependency_state_gap"] = max(
        comparison,
        key=lambda version: comparison[version].get("dependency_minus_state_top1_hit_rate", -1.0)
        if isinstance(comparison[version], dict)
        else -1.0,
    )
    return comparison


def _conclusion(metrics: dict[str, Any]) -> str:
    best = metrics["best_top1_proxy"]
    if best == "full":
        return "full proxy is closest to this pseudo critical bucket by top-1 hit rate"
    if best == "dependency":
        return "dependency-only proxy is closest to this pseudo critical bucket by top-1 hit rate"
    return "state-only remains closest to this pseudo critical bucket by top-1 hit rate"


def _append_goal() -> None:
    if not GOAL_PATH.exists():
        return
    text = GOAL_PATH.read_text(encoding="utf-8")
    marker = "critical_bucket_proxy_v2"
    if marker in text:
        return
    GOAL_PATH.write_text(
        text
        + "\nCritical bucket proxy v2 compares state/dependency/full rankings against both structure-heavy and more load-centric pseudo critical bucket definitions.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
