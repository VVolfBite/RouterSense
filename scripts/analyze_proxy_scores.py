#!/usr/bin/env python3
"""Proxy comparison implementation used by experiment/poc1/proxy_compare.py.

Keep new user-facing POC1 entrypoints under experiment/poc1/.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from routesense_poc1.serialization import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare state-only, dependency-only, and full proxy bucket rankings.")
    parser.add_argument("--input-dir", type=str, default=str(ROOT / "outputs" / "poc1_trace_batch"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "poc1_proxy_compare"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--comment", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    trace_summary_path = input_dir / "trace_batch_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_goal(
        output_dir,
        "POC1 Proxy Compare",
        "Compare state-only, dependency-only, and full proxy bucket rankings from real OLMoE trace batch outputs.",
        args.comment,
    )
    trace_batch_summary = _load_json(trace_summary_path)
    sample_results: list[dict[str, Any]] = []

    for sample in trace_batch_summary.get("samples", []):
        sample_id = sample["sample_id"]
        summary_path = Path(sample["routing_summary_path"])
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        routing_summary = _load_json(summary_path)
        result = _analyze_sample(sample, routing_summary, max(1, args.top_k))
        sample_results.append(result)
        print(
            f"{sample_id}: state={result['state_top1_destination']} "
            f"dep={result['dependency_top1_destination']} "
            f"full={result['full_top1_destination']} "
            f"overlap@{args.top_k}={result['state_dependency_topk_overlap']}"
        )

    aggregate = _aggregate(sample_results)
    payload = {
        "mode": "trace_proxy_comparison",
        "purpose": "POC1-v4 layer-aware trace-based state-only vs dependency-aware proxy comparison",
        "score_version": "v4",
        "input_trace_batch_summary": str(trace_summary_path),
        "top_k": max(1, args.top_k),
        "comment": args.comment,
        "sample_count": len(sample_results),
        "layer_count": len({sample.get("layer_key") for sample in sample_results}),
        "samples": sample_results,
        "aggregate": aggregate,
        "per_layer": _per_layer(sample_results),
        "layer_focus": _layer_focus(sample_results),
    }
    save_json(payload, output_dir / "proxy_comparison_summary.json")
    print(f"wrote {output_dir / 'proxy_comparison_summary.json'}")
    return 0


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_goal(output_dir: Path, title: str, body: str, comment: str | None = None) -> None:
    text = f"# {title}\n\n{body}\n"
    if comment:
        text += f"\nComment: {comment}\n"
    (output_dir / "goal.md").write_text(text, encoding="utf-8")


def _analyze_sample(sample: dict[str, Any], routing_summary: dict[str, Any], top_k: int) -> dict[str, Any]:
    buckets = routing_summary.get("bucket_summaries") or routing_summary.get("buckets") or []
    source_matrix = [[int(dest) for dest in row] for row in routing_summary.get("source_destination_matrix", [])]
    active_destinations = [int(dest) for dest in routing_summary.get("active_destinations", [])]
    total_routes = int(routing_summary.get("total_selected_routes", 0))
    max_bucket_size = max((int(bucket.get("token_count_in_bucket", 0)) for bucket in buckets), default=1)
    avg_bucket_size = total_routes / max(1, len(buckets))
    max_rank = max((int(bucket.get("bucket_rank_by_size") or 1) for bucket in buckets), default=1)
    sequence_length = int(routing_summary.get("sequence_length") or len(source_matrix) or 1)
    bucket_scores = []

    for bucket in buckets:
        destination_id = int(bucket["destination_id"])
        state_score, state_features = _state_score(bucket, max_bucket_size, avg_bucket_size, max_rank, total_routes)
        dependency_score, dependency_features = _dependency_score(
            destination_id, bucket, source_matrix, active_destinations, sequence_length
        )
        full_score = 0.45 * state_score + 0.55 * dependency_score
        bucket_scores.append(
            {
                "destination_id": destination_id,
                "state_only_score": state_score,
                "dependency_only_score": dependency_score,
                "full_score": full_score,
                "state_features": state_features,
                "dependency_features": dependency_features,
            }
        )

    state_ranked = _rank(bucket_scores, "state_only_score")
    dependency_ranked = _rank(bucket_scores, "dependency_only_score")
    full_ranked = _rank(bucket_scores, "full_score")
    state_top = state_ranked[0]["destination_id"] if state_ranked else None
    dependency_top = dependency_ranked[0]["destination_id"] if dependency_ranked else None
    full_top = full_ranked[0]["destination_id"] if full_ranked else None
    state_topk = [item["destination_id"] for item in state_ranked[:top_k]]
    dependency_topk = [item["destination_id"] for item in dependency_ranked[:top_k]]
    full_topk = [item["destination_id"] for item in full_ranked[:top_k]]
    state_dep_overlap = len(set(state_topk) & set(dependency_topk))
    state_full_overlap = len(set(state_topk) & set(full_topk))

    return {
        "sample_id": sample["sample_id"],
        "base_sample_id": sample.get("base_sample_id"),
        "layer_key": sample.get("layer_key"),
        "layer_spec": sample.get("layer_spec"),
        "prompt_category": sample.get("prompt_category"),
        "text": sample.get("text"),
        "max_to_avg_bucket_ratio": sample.get("max_to_avg_bucket_ratio"),
        "bucket_token_count_gini": sample.get("bucket_token_count_gini"),
        "active_destination_count": sample.get("active_destination_count"),
        "hot_bucket_count": sample.get("hot_bucket_count"),
        "top3_bucket_mass": sum(sample.get("top3_bucket_token_counts") or []) / max(1, sample.get("total_selected_routes") or 0),
        "state_top1_destination": state_top,
        "dependency_top1_destination": dependency_top,
        "full_top1_destination": full_top,
        "state_dependency_top1_same": state_top == dependency_top,
        "full_state_top1_same": full_top == state_top,
        "state_topk_destinations": state_topk,
        "dependency_topk_destinations": dependency_topk,
        "full_topk_destinations": full_topk,
        "state_dependency_topk_overlap": state_dep_overlap,
        "state_full_topk_overlap": state_full_overlap,
        "state_dependency_topk_jaccard": _jaccard(state_topk, dependency_topk),
        "state_full_topk_jaccard": _jaccard(state_topk, full_topk),
        "ranking_consistent": state_top == dependency_top == full_top,
        "mean_bridge_score": statistics.fmean(
            item["dependency_features"]["bridge_score"] for item in bucket_scores
        )
        if bucket_scores
        else 0.0,
        "max_bridge_score": max((item["dependency_features"]["bridge_score"] for item in bucket_scores), default=0.0),
        "bucket_scores": bucket_scores,
    }


def _state_score(
    bucket: dict[str, Any],
    max_bucket_size: int,
    avg_bucket_size: float,
    max_rank: int,
    total_routes: int,
) -> tuple[float, dict[str, float]]:
    bucket_size = int(bucket.get("token_count_in_bucket", 0))
    rank = int(bucket.get("bucket_rank_by_size") or max_rank)
    size_norm = bucket_size / max(1, max_bucket_size)
    rank_norm = 1.0 - ((rank - 1) / max(1, max_rank - 1))
    hot = 1.0 if bool(bucket.get("is_hot_bucket")) else 0.0
    route_share = bucket_size / max(1, total_routes)
    density_over_mean = min(1.0, bucket_size / max(1.0, avg_bucket_size) / 3.0)
    score = 0.42 * size_norm + 0.23 * rank_norm + 0.15 * hot + 0.10 * route_share + 0.10 * density_over_mean
    return score, {
        "bucket_size_norm": size_norm,
        "inverse_size_rank_norm": rank_norm,
        "is_hot_bucket": hot,
        "selected_route_share": route_share,
        "density_over_mean_norm": density_over_mean,
    }


def _dependency_score(
    destination_id: int,
    bucket: dict[str, Any],
    source_matrix: list[list[int]],
    active_destinations: list[int],
    sequence_length: int,
) -> tuple[float, dict[str, float]]:
    rows_with_destination = [row for row in source_matrix if destination_id in row]
    source_count = len(source_matrix)
    active_count = len(active_destinations)
    peers: set[int] = set()
    coactive_events = 0
    for row in rows_with_destination:
        row_peers = {int(dest) for dest in row if int(dest) != destination_id}
        peers.update(row_peers)
        coactive_events += len(row_peers)

    source_coverage = len(rows_with_destination) / max(1, source_count)
    coactive_peer_degree = len(peers) / max(1, active_count - 1)
    max_peer_events = max(1, source_count * max(0, _max_row_width(source_matrix) - 1))
    coactive_event_density = coactive_events / max_peer_events
    token_positions = [int(pos) for pos in bucket.get("token_positions", [])]
    if len(token_positions) >= 2:
        position_spread = (max(token_positions) - min(token_positions)) / max(1, sequence_length - 1)
    else:
        position_spread = 0.0
    bridge_score = source_coverage * coactive_peer_degree
    score = (
        0.25 * source_coverage
        + 0.30 * coactive_peer_degree
        + 0.15 * coactive_event_density
        + 0.15 * position_spread
        + 0.15 * bridge_score
    )
    return score, {
        "source_coverage": source_coverage,
        "coactive_peer_degree": coactive_peer_degree,
        "coactive_event_density": coactive_event_density,
        "position_spread": position_spread,
        "bridge_score": bridge_score,
    }


def _max_row_width(source_matrix: list[list[int]]) -> int:
    return max((len(row) for row in source_matrix), default=1)


def _rank(bucket_scores: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    return sorted(bucket_scores, key=lambda item: (-float(item[score_key]), int(item["destination_id"])))


def _jaccard(left: list[int], right: list[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _aggregate(sample_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not sample_results:
        return {}
    state_dependency_diff = [sample for sample in sample_results if not sample["state_dependency_top1_same"]]
    full_state_diff = [sample for sample in sample_results if not sample["full_state_top1_same"]]
    skew_values = [float(sample.get("max_to_avg_bucket_ratio") or 0.0) for sample in sample_results]
    skew_threshold = statistics.median(skew_values)
    high_skew_samples = [sample for sample in sample_results if float(sample.get("max_to_avg_bucket_ratio") or 0.0) >= skew_threshold]
    low_skew_samples = [sample for sample in sample_results if float(sample.get("max_to_avg_bucket_ratio") or 0.0) < skew_threshold]
    divergence_scores = [
        {
            "sample_id": sample["sample_id"],
            "max_to_avg_bucket_ratio": sample.get("max_to_avg_bucket_ratio"),
            "state_dependency_topk_jaccard": sample["state_dependency_topk_jaccard"],
            "divergence_score": 1.0 - sample["state_dependency_topk_jaccard"],
            "state_top1_destination": sample["state_top1_destination"],
            "dependency_top1_destination": sample["dependency_top1_destination"],
            "full_top1_destination": sample["full_top1_destination"],
        }
        for sample in sample_results
    ]
    divergence_scores.sort(key=lambda item: (-item["divergence_score"], -float(item.get("max_to_avg_bucket_ratio") or 0.0)))
    high_skew_diff_rate = _diff_rate(high_skew_samples)
    low_skew_diff_rate = _diff_rate(low_skew_samples)
    top1_diff_rate = len(state_dependency_diff) / len(sample_results)
    grouped_summary = {
        "higher_skew": _group_summary(high_skew_samples),
        "lower_skew": _group_summary(low_skew_samples),
    }
    grouping_results = {
        "max_to_avg_bucket_ratio": _group_by_median(sample_results, "max_to_avg_bucket_ratio"),
        "active_destination_count": _group_by_median(sample_results, "active_destination_count"),
        "hot_bucket_count": _group_by_median(sample_results, "hot_bucket_count"),
        "mean_bridge_score": _group_by_median(sample_results, "mean_bridge_score"),
        "top3_bucket_mass": _group_by_median(sample_results, "top3_bucket_mass"),
    }
    high_skew_topk_jaccard = grouped_summary["higher_skew"]["mean_state_dependency_topk_jaccard"]
    low_skew_topk_jaccard = grouped_summary["lower_skew"]["mean_state_dependency_topk_jaccard"]

    if top1_diff_rate >= 0.5:
        strength = "strong"
        conclusion = "state-only and dependency-aware proxies often diverge; dependency structure appears nontrivial"
    elif top1_diff_rate > 0:
        strength = "moderate"
        conclusion = "state-only and dependency-aware proxies sometimes diverge; dependency structure is worth testing further"
    else:
        strength = "weak"
        conclusion = "state-only and dependency-aware top-1 proxies agree in this batch; inspect top-k divergence before expanding"
    skew_conclusion = (
        "divergence increases in higher-skew samples"
        if high_skew_diff_rate > low_skew_diff_rate or high_skew_topk_jaccard < low_skew_topk_jaccard
        else "divergence does not increase in higher-skew samples in this batch"
    )
    strongest_grouping = _strongest_grouping(grouping_results)
    grouping_recommendation = _grouping_recommendation(strongest_grouping, grouping_results[strongest_grouping])

    return {
        "state_dependency_top1_different_count": len(state_dependency_diff),
        "state_dependency_top1_different_rate": top1_diff_rate,
        "full_state_top1_different_count": len(full_state_diff),
        "full_state_top1_different_rate": len(full_state_diff) / len(sample_results),
        "mean_state_dependency_topk_jaccard": statistics.fmean(
            sample["state_dependency_topk_jaccard"] for sample in sample_results
        ),
        "mean_state_full_topk_jaccard": statistics.fmean(sample["state_full_topk_jaccard"] for sample in sample_results),
        "skew_threshold_median_max_to_avg_bucket_ratio": skew_threshold,
        "high_skew_state_dependency_top1_different_rate": high_skew_diff_rate,
        "low_skew_state_dependency_top1_different_rate": low_skew_diff_rate,
        "divergence_larger_in_high_skew_samples": high_skew_diff_rate > low_skew_diff_rate,
        "grouped_by_skew": grouped_summary,
        "grouped_results": grouping_results,
        "strongest_grouping_dimension": strongest_grouping,
        "grouping_focus_recommendation": grouping_recommendation,
        "largest_divergence_samples": divergence_scores[:5],
        "dependency_signal_strength": strength,
        "skew_conclusion": skew_conclusion,
        "structure_conclusion": f"largest grouped divergence contrast is under {strongest_grouping}",
        "mainline_should_continue": top1_diff_rate > 0 or statistics.fmean(
            sample["state_dependency_topk_jaccard"] for sample in sample_results
        )
        < 0.95,
        "short_conclusion": conclusion,
    }


def _diff_rate(samples: list[dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    return sum(1 for sample in samples if not sample["state_dependency_top1_same"]) / len(samples)


def _group_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "state_dependency_top1_different_rate": 0.0,
            "full_state_top1_different_rate": 0.0,
            "mean_state_dependency_topk_jaccard": None,
            "mean_state_full_topk_jaccard": None,
            "mean_max_to_avg_bucket_ratio": None,
        }
    return {
        "sample_count": len(samples),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "state_dependency_top1_different_rate": _diff_rate(samples),
        "full_state_top1_different_rate": sum(1 for sample in samples if not sample["full_state_top1_same"]) / len(samples),
        "mean_state_dependency_topk_jaccard": statistics.fmean(
            sample["state_dependency_topk_jaccard"] for sample in samples
        ),
        "mean_state_full_topk_jaccard": statistics.fmean(sample["state_full_topk_jaccard"] for sample in samples),
        "mean_max_to_avg_bucket_ratio": statistics.fmean(
            float(sample.get("max_to_avg_bucket_ratio") or 0.0) for sample in samples
        ),
    }


def _group_by_median(samples: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [float(sample.get(metric) or 0.0) for sample in samples]
    if not values:
        return {"metric": metric, "median_threshold": None, "higher": _group_summary([]), "lower": _group_summary([])}
    values_sorted = sorted(values)
    midpoint = len(values_sorted) // 2
    threshold = values_sorted[midpoint] if len(values_sorted) % 2 else (values_sorted[midpoint - 1] + values_sorted[midpoint]) / 2.0
    higher = [sample for sample in samples if float(sample.get(metric) or 0.0) >= threshold]
    lower = [sample for sample in samples if float(sample.get(metric) or 0.0) < threshold]
    higher_summary = _group_summary(higher)
    lower_summary = _group_summary(lower)
    return {
        "metric": metric,
        "median_threshold": threshold,
        "higher": higher_summary,
        "lower": lower_summary,
        "top1_diff_rate_delta_higher_minus_lower": higher_summary["state_dependency_top1_different_rate"]
        - lower_summary["state_dependency_top1_different_rate"],
        "topk_jaccard_delta_higher_minus_lower": (
            higher_summary["mean_state_dependency_topk_jaccard"] or 0.0
        )
        - (lower_summary["mean_state_dependency_topk_jaccard"] or 0.0),
    }


def _strongest_grouping(grouped_results: dict[str, Any]) -> str:
    if not grouped_results:
        return "none"
    return max(
        grouped_results,
        key=lambda key: abs(float(grouped_results[key].get("top1_diff_rate_delta_higher_minus_lower") or 0.0)),
    )


def _per_layer(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for layer_key in sorted({sample.get("layer_key") for sample in samples}):
        layer_samples = [sample for sample in samples if sample.get("layer_key") == layer_key]
        result[str(layer_key)] = _aggregate(layer_samples)
    return result


def _layer_focus(samples: list[dict[str, Any]]) -> dict[str, Any]:
    per_layer = _per_layer(samples)
    if not per_layer:
        return {}
    layer_scores = {
        layer: _layer_signal_score(summary)
        for layer, summary in per_layer.items()
    }
    most = max(layer_scores, key=layer_scores.get)
    least = min(layer_scores, key=layer_scores.get)
    return {
        "most_informative_layer": most,
        "most_informative_layer_score": layer_scores[most],
        "least_informative_layer": least,
        "least_informative_layer_score": layer_scores[least],
        "layer_signal_scores": layer_scores,
        "layer_focus_recommendation": (
            f"Prioritize {most}; it shows the strongest dependency-aware ranking divergence. "
            f"Deprioritize {least} for near-term proxy experiments."
        ),
    }


def _layer_signal_score(summary: dict[str, Any]) -> float:
    top1_diff = float(summary.get("state_dependency_top1_different_rate") or 0.0)
    topk_divergence = 1.0 - float(summary.get("mean_state_dependency_topk_jaccard") or 1.0)
    full_shift = float(summary.get("full_state_top1_different_rate") or 0.0)
    return 0.50 * top1_diff + 0.35 * topk_divergence + 0.15 * full_shift


def _grouping_recommendation(grouping: str, summary: dict[str, Any]) -> str:
    delta = float(summary.get("top1_diff_rate_delta_higher_minus_lower") or 0.0)
    if delta > 0:
        side = "higher"
    elif delta < 0:
        side = "lower"
    else:
        side = "neither"
    return (
        f"{grouping} currently gives the largest top-1 divergence contrast; "
        f"the {side} side shows more state/dependency separation."
    )


if __name__ == "__main__":
    raise SystemExit(main())
