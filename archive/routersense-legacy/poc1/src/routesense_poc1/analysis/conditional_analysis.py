from __future__ import annotations

from collections import defaultdict
from statistics import mean, median

from ..core.schemas import AblationRecord


def _group_records(records: list[AblationRecord]) -> dict[tuple[int, int], list[AblationRecord]]:
    grouped: dict[tuple[int, int], list[AblationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.window_id, record.layer_id)].append(record)
    return grouped


def _pairwise_accuracy(records: list[AblationRecord]) -> dict[str, float | int | None]:
    grouped = _group_records(records)
    total = 0
    correct = 0
    skipped = 0
    for group_records in grouped.values():
        truth_order = sorted(group_records, key=lambda record: (record.delta_nll, record.expert_id))
        predicted_order = sorted(group_records, key=lambda record: (record.router_probability, record.expert_id))
        predicted_index = {id(record): index for index, record in enumerate(predicted_order)}
        for left_index in range(len(truth_order)):
            for right_index in range(left_index + 1, len(truth_order)):
                left = truth_order[left_index]
                right = truth_order[right_index]
                if abs(left.delta_nll - right.delta_nll) < 1e-6:
                    skipped += 1
                    continue
                total += 1
                if predicted_index[id(left)] < predicted_index[id(right)]:
                    correct += 1
    return {
        "pairwise_accuracy": (correct / total) if total else None,
        "pairwise_total": total,
        "pairwise_correct": correct,
        "pairwise_tie_skips": skipped,
        "num_records": len(records),
        "num_groups": len(grouped),
    }


def _split_by_group_stat(
    records: list[AblationRecord],
    stat_fn,
    label_low: str,
    label_high: str,
) -> dict[str, list[AblationRecord]]:
    grouped = _group_records(records)
    group_stats = []
    for key, rows in grouped.items():
        group_stats.append((key, stat_fn(rows)))
    values = sorted(value for _, value in group_stats)
    if not values:
        return {label_low: [], label_high: []}
    median_value = values[len(values) // 2]
    buckets = {label_low: [], label_high: []}
    for key, value in group_stats:
        target = label_high if value >= median_value else label_low
        buckets[target].extend(grouped[key])
    return buckets


def split_by_context_entropy(records: list[AblationRecord]) -> dict[str, dict]:
    buckets = _split_by_group_stat(
        records,
        stat_fn=lambda rows: mean(record.routing_entropy for record in rows),
        label_low="low_entropy",
        label_high="high_entropy",
    )
    return {name: _pairwise_accuracy(bucket_records) for name, bucket_records in buckets.items()}


def split_by_top1_dominance(records: list[AblationRecord]) -> dict[str, dict]:
    buckets = _split_by_group_stat(
        records,
        stat_fn=lambda rows: mean(record.top1_top2_gap for record in rows),
        label_low="small_gap",
        label_high="large_gap",
    )
    return {name: _pairwise_accuracy(bucket_records) for name, bucket_records in buckets.items()}


def split_by_delta_magnitude(records: list[AblationRecord], threshold: float = 0.05) -> dict[str, dict]:
    grouped = _group_records(records)
    selected: list[AblationRecord] = []
    rejected: list[AblationRecord] = []
    for rows in grouped.values():
        oracle_delta = min(record.delta_nll for record in rows)
        if abs(oracle_delta) >= threshold:
            selected.extend(rows)
        else:
            rejected.extend(rows)
    return {
        "large_magnitude_groups": _pairwise_accuracy(selected),
        "small_magnitude_groups": _pairwise_accuracy(rejected),
        "threshold": {"value": threshold},
    }


def analyze_expert_specialization(records: list[AblationRecord]) -> dict[int, dict]:
    grouped: dict[int, list[AblationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.expert_id].append(record)
    payload: dict[int, dict] = {}
    for expert_id, expert_records in grouped.items():
        deltas = [record.delta_nll for record in expert_records]
        payload[expert_id] = {
            "mean_delta_nll": mean(deltas),
            "median_delta_nll": median(deltas),
            "min_delta_nll": min(deltas),
            "max_delta_nll": max(deltas),
            "count": len(expert_records),
            "layer_ids": sorted({record.layer_id for record in expert_records}),
            "topk_ranks": sorted({record.topk_rank for record in expert_records}),
        }
    return payload
