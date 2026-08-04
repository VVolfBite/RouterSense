from __future__ import annotations

from collections import defaultdict
from statistics import mean, median

from ..core.schemas import AblationRecord


def _group_by_window_layer(records: list[AblationRecord]) -> dict[tuple[int, int], list[AblationRecord]]:
    grouped: dict[tuple[int, int], list[AblationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.window_id, record.layer_id)].append(record)
    return grouped


def _pairwise_accuracy(records: list[AblationRecord]) -> dict[str, float | int | None]:
    total = 0
    correct = 0
    skipped = 0
    truth_order = sorted(records, key=lambda record: (record.delta_nll, record.expert_id))
    predicted_order = sorted(records, key=lambda record: (record.router_probability, record.expert_id))
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
    }


def analyze_by_layer(records: list[AblationRecord]) -> dict[int, dict]:
    grouped_by_layer: dict[int, list[AblationRecord]] = defaultdict(list)
    groups_by_layer: dict[int, list[list[AblationRecord]]] = defaultdict(list)
    for group_records in _group_by_window_layer(records).values():
        layer_id = group_records[0].layer_id
        groups_by_layer[layer_id].append(group_records)
        grouped_by_layer[layer_id].extend(group_records)
    payload: dict[int, dict] = {}
    for layer_id, layer_records in grouped_by_layer.items():
        pairwise_total = 0
        pairwise_correct = 0
        pairwise_tie_skips = 0
        for group_records in groups_by_layer[layer_id]:
            pairwise = _pairwise_accuracy(group_records)
            pairwise_total += int(pairwise["pairwise_total"] or 0)
            pairwise_correct += int(pairwise["pairwise_correct"] or 0)
            pairwise_tie_skips += int(pairwise["pairwise_tie_skips"] or 0)
        deltas = [record.delta_nll for record in layer_records]
        payload[layer_id] = {
            "pairwise_accuracy": (pairwise_correct / pairwise_total) if pairwise_total else None,
            "pairwise_total": pairwise_total,
            "pairwise_tie_skips": pairwise_tie_skips,
            "mean_delta_nll": mean(deltas),
            "median_delta_nll": median(deltas),
            "num_records": len(layer_records),
            "num_groups": len(groups_by_layer[layer_id]),
        }
    return payload


def analyze_by_expert(records: list[AblationRecord]) -> dict[int, dict]:
    grouped: dict[int, list[AblationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.expert_id].append(record)
    payload: dict[int, dict] = {}
    for expert_id, expert_records in grouped.items():
        deltas = [record.delta_nll for record in expert_records]
        rank_distribution: dict[int, int] = defaultdict(int)
        layer_distribution: dict[int, int] = defaultdict(int)
        for record in expert_records:
            rank_distribution[record.topk_rank] += 1
            layer_distribution[record.layer_id] += 1
        payload[expert_id] = {
            "mean_delta_nll": mean(deltas),
            "median_delta_nll": median(deltas),
            "min_delta_nll": min(deltas),
            "max_delta_nll": max(deltas),
            "count": len(expert_records),
            "rank_distribution": dict(sorted(rank_distribution.items())),
            "layer_distribution": dict(sorted(layer_distribution.items())),
        }
    return payload


def analyze_oracle_subset(records: list[AblationRecord], top_k: int = 32) -> dict:
    grouped = _group_by_window_layer(records)
    groups = sorted(
        grouped.values(),
        key=lambda rows: min(record.delta_nll for record in rows),
    )
    selected_groups = groups[:top_k]
    pairwise_total = 0
    pairwise_correct = 0
    pairwise_tie_skips = 0
    oracle_deltas: list[float] = []
    random_deltas: list[float] = []
    for group_records in selected_groups:
        pairwise = _pairwise_accuracy(group_records)
        pairwise_total += int(pairwise["pairwise_total"] or 0)
        pairwise_correct += int(pairwise["pairwise_correct"] or 0)
        pairwise_tie_skips += int(pairwise["pairwise_tie_skips"] or 0)
        oracle_deltas.append(min(record.delta_nll for record in group_records))
        random_deltas.append(mean(record.delta_nll for record in group_records))
    return {
        "selected_group_count": len(selected_groups),
        "pairwise_accuracy": (pairwise_correct / pairwise_total) if pairwise_total else None,
        "pairwise_total": pairwise_total,
        "pairwise_tie_skips": pairwise_tie_skips,
        "oracle_mean_delta_nll": mean(oracle_deltas) if oracle_deltas else None,
        "random_mean_delta_nll": mean(random_deltas) if random_deltas else None,
    }


def compute_rank_layer_heatmap(records: list[AblationRecord]) -> list[list[float | None]]:
    max_layer = max(record.layer_id for record in records)
    max_rank = max(record.topk_rank for record in records)
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record.layer_id, record.topk_rank)].append(record.delta_nll)
    matrix: list[list[float | None]] = []
    for layer_id in range(max_layer + 1):
        row: list[float | None] = []
        for rank in range(max_rank + 1):
            deltas = grouped.get((layer_id, rank), [])
            row.append(mean(deltas) if deltas else None)
        matrix.append(row)
    return matrix
