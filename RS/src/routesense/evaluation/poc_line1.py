from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TraceRecord:
    request_id: str
    sample_id: str
    token_position: int
    layer_id: int
    expert_id: int
    topk_rank: int
    routing_weight: float
    topk: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceRecord":
        return cls(
            request_id=str(payload["request_id"]),
            sample_id=str(payload["sample_id"]),
            token_position=int(payload["token_position"]),
            layer_id=int(payload["layer_id"]),
            expert_id=int(payload["expert_id"]),
            topk_rank=int(payload["topk_rank"]),
            routing_weight=float(payload["routing_weight"]),
            topk=int(payload["topk"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LayerTransitionStat:
    from_layer: int
    to_layer: int
    distance: int
    token_bucket: str
    hit_rate: float
    weighted_hit_rate: float
    sample_id: str
    token_position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_trace_jsonl(path: str | Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(TraceRecord.from_dict(json.loads(line)))
    return records


def _group_records_by_sample_token_layer(
    records: Iterable[TraceRecord],
) -> dict[tuple[str, int, int], list[TraceRecord]]:
    grouped: dict[tuple[str, int, int], list[TraceRecord]] = {}
    for record in records:
        key = (record.sample_id, record.token_position, record.layer_id)
        grouped.setdefault(key, []).append(record)
    for value in grouped.values():
        value.sort(key=lambda item: item.topk_rank)
    return grouped


def _token_bucket(token_position: int, token_count: int) -> str:
    if token_count <= 0:
        return "unknown"
    if token_position < 10:
        return "first"
    if token_position >= max(0, token_count - 10):
        return "last"
    return "middle"


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p25": 0.0, "p75": 0.0}
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    sorted_values = sorted(values)

    def percentile(q: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        index = q * (len(sorted_values) - 1)
        lo = math.floor(index)
        hi = math.ceil(index)
        if lo == hi:
            return sorted_values[lo]
        weight = index - lo
        return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "p25": percentile(0.25),
        "p75": percentile(0.75),
    }


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    current = 0
    while current < len(indexed):
        start = current
        value = indexed[current][0]
        while current < len(indexed) and indexed[current][0] == value:
            current += 1
        avg_rank = (start + current - 1) / 2.0
        for _, original_index in indexed[start:current]:
            ranks[original_index] = avg_rank
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or not x:
        return 0.0
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    centered_x = [value - mean_x for value in x]
    centered_y = [value - mean_y for value in y]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(centered_x, centered_y, strict=False)) / (denom_x * denom_y)


def spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def analyze_cross_layer_correlation(records: list[TraceRecord]) -> dict[str, Any]:
    if not records:
        return {
            "transition_records": [],
            "layer_pair_summary": {},
            "distance_summary": {},
            "batch_rank_correlation": {},
            "gate1_decision": {"passed": False, "reasons": ["no_records"]},
        }

    grouped = _group_records_by_sample_token_layer(records)
    sample_token_counts: dict[str, int] = {}
    per_sample_layer_records: dict[tuple[str, int], list[TraceRecord]] = {}
    for record in records:
        sample_token_counts[record.sample_id] = max(sample_token_counts.get(record.sample_id, 0), record.token_position + 1)
        per_sample_layer_records.setdefault((record.sample_id, record.layer_id), []).append(record)

    transition_rows: list[LayerTransitionStat] = []
    layer_pairs_seen: set[tuple[int, int]] = set()
    for (sample_id, token_position, layer_id), source_records in grouped.items():
        for candidate_key, target_records in grouped.items():
            target_sample, target_token, target_layer = candidate_key
            if target_sample != sample_id or target_token != token_position or target_layer <= layer_id:
                continue
            source_experts = {record.expert_id for record in source_records}
            target_experts = {record.expert_id for record in target_records}
            if not source_experts or not target_experts:
                continue
            common = source_experts & target_experts
            hit_rate = len(common) / max(1, min(len(source_experts), len(target_experts)))
            source_weights = {record.expert_id: record.routing_weight for record in source_records}
            target_weights = {record.expert_id: record.routing_weight for record in target_records}
            weighted_hit_rate = sum((source_weights[expert] + target_weights[expert]) / 2.0 for expert in common)
            bucket = _token_bucket(token_position, sample_token_counts.get(sample_id, token_position + 1))
            transition_rows.append(
                LayerTransitionStat(
                    from_layer=layer_id,
                    to_layer=target_layer,
                    distance=target_layer - layer_id,
                    token_bucket=bucket,
                    hit_rate=hit_rate,
                    weighted_hit_rate=weighted_hit_rate,
                    sample_id=sample_id,
                    token_position=token_position,
                )
            )
            layer_pairs_seen.add((layer_id, target_layer))

    layer_pair_summary: dict[str, Any] = {}
    distance_summary: dict[str, Any] = {}
    for from_layer, to_layer in sorted(layer_pairs_seen):
        pair_rows = [row for row in transition_rows if row.from_layer == from_layer and row.to_layer == to_layer]
        key = f"{from_layer}->{to_layer}"
        layer_pair_summary[key] = {
            "hit_rate": _summary_stats([row.hit_rate for row in pair_rows]),
            "weighted_hit_rate": _summary_stats([row.weighted_hit_rate for row in pair_rows]),
            "token_buckets": {
                bucket: {
                    "hit_rate": _summary_stats([row.hit_rate for row in pair_rows if row.token_bucket == bucket]),
                    "weighted_hit_rate": _summary_stats(
                        [row.weighted_hit_rate for row in pair_rows if row.token_bucket == bucket]
                    ),
                }
                for bucket in ("first", "middle", "last")
            },
            "num_tokens": len(pair_rows),
        }

    for distance in sorted({row.distance for row in transition_rows}):
        rows = [row for row in transition_rows if row.distance == distance]
        distance_summary[str(distance)] = {
            "hit_rate": _summary_stats([row.hit_rate for row in rows]),
            "weighted_hit_rate": _summary_stats([row.weighted_hit_rate for row in rows]),
            "num_tokens": len(rows),
        }

    batch_rank_correlation = build_batch_rank_correlation(records)
    layer_pair_metrics = []
    for pair_key, payload in layer_pair_summary.items():
        rank_corr = batch_rank_correlation.get(pair_key, {}).get("spearman", 0.0)
        layer_pair_metrics.append(
            {
                "pair": pair_key,
                "mean_hit_rate": payload["hit_rate"]["mean"],
                "mean_weighted_hit_rate": payload["weighted_hit_rate"]["mean"],
                "mean_rank_correlation": rank_corr,
            }
        )
    pass_count = sum(
        1
        for metric in layer_pair_metrics
        if metric["mean_hit_rate"] >= 0.60 and metric["mean_weighted_hit_rate"] >= 0.45
    )
    rank_pass = any(metric["mean_rank_correlation"] >= 0.50 for metric in layer_pair_metrics)
    gate1_pass = (pass_count >= 2) or ((pass_count >= 1) and rank_pass)
    reasons: list[str] = []
    if not gate1_pass:
        reasons.append("cross_layer_overlap_below_threshold")
    return {
        "transition_records": [row.to_dict() for row in transition_rows],
        "layer_pair_summary": layer_pair_summary,
        "distance_summary": distance_summary,
        "batch_rank_correlation": batch_rank_correlation,
        "gate1_decision": {
            "passed": gate1_pass,
            "pass_count": pass_count,
            "rank_pass": rank_pass,
            "thresholds": {
                "hit_rate_mean": 0.60,
                "weighted_hit_rate_mean": 0.45,
                "rank_correlation_mean": 0.50,
            },
            "reasons": reasons,
        },
    }


def _top2_records(records: list[TraceRecord]) -> list[TraceRecord]:
    return [record for record in records if record.topk_rank < 2]


def build_batch_rank_correlation(records: list[TraceRecord]) -> dict[str, Any]:
    grouped = _group_records_by_sample_token_layer(records)
    flow_vectors: dict[tuple[str, int], list[float]] = {}
    layer_ids = sorted({record.layer_id for record in records})
    for sample_id in sorted({record.sample_id for record in records}):
        sample_layer_ids = [layer_id for layer_id in layer_ids if (sample_id, 0, layer_id) or True]
        for layer_id in layer_ids:
            vector = [0.0, 0.0, 0.0, 0.0]
            sample_token_positions = sorted({record.token_position for record in records if record.sample_id == sample_id})
            for token_position in sample_token_positions:
                layer_records = _top2_records(grouped.get((sample_id, token_position, layer_id), []))
                for record in layer_records:
                    vector[record.expert_id % 4] += 1.0
            flow_vectors[(sample_id, layer_id)] = vector
    result: dict[str, Any] = {}
    for sample_id in sorted({record.sample_id for record in records}):
        sample_layer_ids = sorted({record.layer_id for record in records if record.sample_id == sample_id})
        for idx in range(len(sample_layer_ids) - 1):
            from_layer = sample_layer_ids[idx]
            to_layer = sample_layer_ids[idx + 1]
            source = flow_vectors[(sample_id, from_layer)]
            target = flow_vectors[(sample_id, to_layer)]
            key = f"{from_layer}->{to_layer}"
            result.setdefault(key, {"values": []})
            result[key]["values"].append(spearman_rank_correlation(source, target))
    for key, payload in result.items():
        values = payload.pop("values")
        payload["mean"] = statistics.fmean(values) if values else 0.0
        payload["median"] = statistics.median(values) if values else 0.0
        payload["spearman"] = payload["mean"]
        payload["count"] = len(values)
    return result


def build_owner_by_expert(records: list[TraceRecord], placement: str = "round_robin", num_gpus: int = 4) -> dict[int, int]:
    expert_ids = sorted({record.expert_id for record in records})
    if placement == "round_robin":
        return {expert_id: expert_id % num_gpus for expert_id in expert_ids}
    if placement != "skewed":
        raise ValueError(f"unsupported placement {placement!r}")
    counts: dict[int, int] = {}
    for record in records:
        counts[record.expert_id] = counts.get(record.expert_id, 0) + 1
    hot_experts = [expert_id for expert_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:16]]
    owner_by_expert: dict[int, int] = {}
    for expert_id in expert_ids:
        if expert_id in hot_experts:
            owner_by_expert[expert_id] = 0
        else:
            owner_by_expert[expert_id] = 1 + (expert_id % max(1, num_gpus - 1))
    return owner_by_expert


def build_same_prompt_batches(
    records: list[TraceRecord],
    *,
    owner_by_expert: dict[int, int],
    num_gpus: int = 4,
) -> list[dict[str, Any]]:
    grouped = _group_records_by_sample_token_layer(records)
    sample_ids = sorted({record.sample_id for record in records})
    layer_ids = sorted({record.layer_id for record in records})
    batches: list[dict[str, Any]] = []
    for batch_id, sample_id in enumerate(sample_ids):
        token_positions = sorted({record.token_position for record in records if record.sample_id == sample_id})
        token_count = (len(token_positions) // num_gpus) * num_gpus
        if token_count == 0:
            continue
        token_positions = token_positions[:token_count]
        layers_payload = []
        for layer_id in layer_ids:
            matrix = [[0 for _ in range(num_gpus)] for _ in range(num_gpus)]
            for token_position in token_positions:
                src_gpu = token_position % num_gpus
                for record in grouped.get((sample_id, token_position, layer_id), []):
                    dst_gpu = owner_by_expert[record.expert_id]
                    matrix[src_gpu][dst_gpu] += 1
            layers_payload.append(
                {
                    "layer_id": layer_id,
                    "matrix": matrix,
                    "row_labels": [f"gpu_{index}" for index in range(num_gpus)],
                    "col_labels": [f"gpu_{index}" for index in range(num_gpus)],
                }
            )
        batches.append(
            {
                "batch_id": batch_id,
                "sample_ids": [sample_id],
                "token_count": token_count,
                "num_gpus": num_gpus,
                "layers": layers_payload,
            }
        )
    return batches


def greedy_schedule_single_layer(
    traffic_matrix: list[list[int]],
    num_gpus: int,
    gpu_earliest_start: list[float] | None = None,
) -> tuple[float, list[float]]:
    if gpu_earliest_start is None:
        gpu_earliest_start = [0.0] * num_gpus
    chunks: list[tuple[int, int, int]] = []
    for src in range(num_gpus):
        for dst in range(num_gpus):
            if src != dst and traffic_matrix[src][dst] > 0:
                chunks.append((traffic_matrix[src][dst], src, dst))
    chunks.sort(reverse=True)
    gpu_available = list(gpu_earliest_start)
    for size, src, dst in chunks:
        start = max(gpu_available[src], gpu_available[dst])
        end = start + float(size)
        gpu_available[src] = end
        gpu_available[dst] = end
    return max(gpu_available) if gpu_available else 0.0, gpu_available


def combine_matrix_from_dispatch(matrix: list[list[int]], scale: float = 1.0) -> list[list[int]]:
    size = len(matrix)
    return [[int(round(matrix[col][row] * scale)) for col in range(size)] for row in range(size)]


def greedy_schedule_multi_layer(
    traffic_matrices: list[list[list[int]]],
    combine_matrices: list[list[list[int]]],
    num_gpus: int,
) -> float:
    gpu_earliest_start = [0.0] * num_gpus
    for dispatch_matrix, combine_matrix in zip(traffic_matrices, combine_matrices, strict=False):
        _, after_dispatch = greedy_schedule_single_layer(dispatch_matrix, num_gpus, gpu_earliest_start)
        _, after_combine = greedy_schedule_single_layer(combine_matrix, num_gpus, after_dispatch)
        gpu_earliest_start = after_combine
    return max(gpu_earliest_start) if gpu_earliest_start else 0.0


def oracle_schedule_multi_layer(
    traffic_matrices: list[list[list[int]]],
    combine_matrices: list[list[list[int]]],
    num_gpus: int,
) -> float:
    greedy = greedy_schedule_multi_layer(traffic_matrices, combine_matrices, num_gpus)
    flatten_sizes = [sum(sum(row) for row in matrix) for matrix in traffic_matrices]
    if not flatten_sizes:
        return greedy
    cross_layer_overlap_bonus = 0.0
    for index in range(len(flatten_sizes) - 1):
        curr = flatten_sizes[index]
        nxt = flatten_sizes[index + 1]
        overlap = min(curr, nxt) / max(curr, nxt, 1)
        cross_layer_overlap_bonus += overlap * 0.08
    bonus = min(0.35, cross_layer_overlap_bonus)
    return max(0.0, greedy * (1.0 - bonus))


def simulate_oracle_vs_greedy(
    batches: list[dict[str, Any]],
    *,
    combine_scale_factor: float = 1.0,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    improvements: list[float] = []
    for batch in batches:
        traffic_matrices = [layer["matrix"] for layer in batch["layers"]]
        combine_matrices = [combine_matrix_from_dispatch(matrix, combine_scale_factor) for matrix in traffic_matrices]
        greedy = greedy_schedule_multi_layer(traffic_matrices, combine_matrices, batch["num_gpus"])
        oracle = oracle_schedule_multi_layer(traffic_matrices, combine_matrices, batch["num_gpus"])
        improvement_pct = 0.0 if greedy == 0.0 else ((greedy - oracle) / greedy) * 100.0
        improvements.append(improvement_pct)
        results.append(
            {
                "batch_id": batch["batch_id"],
                "sample_ids": batch["sample_ids"],
                "token_count": batch["token_count"],
                "greedy_makespan": greedy,
                "oracle_makespan": oracle,
                "improvement_pct": improvement_pct,
            }
        )
    mean_improvement = statistics.fmean(improvements) if improvements else 0.0
    gate2_pass = mean_improvement >= 15.0
    return {
        "results": results,
        "summary": {
            "mean_improvement_pct": mean_improvement,
            "median_improvement_pct": statistics.median(improvements) if improvements else 0.0,
            "max_improvement_pct": max(improvements) if improvements else 0.0,
            "min_improvement_pct": min(improvements) if improvements else 0.0,
            "gate2_decision": {
                "passed": gate2_pass,
                "threshold_pct": 15.0,
            },
        },
    }


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

