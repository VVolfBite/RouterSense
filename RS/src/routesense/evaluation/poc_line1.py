from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore


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


@dataclass(frozen=True)
class GatePredictionStat:
    sample_id: str
    from_layer: int
    to_layer: int
    token_position: int
    cosine_similarity: float
    prefetch_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairwiseChunk:
    chunk_id: str
    phase: int
    size: int
    src_gpu: int
    dst_gpu: int

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


def load_hidden_state_bundle(path: str | Path) -> dict[str, dict[int, torch.Tensor]]:
    payload = torch.load(Path(path), map_location="cpu")
    return {
        str(sample_id): {int(layer_id): tensor.float().cpu() for layer_id, tensor in layer_map.items()}
        for sample_id, layer_map in payload.items()
    }


def load_gate_weight_bundle(path: str | Path) -> dict[str, dict[int, torch.Tensor]]:
    payload = torch.load(Path(path), map_location="cpu")
    return {
        str(sample_id): {int(layer_id): tensor.float().cpu() for layer_id, tensor in layer_map.items()}
        for sample_id, layer_map in payload.items()
    }


def _decode_predicted_topk_by_sample(
    *,
    hidden_states_by_sample: dict[str, dict[int, torch.Tensor]],
    gate_weights_by_sample: dict[str, dict[int, torch.Tensor]],
    topk: int,
) -> dict[str, dict[tuple[int, int], list[list[int]]]]:
    predicted: dict[str, dict[tuple[int, int], list[list[int]]]] = {}
    for sample_id, sample_hidden in hidden_states_by_sample.items():
        sample_gates = gate_weights_by_sample[sample_id]
        sample_layer_ids = sorted(layer_id for layer_id in sample_hidden if layer_id in sample_gates)
        sample_payload: dict[tuple[int, int], list[list[int]]] = {}
        for idx in range(len(sample_layer_ids) - 1):
            from_layer = sample_layer_ids[idx]
            to_layer = sample_layer_ids[idx + 1]
            hidden_from = sample_hidden[from_layer].squeeze(0)
            gate_weight = sample_gates[to_layer]
            predicted_logits = hidden_from @ gate_weight.T
            predicted_topk = torch.topk(torch.softmax(predicted_logits, dim=-1), k=topk, dim=-1).indices
            sample_payload[(from_layer, to_layer)] = predicted_topk.tolist()
        predicted[sample_id] = sample_payload
    return predicted


def analyze_cross_layer_predictability(
    records: list[TraceRecord],
    *,
    hidden_states_by_sample: dict[str, dict[int, torch.Tensor]],
    gate_weights_by_sample: dict[str, dict[int, torch.Tensor]],
    topk: int,
    owner_by_expert: dict[int, int] | None = None,
    num_gpus: int = 4,
) -> dict[str, Any]:
    grouped = _group_records_by_sample_token_layer(records)
    prediction_rows: list[GatePredictionStat] = []
    sample_ids = sorted({record.sample_id for record in records})
    layer_ids = sorted({record.layer_id for record in records})
    for sample_id in sample_ids:
        sample_hidden = hidden_states_by_sample[sample_id]
        sample_gates = gate_weights_by_sample[sample_id]
        sample_layer_ids = [layer_id for layer_id in layer_ids if layer_id in sample_hidden and layer_id in sample_gates]
        for idx in range(len(sample_layer_ids) - 1):
            from_layer = sample_layer_ids[idx]
            to_layer = sample_layer_ids[idx + 1]
            hidden_from = sample_hidden[from_layer].squeeze(0)
            hidden_to = sample_hidden[to_layer].squeeze(0)
            gate_weight = sample_gates[to_layer]
            if hidden_from.ndim != 2 or hidden_to.ndim != 2:
                raise RuntimeError(f"unexpected hidden state rank for sample={sample_id}, pair={from_layer}->{to_layer}")
            cosine = F.cosine_similarity(hidden_from, hidden_to, dim=-1)
            predicted_logits = hidden_from @ gate_weight.T
            predicted_topk = torch.topk(torch.softmax(predicted_logits, dim=-1), k=topk, dim=-1).indices
            for token_position in range(hidden_from.shape[0]):
                actual_records = grouped.get((sample_id, token_position, to_layer), [])
                actual_topk = [record.expert_id for record in actual_records[:topk]]
                if len(actual_topk) < topk:
                    continue
                overlap = len(set(predicted_topk[token_position].tolist()) & set(actual_topk))
                prediction_rows.append(
                    GatePredictionStat(
                        sample_id=sample_id,
                        from_layer=from_layer,
                        to_layer=to_layer,
                        token_position=token_position,
                        cosine_similarity=float(cosine[token_position].item()),
                        prefetch_accuracy=float(overlap / topk),
                    )
                )

    pair_summary: dict[str, Any] = {}
    for from_layer in layer_ids:
        for to_layer in layer_ids:
            if to_layer <= from_layer:
                continue
            rows = [row for row in prediction_rows if row.from_layer == from_layer and row.to_layer == to_layer]
            if not rows:
                continue
            pair_summary[f"{from_layer}->{to_layer}"] = {
                "prefetch_accuracy": _summary_stats([row.prefetch_accuracy for row in rows]),
                "cosine_similarity": _summary_stats([row.cosine_similarity for row in rows]),
                "num_tokens": len(rows),
            }

    batch_rank_correlation = build_batch_rank_correlation(
        records,
        owner_by_expert=owner_by_expert,
        num_gpus=num_gpus,
    )
    pass_count = sum(
        1
        for payload in pair_summary.values()
        if payload["prefetch_accuracy"]["mean"] >= 0.70 and payload["cosine_similarity"]["mean"] >= 0.70
    )
    rank_pass = any(value.get("spearman", 0.0) >= 0.50 for value in batch_rank_correlation.values())
    gate1_pass = pass_count >= 1
    reasons: list[str] = []
    if not gate1_pass:
        reasons.append("cross_layer_prefetch_accuracy_below_threshold")
    return {
        "prediction_rows": [row.to_dict() for row in prediction_rows],
        "layer_pair_summary": pair_summary,
        "batch_rank_correlation": batch_rank_correlation,
        "gate1_decision": {
            "passed": gate1_pass,
            "pass_count": pass_count,
            "rank_pass": rank_pass,
            "thresholds": {
                "prefetch_accuracy_mean": 0.70,
                "cosine_similarity_mean": 0.70,
                "rank_correlation_mean": 0.50,
            },
            "reasons": reasons,
        },
    }


def analyze_cross_layer_correlation(
    records: list[TraceRecord],
    *,
    owner_by_expert: dict[int, int] | None = None,
    num_gpus: int = 4,
) -> dict[str, Any]:
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

    batch_rank_correlation = build_batch_rank_correlation(
        records,
        owner_by_expert=owner_by_expert,
        num_gpus=num_gpus,
    )
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
    per_prompt_gate1: dict[str, Any] = {}
    for sample_id in sorted({record.sample_id for record in records}):
        sample_records = [record for record in records if record.sample_id == sample_id]
        sample_grouped = _group_records_by_sample_token_layer(sample_records)
        sample_transition_rows: list[LayerTransitionStat] = []
        sample_layer_pairs_seen: set[tuple[int, int]] = set()
        sample_token_count = sample_token_counts.get(sample_id, 0)
        for (row_sample_id, token_position, layer_id), source_records in sample_grouped.items():
            if row_sample_id != sample_id:
                continue
            for candidate_key, target_records in sample_grouped.items():
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
                sample_transition_rows.append(
                    LayerTransitionStat(
                        from_layer=layer_id,
                        to_layer=target_layer,
                        distance=target_layer - layer_id,
                        token_bucket=_token_bucket(token_position, sample_token_count),
                        hit_rate=hit_rate,
                        weighted_hit_rate=weighted_hit_rate,
                        sample_id=sample_id,
                        token_position=token_position,
                    )
                )
                sample_layer_pairs_seen.add((layer_id, target_layer))
        pair_summary: dict[str, Any] = {}
        for from_layer, to_layer in sorted(sample_layer_pairs_seen):
            rows = [row for row in sample_transition_rows if row.from_layer == from_layer and row.to_layer == to_layer]
            pair_summary[f"{from_layer}->{to_layer}"] = {
                "hit_rate": _summary_stats([row.hit_rate for row in rows]),
                "weighted_hit_rate": _summary_stats([row.weighted_hit_rate for row in rows]),
                "num_tokens": len(rows),
            }
        per_prompt_gate1[sample_id] = {
            "layer_pair_summary": pair_summary,
            "batch_rank_correlation": {
                key: value
                for key, value in batch_rank_correlation.items()
                if any(record.sample_id == sample_id for record in records)
            },
        }

    return {
        "transition_records": [row.to_dict() for row in transition_rows],
        "layer_pair_summary": layer_pair_summary,
        "distance_summary": distance_summary,
        "batch_rank_correlation": batch_rank_correlation,
        "per_prompt_gate1": per_prompt_gate1,
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


def build_batch_rank_correlation(
    records: list[TraceRecord],
    *,
    owner_by_expert: dict[int, int] | None = None,
    num_gpus: int = 4,
) -> dict[str, Any]:
    grouped = _group_records_by_sample_token_layer(records)
    flow_vectors: dict[tuple[str, int], list[float]] = {}
    layer_ids = sorted({record.layer_id for record in records})
    if owner_by_expert is None:
        owner_by_expert = build_owner_by_expert(records, placement="round_robin", num_gpus=num_gpus)
    for sample_id in sorted({record.sample_id for record in records}):
        for layer_id in layer_ids:
            vector = [0.0] * num_gpus
            sample_token_positions = sorted({record.token_position for record in records if record.sample_id == sample_id})
            for token_position in sample_token_positions:
                layer_records = _top2_records(grouped.get((sample_id, token_position, layer_id), []))
                for record in layer_records:
                    vector[owner_by_expert[record.expert_id]] += 1.0
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


def build_sample_layer_matrices(
    records: list[TraceRecord],
    *,
    owner_by_expert: dict[int, int],
    num_gpus: int = 4,
) -> dict[str, dict[int, list[list[int]]]]:
    grouped = _group_records_by_sample_token_layer(records)
    sample_ids = sorted({record.sample_id for record in records})
    layer_ids = sorted({record.layer_id for record in records})
    sample_layer_matrices: dict[str, dict[int, list[list[int]]]] = {}
    for sample_id in sample_ids:
        token_positions = sorted({record.token_position for record in records if record.sample_id == sample_id})
        token_count = (len(token_positions) // num_gpus) * num_gpus
        token_positions = token_positions[:token_count]
        layer_payload: dict[int, list[list[int]]] = {}
        for layer_id in layer_ids:
            matrix = [[0 for _ in range(num_gpus)] for _ in range(num_gpus)]
            for token_position in token_positions:
                src_gpu = token_position % num_gpus
                for record in grouped.get((sample_id, token_position, layer_id), []):
                    dst_gpu = owner_by_expert[record.expert_id]
                    matrix[src_gpu][dst_gpu] += 1
            layer_payload[layer_id] = matrix
        sample_layer_matrices[sample_id] = layer_payload
    return sample_layer_matrices


def build_predicted_traffic(
    *,
    sample_id: str,
    from_layer: int,
    to_layer: int,
    grouped_records: dict[tuple[str, int, int], list[TraceRecord]],
    predicted_topk_by_sample: dict[str, dict[tuple[int, int], list[list[int]]]],
    owner_by_expert: dict[int, int],
    num_gpus: int = 4,
    topk: int = 8,
) -> list[list[int]]:
    token_positions = sorted(
        token_position
        for record_sample_id, token_position, layer_id in grouped_records
        if record_sample_id == sample_id and layer_id == to_layer
    )
    token_count = (len(token_positions) // num_gpus) * num_gpus
    token_positions = token_positions[:token_count]
    matrix = [[0 for _ in range(num_gpus)] for _ in range(num_gpus)]
    predicted_rows = predicted_topk_by_sample.get(sample_id, {}).get((from_layer, to_layer), [])
    for token_position in token_positions:
        if token_position >= len(predicted_rows):
            continue
        src_gpu = token_position % num_gpus
        predicted_experts = predicted_rows[token_position][:topk]
        for expert_id in predicted_experts:
            dst_gpu = owner_by_expert[int(expert_id)]
            matrix[src_gpu][dst_gpu] += 1
    return matrix


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


def greedy_schedule_pairwise(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> float:
    _, after_dispatch = greedy_schedule_single_layer(dispatch_matrix, num_gpus)
    _, after_combine = greedy_schedule_single_layer(combine_matrix, num_gpus, after_dispatch)
    makespan, _ = greedy_schedule_single_layer(next_dispatch_matrix, num_gpus, after_combine)
    return makespan


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


def schedule_single_layer_in_order(
    traffic_matrix: list[list[int]],
    num_gpus: int,
    *,
    gpu_earliest_start: list[float] | None = None,
    chunk_order: list[tuple[int, int, int]] | None = None,
) -> tuple[float, list[float]]:
    if gpu_earliest_start is None:
        gpu_earliest_start = [0.0] * num_gpus
    if chunk_order is None:
        chunk_order = []
        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src != dst and traffic_matrix[src][dst] > 0:
                    chunk_order.append((int(traffic_matrix[src][dst]), src, dst))
        chunk_order.sort(reverse=True)
    gpu_available = list(gpu_earliest_start)
    for size, src, dst in chunk_order:
        start = max(gpu_available[src], gpu_available[dst])
        end = start + float(size)
        gpu_available[src] = end
        gpu_available[dst] = end
    return max(gpu_available) if gpu_available else 0.0, gpu_available


def exact_schedule_single_layer(
    traffic_matrix: list[list[int]],
    num_gpus: int,
    gpu_earliest_start: list[float] | None = None,
) -> tuple[float, list[float]]:
    try:
        import numpy as np
        import scipy.optimize as spo
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for exact oracle scheduling") from exc

    if gpu_earliest_start is None:
        gpu_earliest_start = [0.0] * num_gpus

    chunks: list[tuple[int, int, int]] = []
    for src in range(num_gpus):
        for dst in range(num_gpus):
            if src != dst and traffic_matrix[src][dst] > 0:
                chunks.append((int(traffic_matrix[src][dst]), src, dst))
    if not chunks:
        return max(gpu_earliest_start) if gpu_earliest_start else 0.0, list(gpu_earliest_start)

    chunk_count = len(chunks)
    makespan_index = chunk_count
    variable_count = chunk_count + 1
    bounds = [(0.0, None)] * variable_count
    integrality = [0] * variable_count
    c = [0.0] * variable_count
    c[makespan_index] = 1.0
    constraint_specs: list[tuple[dict[int, float], float]] = []
    total_duration = float(sum(size for size, _, _ in chunks) + sum(gpu_earliest_start))
    precedence_pairs: dict[tuple[int, int], int] = {}

    def add_linear_ub(coeffs: dict[int, float], ub: float) -> None:
        constraint_specs.append((dict(coeffs), ub))

    for idx, (size, src, dst) in enumerate(chunks):
        earliest = max(gpu_earliest_start[src], gpu_earliest_start[dst])
        add_linear_ub({idx: -1.0}, -earliest)
        add_linear_ub({idx: 1.0, makespan_index: -1.0}, -float(size))

    pair_binary_offset = variable_count
    for i in range(chunk_count):
        for j in range(i + 1, chunk_count):
            _, src_i, dst_i = chunks[i]
            _, src_j, dst_j = chunks[j]
            if {src_i, dst_i}.isdisjoint({src_j, dst_j}):
                continue
            precedence_pairs[(i, j)] = pair_binary_offset
            pair_binary_offset += 1

    if precedence_pairs:
        c.extend([0.0] * len(precedence_pairs))
        bounds.extend([(0.0, 1.0)] * len(precedence_pairs))
        integrality.extend([1] * len(precedence_pairs))
        variable_count = len(c)

        for (i, j), binary_index in precedence_pairs.items():
            dur_i = float(chunks[i][0])
            dur_j = float(chunks[j][0])
            add_linear_ub({i: 1.0, j: -1.0, binary_index: total_duration}, total_duration - dur_i)
            add_linear_ub({j: 1.0, i: -1.0, binary_index: -total_duration}, -dur_j)

    constraints: list[Any] = []
    for coeffs, ub in constraint_specs:
        row = np.zeros(variable_count, dtype=float)
        for index, value in coeffs.items():
            row[index] = value
        constraints.append(spo.LinearConstraint(row, -math.inf, ub))

    result = spo.milp(
        c=c,
        bounds=spo.Bounds([low for low, _ in bounds], [math.inf if high is None else high for _, high in bounds]),
        integrality=integrality,
        constraints=constraints,
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"exact single-layer MILP failed: {result.message}")
    starts = result.x[:chunk_count]
    gpu_available = list(gpu_earliest_start)
    for start, (size, src, dst) in sorted(zip(starts, chunks, strict=False), key=lambda item: float(item[0])):
        end = float(start) + float(size)
        gpu_available[src] = max(gpu_available[src], end)
        gpu_available[dst] = max(gpu_available[dst], end)
    return float(result.x[makespan_index]), gpu_available


def _matrix_to_pairwise_chunks(
    matrix: list[list[int]],
    *,
    phase: int,
    num_gpus: int,
) -> list[PairwiseChunk]:
    chunks: list[PairwiseChunk] = []
    for src in range(num_gpus):
        for dst in range(num_gpus):
            size = int(matrix[src][dst])
            if src == dst or size <= 0:
                continue
            chunks.append(
                PairwiseChunk(
                    chunk_id=f"phase{phase}_src{src}_dst{dst}",
                    phase=phase,
                    size=size,
                    src_gpu=src,
                    dst_gpu=dst,
                )
            )
    return chunks


def pairwise_oracle(
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
) -> dict[str, Any]:
    schedule: list[dict[str, Any]] = []
    gpu_state = [0.0] * num_gpus
    total_chunk_count = 0
    current_offset = 0.0
    for phase, matrix in enumerate((dispatch_matrix, combine_matrix, next_dispatch_matrix)):
        phase_chunks = _matrix_to_pairwise_chunks(matrix, phase=phase, num_gpus=num_gpus)
        total_chunk_count += len(phase_chunks)
        if not phase_chunks:
            continue
        _, after_phase = exact_schedule_single_layer(matrix, num_gpus, gpu_state)
        phase_schedule = []
        phase_gpu = list(gpu_state)
        sorted_chunks = sorted(phase_chunks, key=lambda item: (-item.size, item.src_gpu, item.dst_gpu))
        for chunk in sorted_chunks:
            start = max(phase_gpu[chunk.src_gpu], phase_gpu[chunk.dst_gpu])
            end = start + float(chunk.size)
            phase_gpu[chunk.src_gpu] = end
            phase_gpu[chunk.dst_gpu] = end
            phase_schedule.append(
                {
                    **chunk.to_dict(),
                    "start": start,
                    "end": end,
                }
            )
        phase_makespan = max(after_phase) if after_phase else current_offset
        if phase_schedule:
            latest_end = max(item["end"] for item in phase_schedule)
            if latest_end > phase_makespan:
                phase_makespan = latest_end
        schedule.extend(phase_schedule)
        current_offset = phase_makespan
        gpu_state = [phase_makespan] * num_gpus

    return {
        "makespan": max(gpu_state) if gpu_state else 0.0,
        "schedule": schedule,
        "chunk_count": total_chunk_count,
    }


def oracle_schedule_multi_layer(
    traffic_matrices: list[list[list[int]]],
    combine_matrices: list[list[list[int]]],
    num_gpus: int,
) -> float:
    gpu_earliest_start = [0.0] * num_gpus
    for dispatch_matrix, combine_matrix in zip(traffic_matrices, combine_matrices, strict=False):
        _, after_dispatch = exact_schedule_single_layer(dispatch_matrix, num_gpus, gpu_earliest_start)
        _, after_combine = exact_schedule_single_layer(combine_matrix, num_gpus, after_dispatch)
        gpu_earliest_start = after_combine
    return max(gpu_earliest_start) if gpu_earliest_start else 0.0


def annealed_oracle_schedule_multi_layer(
    traffic_matrices: list[list[list[int]]],
    combine_matrices: list[list[list[int]]],
    num_gpus: int,
    *,
    iterations: int = 200,
) -> float:
    chunk_orders: list[list[tuple[int, int, int]]] = []
    for matrix in [*traffic_matrices, *combine_matrices]:
        chunks: list[tuple[int, int, int]] = []
        for src in range(num_gpus):
            for dst in range(num_gpus):
                if src != dst and matrix[src][dst] > 0:
                    chunks.append((int(matrix[src][dst]), src, dst))
        chunks.sort(reverse=True)
        chunk_orders.append(chunks)

    def evaluate(all_orders: list[list[tuple[int, int, int]]]) -> float:
        gpu_state = [0.0] * num_gpus
        order_index = 0
        for dispatch_matrix, combine_matrix in zip(traffic_matrices, combine_matrices, strict=False):
            _, gpu_state = schedule_single_layer_in_order(
                dispatch_matrix,
                num_gpus,
                gpu_earliest_start=gpu_state,
                chunk_order=all_orders[order_index],
            )
            order_index += 1
            _, gpu_state = schedule_single_layer_in_order(
                combine_matrix,
                num_gpus,
                gpu_earliest_start=gpu_state,
                chunk_order=all_orders[order_index],
            )
            order_index += 1
        return max(gpu_state) if gpu_state else 0.0

    best_orders = [list(order) for order in chunk_orders]
    best_value = evaluate(best_orders)
    current_orders = [list(order) for order in best_orders]
    current_value = best_value
    temperature = max(1.0, best_value * 0.1)
    for step in range(max(1, iterations)):
        candidate_orders = [list(order) for order in current_orders]
        mutated = False
        for order in candidate_orders:
            if len(order) < 2:
                continue
            i = step % len(order)
            j = (step * 7 + 3) % len(order)
            if i == j:
                continue
            order[i], order[j] = order[j], order[i]
            mutated = True
            break
        if not mutated:
            continue
        candidate_value = evaluate(candidate_orders)
        delta = candidate_value - current_value
        accept = delta <= 0.0 or math.exp(-delta / max(temperature, 1e-6)) > 0.5
        if accept:
            current_orders = candidate_orders
            current_value = candidate_value
            if current_value < best_value:
                best_orders = [list(order) for order in current_orders]
                best_value = current_value
        temperature *= 0.98
    return best_value


def simulate_oracle_vs_greedy(
    batches: list[dict[str, Any]],
    *,
    combine_scale_factor: float = 1.0,
    exact_layer_limit: int = 4,
    exact_chunk_limit: int = 16,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    improvements: list[float] = []
    oracle_modes: list[str] = []
    for batch in batches:
        traffic_matrices = [layer["matrix"] for layer in batch["layers"]]
        combine_matrices = [combine_matrix_from_dispatch(matrix, combine_scale_factor) for matrix in traffic_matrices]
        greedy = greedy_schedule_multi_layer(traffic_matrices, combine_matrices, batch["num_gpus"])
        max_chunks = 0
        for matrix in [*traffic_matrices, *combine_matrices]:
            chunk_count = sum(1 for src in range(batch["num_gpus"]) for dst in range(batch["num_gpus"]) if src != dst and matrix[src][dst] > 0)
            max_chunks = max(max_chunks, chunk_count)
        if len(traffic_matrices) <= exact_layer_limit and max_chunks <= exact_chunk_limit:
            oracle = oracle_schedule_multi_layer(traffic_matrices, combine_matrices, batch["num_gpus"])
            oracle_mode = "exact_milp"
        else:
            oracle = annealed_oracle_schedule_multi_layer(traffic_matrices, combine_matrices, batch["num_gpus"])
            oracle_mode = "annealed_fallback"
        improvement_pct = 0.0 if greedy == 0.0 else ((greedy - oracle) / greedy) * 100.0
        improvements.append(improvement_pct)
        oracle_modes.append(oracle_mode)
        results.append(
            {
                "batch_id": batch["batch_id"],
                "sample_ids": batch["sample_ids"],
                "token_count": batch["token_count"],
                "greedy_makespan": greedy,
                "oracle_makespan": oracle,
                "improvement_pct": improvement_pct,
                "oracle_mode": oracle_mode,
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
            "oracle_modes": sorted(set(oracle_modes)),
            "gate2_decision": {
                "passed": gate2_pass,
                "threshold_pct": 15.0,
            },
        },
    }


def run_pairwise_analysis(
    records: list[TraceRecord],
    *,
    hidden_states_by_sample: dict[str, dict[int, torch.Tensor]],
    gate_weights_by_sample: dict[str, dict[int, torch.Tensor]],
    owner_by_expert: dict[int, int],
    num_gpus: int = 4,
    topk: int = 8,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    grouped = _group_records_by_sample_token_layer(records)
    sample_layer_matrices = build_sample_layer_matrices(
        records,
        owner_by_expert=owner_by_expert,
        num_gpus=num_gpus,
    )
    predicted_topk_by_sample = _decode_predicted_topk_by_sample(
        hidden_states_by_sample=hidden_states_by_sample,
        gate_weights_by_sample=gate_weights_by_sample,
        topk=topk,
    )
    sample_ids = sorted({record.sample_id for record in records})
    if sample_limit is not None:
        sample_ids = sample_ids[:sample_limit]
    layer_ids = sorted({record.layer_id for record in records})
    results: list[dict[str, Any]] = []
    predicted_improvements: list[float] = []
    perfect_improvements: list[float] = []
    traffic_correlations: list[float] = []
    pairwise_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def matrix_key(matrix: list[list[int]]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in row) for row in matrix)

    def solve_pairwise_cached(
        phase0: list[list[int]],
        phase1: list[list[int]],
        phase2: list[list[int]],
    ) -> dict[str, Any]:
        key = (matrix_key(phase0), matrix_key(phase1), matrix_key(phase2), num_gpus)
        cached = pairwise_cache.get(key)
        if cached is None:
            cached = pairwise_oracle(phase0, phase1, phase2, num_gpus)
            pairwise_cache[key] = cached
        return cached

    for sample_id in sample_ids:
        for idx in range(len(layer_ids) - 1):
            from_layer = layer_ids[idx]
            to_layer = layer_ids[idx + 1]
            layer_map = sample_layer_matrices.get(sample_id, {})
            if from_layer not in layer_map or to_layer not in layer_map:
                continue
            dispatch_matrix = layer_map[from_layer]
            next_actual_matrix = layer_map[to_layer]
            combine_matrix = combine_matrix_from_dispatch(dispatch_matrix, 1.0)
            next_predicted_matrix = build_predicted_traffic(
                sample_id=sample_id,
                from_layer=from_layer,
                to_layer=to_layer,
                grouped_records=grouped,
                predicted_topk_by_sample=predicted_topk_by_sample,
                owner_by_expert=owner_by_expert,
                num_gpus=num_gpus,
                topk=topk,
            )

            greedy_makespan = greedy_schedule_pairwise(dispatch_matrix, combine_matrix, next_actual_matrix, num_gpus)
            oracle_perfect = solve_pairwise_cached(dispatch_matrix, combine_matrix, next_actual_matrix)
            oracle_predicted = solve_pairwise_cached(dispatch_matrix, combine_matrix, next_predicted_matrix)
            perfect_improvement_pct = (
                0.0 if greedy_makespan == 0.0 else ((greedy_makespan - oracle_perfect["makespan"]) / greedy_makespan) * 100.0
            )
            predicted_improvement_pct = (
                0.0
                if greedy_makespan == 0.0
                else ((greedy_makespan - oracle_predicted["makespan"]) / greedy_makespan) * 100.0
            )
            flat_actual = [float(value) for row in next_actual_matrix for value in row]
            flat_predicted = [float(value) for row in next_predicted_matrix for value in row]
            traffic_corr = spearman_rank_correlation(flat_actual, flat_predicted)

            perfect_improvements.append(perfect_improvement_pct)
            predicted_improvements.append(predicted_improvement_pct)
            traffic_correlations.append(traffic_corr)
            results.append(
                {
                    "sample_id": sample_id,
                    "layer_pair": f"{from_layer}->{to_layer}",
                    "dispatch_matrix": dispatch_matrix,
                    "combine_matrix": combine_matrix,
                    "next_actual_matrix": next_actual_matrix,
                    "next_predicted_matrix": next_predicted_matrix,
                    "greedy_makespan": greedy_makespan,
                    "oracle_perfect_makespan": oracle_perfect["makespan"],
                    "oracle_predicted_makespan": oracle_predicted["makespan"],
                    "perfect_improvement_pct": perfect_improvement_pct,
                    "predicted_improvement_pct": predicted_improvement_pct,
                    "traffic_correlation": traffic_corr,
                    "oracle_perfect_chunk_count": oracle_perfect["chunk_count"],
                    "oracle_predicted_chunk_count": oracle_predicted["chunk_count"],
                    "oracle_perfect_schedule": oracle_perfect["schedule"],
                    "oracle_predicted_schedule": oracle_predicted["schedule"],
                }
            )

    predicted_vs_corr = (
        spearman_rank_correlation(predicted_improvements, traffic_correlations)
        if len(predicted_improvements) >= 2
        else 0.0
    )
    summary = {
        "pair_count": len(results),
        "perfect_improvement_pct": _summary_stats(perfect_improvements),
        "predicted_improvement_pct": _summary_stats(predicted_improvements),
        "traffic_correlation": _summary_stats(traffic_correlations),
        "predicted_improvement_vs_traffic_correlation": predicted_vs_corr,
        "gate2_decision": {
            "passed": (
                _summary_stats(perfect_improvements)["mean"] >= 15.0
                and _summary_stats(predicted_improvements)["mean"] >= 10.0
            ),
            "thresholds": {
                "oracle_perfect_mean_improvement_pct": 15.0,
                "oracle_predicted_mean_improvement_pct": 10.0,
            },
        },
    }
    return {
        "results": results,
        "summary": summary,
    }


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
