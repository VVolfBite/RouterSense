from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore

from .traffic_matrix import (
    TraceRecord,
    _group_records_by_sample_token_layer,
    _top2_records,
    build_owner_by_expert,
)


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
    with torch.no_grad():
        for sample_id, sample_hidden in hidden_states_by_sample.items():
            sample_gates = gate_weights_by_sample[sample_id]
            sample_layer_ids = sorted(layer_id for layer_id in sample_hidden if layer_id in sample_gates)
            sample_payload: dict[tuple[int, int], list[list[int]]] = {}
            for idx in range(len(sample_layer_ids) - 1):
                from_layer = sample_layer_ids[idx]
                to_layer = sample_layer_ids[idx + 1]
                hidden_from = sample_hidden[from_layer].squeeze(0)
                gate_weight = sample_gates[to_layer]
                if hidden_from.device != gate_weight.device:
                    hidden_from = hidden_from.to(gate_weight.device)
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

    batch_rank_correlation = build_batch_rank_correlation(records, owner_by_expert=owner_by_expert, num_gpus=num_gpus)
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
    for record in records:
        sample_token_counts[record.sample_id] = max(sample_token_counts.get(record.sample_id, 0), record.token_position + 1)

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
            transition_rows.append(LayerTransitionStat(
                from_layer=layer_id,
                to_layer=target_layer,
                distance=target_layer - layer_id,
                token_bucket=bucket,
                hit_rate=hit_rate,
                weighted_hit_rate=weighted_hit_rate,
                sample_id=sample_id,
                token_position=token_position,
            ))
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
                    "weighted_hit_rate": _summary_stats([row.weighted_hit_rate for row in pair_rows if row.token_bucket == bucket]),
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

    batch_rank_correlation = build_batch_rank_correlation(records, owner_by_expert=owner_by_expert, num_gpus=num_gpus)
    layer_pair_metrics = []
    for pair_key, payload in layer_pair_summary.items():
        rank_corr = batch_rank_correlation.get(pair_key, {}).get("spearman", 0.0)
        layer_pair_metrics.append({
            "pair": pair_key,
            "mean_hit_rate": payload["hit_rate"]["mean"],
            "mean_weighted_hit_rate": payload["weighted_hit_rate"]["mean"],
            "mean_rank_correlation": rank_corr,
        })
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
                sample_transition_rows.append(LayerTransitionStat(
                    from_layer=layer_id,
                    to_layer=target_layer,
                    distance=target_layer - layer_id,
                    token_bucket=_token_bucket(token_position, sample_token_count),
                    hit_rate=hit_rate,
                    weighted_hit_rate=weighted_hit_rate,
                    sample_id=sample_id,
                    token_position=token_position,
                ))
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


def evaluate_gate2(perfect_improvements: list[float], predicted_improvements: list[float]) -> dict[str, Any]:
    mean_perfect = statistics.fmean(perfect_improvements) if perfect_improvements else 0.0
    mean_predicted = statistics.fmean(predicted_improvements) if predicted_improvements else 0.0

    if mean_perfect >= 15.0 and mean_predicted >= 10.0:
        decision = "PASS"
        reason = "Oracle-perfect >= 15% and Oracle-predicted >= 10%"
    elif mean_perfect >= 15.0 and mean_predicted >= 5.0:
        decision = "MARGINAL"
        reason = "Scheduling valuable but prediction quality insufficient"
    elif mean_perfect >= 15.0 and mean_predicted < 5.0:
        decision = "PREDICTION_BOTTLENECK"
        reason = "Scheduling space exists but prediction too inaccurate"
    elif mean_perfect < 5.0:
        decision = "NO_SCHEDULING_SPACE"
        reason = "Insufficient scheduling optimization space"
    else:
        decision = "WEAK"
        reason = f"perfect={mean_perfect:.1f}%, predicted={mean_predicted:.1f}%"

    return {
        "decision": decision,
        "mean_perfect_improvement_pct": mean_perfect,
        "mean_predicted_improvement_pct": mean_predicted,
        "reason": reason,
        "thresholds": {
            "oracle_perfect_mean_improvement_pct": 15.0,
            "oracle_predicted_mean_improvement_pct": 10.0,
        },
        "passed": decision == "PASS",
    }
