from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore

from rs.runtime.offline.traffic.matrix_builder import (
    TraceRecord,
    _group_records_by_sample_token_layer,
    build_owner_by_expert,
    build_sample_layer_matrices,
    build_predicted_traffic,
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
    return {str(sample_id): {int(layer_id): tensor.float().cpu() for layer_id, tensor in layer_map.items()} for sample_id, layer_map in payload.items()}


def load_gate_weight_bundle(path: str | Path) -> dict[str, dict[int, torch.Tensor]]:
    payload = torch.load(Path(path), map_location="cpu")
    return {str(sample_id): {int(layer_id): tensor.float().cpu() for layer_id, tensor in layer_map.items()} for sample_id, layer_map in payload.items()}


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


def build_batch_rank_correlation(records: list[TraceRecord], *, owner_by_expert: dict[int, int] | None = None, num_gpus: int = 4) -> dict[str, dict[str, float]]:
    if owner_by_expert is None:
        owner_by_expert = build_owner_by_expert(records, num_gpus=num_gpus)
    sample_layer_matrices = build_sample_layer_matrices(records, owner_by_expert=owner_by_expert, num_gpus=num_gpus)
    pair_correlations: dict[str, list[float]] = {}
    for sample_id, layer_map in sample_layer_matrices.items():
        layer_ids = sorted(layer_map)
        for idx in range(len(layer_ids) - 1):
            left = layer_map[layer_ids[idx]]
            right = layer_map[layer_ids[idx + 1]]
            flat_left = [float(item) for row in left for item in row]
            flat_right = [float(item) for row in right for item in row]
            pair_correlations.setdefault(f"{layer_ids[idx]}->{layer_ids[idx + 1]}", []).append(
                spearman_rank_correlation(flat_left, flat_right)
            )
    return {pair: {"spearman": statistics.fmean(values) if values else 0.0} for pair, values in pair_correlations.items()}


def evaluate_gate2(report: dict[str, Any]) -> dict[str, Any]:
    pair_summary = report.get("layer_pair_summary", {})
    batch_rank = report.get("batch_rank_correlation", {})
    pass_count = sum(
        1
        for payload in pair_summary.values()
        if payload["prefetch_accuracy"]["mean"] >= 0.70 and payload["cosine_similarity"]["mean"] >= 0.70
    )
    rank_pass = any(value.get("spearman", 0.0) >= 0.50 for value in batch_rank.values())
    gate1_pass = pass_count >= 1
    reasons: list[str] = []
    if not gate1_pass:
        reasons.append("cross_layer_prefetch_accuracy_below_threshold")
    if not rank_pass:
        reasons.append("batch_rank_correlation_below_threshold")
    return {
        "passed": gate1_pass and rank_pass,
        "pass_count": pass_count,
        "rank_pass": rank_pass,
        "thresholds": {
            "prefetch_accuracy_mean": 0.70,
            "cosine_similarity_mean": 0.70,
            "rank_correlation_mean": 0.50,
        },
        "reasons": reasons,
    }


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
    report = {
        "prediction_rows": [row.to_dict() for row in prediction_rows],
        "layer_pair_summary": pair_summary,
        "batch_rank_correlation": batch_rank_correlation,
    }
    report["gate1_decision"] = evaluate_gate2(report)
    return report


def analyze_cross_layer_correlation(
    records: list[TraceRecord],
    *,
    owner_by_expert: dict[int, int] | None = None,
    num_gpus: int = 4,
) -> dict[str, Any]:
    if owner_by_expert is None:
        owner_by_expert = build_owner_by_expert(records, num_gpus=num_gpus)
    sample_layer_matrices = build_sample_layer_matrices(records, owner_by_expert=owner_by_expert, num_gpus=num_gpus)
    transition_records: list[LayerTransitionStat] = []
    for sample_id, layer_map in sample_layer_matrices.items():
        layer_ids = sorted(layer_map)
        token_count = len({record.token_position for record in records if record.sample_id == sample_id})
        for idx in range(len(layer_ids) - 1):
            left = layer_map[layer_ids[idx]]
            right = layer_map[layer_ids[idx + 1]]
            flat_left = [float(item) for row in left for item in row]
            flat_right = [float(item) for row in right for item in row]
            score = spearman_rank_correlation(flat_left, flat_right)
            transition_records.append(
                LayerTransitionStat(
                    from_layer=layer_ids[idx],
                    to_layer=layer_ids[idx + 1],
                    distance=1,
                    token_bucket=_token_bucket(0, token_count),
                    hit_rate=score,
                    weighted_hit_rate=score,
                    sample_id=sample_id,
                    token_position=0,
                )
            )
    layer_pair_summary: dict[str, Any] = {}
    for row in transition_records:
        layer_pair_summary.setdefault(f"{row.from_layer}->{row.to_layer}", []).append(row.hit_rate)
    summary = {pair: _summary_stats(values) for pair, values in layer_pair_summary.items()}
    return {
        "transition_records": [row.to_dict() for row in transition_records],
        "layer_pair_summary": summary,
        "distance_summary": {"1": _summary_stats([row.hit_rate for row in transition_records]) if transition_records else _summary_stats([])},
        "batch_rank_correlation": build_batch_rank_correlation(records, owner_by_expert=owner_by_expert, num_gpus=num_gpus),
        "gate1_decision": {"passed": bool(transition_records), "reasons": []},
    }
