from __future__ import annotations

import json
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


def _top2_records(records: list[TraceRecord]) -> list[TraceRecord]:
    return [record for record in records if record.topk_rank < 2]


def _normalize_token_positions(token_positions: list[int], *, num_gpus: int, max_tokens: int | None = None) -> list[int]:
    if len(token_positions) < num_gpus:
        normalized = list(token_positions)
    else:
        token_count = (len(token_positions) // num_gpus) * num_gpus
        normalized = token_positions[:token_count]
    if max_tokens is not None:
        normalized = normalized[:max_tokens]
    return normalized


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
        token_positions = _normalize_token_positions(token_positions, num_gpus=num_gpus)
        token_count = len(token_positions)
        if token_count == 0:
            continue
        layers_payload = []
        for layer_id in layer_ids:
            matrix = [[0 for _ in range(num_gpus)] for _ in range(num_gpus)]
            for token_position in token_positions:
                src_gpu = token_position % num_gpus
                for record in grouped.get((sample_id, token_position, layer_id), []):
                    dst_gpu = owner_by_expert[record.expert_id]
                    matrix[src_gpu][dst_gpu] += 1
            layers_payload.append({
                "layer_id": layer_id,
                "matrix": matrix,
                "row_labels": [f"gpu_{index}" for index in range(num_gpus)],
                "col_labels": [f"gpu_{index}" for index in range(num_gpus)],
            })
        batches.append({
            "batch_id": batch_id,
            "sample_ids": [sample_id],
            "token_count": token_count,
            "num_gpus": num_gpus,
            "layers": layers_payload,
        })
    return batches


def build_sample_layer_matrices(
    records: list[TraceRecord],
    *,
    owner_by_expert: dict[int, int],
    num_gpus: int = 4,
    grouped: dict[tuple[str, int, int], list[TraceRecord]] | None = None,
) -> dict[str, dict[int, list[list[int]]]]:
    if grouped is None:
        grouped = _group_records_by_sample_token_layer(records)
    sample_ids = sorted({record.sample_id for record in records})
    layer_ids = sorted({record.layer_id for record in records})
    tokens_by_sample: dict[str, set[int]] = {}
    for record in records:
        tokens_by_sample.setdefault(record.sample_id, set()).add(record.token_position)
    normalized_tokens_by_sample = {
        sample_id: _normalize_token_positions(sorted(token_positions), num_gpus=num_gpus)
        for sample_id, token_positions in tokens_by_sample.items()
    }
    sample_layer_matrices: dict[str, dict[int, list[list[int]]]] = {}
    for sample_id in sample_ids:
        token_positions = normalized_tokens_by_sample.get(sample_id, [])
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
    token_index: dict[tuple[str, int], list[int]] | None = None,
) -> list[list[int]]:
    if token_index is not None:
        token_positions = token_index.get((sample_id, to_layer), [])
    else:
        token_positions = sorted(
            token_position
            for record_sample_id, token_position, layer_id in grouped_records
            if record_sample_id == sample_id and layer_id == to_layer
        )
        token_positions = _normalize_token_positions(token_positions, num_gpus=num_gpus, max_tokens=256)
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


def combine_matrix_from_dispatch(matrix: list[list[int]], scale: float = 1.0) -> list[list[int]]:
    size = len(matrix)
    return [[int(round(matrix[col][row] * scale)) for col in range(size)] for row in range(size)]
