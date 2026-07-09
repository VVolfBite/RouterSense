"""Expert-route trace schema and compact source-rank x expert aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExpertRouteRecord:
    layer_id: int
    rank: int
    token_count: int
    top_k: int
    expert_ids: tuple[tuple[int, ...], ...]
    routing_weights: tuple[tuple[float, ...], ...] | None
    source_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceExpertCountMatrix:
    layer_id: int
    world_size: int
    num_experts: int
    counts: tuple[tuple[int, ...], ...]
    weighted_counts: tuple[tuple[float, ...], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_route_records(
    records: tuple[ExpertRouteRecord, ...] | list[ExpertRouteRecord],
    *,
    world_size: int,
    num_experts: int,
    use_routing_weights: bool = False,
) -> SourceExpertCountMatrix:
    if not records:
        return SourceExpertCountMatrix(
            layer_id=0,
            world_size=int(world_size),
            num_experts=int(num_experts),
            counts=tuple(tuple(0 for _ in range(num_experts)) for _ in range(world_size)),
            weighted_counts=tuple(tuple(0.0 for _ in range(num_experts)) for _ in range(world_size)) if use_routing_weights else None,
        )
    layer_id = int(records[0].layer_id)
    counts = [[0 for _ in range(num_experts)] for _ in range(world_size)]
    weighted = [[0.0 for _ in range(num_experts)] for _ in range(world_size)] if use_routing_weights else None
    for record in records:
        source_rank = int(record.source_rank)
        for token_idx, expert_row in enumerate(record.expert_ids):
            weight_row = None if record.routing_weights is None else record.routing_weights[token_idx]
            for expert_pos, expert_id in enumerate(expert_row):
                expert_idx = int(expert_id)
                counts[source_rank][expert_idx] += 1
                if weighted is not None:
                    weight = 1.0
                    if weight_row is not None and expert_pos < len(weight_row):
                        weight = float(weight_row[expert_pos])
                    weighted[source_rank][expert_idx] += weight
    return SourceExpertCountMatrix(
        layer_id=layer_id,
        world_size=int(world_size),
        num_experts=int(num_experts),
        counts=tuple(tuple(int(value) for value in row) for row in counts),
        weighted_counts=None if weighted is None else tuple(tuple(float(value) for value in row) for row in weighted),
    )


def write_expert_route_jsonl(path: Path, records: tuple[ExpertRouteRecord, ...] | list[ExpertRouteRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record.to_dict(), ensure_ascii=False) for record in records) + ("\n" if records else ""), encoding="utf-8")


def load_expert_route_jsonl(path: Path) -> tuple[ExpertRouteRecord, ...]:
    rows: list[ExpertRouteRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            ExpertRouteRecord(
                layer_id=int(payload["layer_id"]),
                rank=int(payload["rank"]),
                token_count=int(payload["token_count"]),
                top_k=int(payload["top_k"]),
                expert_ids=tuple(tuple(int(v) for v in row) for row in payload["expert_ids"]),
                routing_weights=None
                if payload.get("routing_weights") is None
                else tuple(tuple(float(v) for v in row) for row in payload["routing_weights"]),
                source_rank=int(payload["source_rank"]),
            )
        )
    return tuple(rows)


def write_source_expert_counts_jsonl(path: Path, rows: tuple[SourceExpertCountMatrix, ...] | list[SourceExpertCountMatrix]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row.to_dict(), ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def load_source_expert_counts_jsonl(path: Path) -> tuple[SourceExpertCountMatrix, ...]:
    rows: list[SourceExpertCountMatrix] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            SourceExpertCountMatrix(
                layer_id=int(payload["layer_id"]),
                world_size=int(payload["world_size"]),
                num_experts=int(payload["num_experts"]),
                counts=tuple(tuple(int(v) for v in row) for row in payload["counts"]),
                weighted_counts=None
                if payload.get("weighted_counts") is None
                else tuple(tuple(float(v) for v in row) for row in payload["weighted_counts"]),
            )
        )
    return tuple(rows)


__all__ = [
    "ExpertRouteRecord",
    "SourceExpertCountMatrix",
    "aggregate_route_records",
    "load_expert_route_jsonl",
    "load_source_expert_counts_jsonl",
    "write_expert_route_jsonl",
    "write_source_expert_counts_jsonl",
]
