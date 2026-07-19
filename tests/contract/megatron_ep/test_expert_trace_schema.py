from __future__ import annotations

from pathlib import Path

from rs.runtime.online.megatron_ep.prediction.expert_trace import (
    ExpertRouteRecord,
    aggregate_route_records,
    load_expert_route_jsonl,
    load_source_expert_counts_jsonl,
    write_expert_route_jsonl,
    write_source_expert_counts_jsonl,
)


def test_expert_route_schema_roundtrip_and_aggregation(tmp_path: Path) -> None:
    records = (
        ExpertRouteRecord(
            layer_id=3,
            rank=0,
            token_count=2,
            top_k=2,
            expert_ids=((0, 3), (1, 3)),
            routing_weights=((0.7, 0.3), (0.4, 0.6)),
            source_rank=0,
        ),
        ExpertRouteRecord(
            layer_id=3,
            rank=1,
            token_count=1,
            top_k=2,
            expert_ids=((2, 1),),
            routing_weights=((0.8, 0.2),),
            source_rank=1,
        ),
    )
    route_path = tmp_path / "expert_route.jsonl"
    write_expert_route_jsonl(route_path, records)
    loaded = load_expert_route_jsonl(route_path)
    assert loaded == records
    counts = aggregate_route_records(loaded, world_size=2, num_experts=4, use_routing_weights=True)
    assert counts.counts[0] == (1, 1, 0, 2)
    assert counts.counts[1] == (0, 1, 1, 0)
    assert counts.weighted_counts is not None
    count_path = tmp_path / "source_expert_counts.jsonl"
    write_source_expert_counts_jsonl(count_path, (counts,))
    reloaded = load_source_expert_counts_jsonl(count_path)
    assert reloaded == (counts,)
