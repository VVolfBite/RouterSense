from __future__ import annotations

from experiments.paper.contracts import RecordMetadata, TraceSample
from experiments.paper.traffic_builder import build_traffic_instance


def test_virtual_ep_mapping_and_digest_are_stable() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    sample = TraceSample(
        schema_version="v1",
        model_id="m",
        model_revision="rev",
        prompt_id="p",
        batch_id="b",
        sequence_length=4,
        layer_id="1",
        num_experts=2,
        top_k=0,
        router_logits_digest="na",
        selected_experts_digest="x",
        routing_weights_digest="na",
        compact_route_counts=((0, 1), (2, 0)),
        capture_timestamp="offline",
        metadata=metadata,
        trace_sample_id="ts",
    )
    instance = build_traffic_instance(
        trace_sample=sample,
        p0_matrix=((0, 1), (2, 0)),
        p1_matrix=((0, 2), (3, 0)),
        p2_matrix=((0, 4), (1, 0)),
        virtual_ep_size=2,
        metadata=metadata,
    )
    assert instance.expert_to_rank_mapping == (0, 1)
    assert instance.traffic_digest
