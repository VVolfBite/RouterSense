from __future__ import annotations

from experiments.paper.contracts import RecordMetadata, TraceSample
from experiments.paper.traffic_builder import build_traffic_instance, deterministic_expert_to_rank_mapping


def test_virtual_ep_mapping_matches_num_experts_and_digest_is_stable() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    sample = TraceSample(
        schema_version="v1",
        model_id="m",
        model_revision="rev",
        prompt_id="p",
        batch_id="b",
        sequence_length=4,
        layer_id="1",
        num_experts=4,
        top_k=0,
        router_logits_digest=None,
        selected_experts_digest="x",
        routing_weights_digest=None,
        compact_route_counts=((0, 1, 0, 0),),
        capture_timestamp="offline",
        metadata=metadata,
        trace_sample_id="ts",
    )
    mapping = deterministic_expert_to_rank_mapping(
        model_id=sample.model_id,
        layer_id=sample.layer_id,
        num_experts=sample.num_experts,
        virtual_ep_size=2,
    )
    instance = build_traffic_instance(
        trace_sample=sample,
        p0_matrix=((0, 1), (2, 0)),
        p1_matrix=((0, 2), (1, 0)),
        p2_matrix=((0, 1), (1, 0)),
        virtual_ep_size=2,
        metadata=metadata,
        mapping=mapping,
    )
    assert len(instance.expert_to_rank_mapping) == sample.num_experts
    assert all(0 <= value < 2 for value in instance.expert_to_rank_mapping)
    assert instance.mapping_digest
    assert instance.traffic_digest
