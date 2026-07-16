from __future__ import annotations

from experiments.paper.contracts import RecordMetadata, TraceSample


def test_trace_sample_contract_roundtrip() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    sample = TraceSample(
        schema_version="v1",
        model_id="m",
        model_revision="rev",
        prompt_id="p",
        batch_id="b",
        sequence_length=8,
        layer_id="1",
        num_experts=2,
        top_k=2,
        router_logits_digest="a",
        selected_experts_digest="b",
        routing_weights_digest="c",
        compact_route_counts=((0, 1), (2, 0)),
        capture_timestamp="now",
        metadata=metadata,
        trace_sample_id="ts",
    )
    payload = sample.to_dict()
    assert payload["trace_sample_id"] == "ts"
    assert payload["metadata"]["branch"] == "b"
