from __future__ import annotations

from experiments.paper.configuration import consumed_config_payload
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


def test_consumed_config_preserves_requested_virtual_ep_sizes(tmp_path) -> None:
    payload = consumed_config_payload(
        {
            "schema_version": "paper_eval_config.v1",
            "claim_scope": "x",
            "models": {"model_id": "m", "model_revision": "r"},
            "inputs": {"source": "trace_bundle"},
            "layers": [0],
            "virtual_ep_sizes": [2, 4],
            "physical_world_size": 1,
            "policies": [],
            "predictors": [],
            "cost_model": "c",
            "seeds": {"default": 0},
            "splits": {"development": [], "validation": [], "frozen_evaluation": []},
            "measurement": {"mode": "x"},
            "eligibility": {"runtime_performance": False},
            "output": {"dir": "outputs/paper"},
        },
        output_dir=tmp_path,
        input_path="bundle",
    )
    assert payload["virtual_ep_sizes"] == [2, 4]
    assert payload["inputs"]["resolved_input"] == "bundle"
