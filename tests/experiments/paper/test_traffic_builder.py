from __future__ import annotations

import json

from experiments.paper.contracts import RecordMetadata, TraceSample
from experiments.paper.traffic_builder import (
    build_traffic_instances_from_trace_bundle,
    build_traffic_instance,
    contiguous_balanced_expert_to_rank_mapping,
    round_robin_expert_to_rank_mapping,
    stable_token_owner_v1,
)


def _sample(*, layer_id: str) -> TraceSample:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    return TraceSample(
        schema_version="v1",
        model_id="m",
        model_revision="rev",
        prompt_id="sample-0",
        batch_id="req-0",
        sequence_length=4,
        layer_id=layer_id,
        num_experts=8,
        top_k=0,
        router_logits_digest=None,
        selected_experts_digest="x",
        routing_weights_digest=None,
        compact_route_counts=((0, 1, 0, 0, 0, 0, 0, 0),),
        capture_timestamp="offline",
        metadata=metadata,
        trace_sample_id=f"ts:{layer_id}",
    )


def test_token_owner_is_stable_across_layers() -> None:
    a = stable_token_owner_v1(
        model_id="m",
        model_revision="rev",
        prompt_id="sample-0",
        batch_id="req-0",
        token_position=7,
        virtual_ep_size=4,
    )
    b = stable_token_owner_v1(
        model_id="m",
        model_revision="rev",
        prompt_id="sample-0",
        batch_id="req-0",
        token_position=7,
        virtual_ep_size=4,
    )
    assert a == b


def test_contiguous_balanced_mapping_is_balanced_and_legal() -> None:
    mapping = contiguous_balanced_expert_to_rank_mapping(num_experts=8, virtual_ep_size=4)
    assert len(mapping) == 8
    assert all(0 <= value < 4 for value in mapping)
    counts = [mapping.count(rank) for rank in range(4)]
    assert max(counts) - min(counts) <= 1


def test_round_robin_mapping_is_balanced_and_legal() -> None:
    mapping = round_robin_expert_to_rank_mapping(num_experts=8, virtual_ep_size=4)
    assert len(mapping) == 8
    assert all(0 <= value < 4 for value in mapping)
    counts = [mapping.count(rank) for rank in range(4)]
    assert max(counts) - min(counts) <= 1


def test_build_traffic_instance_records_separate_current_and_target_mapping_digests() -> None:
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    sample = _sample(layer_id="1")
    mapping = contiguous_balanced_expert_to_rank_mapping(num_experts=sample.num_experts, virtual_ep_size=4)
    instance = build_traffic_instance(
        trace_sample=sample,
        p0_matrix=((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        p1_matrix=((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        p2_matrix=((0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0)),
        virtual_ep_size=4,
        metadata=metadata,
        mapping=mapping,
        placement_policy_id="contiguous_balanced",
        current_layer_mapping_digest="current-digest",
        target_layer_mapping_digest="target-digest",
    )
    assert instance.source_ownership_policy_id == "stable_token_owner_v1"
    assert instance.placement_policy_id == "contiguous_balanced"
    assert instance.current_layer_mapping_digest == "current-digest"
    assert instance.target_layer_mapping_digest == "target-digest"


def test_build_traffic_uses_target_layer_mapping_for_p2(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "summary.json").write_text(
        json.dumps({"sample_ids": ["sample-0"], "request_ids": ["req-0"], "moe_layer_count": 2}),
        encoding="utf-8",
    )
    (bundle / "architecture_probe.json").write_text(
        json.dumps(
            {
                "layers": [
                    {"layer_index": 0, "gate_weight_shape": [4, 8]},
                    {"layer_index": 1, "gate_weight_shape": [8, 8]},
                ],
                "moe_layer_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (bundle / "trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"request_id": "req-0", "sample_id": "sample-0", "token_position": 0, "layer_id": 0, "expert_id": 0, "topk_rank": 0, "routing_weight": 1.0, "topk": 1}),
                json.dumps({"request_id": "req-0", "sample_id": "sample-0", "token_position": 0, "layer_id": 1, "expert_id": 7, "topk_rank": 0, "routing_weight": 1.0, "topk": 1}),
            ]
        ),
        encoding="utf-8",
    )
    metadata = RecordMetadata("b", "c", "d", "r", 0, "m", "rev")
    _samples, traffic_instances = build_traffic_instances_from_trace_bundle(
        bundle_dir=bundle,
        virtual_ep_sizes=(2,),
        selected_layers={"0"},
        metadata=metadata,
        cost_model_id="formal_replay_makespan",
    )
    instance = traffic_instances[0]
    assert instance.current_layer_mapping_digest != instance.target_layer_mapping_digest
