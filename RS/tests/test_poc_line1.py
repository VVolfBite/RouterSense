from __future__ import annotations

from routesense.evaluation import (
    analyze_cross_layer_correlation,
    analyze_cross_layer_predictability,
    build_owner_by_expert,
    build_same_prompt_batches,
    combine_matrix_from_dispatch,
    greedy_schedule_single_layer,
    oracle_schedule_multi_layer,
    simulate_oracle_vs_greedy,
    spearman_rank_correlation,
)
from routesense.trace.olmoe_router_trace import discover_moe_layer_ids


def _make_record(
    sample_id: str,
    token_position: int,
    layer_id: int,
    expert_id: int,
    topk_rank: int,
    routing_weight: float,
    *,
    topk: int = 8,
) -> dict[str, object]:
    return {
        "request_id": f"req-{sample_id}",
        "sample_id": sample_id,
        "token_position": token_position,
        "layer_id": layer_id,
        "expert_id": expert_id,
        "topk_rank": topk_rank,
        "routing_weight": routing_weight,
        "topk": topk,
    }


def test_discover_moe_layer_ids_from_dummy_model():
    class DummyMLP:
        def __init__(self, moe: bool) -> None:
            if moe:
                self.gate = object()
                self.experts = object()

    class DummyLayer:
        def __init__(self, moe: bool) -> None:
            self.mlp = DummyMLP(moe)

    class DummyInnerModel:
        def __init__(self) -> None:
            self.layers = [DummyLayer(True), DummyLayer(False), DummyLayer(True)]

    class DummyModel:
        def __init__(self) -> None:
            self.model = DummyInnerModel()

    assert discover_moe_layer_ids(DummyModel()) == [0, 2]


def test_cross_layer_analysis_reports_expected_overlap():
    from routesense.evaluation.poc_line1 import TraceRecord

    records = []
    for token_position in range(12):
        for topk_rank, (expert_id, weight) in enumerate([(0, 0.6), (1, 0.3), (2, 0.1)]):
            records.append(TraceRecord("req", "sample-0", token_position, 0, expert_id, topk_rank, weight, 8))
            records.append(TraceRecord("req", "sample-0", token_position, 5, expert_id, topk_rank, weight, 8))
    owner_by_expert = build_owner_by_expert(records, placement="round_robin", num_gpus=4)
    report = analyze_cross_layer_correlation(records, owner_by_expert=owner_by_expert, num_gpus=4)
    assert report["gate1_decision"]["passed"] is True
    pair = report["layer_pair_summary"]["0->5"]
    assert pair["hit_rate"]["mean"] == 1.0
    assert pair["weighted_hit_rate"]["mean"] > 0.9


def test_spearman_rank_correlation_basic():
    assert abs(spearman_rank_correlation([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12
    assert abs(spearman_rank_correlation([1, 2, 3], [6, 4, 2]) + 1.0) < 1e-12


def test_round_robin_and_skewed_placement():
    from routesense.evaluation.poc_line1 import TraceRecord

    records = [TraceRecord("req", "sample", 0, 0, expert_id, 0, 0.5, 8) for expert_id in range(20)]
    rr = build_owner_by_expert(records, placement="round_robin", num_gpus=4)
    skewed = build_owner_by_expert(records, placement="skewed", num_gpus=4)
    assert rr[0] == 0 and rr[1] == 1
    assert skewed[0] == 0
    assert set(skewed.values()).issubset({0, 1, 2, 3})


def test_build_same_prompt_batches_uses_token_position_for_source_rank():
    from routesense.evaluation.poc_line1 import TraceRecord

    records = [
        TraceRecord("req", "sample-0", token_position, 0, token_position, 0, 0.5, 8)
        for token_position in range(8)
    ]
    owner = {record.expert_id: record.expert_id % 4 for record in records}
    batches = build_same_prompt_batches(records, owner_by_expert=owner, num_gpus=4)
    assert len(batches) == 1
    matrix = batches[0]["layers"][0]["matrix"]
    assert matrix[0][0] == 2
    assert matrix[1][1] == 2
    assert matrix[2][2] == 2
    assert matrix[3][3] == 2


def test_combine_matrix_transpose():
    matrix = [
        [0, 2, 0, 1],
        [3, 0, 1, 0],
        [0, 0, 0, 4],
        [2, 0, 0, 0],
    ]
    combine = combine_matrix_from_dispatch(matrix, 1.0)
    assert combine[1][0] == 2
    assert combine[0][1] == 3
    assert combine[3][2] == 4


def test_greedy_schedule_single_layer_has_positive_makespan():
    makespan, per_gpu = greedy_schedule_single_layer(
        [
            [0, 4, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 2],
            [1, 0, 0, 0],
        ],
        4,
    )
    assert makespan > 0
    assert len(per_gpu) == 4


def test_oracle_schedule_improves_over_greedy():
    traffic = [
        [
            [0, 4, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 2],
            [1, 0, 0, 0],
        ],
        [
            [0, 0, 5, 0],
            [2, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ],
    ]
    combine = [combine_matrix_from_dispatch(matrix, 1.0) for matrix in traffic]
    from routesense.evaluation import greedy_schedule_multi_layer

    greedy = greedy_schedule_multi_layer(traffic, combine, 4)
    oracle = oracle_schedule_multi_layer(traffic, combine, 4)
    assert oracle <= greedy
    assert oracle >= 0.0


def test_simulate_oracle_vs_greedy_returns_gate2_summary():
    from routesense.evaluation.poc_line1 import TraceRecord

    records = []
    for sample_id in ("sample-0", "sample-1"):
        for token_position in range(8):
            for layer_id, expert_base in ((0, 0), (5, 4), (10, 8), (15, 12)):
                records.append(
                    TraceRecord("req", sample_id, token_position, layer_id, expert_base + (token_position % 4), 0, 0.8, 8)
                )
                records.append(
                    TraceRecord("req", sample_id, token_position, layer_id, expert_base + ((token_position + 1) % 4), 1, 0.2, 8)
                )
    owner = build_owner_by_expert(records, placement="round_robin", num_gpus=4)
    batches = build_same_prompt_batches(records, owner_by_expert=owner, num_gpus=4)
    report = simulate_oracle_vs_greedy(batches, combine_scale_factor=1.0)
    assert "summary" in report
    assert "gate2_decision" in report["summary"]
    assert len(report["results"]) == len(batches)


def test_batch_rank_correlation_uses_placement():
    from routesense.evaluation.poc_line1 import TraceRecord, build_batch_rank_correlation

    records = [
        TraceRecord("req", "sample-0", 0, 0, 10, 0, 0.8, 8),
        TraceRecord("req", "sample-0", 0, 1, 20, 0, 0.8, 8),
        TraceRecord("req", "sample-0", 1, 0, 11, 0, 0.8, 8),
        TraceRecord("req", "sample-0", 1, 1, 21, 0, 0.8, 8),
    ]
    owner_by_expert = {10: 3, 11: 3, 20: 1, 21: 1}
    report = build_batch_rank_correlation(records, owner_by_expert=owner_by_expert, num_gpus=4)
    assert "0->1" in report
    assert report["0->1"]["count"] == 1


def test_cross_layer_predictability_prefetch_accuracy():
    from routesense.evaluation.poc_line1 import TraceRecord
    import torch

    records = [
        TraceRecord("req", "sample-0", 0, 0, 1, 0, 0.8, 2),
        TraceRecord("req", "sample-0", 0, 0, 0, 1, 0.2, 2),
        TraceRecord("req", "sample-0", 0, 1, 1, 0, 0.8, 2),
        TraceRecord("req", "sample-0", 0, 1, 0, 1, 0.2, 2),
        TraceRecord("req", "sample-0", 1, 0, 0, 0, 0.8, 2),
        TraceRecord("req", "sample-0", 1, 0, 1, 1, 0.2, 2),
        TraceRecord("req", "sample-0", 1, 1, 0, 0, 0.8, 2),
        TraceRecord("req", "sample-0", 1, 1, 1, 1, 0.2, 2),
    ]
    hidden_states = {
        "sample-0": {
            0: torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32),
            1: torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32),
        }
    }
    gate_weights = {
        "sample-0": {
            0: torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            1: torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        }
    }
    owner_by_expert = {0: 0, 1: 1}
    report = analyze_cross_layer_predictability(
        records,
        hidden_states_by_sample=hidden_states,
        gate_weights_by_sample=gate_weights,
        topk=2,
        owner_by_expert=owner_by_expert,
        num_gpus=4,
    )
    assert report["gate1_decision"]["passed"] is True
    pair = report["layer_pair_summary"]["0->1"]
    assert pair["prefetch_accuracy"]["mean"] == 1.0
