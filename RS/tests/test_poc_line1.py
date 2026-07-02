from __future__ import annotations

from routesense.evaluation import (
    analyze_cross_layer_correlation,
    analyze_cross_layer_predictability,
    build_owner_by_expert,
    build_predicted_traffic,
    build_same_prompt_batches,
    combine_matrix_from_dispatch,
    fast_schedule_barrier_aware_birkhoff,
    fast_schedule_birkhoff,
    fast_schedule_completion_balanced,
    fast_schedule_cp_local_swap,
    fast_schedule_cp_lpt,
    fast_schedule_critical_path_compression,
    fast_schedule_grasp,
    fast_schedule_ibbr,
    fast_schedule_iterated_greedy,
    fast_schedule_lagrangian,
    fast_schedule_lns,
    fast_schedule_lookahead_lpt,
    fast_schedule_phase_aware_greedy,
    fast_schedule_pairwise,
    fast_schedule_randomized_multistart_birkhoff,
    fast_schedule_simulated_annealing,
    fast_schedule_tabu_search,
    fast_schedule_two_stage,
    greedy_schedule_single_layer,
    greedy_schedule_pairwise,
    evaluate_gate2,
    pairwise_oracle,
    run_pairwise_analysis,
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


def test_build_predicted_traffic_uses_predicted_topk_layout():
    from routesense.evaluation.poc_line1 import TraceRecord

    grouped = {
        ("sample-0", 0, 1): [TraceRecord("req", "sample-0", 0, 1, 0, 0, 1.0, 2)],
        ("sample-0", 1, 1): [TraceRecord("req", "sample-0", 1, 1, 1, 0, 1.0, 2)],
        ("sample-0", 2, 1): [TraceRecord("req", "sample-0", 2, 1, 2, 0, 1.0, 2)],
        ("sample-0", 3, 1): [TraceRecord("req", "sample-0", 3, 1, 3, 0, 1.0, 2)],
    }
    predicted = {
        "sample-0": {
            (0, 1): [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
            ]
        }
    }
    owner = {0: 0, 1: 1, 2: 2, 3: 3}
    matrix = build_predicted_traffic(
        sample_id="sample-0",
        from_layer=0,
        to_layer=1,
        grouped_records=grouped,
        predicted_topk_by_sample=predicted,
        owner_by_expert=owner,
        num_gpus=4,
        topk=2,
    )
    assert matrix[0][0] == 1 and matrix[0][1] == 1
    assert matrix[1][1] == 1 and matrix[1][2] == 1
    assert matrix[2][2] == 1 and matrix[2][3] == 1
    assert matrix[3][3] == 1 and matrix[3][0] == 1


def test_pairwise_oracle_improves_or_matches_greedy():
    dispatch = [
        [0, 4, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 2],
        [1, 0, 0, 0],
    ]
    combine = combine_matrix_from_dispatch(dispatch, 1.0)
    next_dispatch = [
        [0, 0, 5, 0],
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    greedy = greedy_schedule_pairwise(dispatch, combine, next_dispatch, 4)
    oracle = pairwise_oracle(dispatch, combine, next_dispatch, 4)
    assert oracle["makespan"] <= greedy
    assert oracle["chunk_count"] > 0
    assert oracle["schedule"]


def test_relaxed_models_do_not_worsen_greedy_makespan():
    dispatch = [
        [0, 4, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 2],
        [1, 0, 0, 0],
    ]
    combine = combine_matrix_from_dispatch(dispatch, 1.0)
    next_dispatch = [
        [0, 0, 5, 0],
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    half = greedy_schedule_pairwise(dispatch, combine, next_dispatch, 4, model="half_duplex")
    full = greedy_schedule_pairwise(dispatch, combine, next_dispatch, 4, model="full_duplex")
    incast = greedy_schedule_pairwise(dispatch, combine, next_dispatch, 4, model="incast_only")
    assert full <= half
    assert incast <= full


def test_relaxed_models_do_not_worsen_oracle_makespan():
    dispatch = [
        [0, 4, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 2],
        [1, 0, 0, 0],
    ]
    combine = combine_matrix_from_dispatch(dispatch, 1.0)
    next_dispatch = [
        [0, 0, 5, 0],
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    half = pairwise_oracle(dispatch, combine, next_dispatch, 4, model="half_duplex")
    full = pairwise_oracle(dispatch, combine, next_dispatch, 4, model="full_duplex")
    incast = pairwise_oracle(dispatch, combine, next_dispatch, 4, model="incast_only")
    assert full["makespan"] <= half["makespan"]
    assert incast["makespan"] <= full["makespan"]


def test_fast_schedule_pairwise_returns_valid_schedule():
    dispatch = [
        [0, 8, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 2],
        [2, 0, 0, 0],
    ]
    combine = combine_matrix_from_dispatch(dispatch, 1.0)
    next_dispatch = [
        [0, 0, 9, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [4, 0, 0, 0],
    ]
    result = fast_schedule_pairwise(dispatch, combine, next_dispatch, 4)
    assert result["makespan"] >= 0
    assert result["schedule"]
    assert result["candidates"]


def test_all_fast_scheduler_variants_return_valid_schedule():
    dispatch = [
        [0, 8, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 2],
        [2, 0, 0, 0],
    ]
    combine = combine_matrix_from_dispatch(dispatch, 1.0)
    next_dispatch = [
        [0, 0, 9, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [4, 0, 0, 0],
    ]
    for scheduler in (
        fast_schedule_lookahead_lpt,
        fast_schedule_cp_lpt,
        fast_schedule_birkhoff,
        fast_schedule_phase_aware_greedy,
        fast_schedule_barrier_aware_birkhoff,
        fast_schedule_randomized_multistart_birkhoff,
        fast_schedule_tabu_search,
        fast_schedule_lns,
        fast_schedule_simulated_annealing,
        fast_schedule_lagrangian,
        fast_schedule_grasp,
        fast_schedule_completion_balanced,
        fast_schedule_two_stage,
        fast_schedule_critical_path_compression,
        fast_schedule_ibbr,
        fast_schedule_iterated_greedy,
        fast_schedule_cp_local_swap,
    ):
        payload = scheduler(dispatch, combine, next_dispatch, 4)
        assert payload["makespan"] >= 0
        assert payload["schedule"]
        assert payload["solve_time_ms"] >= 0
        assert payload["strategy"]


def test_pairwise_oracle_has_no_phase_barrier_in_schedule():
    dispatch = [
        [0, 0, 4, 0],
        [0, 0, 0, 10],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    combine = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    next_dispatch = [
        [0, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    oracle = pairwise_oracle(dispatch, combine, next_dispatch, 4)
    phase0_end = max(item["end"] for item in oracle["schedule"] if item["phase"] == 0)
    phase1_start = min(item["start"] for item in oracle["schedule"] if item["phase"] == 1)
    # no global phase barrier: later phase can start before every phase0 chunk globally ends
    assert phase1_start < phase0_end


def test_evaluate_gate2_decision_labels():
    payload = evaluate_gate2([20.0, 18.0], [12.0, 11.0])
    assert payload["decision"] == "PASS"
    payload = evaluate_gate2([20.0, 18.0], [6.0, 7.0])
    assert payload["decision"] == "MARGINAL"
    payload = evaluate_gate2([20.0, 18.0], [2.0, 3.0])
    assert payload["decision"] == "PREDICTION_BOTTLENECK"
    payload = evaluate_gate2([3.0, 4.0], [2.0, 3.0])
    assert payload["decision"] == "NO_SCHEDULING_SPACE"


def test_run_pairwise_analysis_reports_gate2_summary():
    from routesense.evaluation.poc_line1 import TraceRecord
    import torch

    records = []
    for token_position, expert_pair in enumerate(((0, 1), (1, 2), (2, 3), (3, 0))):
        records.append(TraceRecord("req", "sample-0", token_position, 0, expert_pair[0], 0, 0.8, 2))
        records.append(TraceRecord("req", "sample-0", token_position, 0, expert_pair[1], 1, 0.2, 2))
        records.append(TraceRecord("req", "sample-0", token_position, 1, expert_pair[0], 0, 0.8, 2))
        records.append(TraceRecord("req", "sample-0", token_position, 1, expert_pair[1], 1, 0.2, 2))

    hidden_states = {
        "sample-0": {
            0: torch.tensor([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32),
            1: torch.tensor([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32),
        }
    }
    gate_weights = {
        "sample-0": {
            0: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            1: torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        }
    }
    owner = {0: 0, 1: 1, 2: 2, 3: 3}
    report = run_pairwise_analysis(
        records,
        hidden_states_by_sample=hidden_states,
        gate_weights_by_sample=gate_weights,
        owner_by_expert=owner,
        num_gpus=4,
        topk=2,
    )
    assert report["summary"]["pair_count"] == 1
    assert "fast_improvement_pct" in report["summary"]
    assert "phase_aware_greedy_improvement_pct" in report["summary"]
    assert "birkhoff_improvement_pct" in report["summary"]
    assert "lagrangian_improvement_pct" in report["summary"]
    assert "iterated_greedy_improvement_pct" in report["summary"]
    assert "oracle_prediction_gap_pct" in report["summary"]
    assert "fast_latency_ms" in report["summary"]
    assert "phase_aware_greedy_latency_ms" in report["summary"]
    assert "lagrangian_latency_ms" in report["summary"]
    assert "oracle_perfect_solver_statuses" in report["summary"]
    assert "oracle_predicted_solver_statuses" in report["summary"]
    assert "gate2_decision" in report["summary"]
    assert "decision" in report["summary"]["gate2_decision"]
    assert len(report["results"]) == 1
    assert "fast_makespan" in report["results"][0]
    assert "phase_aware_greedy_makespan" in report["results"][0]
    assert "birkhoff_makespan" in report["results"][0]
    assert "lagrangian_makespan" in report["results"][0]
    assert "iterated_greedy_makespan" in report["results"][0]
    assert "oracle_prediction_gap_pct" in report["results"][0]
    assert "oracle_perfect_latency_ms" in report["results"][0]
