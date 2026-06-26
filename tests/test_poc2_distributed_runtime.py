from __future__ import annotations

from routesense_poc2.distributed_runtime import (
    PlacementEntry,
    WorkloadBucket,
    WorkloadPlan,
    _go_no_go,
    _paired,
    _position_effects,
    dependency_score_for_bucket,
    available_audit_strategies,
    apply_placement,
    build_placement_map,
    counterbalanced_orders,
    load_plan_snapshot,
    mark_state_meaningful,
    release_rounds_for_order,
    resolve_strategy_orders,
    save_plan_snapshot,
    shuffled_dependency_order,
    strong_state_score_for_bucket,
    lina_inspired_score_for_bucket,
)
from routesense_poc2.scheduler import RuntimeState
from routesense_poc2.scheduler import BucketRecord


def test_build_placement_map_is_deterministic():
    left = build_placement_map(8, 4, "modulo")
    right = build_placement_map(8, 4, "modulo")
    assert [item.destination_rank for item in left] == [item.destination_rank for item in right]


def test_balanced_mapping_spreads_experts():
    placement = build_placement_map(10, 4, "balanced")
    ranks = [item.destination_rank for item in placement]
    assert min(ranks) == 0
    assert max(ranks) == 3


def test_apply_placement_updates_destination_rank():
    buckets = [
        BucketRecord(bucket_id="b0", destination_id=3, destination_rank=0, arrival_index=0, token_count=2),
        BucketRecord(bucket_id="b1", destination_id=4, destination_rank=0, arrival_index=1, token_count=2),
    ]
    placement = [PlacementEntry(expert_id=index, destination_rank=index % 2) for index in range(8)]
    updated = apply_placement(buckets, placement)
    assert updated[0].destination_rank == 1
    assert updated[1].destination_rank == 0


def test_counterbalanced_orders_rotate():
    orders = counterbalanced_orders(["fifo", "state-only", "dependency-only", "full"], 4)
    assert orders[0][0] == "fifo"
    assert orders[1][0] == "state-only"
    assert orders[2][0] == "dependency-only"
    assert orders[3][0] == "full"


def test_resolve_strategy_orders_randomized_has_right_count():
    orders = resolve_strategy_orders("randomized", 3, 42)
    assert len(orders) == 3
    assert all(len(order) == 9 for order in orders)


def test_state_meaningful_flag():
    assert mark_state_meaningful(1) is False
    assert mark_state_meaningful(2) is True


def test_workload_plan_roundtrip_dict():
    plan = WorkloadPlan(
        plan_id="p0",
        model_key="olmoe",
        seed=42,
        microbatch_id=0,
        layer_id=1,
        layer_path="layer",
        hidden_dim=128,
        intermediate_dim=256,
        placement_policy="balanced",
        placement_mapping=[PlacementEntry(expert_id=0, destination_rank=0)],
        bucket_workloads=[
            WorkloadBucket(
                bucket_id="b0",
                destination_id=0,
                destination_rank=0,
                token_count=2,
                payload_rows=2,
                hidden_dim=128,
                intermediate_dim=256,
                estimated_service_units=2.0,
                source_count=2,
                source_coverage=0.5,
                coactive_peer_degree=0.1,
                coactive_event_density=0.1,
                position_spread=0.0,
                bridge_score=0.05,
                route_share=0.2,
                density_over_mean=0.3,
                size_norm=0.4,
                inverse_size_rank_norm=1.0,
                is_hot_bucket=True,
            )
        ],
        routing_summary={"x": 1},
    )
    payload = plan.to_dict()
    assert payload["plan_id"] == "p0"
    assert payload["bucket_workloads"][0]["bucket_id"] == "b0"


def test_hotspot_mapping_concentrates_prefix_experts():
    placement = build_placement_map(16, 4, "hotspot")
    first_half = [item.destination_rank for item in placement[:8]]
    assert set(first_half).issubset({0, 1})


def test_shuffled_dependency_preserves_distribution_changes_assignment():
    buckets = [
        WorkloadBucket(
            bucket_id=f"b{i}",
            destination_id=i,
            destination_rank=i % 2,
            token_count=2 + i,
            payload_rows=2 + i,
            hidden_dim=64,
            intermediate_dim=128,
            estimated_service_units=1.0 + i,
            source_count=1,
            source_coverage=0.1 * i,
            coactive_peer_degree=0.2 * i,
            coactive_event_density=0.05 * i,
            position_spread=0.03 * i,
            bridge_score=0.02 * i,
            route_share=0.1,
            density_over_mean=0.2,
            size_norm=0.3,
            inverse_size_rank_norm=1.0,
            is_hot_bucket=False,
        )
        for i in range(1, 6)
    ]
    baseline_scores = sorted(dependency_score_for_bucket(bucket) for bucket in buckets)
    shuffled = shuffled_dependency_order(buckets, RuntimeState(), seed=7, combine_with_state=False)
    shuffled_scores = sorted(dependency_score_for_bucket(bucket) for bucket in buckets)
    assert baseline_scores == shuffled_scores
    assert shuffled != [bucket.bucket_id for bucket in buckets]


def test_strong_state_and_lina_scores_do_not_depend_on_dependency_fields_when_state_same():
    state = RuntimeState(rank_backlog_work={0: 2.0}, rank_queue_depth={0: 1}, destination_hotness={0: 3.0})
    left = WorkloadBucket(
        bucket_id="x",
        destination_id=0,
        destination_rank=0,
        token_count=4,
        payload_rows=4,
        hidden_dim=64,
        intermediate_dim=128,
        estimated_service_units=2.0,
        source_count=2,
        source_coverage=0.1,
        coactive_peer_degree=0.1,
        coactive_event_density=0.1,
        position_spread=0.1,
        bridge_score=0.1,
        route_share=0.4,
        density_over_mean=0.5,
        size_norm=0.6,
        inverse_size_rank_norm=0.7,
        is_hot_bucket=False,
    )
    right = WorkloadBucket(**{**left.__dict__, "source_coverage": 0.9, "coactive_peer_degree": 0.9, "bridge_score": 0.9})
    assert strong_state_score_for_bucket(left, state) == strong_state_score_for_bucket(right, state)
    assert lina_inspired_score_for_bucket(left, state) == lina_inspired_score_for_bucket(right, state)


def test_paired_matches_by_key_and_flags_distribution_asymmetry():
    left = [
        {"repetition_index": 0, "microbatch_id": 0, "plan_id": "p0", "end_to_end_batch_completion_ms": -10.0, "strategy": "a", "strategy_order": ["a", "b"]},
        {"repetition_index": 1, "microbatch_id": 0, "plan_id": "p0", "end_to_end_batch_completion_ms": 10.0, "strategy": "a", "strategy_order": ["b", "a"]},
    ]
    right = [
        {"repetition_index": 0, "microbatch_id": 0, "plan_id": "p0", "end_to_end_batch_completion_ms": 0.0, "strategy": "b", "strategy_order": ["a", "b"]},
        {"repetition_index": 1, "microbatch_id": 0, "plan_id": "p0", "end_to_end_batch_completion_ms": 9.0, "strategy": "b", "strategy_order": ["b", "a"]},
    ]
    paired = _paired(left, right)
    assert paired["paired_count"] == 2
    assert "position_adjusted_mean_delta_ms" in paired
    assert "per_repetition_pairs" in paired


def test_position_effects_include_correlation():
    records = [
        {"warmup": False, "strategy": "full", "strategy_order": ["full", "fifo"], "end_to_end_batch_completion_ms": 10.0},
        {"warmup": False, "strategy": "full", "strategy_order": ["fifo", "full"], "end_to_end_batch_completion_ms": 20.0},
    ]
    effects = _position_effects(records)
    assert "position_correlation" in effects["full"]


def test_go_no_go_rules():
    no_go = _go_no_go(
        {
            "dependency_only_vs_shuffled_dependency": {"paired_count": 6, "bootstrap_ci95": [-0.1, 0.1], "win_rate": 0.5},
            "full_vs_full_with_shuffled_dependency": {"paired_count": 6, "bootstrap_ci95": [-0.1, 0.1], "win_rate": 0.5},
            "full_vs_strong_state": {"paired_count": 6, "bootstrap_ci95": [-0.1, 0.1], "win_rate": 0.5},
            "full_vs_lina_inspired": {"paired_count": 6, "bootstrap_ci95": [-0.1, 0.1], "win_rate": 0.5},
        },
        True,
    )
    assert no_go["decision"] == "no_go"
    go = _go_no_go(
        {
            "dependency_only_vs_shuffled_dependency": {"paired_count": 6, "bootstrap_ci95": [-2.0, -0.5], "win_rate": 0.8},
            "full_vs_full_with_shuffled_dependency": {"paired_count": 6, "bootstrap_ci95": [-1.0, -0.1], "win_rate": 0.7},
            "full_vs_strong_state": {"paired_count": 6, "bootstrap_ci95": [-2.0, -0.3], "win_rate": 0.8},
            "full_vs_lina_inspired": {"paired_count": 6, "bootstrap_ci95": [-1.0, 0.1], "win_rate": 0.55},
        },
        True,
    )
    assert go["decision"] == "go"


def test_plan_snapshot_roundtrip(tmp_path):
    plan = WorkloadPlan(
        plan_id="p0",
        model_key="olmoe",
        seed=1,
        microbatch_id=0,
        layer_id=0,
        layer_path="layer",
        hidden_dim=64,
        intermediate_dim=128,
        placement_policy="balanced",
        placement_mapping=[PlacementEntry(expert_id=0, destination_rank=0)],
        bucket_workloads=[
            WorkloadBucket(
                bucket_id="b0",
                destination_id=0,
                destination_rank=0,
                token_count=4,
                payload_rows=4,
                hidden_dim=64,
                intermediate_dim=128,
                estimated_service_units=2.0,
                source_count=1,
                source_coverage=0.1,
                coactive_peer_degree=0.1,
                coactive_event_density=0.1,
                position_spread=0.1,
                bridge_score=0.1,
                route_share=0.2,
                density_over_mean=0.3,
                size_norm=0.4,
                inverse_size_rank_norm=0.5,
                is_hot_bucket=False,
            )
        ],
        routing_summary={"token_count": 4},
    )
    path = save_plan_snapshot(tmp_path, [plan])
    loaded = load_plan_snapshot(path)
    assert loaded[0].plan_id == "p0"
    assert loaded[0].bucket_workloads[0].payload_rows == 4


def test_release_rounds_for_order():
    rounds = release_rounds_for_order(["a", "b", "c", "d", "e"], 2)
    assert rounds == [["a", "b"], ["c", "d"], ["e"]]


def test_available_audit_strategies():
    assert available_audit_strategies() == ["fifo", "random-order", "strong-state", "full"]
