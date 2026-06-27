from __future__ import annotations

import copy

from routesense_poc2.distributed_runtime import (
    benchmark_preflight,
    build_canonical_round_layout,
    canonical_global_plan_hash,
    canonical_global_plan_payload,
    decode_metadata_tensor_rows,
    expected_route_item_manifest,
    GlobalDispatchPlan,
    GlobalRuntimeState,
    PlacementEntry,
    WorkloadBucket,
    WorkloadPlan,
    ProtocolConfig,
    _pack_return_segments,
    _pack_return_payload_and_metadata,
    _planned_received_metadata_for_rank,
    _metadata_tensor_to_dicts,
    summarize_route_manifests,
    _hash_payload,
    _go_no_go,
    _planned_round_matrix,
    _paired,
    _position_effects,
    assign_origin_rank_from_ordinal,
    build_global_dispatch_plan,
    broadcast_global_dispatch_plan,
    build_workload_plan,
    dependency_score_for_bucket,
    available_audit_strategies,
    apply_placement,
    build_placement_map,
    counterbalanced_orders,
    latin_square_orders,
    load_plan_snapshot,
    mark_state_meaningful,
    release_rounds_for_order,
    resolve_strategy_orders,
    save_plan_snapshot,
    shuffled_dependency_order,
    strong_state_score_for_bucket,
    lina_inspired_score_for_bucket,
    _greedy_minimax_round_pack,
    update_global_runtime_state,
)
from routesense_poc2.stress import (
    build_burst_episode_schedule,
    choose_skewed_placement,
    dependency_predictiveness_gate,
    oracle_minimax_round_packer,
    scenario_manifest,
    split_calibration_and_evaluation,
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
    assert all(len(order) == len(orders[0]) for order in orders)
    assert len(orders[0]) >= 9


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
        origin_sharding="round-robin",
        placement_mapping=[PlacementEntry(expert_id=0, destination_rank=0)],
        bucket_workloads=[
            WorkloadBucket(
                bucket_id="b0",
                route_id="r0",
                token_ids=[0, 1],
                token_positions=[0, 1],
                origin_rank=0,
                destination_id=0,
                destination_rank=0,
                expert_id=0,
                layer_id=1,
                microbatch_id=0,
                token_count=2,
                payload_rows=2,
                hidden_dim=128,
                intermediate_dim=256,
                estimated_service_units=2.0,
                payload_bytes=512,
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


def test_assign_origin_rank_round_robin_and_contiguous():
    assert assign_origin_rank_from_ordinal(0, 8, 4, "round-robin") == 0
    assert assign_origin_rank_from_ordinal(5, 8, 4, "round-robin") == 1
    assert assign_origin_rank_from_ordinal(0, 8, 4, "contiguous") == 0
    assert assign_origin_rank_from_ordinal(7, 8, 4, "contiguous") == 3


def test_build_workload_plan_preserves_origin_and_route_identity():
    protocol = ProtocolConfig(
        model_key="olmoe",
        model_path="/tmp/model",
        output_dir="/tmp/out",
        artifact_dir="/tmp/art",
        world_size=4,
        origin_sharding="round-robin",
    )
    routing = {
        "layer_id": 0,
        "layer_path": "layer0",
        "bucket_records": [
            BucketRecord(bucket_id="dest1", destination_id=1, destination_rank=1, arrival_index=0, token_count=2, token_positions=[0, 1]),
            BucketRecord(bucket_id="dest2", destination_id=2, destination_rank=2, arrival_index=1, token_count=2, token_positions=[0, 1]),
        ],
        "route_items": [
            {"route_id": "r0", "token_id": 11, "token_pos": 0, "route_rank": 0, "expert_id": 1},
            {"route_id": "r1", "token_id": 23, "token_pos": 1, "route_rank": 0, "expert_id": 2},
        ],
        "routing_summary": {"token_ids": [11, 23]},
        "route_item_index_map": {
            str((0, 0, 0, 11, 0, 1, "r0")): 0,
            str((0, 0, 1, 23, 0, 2, "r1")): 1,
        },
    }
    placement = [PlacementEntry(expert_id=1, destination_rank=1), PlacementEntry(expert_id=2, destination_rank=2)]
    plan = build_workload_plan(
        config=protocol,
        microbatch_id=0,
        routing=routing,
        hidden_dim=64,
        intermediate_dim=128,
        placement=placement,
    )
    assert {bucket.origin_rank for bucket in plan.bucket_workloads} == {0, 1}
    assert {bucket.route_id for bucket in plan.bucket_workloads} == {"r0", "r1"}
    assert plan.origin_sharding == "round-robin"
    assert sorted(index for bucket in plan.bucket_workloads for index in bucket.route_item_indices) == [0, 1]


def test_build_workload_plan_preserves_topk_route_tuple_alignment():
    protocol = ProtocolConfig(
        model_key="olmoe",
        model_path="/tmp/model",
        output_dir="/tmp/out",
        artifact_dir="/tmp/art",
        world_size=2,
        origin_sharding="round-robin",
    )
    routing = {
        "layer_id": 0,
        "layer_path": "layer0",
        "bucket_records": [
            BucketRecord(bucket_id="dest1", destination_id=1, destination_rank=1, arrival_index=0, token_count=2, token_positions=[4, 7]),
            BucketRecord(bucket_id="dest2", destination_id=2, destination_rank=0, arrival_index=1, token_count=2, token_positions=[4, 7]),
        ],
        "route_items": [
            {"route_id": "r_topk_1", "token_id": 100, "token_pos": 7, "route_rank": 1, "expert_id": 2},
            {"route_id": "r_topk_0", "token_id": 100, "token_pos": 7, "route_rank": 0, "expert_id": 1},
        ],
        "routing_summary": {"token_ids": [100]},
        "route_item_index_map": {
            str((0, 0, 7, 100, 0, 1, "r_topk_0")): 0,
            str((0, 0, 7, 100, 1, 2, "r_topk_1")): 1,
        },
    }
    placement = [PlacementEntry(expert_id=1, destination_rank=1), PlacementEntry(expert_id=2, destination_rank=0)]
    plan = build_workload_plan(
        config=protocol,
        microbatch_id=0,
        routing=routing,
        hidden_dim=64,
        intermediate_dim=128,
        placement=placement,
    )
    route_pairs = sorted(
        (
            bucket.route_item_indices[0],
            bucket.route_item_ids[0],
            bucket.route_ranks[0],
            bucket.expert_id,
            bucket.token_ids[0],
        )
        for bucket in plan.bucket_workloads
    )
    assert route_pairs[0][2:] == (0, 1, 100)
    assert route_pairs[1][2:] == (1, 2, 100)


def test_contiguous_origin_sharding_uses_token_ordinal_not_token_id():
    protocol = ProtocolConfig(
        model_key="olmoe",
        model_path="/tmp/model",
        output_dir="/tmp/out",
        artifact_dir="/tmp/art",
        world_size=2,
        origin_sharding="contiguous",
    )
    routing = {
        "layer_id": 0,
        "layer_path": "layer0",
        "bucket_records": [
            BucketRecord(bucket_id="dest1", destination_id=1, destination_rank=1, arrival_index=0, token_count=4, token_positions=[9, 1, 7, 3]),
        ],
        "route_items": [
            {"route_id": "r0", "token_id": 42, "token_pos": 9, "route_rank": 0, "expert_id": 1},
            {"route_id": "r1", "token_id": 100, "token_pos": 1, "route_rank": 0, "expert_id": 1},
            {"route_id": "r2", "token_id": 7, "token_pos": 7, "route_rank": 0, "expert_id": 1},
            {"route_id": "r3", "token_id": 55, "token_pos": 3, "route_rank": 0, "expert_id": 1},
        ],
        "routing_summary": {"token_ids": [42, 100, 7, 55]},
        "route_item_index_map": {
            str((0, 0, 1, 100, 0, 1, "r1")): 0,
            str((0, 0, 3, 55, 0, 1, "r3")): 1,
            str((0, 0, 7, 7, 0, 1, "r2")): 2,
            str((0, 0, 9, 42, 0, 1, "r0")): 3,
        },
    }
    placement = [PlacementEntry(expert_id=1, destination_rank=1)]
    plan = build_workload_plan(
        config=protocol,
        microbatch_id=0,
        routing=routing,
        hidden_dim=64,
        intermediate_dim=128,
        placement=placement,
    )
    detail = plan.routing_summary["token_origin_rank_detail"]
    assert detail["7"]["origin_rank"] == 0
    assert detail["42"]["origin_rank"] == 0
    assert detail["55"]["origin_rank"] == 1
    assert detail["100"]["origin_rank"] == 1


def test_total_dispatch_matrix_invariant_across_policy_orders():
    buckets = [
        WorkloadBucket(
            bucket_id="b0",
            route_id="r0",
            token_ids=[0],
            token_positions=[0],
            origin_rank=0,
            destination_id=3,
            destination_rank=1,
            expert_id=3,
            layer_id=0,
            microbatch_id=0,
            token_count=1,
            payload_rows=2,
            hidden_dim=64,
            intermediate_dim=128,
            estimated_service_units=2.0,
            payload_bytes=256,
            source_count=2,
            source_coverage=0.2,
            coactive_peer_degree=0.1,
            coactive_event_density=0.1,
            position_spread=0.0,
            bridge_score=0.1,
            route_share=0.2,
            density_over_mean=0.3,
            size_norm=0.4,
            inverse_size_rank_norm=1.0,
            is_hot_bucket=False,
        ),
        WorkloadBucket(
            bucket_id="b1",
            route_id="r1",
            token_ids=[1],
            token_positions=[1],
            origin_rank=1,
            destination_id=2,
            destination_rank=0,
            expert_id=2,
            layer_id=0,
            microbatch_id=0,
            token_count=1,
            payload_rows=3,
            hidden_dim=64,
            intermediate_dim=128,
            estimated_service_units=2.0,
            payload_bytes=384,
            source_count=2,
            source_coverage=0.2,
            coactive_peer_degree=0.1,
            coactive_event_density=0.1,
            position_spread=0.0,
            bridge_score=0.1,
            route_share=0.2,
            density_over_mean=0.3,
            size_norm=0.4,
            inverse_size_rank_norm=0.5,
            is_hot_bucket=False,
        ),
    ]
    fifo_rows, _ = _planned_round_matrix(buckets, 2, 64)
    reordered_rows, _ = _planned_round_matrix([buckets[1], buckets[0]], 2, 64)
    assert fifo_rows == reordered_rows


def test_shuffled_dependency_preserves_distribution_changes_assignment():
    buckets = [
        WorkloadBucket(
            bucket_id=f"b{i}",
            route_id=f"r{i}",
            token_ids=[i],
            token_positions=[i],
            origin_rank=i % 2,
            destination_id=i,
            destination_rank=i % 2,
            expert_id=i,
            layer_id=0,
            microbatch_id=0,
            token_count=2 + i,
            payload_rows=2 + i,
            hidden_dim=64,
            intermediate_dim=128,
            estimated_service_units=1.0 + i,
            payload_bytes=(2 + i) * 64 * 2,
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
        route_id="rx",
        token_ids=[0],
        token_positions=[0],
        origin_rank=0,
        destination_id=0,
        destination_rank=0,
        expert_id=0,
        layer_id=0,
        microbatch_id=0,
        token_count=4,
        payload_rows=4,
        hidden_dim=64,
        intermediate_dim=128,
        estimated_service_units=2.0,
        payload_bytes=4 * 64 * 2,
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


def test_strong_state_changes_with_global_flow_fields():
    base = WorkloadBucket(
        bucket_id="x",
        route_id="rx",
        token_ids=[0],
        token_positions=[0],
        origin_rank=0,
        destination_id=0,
        destination_rank=0,
        expert_id=0,
        layer_id=0,
        microbatch_id=0,
        token_count=4,
        payload_rows=4,
        hidden_dim=64,
        intermediate_dim=128,
        estimated_service_units=2.0,
        payload_bytes=4 * 64 * 2,
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
    low = RuntimeState(
        source_outbound_pending_bytes={0: 0.0},
        destination_inbound_pending_bytes={0: 0.0},
        flow_pending_bytes={"0->0": 0.0},
        return_flow_pending_bytes={"0->0": 0.0},
        rank_compute_queue_rows={0: 0.0},
        expert_pending_rows={0: 0.0},
    )
    high = RuntimeState(
        source_outbound_pending_bytes={0: 1 << 20},
        destination_inbound_pending_bytes={0: 1 << 20},
        flow_pending_bytes={"0->0": 1 << 20},
        return_flow_pending_bytes={"0->0": 1 << 20},
        rank_compute_queue_rows={0: 128.0},
        expert_pending_rows={0: 128.0},
        rank_dispatch_time_ms={0: 10.0},
        rank_compute_time_ms={0: 20.0},
        rank_return_time_ms={0: 10.0},
    )
    assert strong_state_score_for_bucket(base, low) != strong_state_score_for_bucket(base, high)


def test_strong_state_sort_defect_projected_bucket_fields_do_not_change_score():
    runtime_state = RuntimeState()
    base = WorkloadBucket(
        bucket_id="x",
        route_id="rx",
        token_ids=[0],
        token_positions=[0],
        origin_rank=0,
        destination_id=0,
        destination_rank=1,
        expert_id=2,
        layer_id=0,
        microbatch_id=0,
        token_count=4,
        payload_rows=4,
        hidden_dim=64,
        intermediate_dim=128,
        estimated_service_units=2.0,
        payload_bytes=4 * 64 * 2,
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
    mutated = copy.deepcopy(base)
    mutated.flow_pending_bytes = float(1 << 20)
    mutated.destination_inbound_pending_bytes = float(1 << 20)
    assert strong_state_score_for_bucket(base, runtime_state) == strong_state_score_for_bucket(mutated, runtime_state)


def test_strong_state_packer_changes_choice_with_projected_pressure_not_dependency():
    buckets = [
        BucketRecord(
            bucket_id="hot",
            route_item_ids=["r0"],
            route_item_indices=[0],
            origin_rank=0,
            destination_id=3,
            destination_rank=1,
            expert_id=3,
            arrival_index=0,
            token_count=1,
            payload_rows=4,
            payload_bytes=512,
            source_coverage=0.0,
            coactive_peer_degree=0.0,
            coactive_event_density=0.0,
            position_spread=0.0,
            bridge_score=0.0,
            route_share=0.2,
            density_over_mean=0.2,
            size_norm=0.3,
            inverse_size_rank_norm=0.6,
            flow_pending_bytes=float(1 << 20),
            destination_inbound_pending_bytes=float(1 << 20),
        ),
        BucketRecord(
            bucket_id="cool",
            route_item_ids=["r1"],
            route_item_indices=[1],
            origin_rank=1,
            destination_id=4,
            destination_rank=0,
            expert_id=4,
            arrival_index=1,
            token_count=1,
            payload_rows=4,
            payload_bytes=512,
            source_coverage=0.9,
            coactive_peer_degree=0.9,
            coactive_event_density=0.9,
            position_spread=0.9,
            bridge_score=0.9,
            route_share=0.2,
            density_over_mean=0.2,
            size_norm=0.3,
            inverse_size_rank_norm=0.6,
        ),
    ]
    packed = _greedy_minimax_round_pack(buckets, round_size=1, runtime_state=RuntimeState(), policy_seed=7, arrival_jitter_mode="none", oracle=False)
    assert set(packed["release_order"]) == {"hot", "cool"}
    dep_mutated = [copy.deepcopy(bucket) for bucket in buckets]
    dep_mutated[0].source_coverage = 0.99
    dep_mutated[0].coactive_peer_degree = 0.99
    dep_mutated[0].bridge_score = 0.99
    dep_mutated[1].source_coverage = 0.01
    dep_mutated[1].coactive_peer_degree = 0.01
    dep_mutated[1].bridge_score = 0.01
    packed_dep = _greedy_minimax_round_pack(dep_mutated, round_size=1, runtime_state=RuntimeState(), policy_seed=7, arrival_jitter_mode="none", oracle=False)
    assert packed_dep["release_order"] == packed["release_order"]


def test_source_arrival_jitter_does_not_depend_on_destination_rank():
    left = WorkloadBucket(
        bucket_id="a",
        route_id="ra",
        token_ids=[0],
        token_positions=[5],
        origin_rank=1,
        destination_id=1,
        destination_rank=0,
        expert_id=1,
        layer_id=0,
        microbatch_id=2,
        token_count=1,
        payload_rows=1,
        hidden_dim=64,
        intermediate_dim=128,
        estimated_service_units=1.0,
        payload_bytes=128,
        source_count=1,
        source_coverage=0.0,
        coactive_peer_degree=0.0,
        coactive_event_density=0.0,
        position_spread=0.0,
        bridge_score=0.0,
        route_share=0.1,
        density_over_mean=0.1,
        size_norm=0.1,
        inverse_size_rank_norm=0.1,
        is_hot_bucket=False,
    )
    right = WorkloadBucket(**{**left.__dict__, "destination_rank": 3, "destination_id": 9, "expert_id": 9})
    packed_left = _greedy_minimax_round_pack([left], round_size=1, runtime_state=RuntimeState(), policy_seed=11, arrival_jitter_mode="moderate", oracle=False)
    packed_right = _greedy_minimax_round_pack([right], round_size=1, runtime_state=RuntimeState(), policy_seed=11, arrival_jitter_mode="moderate", oracle=False)
    assert packed_left["release_order"] == packed_right["release_order"]


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
        {"warmup": False, "strategy": "full", "strategy_order": ["full", "fifo"], "global_end_to_end_completion_ms": 10.0},
        {"warmup": False, "strategy": "full", "strategy_order": ["fifo", "full"], "global_end_to_end_completion_ms": 20.0},
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
        origin_sharding="round-robin",
        placement_mapping=[PlacementEntry(expert_id=0, destination_rank=0)],
        bucket_workloads=[
            WorkloadBucket(
                bucket_id="b0",
                route_id="r0",
                token_ids=[0, 1, 2, 3],
                token_positions=[0, 1, 2, 3],
                origin_rank=0,
                destination_id=0,
                destination_rank=0,
                expert_id=0,
                layer_id=0,
                microbatch_id=0,
                token_count=4,
                payload_rows=4,
                hidden_dim=64,
                intermediate_dim=128,
                estimated_service_units=2.0,
                payload_bytes=4 * 64 * 2,
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


def test_non_rank0_may_not_generate_global_dispatch_plan():
    import routesense_poc2.distributed_runtime as runtime

    protocol = ProtocolConfig(model_key="olmoe", model_path="/tmp/model", output_dir="/tmp/out", artifact_dir="/tmp/art")
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
        origin_sharding="round-robin",
        placement_mapping=[],
        bucket_workloads=[],
        routing_summary={},
    )
    original = runtime.dist.get_rank
    runtime.dist.get_rank = lambda: 1
    try:
        try:
            build_global_dispatch_plan(
                protocol=protocol,
                plan=plan,
                strategy="fifo",
                global_state=GlobalRuntimeState(),
                repetition_index=0,
            )
            assert False
        except RuntimeError:
            pass
    finally:
        runtime.dist.get_rank = original


def test_global_dispatch_plan_round_diagnostics_survive_dict_roundtrip():
    plan = GlobalDispatchPlan(
        plan_id="p",
        workload_plan_id="w",
        policy_name="strong-state-packer",
        policy_seed=1,
        microbatch_id=0,
        release_order=["b0"],
        release_rounds=[["b0"]],
        total_dispatch_matrix=[[1]],
        per_round_dispatch_matrices=[[[1]]],
        decision_hash="abc",
        scheduler_state_snapshot={"x": 1},
        scheduler_decision={"y": 2},
        round_diagnostics=[{"round_index": 0, "objective": 1.25}],
        round_pressure_score=1.25,
    )
    cloned = GlobalDispatchPlan.from_dict(plan.to_dict())
    assert cloned.round_diagnostics == [{"round_index": 0, "objective": 1.25}]
    assert cloned.round_pressure_score == 1.25
    assert canonical_global_plan_hash(cloned) == canonical_global_plan_hash(plan)


def test_global_dispatch_plan_hash_changes_with_round_diagnostics_and_pressure():
    plan = GlobalDispatchPlan(
        plan_id="p",
        workload_plan_id="w",
        policy_name="strong-state-packer",
        policy_seed=1,
        microbatch_id=0,
        release_order=["b0"],
        release_rounds=[["b0"]],
        total_dispatch_matrix=[[1]],
        per_round_dispatch_matrices=[[[1]]],
        decision_hash="",
        scheduler_state_snapshot={"x": 1},
        scheduler_decision={"y": 2},
        round_diagnostics=[{"round_index": 0, "objective": 1.25}],
        round_pressure_score=1.25,
    )
    plan.decision_hash = canonical_global_plan_hash(plan)
    mutated_diag = GlobalDispatchPlan.from_dict(plan.to_dict())
    mutated_diag.round_diagnostics = [{"round_index": 0, "objective": 2.25}]
    mutated_pressure = GlobalDispatchPlan.from_dict(plan.to_dict())
    mutated_pressure.round_pressure_score = 2.25
    assert canonical_global_plan_hash(mutated_diag) != plan.decision_hash
    assert canonical_global_plan_hash(mutated_pressure) != plan.decision_hash


def test_canonical_global_plan_payload_excludes_decision_hash():
    plan = GlobalDispatchPlan(
        plan_id="p",
        workload_plan_id="w",
        policy_name="fifo",
        policy_seed=1,
        microbatch_id=0,
        release_order=["b0"],
        release_rounds=[["b0"]],
        total_dispatch_matrix=[[1]],
        per_round_dispatch_matrices=[[[1]]],
        decision_hash="abc",
        scheduler_state_snapshot={},
        scheduler_decision={},
    )
    payload = canonical_global_plan_payload(plan)
    assert "decision_hash" not in payload


def test_global_dispatch_plan_hash_is_stable():
    payload = {"a": 1, "b": [1, 2, 3]}
    assert _hash_payload(payload) == _hash_payload(payload)


def test_global_completion_uses_max_rank_completion():
    state = GlobalRuntimeState()
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
        origin_sharding="round-robin",
        placement_mapping=[],
        bucket_workloads=[],
        routing_summary={},
    )
    dispatch_plan = type("DummyPlan", (), {"release_order": [], "policy_name": "fifo"})()
    updated = update_global_runtime_state(
        state,
        dispatch_plan,  # type: ignore[arg-type]
        plan,
        [
            {"rank": 0, "local_completion_ms": 3.0, "dispatch_total_ms": 1.0, "compute_total_ms": 1.0, "return_total_ms": 1.0},
            {"rank": 1, "local_completion_ms": 5.0, "dispatch_total_ms": 1.0, "compute_total_ms": 2.0, "return_total_ms": 1.0},
        ],
    )
    assert updated.global_completion_time_ms == 5.0
    assert updated.source_outbound_pending_bytes == {}
    assert updated.destination_inbound_pending_bytes == {}
    assert updated.flow_pending_bytes == {}
    assert updated.return_flow_pending_bytes == {}


def test_global_runtime_state_default_to_scheduler_state():
    state = GlobalRuntimeState()
    scheduler_state = state.to_scheduler_state()
    assert scheduler_state.return_flow_pending_bytes == {}


def test_return_flow_pending_bytes_affects_strong_state():
    bucket = WorkloadBucket(
        bucket_id="x",
        route_id="rx",
        token_ids=[0],
        token_positions=[0],
        origin_rank=0,
        destination_id=0,
        destination_rank=1,
        expert_id=0,
        layer_id=0,
        microbatch_id=0,
        token_count=1,
        payload_rows=1,
        hidden_dim=64,
        intermediate_dim=128,
        estimated_service_units=1.0,
        payload_bytes=128,
        source_count=1,
        source_coverage=0.1,
        coactive_peer_degree=0.1,
        coactive_event_density=0.1,
        position_spread=0.1,
        bridge_score=0.1,
        route_share=0.1,
        density_over_mean=0.1,
        size_norm=0.1,
        inverse_size_rank_norm=1.0,
        is_hot_bucket=False,
    )
    left = RuntimeState(return_flow_pending_bytes={"1->0": 0.0})
    right = RuntimeState(return_flow_pending_bytes={"1->0": 1 << 20})
    assert strong_state_score_for_bucket(bucket, left) != strong_state_score_for_bucket(bucket, right)


def test_latin_square_requires_multiple_of_strategy_count():
    try:
        latin_square_orders(["fifo", "random-order", "strong-state", "full"], 6)
        assert False
    except ValueError:
        pass


def test_pack_return_segments_reorders_by_origin():
    import torch

    payload = torch.arange(12, dtype=torch.float16).reshape(6, 2)
    metadata = [
        {"origin_rank": 2, "rows": 2, "route_id": "r0"},
        {"origin_rank": 0, "rows": 1, "route_id": "r1"},
        {"origin_rank": 1, "rows": 3, "route_id": "r2"},
    ]
    packed, send_splits, packed_metadata = _pack_return_segments(payload, metadata, 4)
    assert send_splits == [1, 3, 2, 0]
    assert [item["route_id"] for item in packed_metadata] == ["r1", "r2", "r0"]
    assert packed.shape[0] == 6


def test_canonical_receive_order_follows_source_rank_not_release_order():
    buckets = [
        WorkloadBucket(
            bucket_id="b_gpu2",
            route_id="r2",
            token_ids=[20],
            token_positions=[2],
            origin_rank=2,
            destination_id=11,
            destination_rank=1,
            expert_id=11,
            layer_id=0,
            microbatch_id=0,
            token_count=1,
            payload_rows=1,
            hidden_dim=64,
            intermediate_dim=128,
            estimated_service_units=1.0,
            payload_bytes=128,
            source_count=1,
            source_coverage=0.1,
            coactive_peer_degree=0.1,
            coactive_event_density=0.1,
            position_spread=0.1,
            bridge_score=0.1,
            route_share=0.1,
            density_over_mean=0.1,
            size_norm=0.1,
            inverse_size_rank_norm=1.0,
            is_hot_bucket=False,
            route_item_ids=["r2"],
            route_ranks=[0],
        ),
        WorkloadBucket(
            bucket_id="b_gpu0",
            route_id="r0",
            token_ids=[0],
            token_positions=[0],
            origin_rank=0,
            destination_id=10,
            destination_rank=1,
            expert_id=10,
            layer_id=0,
            microbatch_id=0,
            token_count=1,
            payload_rows=1,
            hidden_dim=64,
            intermediate_dim=128,
            estimated_service_units=1.0,
            payload_bytes=128,
            source_count=1,
            source_coverage=0.1,
            coactive_peer_degree=0.1,
            coactive_event_density=0.1,
            position_spread=0.1,
            bridge_score=0.1,
            route_share=0.1,
            density_over_mean=0.1,
            size_norm=0.1,
            inverse_size_rank_norm=1.0,
            is_hot_bucket=False,
            route_item_ids=["r0"],
            route_ranks=[0],
        ),
        WorkloadBucket(
            bucket_id="b_gpu3",
            route_id="r3",
            token_ids=[30],
            token_positions=[3],
            origin_rank=3,
            destination_id=12,
            destination_rank=1,
            expert_id=12,
            layer_id=0,
            microbatch_id=0,
            token_count=1,
            payload_rows=1,
            hidden_dim=64,
            intermediate_dim=128,
            estimated_service_units=1.0,
            payload_bytes=128,
            source_count=1,
            source_coverage=0.1,
            coactive_peer_degree=0.1,
            coactive_event_density=0.1,
            position_spread=0.1,
            bridge_score=0.1,
            route_share=0.1,
            density_over_mean=0.1,
            size_norm=0.1,
            inverse_size_rank_norm=1.0,
            is_hot_bucket=False,
            route_item_ids=["r3"],
            route_ranks=[0],
        ),
        WorkloadBucket(
            bucket_id="b_gpu1",
            route_id="r1",
            token_ids=[10],
            token_positions=[1],
            origin_rank=1,
            destination_id=13,
            destination_rank=1,
            expert_id=13,
            layer_id=0,
            microbatch_id=0,
            token_count=1,
            payload_rows=1,
            hidden_dim=64,
            intermediate_dim=128,
            estimated_service_units=1.0,
            payload_bytes=128,
            source_count=1,
            source_coverage=0.1,
            coactive_peer_degree=0.1,
            coactive_event_density=0.1,
            position_spread=0.1,
            bridge_score=0.1,
            route_share=0.1,
            density_over_mean=0.1,
            size_norm=0.1,
            inverse_size_rank_norm=1.0,
            is_hot_bucket=False,
            route_item_ids=["r1"],
            route_ranks=[0],
        ),
    ]
    layout = build_canonical_round_layout(buckets, 4, correctness_mode=True, round_index=0)
    lookup = {bucket.bucket_id: bucket for bucket in buckets}
    received = _planned_received_metadata_for_rank(layout, lookup, 1)
    assert [item["bucket_id"] for item in received] == ["b_gpu0", "b_gpu1", "b_gpu2", "b_gpu3"]


def test_pack_return_payload_and_metadata_aligns_segments():
    import torch

    payload = torch.arange(12, dtype=torch.float16).reshape(6, 2)
    metadata_tensor = torch.tensor(
        [
            [1, 0, 2, 0, 9, 0],
            [2, 1, 2, 0, 9, 0],
            [3, 0, 0, 0, 7, 0],
            [4, 0, 1, 0, 8, 0],
            [5, 1, 1, 0, 8, 0],
            [6, 2, 1, 0, 8, 0],
        ],
        dtype=torch.int32,
    )
    metadata = [
        {"origin_rank": 2, "rows": 2, "route_id": "r0", "token_ids": [1, 2]},
        {"origin_rank": 0, "rows": 1, "route_id": "r1", "token_ids": [3]},
        {"origin_rank": 1, "rows": 3, "route_id": "r2", "token_ids": [4, 5, 6]},
    ]
    packed_payload, packed_metadata, send_splits, packed_dicts = _pack_return_payload_and_metadata(payload, metadata_tensor, metadata, 4)
    assert send_splits == [1, 3, 2, 0]
    assert packed_payload.shape[0] == packed_metadata.shape[0] == 6
    assert [item["route_id"] for item in packed_dicts] == ["r1", "r2", "r0"]


def test_metadata_tensor_decoding_matches_planned_rows():
    import torch

    planned = [
        {
            "bucket_id": "b0",
            "route_id": "r0",
            "route_item_ids": ["r0"],
            "route_ranks": [0],
            "origin_rank": 0,
            "destination_rank": 1,
            "rows": 2,
            "destination_id": 3,
            "expert_id": 3,
            "estimated_service_units": 2.0,
            "token_count": 2,
            "token_ids": [7, 8],
            "token_positions": [0, 1],
            "payload_bytes": 256,
        }
    ]
    tensor = torch.tensor([[7, 0, 0, 1, 3, 0], [8, 1, 0, 1, 3, 0]], dtype=torch.int32)
    decoded = _metadata_tensor_to_dicts(tensor, planned)
    assert decoded[0]["token_ids"] == [7, 8]


def test_decode_metadata_tensor_rows_roundtrip():
    import torch

    tensor = torch.tensor([[7, 101, 0, 1, 3, 0], [8, 102, 0, 1, 3, 1]], dtype=torch.int32)
    rows = decode_metadata_tensor_rows(tensor)
    assert rows[0].global_route_item_index == 101
    assert rows[1].route_rank == 1


def test_summarize_route_manifests_uses_expected_not_observed_union():
    expected = [{"global_route_item_index": 0, "token_id": 7, "origin_rank": 0, "destination_rank": 1, "expert_id": 3, "route_rank": 0, "microbatch_id": 0, "layer_id": 0}]
    summary = summarize_route_manifests(expected, [], [], [], [])
    assert summary["expected_route_item_count"] == 1
    assert summary["missing_dispatch_count"] == 1


def test_global_route_item_manifest_separates_microbatches():
    plans = [
        WorkloadPlan(
            plan_id="p0",
            model_key="olmoe",
            seed=1,
            microbatch_id=0,
            layer_id=0,
            layer_path="layer",
            hidden_dim=64,
            intermediate_dim=128,
            placement_policy="balanced",
            origin_sharding="round-robin",
            placement_mapping=[],
            bucket_workloads=[
                WorkloadBucket(
                    bucket_id="b0",
                    route_id="r0",
                    token_ids=[11],
                    token_positions=[0],
                    origin_rank=0,
                    destination_id=1,
                    destination_rank=1,
                    expert_id=1,
                    layer_id=0,
                    microbatch_id=0,
                    token_count=1,
                    payload_rows=1,
                    hidden_dim=64,
                    intermediate_dim=128,
                    estimated_service_units=1.0,
                    payload_bytes=128,
                    source_count=1,
                    source_coverage=0.1,
                    coactive_peer_degree=0.1,
                    coactive_event_density=0.1,
                    position_spread=0.1,
                    bridge_score=0.1,
                    route_share=0.1,
                    density_over_mean=0.1,
                    size_norm=0.1,
                    inverse_size_rank_norm=1.0,
                    is_hot_bucket=False,
                    route_item_indices=[0],
                    route_item_ids=["r0"],
                    route_ranks=[0],
                )
            ],
            routing_summary={},
        ),
        WorkloadPlan(
            plan_id="p1",
            model_key="olmoe",
            seed=1,
            microbatch_id=1,
            layer_id=0,
            layer_path="layer",
            hidden_dim=64,
            intermediate_dim=128,
            placement_policy="balanced",
            origin_sharding="round-robin",
            placement_mapping=[],
            bucket_workloads=[
                WorkloadBucket(
                    bucket_id="b1",
                    route_id="r1",
                    token_ids=[22],
                    token_positions=[1],
                    origin_rank=1,
                    destination_id=2,
                    destination_rank=0,
                    expert_id=2,
                    layer_id=0,
                    microbatch_id=1,
                    token_count=1,
                    payload_rows=1,
                    hidden_dim=64,
                    intermediate_dim=128,
                    estimated_service_units=1.0,
                    payload_bytes=128,
                    source_count=1,
                    source_coverage=0.1,
                    coactive_peer_degree=0.1,
                    coactive_event_density=0.1,
                    position_spread=0.1,
                    bridge_score=0.1,
                    route_share=0.1,
                    density_over_mean=0.1,
                    size_norm=0.1,
                    inverse_size_rank_norm=1.0,
                    is_hot_bucket=False,
                    route_item_indices=[1],
                    route_item_ids=["r1"],
                    route_ranks=[0],
                )
            ],
            routing_summary={},
        ),
    ]
    manifest = expected_route_item_manifest(plans)
    assert [item["global_route_item_index"] for item in manifest] == [0, 1]


def test_metadata_tensor_to_dicts_fails_on_tampered_destination():
    import torch

    planned = [
        {
            "bucket_id": "b0",
            "route_id": "r0",
            "route_item_ids": ["r0"],
            "route_item_indices": [101],
            "route_ranks": [0],
            "origin_rank": 0,
            "destination_rank": 1,
            "rows": 1,
            "destination_id": 3,
            "expert_id": 3,
            "estimated_service_units": 1.0,
            "token_count": 1,
            "token_ids": [7],
            "token_positions": [0],
            "payload_bytes": 128,
        }
    ]
    tensor = torch.tensor([[7, 101, 0, 2, 3, 0]], dtype=torch.int32)
    try:
        _metadata_tensor_to_dicts(tensor, planned, current_destination_rank=1)
        assert False
    except RuntimeError:
        pass


def test_benchmark_preflight_reports_stable_contract():
    import os
    import torch

    plan = WorkloadPlan(
        plan_id="p0",
        model_key="olmoe",
        seed=1,
        microbatch_id=0,
        layer_id=0,
        layer_path="layer",
        hidden_dim=8,
        intermediate_dim=16,
        placement_policy="balanced",
        origin_sharding="round-robin",
        placement_mapping=[PlacementEntry(expert_id=0, destination_rank=0)],
        bucket_workloads=[
            WorkloadBucket(
                bucket_id="b0",
                route_id="r0",
                route_item_indices=[0],
                token_ids=[0],
                token_positions=[0],
                origin_rank=0,
                destination_id=0,
                destination_rank=0,
                expert_id=0,
                layer_id=0,
                microbatch_id=0,
                token_count=1,
                payload_rows=1,
                hidden_dim=8,
                intermediate_dim=16,
                estimated_service_units=1.0,
                payload_bytes=16,
                source_count=1,
                source_coverage=0.1,
                coactive_peer_degree=0.1,
                coactive_event_density=0.1,
                position_spread=0.1,
                bridge_score=0.1,
                route_share=0.1,
                density_over_mean=0.1,
                size_norm=0.1,
                inverse_size_rank_norm=1.0,
                is_hot_bucket=False,
                route_item_ids=["r0"],
                route_ranks=[0],
            )
        ],
        routing_summary={},
    )
    cache = type(
        "Cache",
        (),
        {
            "payload_by_bucket_id": {"b0": torch.ones((1, 8), dtype=torch.float16)},
            "metadata_by_bucket_id": {"b0": torch.ones((1, 6), dtype=torch.int32)},
            "expert_weight1": {0: torch.ones((16, 8), dtype=torch.float16)},
            "expert_weight2": {0: torch.ones((8, 16), dtype=torch.float16)},
        },
    )()
    protocol = ProtocolConfig(model_key="olmoe", model_path="/tmp", output_dir="/tmp/out", artifact_dir="/tmp/art")
    preflight = benchmark_preflight(
        protocol,
        plan,
        cache,
        allowed_pids=[os.getpid()],
        allowed_cmd_substrings=["python"],
    )  # type: ignore[arg-type]
    assert preflight["used_python_metadata_gather_expected"] is False
    assert preflight["legacy_payload_function_allowed"] is False


def _stress_plan(plan_id: str, microbatch_id: int, placement_mapping: list[PlacementEntry]) -> WorkloadPlan:
    destination_rank_map = {item.expert_id: item.destination_rank for item in placement_mapping}
    buckets = []
    for index, (expert_id, token_count, rows, origin_rank) in enumerate(
        [
            (0, 4, 8, 0),
            (1, 6, 12, 1),
            (2, 8, 16, 2),
            (3, 5, 10, 3),
        ]
    ):
        destination_rank = destination_rank_map[expert_id]
        buckets.append(
            WorkloadBucket(
                bucket_id=f"{plan_id}_b{index}",
                route_id=f"{plan_id}_r{index}",
                route_item_indices=[microbatch_id * 10 + index],
                token_ids=[1000 + microbatch_id * 100 + index],
                token_positions=[index],
                origin_rank=origin_rank,
                destination_id=expert_id,
                destination_rank=destination_rank,
                expert_id=expert_id,
                layer_id=0,
                microbatch_id=microbatch_id,
                token_count=token_count,
                payload_rows=rows,
                hidden_dim=64,
                intermediate_dim=128,
                estimated_service_units=float(rows),
                payload_bytes=rows * 64 * 2,
                source_count=2,
                source_coverage=0.5 + 0.1 * index,
                coactive_peer_degree=0.2 + 0.1 * index,
                coactive_event_density=0.1 + 0.05 * index,
                position_spread=0.05 * index,
                bridge_score=0.1 * index,
                route_share=0.2 + 0.05 * index,
                density_over_mean=0.3 + 0.05 * index,
                size_norm=0.4 + 0.1 * index,
                inverse_size_rank_norm=max(0.0, 1.0 - 0.2 * index),
                is_hot_bucket=index >= 2,
                route_item_ids=[f"{plan_id}_ri{index}"],
                route_ranks=[0],
            )
        )
    return WorkloadPlan(
        plan_id=plan_id,
        model_key="olmoe",
        seed=42,
        microbatch_id=microbatch_id,
        layer_id=0,
        layer_path="layer0",
        hidden_dim=64,
        intermediate_dim=128,
        placement_policy="balanced",
        origin_sharding="round-robin",
        placement_mapping=placement_mapping,
        bucket_workloads=buckets,
        routing_summary={"token_origin_rank_detail": {str(1000 + microbatch_id * 100 + i): {"origin_rank": i} for i in range(4)}},
    )


def test_stress_split_and_manifest_hash_stability():
    placement = build_placement_map(4, 4, "balanced")
    plans = [_stress_plan(f"p{i}", i, placement) for i in range(4)]
    calibration, evaluation = split_calibration_and_evaluation(plans)
    assert len(calibration) >= 1
    assert len(evaluation) >= 1
    manifest = scenario_manifest(
        scenario_id="balanced",
        plans=plans,
        placement=placement,
        episode_rows=[{"episode_index": i, "phase": "benign"} for i in range(len(plans))],
        target_rank=2,
    )
    assert manifest["trace_bundle_hash"]
    assert manifest["placement_hash"]
    assert manifest["route_item_set_hash"]


def test_choose_skewed_placement_preserves_expert_count_per_rank():
    placement = build_placement_map(4, 4, "balanced")
    plans = [_stress_plan(f"p{i}", i, placement) for i in range(4)]
    next_placement, metrics = choose_skewed_placement(placement, plans[:2], plans[2:], target_rank=2, strength="mild-skew")
    base_counts = {}
    next_counts = {}
    for item in placement:
        base_counts[item.destination_rank] = base_counts.get(item.destination_rank, 0) + 1
    for item in next_placement:
        next_counts[item.destination_rank] = next_counts.get(item.destination_rank, 0) + 1
    assert sorted(base_counts.values()) == sorted(next_counts.values())
    assert "target_rank_inbound_share" in metrics


def test_burst_episode_schedule_preserves_phase_labels():
    placement = build_placement_map(4, 4, "balanced")
    plans = [_stress_plan(f"p{i}", i, placement) for i in range(8)]
    selected, rows = build_burst_episode_schedule(plans, target_rank=2)
    assert len(selected) == len(rows)
    assert {row["phase"] for row in rows} == {"benign", "ramp", "burst", "recovery"}


def test_dependency_predictiveness_gate_emits_pass_or_fail():
    placement = build_placement_map(4, 4, "balanced")
    plans = [_stress_plan(f"p{i}", i, placement) for i in range(8)]
    gate = dependency_predictiveness_gate(plans[:4], plans[4:], target_rank=2)
    assert gate["gate"] in {"pass", "fail"}
    assert "state_only" in gate
    assert "state_plus_dependency" in gate


def test_oracle_minimax_round_packer_preserves_bucket_ids():
    placement = build_placement_map(4, 4, "balanced")
    plan = _stress_plan("oracle", 0, placement)
    oracle = oracle_minimax_round_packer(plan, 2)
    flattened = [bucket_id for round_ids in oracle["release_rounds"] for bucket_id in round_ids]
    assert sorted(flattened) == sorted(bucket.bucket_id for bucket in plan.bucket_workloads)
    assert oracle["peak_bottleneck_pressure"] >= 0
