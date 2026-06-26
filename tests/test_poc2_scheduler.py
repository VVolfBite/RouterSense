from __future__ import annotations

from routesense_poc2.scheduler import BucketRecord, RuntimeState, SchedulerInput, create_scheduler


def _make_bucket(bucket_id: str, destination_id: int, arrival_index: int, token_count: int, bridge: float) -> BucketRecord:
    return BucketRecord(
        bucket_id=bucket_id,
        destination_id=destination_id,
        destination_rank=destination_id % 4,
        arrival_index=arrival_index,
        token_count=token_count,
        source_count=16,
        active_destinations=[0, 1, 2, 3],
        coactive_destinations=[item for item in [0, 1, 2, 3] if item != destination_id],
        source_coverage=token_count / 16.0,
        coactive_peer_degree=bridge,
        coactive_event_density=bridge / 2.0,
        position_spread=0.3,
        bridge_score=bridge,
        route_share=token_count / 32.0,
        density_over_mean=0.5,
        size_norm=token_count / 8.0,
        inverse_size_rank_norm=1.0 - arrival_index / 2.0,
        is_hot_bucket=token_count > 1,
        estimated_service_units=float(token_count),
    )


def test_fifo_preserves_arrival_order():
    scheduler = create_scheduler("fifo")
    state = RuntimeState()
    buckets = [
        _make_bucket("b0", 0, 0, 8, 0.2),
        _make_bucket("b1", 1, 1, 4, 0.9),
        _make_bucket("b2", 2, 2, 6, 0.5),
    ]
    decision = scheduler.build_decision(SchedulerInput("fifo", 0, 0, "layer", buckets, state))
    assert decision.release_order == ["b0", "b1", "b2"]


def test_dependency_only_can_overrule_size():
    scheduler = create_scheduler("dependency-only")
    state = RuntimeState()
    buckets = [
        _make_bucket("large_low_dep", 0, 0, 8, 0.1),
        _make_bucket("small_high_dep", 1, 1, 4, 1.0),
    ]
    decision = scheduler.build_decision(SchedulerInput("dependency-only", 0, 0, "layer", buckets, state))
    assert decision.release_order[0] == "small_high_dep"
