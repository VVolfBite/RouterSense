from __future__ import annotations

from routesense.runtime.distributed_ep.adapter.expert_store import plan_local_expert_ids, summarize_residency
from routesense.runtime.distributed_ep.adapter.olmoe_adapter import build_dispatch_plan_from_trace
from routesense.runtime.distributed_ep.core.collective import CollectiveOps
from routesense.runtime.distributed_ep.core.correctness import summarize_dispatch_plans
from routesense.runtime.distributed_ep.core.placement import PlacementStrategy
from routesense.runtime.distributed_ep.core.worker_loop import WorkerLoop


def test_round_robin_placement_balanced():
    owner = PlacementStrategy.round_robin(num_experts=8, num_gpus=4)
    assert owner[0] == 0
    assert owner[1] == 1
    assert owner[4] == 0


def test_plan_local_expert_ids():
    owner = {0: 0, 1: 1, 2: 0, 3: 1}
    assert plan_local_expert_ids(owner, 0) == [0, 2]
    assert plan_local_expert_ids(owner, 1) == [1, 3]


def test_build_dispatch_plan_from_trace_counts_cross_node():
    owner = {0: 0, 1: 1, 2: 2, 3: 3}
    trace_records = [
        {"token_position": 0, "expert_id": 0, "expert_rank_within_topk": 0, "routing_weight": 0.7},
        {"token_position": 0, "expert_id": 2, "expert_rank_within_topk": 1, "routing_weight": 0.3},
    ]
    plan = build_dispatch_plan_from_trace(
        trace_records,
        owner_by_expert=owner,
        origin_rank=0,
        layer_id=5,
        world_size=4,
    )
    assert plan.layer_id == 5
    assert plan.total_cross_node_routes() == 1
    assert plan.send_counts_for_rank(0) == [1, 0, 1, 0]


def test_collective_ops_use_dispatch_plan():
    owner = {0: 0, 1: 1}
    trace_records = [
        {"token_position": 0, "expert_id": 0, "expert_rank_within_topk": 0, "routing_weight": 0.5},
        {"token_position": 1, "expert_id": 1, "expert_rank_within_topk": 0, "routing_weight": 0.5},
    ]
    plan = build_dispatch_plan_from_trace(
        trace_records,
        owner_by_expert=owner,
        origin_rank=0,
        layer_id=1,
        world_size=2,
    )
    ops = CollectiveOps(bytes_per_row=16)
    ops.dispatch(payload=None, plan=plan, rank=0)
    ops.return_results(payload=None, plan=plan, rank=0)
    assert len(ops.records) == 2
    assert ops.records[0].send_bytes == 32


def test_worker_loop_records_processed_routes():
    owner = {0: 0, 1: 1}
    trace_records = [
        {"token_position": 0, "expert_id": 1, "expert_rank_within_topk": 0, "routing_weight": 0.5},
        {"token_position": 1, "expert_id": 1, "expert_rank_within_topk": 0, "routing_weight": 0.5},
    ]
    plan = build_dispatch_plan_from_trace(
        trace_records,
        owner_by_expert=owner,
        origin_rank=0,
        layer_id=1,
        world_size=2,
    )
    loop = WorkerLoop()
    loop.record_plan(plan, rank=1)
    assert loop.state.processed_route_count == 2


def test_correctness_summary_counts_routes():
    owner = {0: 0, 1: 1}
    plan = build_dispatch_plan_from_trace(
        [{"token_position": 0, "expert_id": 1, "expert_rank_within_topk": 0, "routing_weight": 1.0}],
        owner_by_expert=owner,
        origin_rank=0,
        layer_id=0,
        world_size=2,
    )
    summary = summarize_dispatch_plans([plan])
    assert summary["cross_node_routes"] == 1


def test_residency_summary_is_physically_sharded():
    residency = summarize_residency([1, 3], local_parameter_count=1024)
    assert residency.weight_residency_mode == "physically_sharded_experts"
    assert residency.non_owner_parameter_count == 0
