from __future__ import annotations

from dataclasses import dataclass

from rs.runtime.distributed_ep.adapter.runner import (
    _aggregate_matrix,
    _build_matrix_from_plan,
    _build_rank_local_matrix_from_plan,
    build_distributed_runner_plan,
)
from rs.runtime.distributed_ep.core.manifest import DispatchPlan, DispatchShard


class _FakeReduceOp:
    SUM = "sum"


@dataclass
class _FakeDist:
    delta: float
    backend: str = "gloo"
    ReduceOp = _FakeReduceOp()

    def all_reduce(self, tensor, op=None) -> None:
        del op
        tensor.add_(self.delta)

    def get_backend(self) -> str:
        return self.backend


def test_aggregate_matrix_uses_all_reduce_result() -> None:
    matrix = [[1, 2], [3, 4]]
    aggregated = _aggregate_matrix(
        matrix,
        use_distributed=True,
        world_size=2,
        dist_module=_FakeDist(delta=10.0),
    )
    assert aggregated == [[11, 12], [13, 14]]


def test_build_distributed_runner_plan_builds_global_dispatch_plan() -> None:
    class _AdapterConfig:
        num_experts = 4

        def to_dict(self):
            return {"num_experts": 4, "hidden_size": 8}

    class _LocalWeights:
        def to_dict(self):
            return {}

    class _Experts:
        pass

    class _MLP:
        experts = _Experts()

    class _Layer:
        mlp = _MLP()

    class _Model:
        class model:
            layers = [_Layer()]

    trace = {"records": [{"layer_id": 0, "expert_id": 0, "token_flat_index": 0, "topk_rank": 0, "routing_weight": 1.0}]}

    import rs.runtime.distributed_ep.adapter.runner as runner

    original_probe = runner.probe_olmoe_adapter_config
    original_count = runner.count_local_expert_parameters
    original_residency = runner.summarize_residency
    original_extract = runner.extract_local_expert_weights
    try:
        runner.probe_olmoe_adapter_config = lambda model: _AdapterConfig()
        runner.count_local_expert_parameters = lambda experts_module, local_expert_ids: 0
        runner.summarize_residency = lambda local_expert_ids, local_parameter_count=0: type("Residency", (), {"to_dict": lambda self: {"local_expert_ids": list(local_expert_ids)}})()
        runner.extract_local_expert_weights = lambda experts_module, local_expert_ids: _LocalWeights()
        config = runner.DistributedRunnerConfig(world_size=4, node_rank=0, model_id="test", origin_rank=0)
        plan = build_distributed_runner_plan(
            model=_Model(),
            trace={
                "records": [
                    {"layer_id": 0, "expert_id": 1, "token_position": 0, "topk_rank": 0, "routing_weight": 1.0},
                    {"layer_id": 0, "expert_id": 2, "token_position": 1, "topk_rank": 0, "routing_weight": 1.0},
                    {"layer_id": 0, "expert_id": 3, "token_position": 2, "topk_rank": 0, "routing_weight": 1.0},
                    {"layer_id": 0, "expert_id": 0, "token_position": 3, "topk_rank": 0, "routing_weight": 1.0},
                ]
            },
            config=config,
            rank=3,
            host="host",
            gpu_name="cuda:3",
        )
    finally:
        runner.probe_olmoe_adapter_config = original_probe
        runner.count_local_expert_parameters = original_count
        runner.summarize_residency = original_residency
        runner.extract_local_expert_weights = original_extract

    dispatch_plan = plan.dispatch_plans[0]
    pairs = {(shard.source_rank, shard.destination_rank) for shard in dispatch_plan.shards}
    assert pairs == {(0, 1), (1, 2), (2, 3), (3, 0)}


def test_build_rank_local_matrix_from_plan_only_emits_rank_contribution() -> None:
    plan = DispatchPlan(
        layer_id=0,
        world_size=3,
        shards=[
            DispatchShard(source_rank=0, destination_rank=1, route_items=[type("Item", (), {"payload_rows": 2})()]),
            DispatchShard(source_rank=1, destination_rank=2, route_items=[type("Item", (), {"payload_rows": 3})()]),
            DispatchShard(source_rank=2, destination_rank=0, route_items=[type("Item", (), {"payload_rows": 5})()]),
        ],
    )

    send_rank1 = _build_rank_local_matrix_from_plan(plan, 1, "send")
    recv_rank1 = _build_rank_local_matrix_from_plan(plan, 1, "recv")
    global_send = _build_matrix_from_plan(plan, 0, "send")
    global_recv = _build_matrix_from_plan(plan, 0, "recv")

    assert send_rank1 == [
        [0, 0, 0],
        [0, 0, 3],
        [0, 0, 0],
    ]
    assert recv_rank1 == [
        [0, 0, 0],
        [2, 0, 0],
        [0, 0, 0],
    ]
    assert global_send == [
        [0, 2, 0],
        [0, 0, 3],
        [5, 0, 0],
    ]
    assert global_recv == [
        [0, 0, 5],
        [2, 0, 0],
        [0, 3, 0],
    ]
