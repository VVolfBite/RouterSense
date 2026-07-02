from __future__ import annotations

from dataclasses import dataclass, field

from routesense.runtime.distributed_ep.adapter.runner import _build_matrix_from_plan
from routesense.runtime.distributed_ep.core.collective import CollectiveOps
from routesense.runtime.distributed_ep.core.manifest import DispatchPlan, DispatchShard
from routesense.runtime.distributed_ep.core.nccl_executor import NCCLExecutionResult, NCCLOpRecord


@dataclass
class FakeExecutor:
    calls: list[dict] = field(default_factory=list)

    def execute_phase(self, send_chunks, recv_chunks, **kwargs):
        self.calls.append(
            {
                "send_chunks": list(send_chunks),
                "recv_chunks": list(recv_chunks),
                "kwargs": kwargs,
            }
        )
        return NCCLExecutionResult(
            ops=[
                NCCLOpRecord(
                    op="send",
                    peer_rank=send_chunks[0][0] if send_chunks else -1,
                    tensor_size=0,
                    start_us=0.0,
                    end_us=1.0,
                    duration_us=1.0,
                )
            ],
            total_wall_time_us=10.0,
            phase=int(kwargs.get("phase", 0)),
            direction=str(kwargs.get("direction", "dispatch")),
        )


def _sample_plan() -> DispatchPlan:
    return DispatchPlan(
        layer_id=3,
        world_size=4,
        shards=[
            DispatchShard(source_rank=0, destination_rank=1, route_items=[]),
            DispatchShard(source_rank=0, destination_rank=2, route_items=[]),
            DispatchShard(source_rank=2, destination_rank=0, route_items=[]),
        ],
    )


def test_build_matrix_from_plan_send_and_recv() -> None:
    plan = _sample_plan()
    plan.shards[0].route_items = []  # explicit no-op, keeps rows=0
    plan.shards[1].route_items = []
    plan.shards[2].route_items = []

    # Rebuild with implicit row counts via subclassed property substitute is not needed here.
    plan = DispatchPlan(
        layer_id=3,
        world_size=4,
        shards=[
            type("Shard", (), {"source_rank": 0, "destination_rank": 1, "rows": 5})(),
            type("Shard", (), {"source_rank": 0, "destination_rank": 2, "rows": 7})(),
            type("Shard", (), {"source_rank": 2, "destination_rank": 0, "rows": 11})(),
        ],
    )

    send = _build_matrix_from_plan(plan, 0, "send")
    recv = _build_matrix_from_plan(plan, 0, "recv")
    assert send[0][1] == 5
    assert send[0][2] == 7
    assert send[2][0] == 11
    assert recv[1][0] == 5
    assert recv[2][0] == 7
    assert recv[0][2] == 11


def test_collective_execute_scheduled_phase_filters_rank_chunks() -> None:
    plan = DispatchPlan(
        layer_id=5,
        world_size=4,
        shards=[],
    )
    schedule = [
        {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 4},
        {"phase": 0, "src_gpu": 2, "dst_gpu": 0, "size": 6},
        {"phase": 0, "src_gpu": 1, "dst_gpu": 3, "size": 8},
    ]
    executor = FakeExecutor()
    collective = CollectiveOps(bytes_per_row=16)
    result = collective.execute_scheduled_phase(
        plan=plan,
        rank=0,
        schedule=schedule,
        phase=0,
        direction="dispatch",
        executor=executor,
        hidden_size=32,
    )
    assert result.total_wall_time_us == 10.0
    assert executor.calls[0]["send_chunks"] == [(1, 4)]
    assert executor.calls[0]["recv_chunks"] == [(2, 6)]
    assert collective.records[0].send_bytes == 4 * 16
    assert collective.records[0].recv_bytes == 6 * 16
