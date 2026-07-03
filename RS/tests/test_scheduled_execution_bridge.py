from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rs.runtime.distributed_ep.adapter.runner import _build_matrix_from_plan
import rs.runtime.distributed_ep.adapter.runner as runner_module
from rs.runtime.distributed_ep.core.collective import CollectiveOps
from rs.runtime.distributed_ep.core.manifest import DispatchPlan, DispatchShard
from rs.runtime.distributed_ep.core.nccl_executor import NCCLExecutionResult, NCCLOpRecord
from rs.runtime.distributed_ep.core.wave_executor import WaveExecutionResult


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


@dataclass
class _DummyStrategyResult:
    makespan: float = 1.0
    solve_time_ms: float = 0.5
    schedule: list[dict] = field(default_factory=list)


@dataclass
class _DummyStrategy:
    result: _DummyStrategyResult

    def solve(self, ctx):
        return self.result


@dataclass
class _DummyWaveBundle:
    dispatch_waves: list[object]
    combine_waves: list[object]
    dispatch_token_indices: list[list[int]]
    combine_token_indices: list[list[int]]


@dataclass
class _RecordingTransport:
    calls: list[dict] = field(default_factory=list)
    transport_name: str = "scheduled_all_to_all"

    def execute_schedule(self, wave_schedule, *, phase, direction, token_buffer, hidden_size):
        self.calls.append(
            {
                "phase": phase,
                "direction": direction,
                "wave_count": len(wave_schedule),
                "hidden_size": hidden_size,
                "rows": int(token_buffer.shape[0]),
            }
        )
        rows = token_buffer.shape[0]
        return WaveExecutionResult(
            phase=phase,
            direction=direction,
            total_comm_ms=1.0,
            total_pack_ms=2.0,
            total_unpack_ms=3.0,
            wave_count=len(wave_schedule),
            received_route_items=[],
            received_tensor=torch.zeros((rows, hidden_size), dtype=token_buffer.dtype),
        )


def test_execute_scheduled_inference_uses_scheduled_transport(monkeypatch) -> None:
    recorded = _RecordingTransport()
    strategy_result = _DummyStrategyResult(
        schedule=[
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 0},
        ]
    )
    dispatch_plan = DispatchPlan(layer_id=0, world_size=2, shards=[])

    monkeypatch.setattr(runner_module, "get_strategy", lambda name: _DummyStrategy(strategy_result))
    monkeypatch.setattr(
        runner_module,
        "scheduling_result_to_wave_schedule",
        lambda *args, **kwargs: _DummyWaveBundle(
            dispatch_waves=[object()],
            combine_waves=[object()],
            dispatch_token_indices=[[0]],
            combine_token_indices=[[0]],
        ),
    )
    monkeypatch.setattr(runner_module, "verify_wave_conservation", lambda *args, **kwargs: {"pass": True})
    monkeypatch.setattr(runner_module, "execute_local_experts", lambda tensor, route_items, local_weights: tensor)
    monkeypatch.setattr(
        runner_module,
        "execute_native_baseline",
        lambda **kwargs: type(
            "Native",
            (),
            {
                "final_output": torch.zeros((2, 4), dtype=torch.float16),
                "combine_result": type("Combine", (), {"received_route_items": [], "total_comm_ms": 0.2})(),
                "dispatch_result": type("Dispatch", (), {"total_comm_ms": 0.1})(),
            },
        )(),
    )
    monkeypatch.setattr(
        runner_module,
        "verify_token_conservation",
        lambda *args, **kwargs: {"token_conservation_pass": True, "gate_weight_conservation_pass": True},
    )
    monkeypatch.setattr(runner_module, "ScheduledAllToAllTransport", lambda executor, *, split_into_micro_ops: recorded)

    result = runner_module.execute_scheduled_inference(
        dispatch_plans=[dispatch_plan],
        rank=0,
        world_size=2,
        strategy_name="greedy",
        hidden_size=4,
        local_expert_weights=object(),
        hidden_state_rows=torch.zeros((2, 4), dtype=torch.float16),
        execution_mode="scheduled_transport",
    )

    assert result["execution_mode"] == "scheduled_transport"
    assert result["wave_execution"]["transport"] == "scheduled_all_to_all"
    assert result["wave_execution"]["transport_granularity"] == "wave"
    assert [call["direction"] for call in recorded.calls] == ["dispatch", "combine"]


def test_execute_scheduled_inference_respects_atomic_transport_granularity(monkeypatch) -> None:
    strategy_result = _DummyStrategyResult(
        schedule=[
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 0},
        ]
    )
    dispatch_plan = DispatchPlan(layer_id=0, world_size=2, shards=[])
    captured: dict[str, object] = {}

    class _CapturingTransport(_RecordingTransport):
        def __init__(self, executor, *, split_into_micro_ops: bool) -> None:
            super().__init__()
            captured["split_into_micro_ops"] = split_into_micro_ops

    monkeypatch.setattr(runner_module, "get_strategy", lambda name: _DummyStrategy(strategy_result))
    monkeypatch.setattr(
        runner_module,
        "scheduling_result_to_wave_schedule",
        lambda *args, **kwargs: _DummyWaveBundle(
            dispatch_waves=[object()],
            combine_waves=[object()],
            dispatch_token_indices=[[0]],
            combine_token_indices=[[0]],
        ),
    )
    monkeypatch.setattr(runner_module, "verify_wave_conservation", lambda *args, **kwargs: {"pass": True})
    monkeypatch.setattr(runner_module, "execute_local_experts", lambda tensor, route_items, local_weights: tensor)
    monkeypatch.setattr(
        runner_module,
        "execute_native_baseline",
        lambda **kwargs: type(
            "Native",
            (),
            {
                "final_output": torch.zeros((2, 4), dtype=torch.float16),
                "combine_result": type("Combine", (), {"received_route_items": [], "total_comm_ms": 0.2})(),
                "dispatch_result": type("Dispatch", (), {"total_comm_ms": 0.1})(),
            },
        )(),
    )
    monkeypatch.setattr(
        runner_module,
        "verify_token_conservation",
        lambda *args, **kwargs: {"token_conservation_pass": True, "gate_weight_conservation_pass": True},
    )
    monkeypatch.setattr(runner_module, "ScheduledAllToAllTransport", _CapturingTransport)

    runner_module.execute_scheduled_inference(
        dispatch_plans=[dispatch_plan],
        rank=0,
        world_size=2,
        strategy_name="birkhoff",
        hidden_size=4,
        local_expert_weights=object(),
        hidden_state_rows=torch.zeros((2, 4), dtype=torch.float16),
        execution_mode="scheduled_transport",
        transport_granularity="atomic",
    )

    assert captured["split_into_micro_ops"] is True


def test_execute_scheduled_inference_respects_wave_transport_granularity(monkeypatch) -> None:
    strategy_result = _DummyStrategyResult(
        schedule=[
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 0},
        ]
    )
    dispatch_plan = DispatchPlan(layer_id=0, world_size=2, shards=[])
    captured: dict[str, object] = {}

    class _CapturingTransport(_RecordingTransport):
        def __init__(self, executor, *, split_into_micro_ops: bool) -> None:
            super().__init__()
            captured["split_into_micro_ops"] = split_into_micro_ops

    monkeypatch.setattr(runner_module, "get_strategy", lambda name: _DummyStrategy(strategy_result))
    monkeypatch.setattr(
        runner_module,
        "scheduling_result_to_wave_schedule",
        lambda *args, **kwargs: _DummyWaveBundle(
            dispatch_waves=[object()],
            combine_waves=[object()],
            dispatch_token_indices=[[0]],
            combine_token_indices=[[0]],
        ),
    )
    monkeypatch.setattr(runner_module, "verify_wave_conservation", lambda *args, **kwargs: {"pass": True})
    monkeypatch.setattr(runner_module, "execute_local_experts", lambda tensor, route_items, local_weights: tensor)
    monkeypatch.setattr(
        runner_module,
        "execute_native_baseline",
        lambda **kwargs: type(
            "Native",
            (),
            {
                "final_output": torch.zeros((2, 4), dtype=torch.float16),
                "combine_result": type("Combine", (), {"received_route_items": [], "total_comm_ms": 0.2})(),
                "dispatch_result": type("Dispatch", (), {"total_comm_ms": 0.1})(),
            },
        )(),
    )
    monkeypatch.setattr(
        runner_module,
        "verify_token_conservation",
        lambda *args, **kwargs: {"token_conservation_pass": True, "gate_weight_conservation_pass": True},
    )
    monkeypatch.setattr(runner_module, "ScheduledAllToAllTransport", _CapturingTransport)

    runner_module.execute_scheduled_inference(
        dispatch_plans=[dispatch_plan],
        rank=0,
        world_size=2,
        strategy_name="U_gated_maxweight_matching_atomic",
        hidden_size=4,
        local_expert_weights=object(),
        hidden_state_rows=torch.zeros((2, 4), dtype=torch.float16),
        execution_mode="scheduled_transport",
        transport_granularity="wave",
    )

    assert captured["split_into_micro_ops"] is False
