from __future__ import annotations

from dataclasses import dataclass

from rs.runtime.distributed_ep.core import (
    DispatchPlan,
    DispatchShard,
    RouteItem,
    ScheduledAllToAllTransport,
    build_token_wave_mapping,
    scheduling_result_to_wave_schedule,
    verify_wave_conservation,
    verify_token_conservation,
)


def _route_item(token_idx: int, *, origin: int, destination: int, expert: int) -> RouteItem:
    return RouteItem(
        request_id="req",
        generation_step=0,
        layer_id=0,
        token_flat_index=token_idx,
        route_rank_within_topk=0,
        origin_rank=origin,
        destination_rank=destination,
        expert_id=expert,
        payload_rows=1,
        routing_weight=1.0,
        is_cross_node=origin != destination,
    )


def _sample_dispatch_plan() -> DispatchPlan:
    return DispatchPlan(
        layer_id=0,
        world_size=4,
        shards=[
            DispatchShard(source_rank=0, destination_rank=1, route_items=[_route_item(0, origin=0, destination=1, expert=10), _route_item(1, origin=0, destination=1, expert=10)]),
            DispatchShard(source_rank=0, destination_rank=2, route_items=[_route_item(2, origin=0, destination=2, expert=20)]),
            DispatchShard(source_rank=3, destination_rank=0, route_items=[_route_item(3, origin=3, destination=0, expert=30)]),
        ],
    )


@dataclass
class DummySchedulingResult:
    schedule: list[dict]


def test_scheduling_result_to_wave_schedule_preserves_counts() -> None:
    plan = _sample_dispatch_plan()
    result = DummySchedulingResult(
        schedule=[
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 1},
            {"phase": 0, "src_gpu": 0, "dst_gpu": 2, "size": 1, "served_volume": 1, "wave_id": 1},
            {"phase": 0, "src_gpu": 3, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "size": 2, "served_volume": 2, "wave_id": 0},
            {"phase": 1, "src_gpu": 2, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 1},
            {"phase": 1, "src_gpu": 0, "dst_gpu": 3, "size": 1, "served_volume": 1, "wave_id": 0},
        ]
    )
    bundle = scheduling_result_to_wave_schedule(result, dispatch_plan=plan, rank=0, world_size=4)
    dispatch_report = verify_wave_conservation(bundle.dispatch_waves, rank=0, dispatch_plan=plan, phase=0)
    combine_report = verify_wave_conservation(bundle.combine_waves, rank=0, dispatch_plan=plan, phase=1)
    assert dispatch_report["pass"] is True
    assert combine_report["pass"] is True


def test_build_token_wave_mapping_covers_expected_tokens() -> None:
    plan = _sample_dispatch_plan()
    result = DummySchedulingResult(
        schedule=[
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 2, "served_volume": 2, "wave_id": 0},
            {"phase": 0, "src_gpu": 0, "dst_gpu": 2, "size": 1, "served_volume": 1, "wave_id": 1},
            {"phase": 0, "src_gpu": 3, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 1},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "size": 2, "served_volume": 2, "wave_id": 0},
            {"phase": 1, "src_gpu": 2, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 1},
            {"phase": 1, "src_gpu": 0, "dst_gpu": 3, "size": 1, "served_volume": 1, "wave_id": 0},
        ]
    )
    bundle = scheduling_result_to_wave_schedule(result, dispatch_plan=plan, rank=0, world_size=4)
    dispatch_send = build_token_wave_mapping(bundle.dispatch_waves, direction="send")
    combine_send = build_token_wave_mapping(bundle.combine_waves, direction="send")
    assert dispatch_send == [[0, 1], [2]]
    assert combine_send == [[3], []]


def test_scheduling_result_to_wave_schedule_respects_max_waves() -> None:
    plan = _sample_dispatch_plan()
    result = DummySchedulingResult(
        schedule=[
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 1},
            {"phase": 0, "src_gpu": 0, "dst_gpu": 2, "size": 1, "served_volume": 1, "wave_id": 2},
            {"phase": 0, "src_gpu": 3, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 2},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "size": 2, "served_volume": 2, "wave_id": 0},
            {"phase": 1, "src_gpu": 2, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 2},
            {"phase": 1, "src_gpu": 0, "dst_gpu": 3, "size": 1, "served_volume": 1, "wave_id": 1},
        ]
    )
    bundle = scheduling_result_to_wave_schedule(result, dispatch_plan=plan, rank=0, world_size=4, max_waves=2)
    assert len(bundle.dispatch_waves) == 2
    assert len(bundle.combine_waves) == 2
    dispatch_report = verify_wave_conservation(bundle.dispatch_waves, rank=0, dispatch_plan=plan, phase=0)
    combine_report = verify_wave_conservation(bundle.combine_waves, rank=0, dispatch_plan=plan, phase=1)
    assert dispatch_report["pass"] is True
    assert combine_report["pass"] is True


def test_scheduling_result_to_wave_schedule_falls_back_to_plan_when_schedule_missing() -> None:
    plan = _sample_dispatch_plan()
    result = DummySchedulingResult(schedule=[])

    bundle = scheduling_result_to_wave_schedule(result, dispatch_plan=plan, rank=0, world_size=4)

    dispatch_report = verify_wave_conservation(bundle.dispatch_waves, rank=0, dispatch_plan=plan, phase=0)
    combine_report = verify_wave_conservation(bundle.combine_waves, rank=0, dispatch_plan=plan, phase=1)

    assert len(bundle.dispatch_waves) == 1
    assert len(bundle.combine_waves) == 1
    assert dispatch_report["pass"] is True
    assert combine_report["pass"] is True


def test_verify_token_conservation_checks_gate_weights() -> None:
    import torch

    plan = _sample_dispatch_plan()
    native = torch.ones((4, 2), dtype=torch.float32)
    wave = torch.ones((4, 2), dtype=torch.float32)
    native_items = [
        _route_item(0, origin=0, destination=1, expert=10),
        _route_item(1, origin=0, destination=1, expert=10),
    ]
    wave_items = [
        _route_item(0, origin=0, destination=1, expert=10),
        _route_item(1, origin=0, destination=1, expert=10),
    ]
    report = verify_token_conservation(
        native,
        wave,
        plan,
        native_route_items=native_items,
        wave_route_items=wave_items,
    )
    assert report["gate_weight_conservation_pass"] is True

    wave_items[1].routing_weight = 0.5
    report = verify_token_conservation(
        native,
        wave,
        plan,
        native_route_items=native_items,
        wave_route_items=wave_items,
    )
    assert report["gate_weight_conservation_pass"] is False


def test_scheduled_transport_splits_wave_by_transfer_order() -> None:
    import torch

    plan = _sample_dispatch_plan()
    result = DummySchedulingResult(
        schedule=[
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 0, "src_gpu": 0, "dst_gpu": 2, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 0, "src_gpu": 3, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "size": 2, "served_volume": 2, "wave_id": 0},
            {"phase": 1, "src_gpu": 2, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 1, "src_gpu": 0, "dst_gpu": 3, "size": 1, "served_volume": 1, "wave_id": 0},
        ]
    )
    bundle = scheduling_result_to_wave_schedule(result, dispatch_plan=plan, rank=0, world_size=4)

    class _StubExecutor:
        def __init__(self) -> None:
            self.rank = 0
            self.world_size = 4
            self.dtype = torch.float16
            self.device = torch.device("cpu")
            self.calls: list[list[int]] = []

        def execute_waves(self, wave_schedule, *, phase, direction, token_buffer, hidden_size):
            wave = wave_schedule[0]
            self.calls.append(list(wave.output_split_sizes))
            recv_rows = int(sum(wave.input_split_sizes))
            return type(
                "Result",
                (),
                {
                    "phase": phase,
                    "direction": direction,
                    "total_comm_ms": 1.0,
                    "total_pack_ms": 1.0,
                    "total_unpack_ms": 1.0,
                    "wave_count": 1,
                    "timings": [],
                    "received_route_items": [],
                    "received_tensor": torch.empty((recv_rows, hidden_size), dtype=token_buffer.dtype),
                },
            )()

    stub = _StubExecutor()
    transport = ScheduledAllToAllTransport(stub)
    result = transport.execute_schedule(
        bundle.dispatch_waves,
        phase=0,
        direction="dispatch",
        token_buffer=torch.zeros((4, 8), dtype=torch.float16),
        hidden_size=8,
    )

    assert stub.calls == [
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ]
    assert result.wave_count == 4


def test_scheduled_transport_prioritizes_release_then_priority() -> None:
    import torch

    plan = _sample_dispatch_plan()
    result = DummySchedulingResult(
        schedule=[
            {"phase": 0, "src_gpu": 0, "dst_gpu": 2, "size": 1, "served_volume": 1, "wave_id": 0, "release_time": 2.0, "priority": [1.0]},
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 0, "release_time": 1.0, "priority": [1.0]},
            {"phase": 0, "src_gpu": 0, "dst_gpu": 1, "size": 1, "served_volume": 1, "wave_id": 0, "release_time": 1.0, "priority": [3.0]},
            {"phase": 0, "src_gpu": 3, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 0, "release_time": 3.0, "priority": [5.0]},
            {"phase": 1, "src_gpu": 1, "dst_gpu": 0, "size": 2, "served_volume": 2, "wave_id": 0},
            {"phase": 1, "src_gpu": 2, "dst_gpu": 0, "size": 1, "served_volume": 1, "wave_id": 0},
            {"phase": 1, "src_gpu": 0, "dst_gpu": 3, "size": 1, "served_volume": 1, "wave_id": 0},
        ]
    )
    bundle = scheduling_result_to_wave_schedule(result, dispatch_plan=plan, rank=0, world_size=4)

    class _StubExecutor:
        def __init__(self) -> None:
            self.rank = 0
            self.world_size = 4
            self.dtype = torch.float16
            self.device = torch.device("cpu")
            self.calls: list[list[int]] = []

        def execute_waves(self, wave_schedule, *, phase, direction, token_buffer, hidden_size):
            wave = wave_schedule[0]
            self.calls.append(list(wave.output_split_sizes))
            recv_rows = int(sum(wave.input_split_sizes))
            return type(
                "Result",
                (),
                {
                    "phase": phase,
                    "direction": direction,
                    "total_comm_ms": 1.0,
                    "total_pack_ms": 1.0,
                    "total_unpack_ms": 1.0,
                    "wave_count": 1,
                    "timings": [],
                    "received_route_items": [],
                    "received_tensor": torch.empty((recv_rows, hidden_size), dtype=token_buffer.dtype),
                },
            )()

    stub = _StubExecutor()
    transport = ScheduledAllToAllTransport(stub)
    transport.execute_schedule(
        bundle.dispatch_waves,
        phase=0,
        direction="dispatch",
        token_buffer=torch.zeros((4, 8), dtype=torch.float16),
        hidden_size=8,
    )

    assert stub.calls == [
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ]
