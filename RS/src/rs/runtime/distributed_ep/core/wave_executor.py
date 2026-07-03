from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import torch
import torch.distributed as dist

from .manifest import DispatchPlan, RouteItem
from .wave_planner import ScheduledTransferOp, WaveSpec


@dataclass
class WaveTimingRecord:
    wave_id: int
    pack_ms: float
    comm_ms: float
    unpack_ms: float
    cpu_pack_ms: float = 0.0
    cpu_unpack_ms: float = 0.0


@dataclass
class WaveExecutionResult:
    phase: int
    direction: str
    total_comm_ms: float
    total_pack_ms: float
    total_unpack_ms: float
    wave_count: int
    timings: list[WaveTimingRecord] = field(default_factory=list)
    received_route_items: list[RouteItem] = field(default_factory=list)
    received_tensor: torch.Tensor | None = None
    aggregated_output: torch.Tensor | None = None


@dataclass
class NativeBaselineResult:
    dispatch_result: WaveExecutionResult
    combine_result: WaveExecutionResult
    final_output: torch.Tensor
    dispatch_received_route_items: list[RouteItem]
    local_expert_output: torch.Tensor


class WaveTransportExecutor:
    """Transport-level interface for scheduled collective execution."""

    transport_name = "transport"

    def execute_schedule(
        self,
        wave_schedule: list[WaveSpec],
        *,
        phase: int,
        direction: str,
        token_buffer: torch.Tensor,
        hidden_size: int,
    ) -> WaveExecutionResult:
        raise NotImplementedError


class CollectiveWaveExecutor:
    """Execute one all_to_all_single per wave."""

    def __init__(self, rank: int, world_size: int, dtype: torch.dtype, device: torch.device | str) -> None:
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        self.device = torch.device(device)

    def execute_waves(
        self,
        wave_schedule: list[WaveSpec],
        *,
        phase: int,
        direction: str,
        token_buffer: torch.Tensor,
        hidden_size: int,
    ) -> WaveExecutionResult:
        timings: list[WaveTimingRecord] = []
        received_route_items: list[RouteItem] = []
        received_parts: list[torch.Tensor] = []
        total_pack_ms = 0.0
        total_comm_ms = 0.0
        total_unpack_ms = 0.0
        for wave in wave_schedule:
            pack_start, comm_start, comm_end, unpack_end = _make_events(self.device)
            _record_event(pack_start)
            cpu_pack_start = time.perf_counter()
            input_tensor = _pack_wave_tensor(
                wave=wave,
                rank=self.rank,
                token_buffer=token_buffer,
                hidden_size=hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            cpu_pack_ms = (time.perf_counter() - cpu_pack_start) * 1000.0
            recv_rows = int(sum(wave.input_split_sizes))
            output_tensor = torch.empty(recv_rows * hidden_size, dtype=self.dtype, device=self.device)
            _record_event(comm_start)
            dist.all_to_all_single(
                output_tensor,
                input_tensor,
                output_split_sizes=[int(value) * hidden_size for value in wave.input_split_sizes],
                input_split_sizes=[int(value) * hidden_size for value in wave.output_split_sizes],
            )
            _record_event(comm_end)
            cpu_unpack_start = time.perf_counter()
            unpacked = torch.empty((0, hidden_size), dtype=self.dtype, device=self.device)
            if recv_rows > 0:
                unpacked = output_tensor.view(recv_rows, hidden_size).clone()
                received_parts.append(unpacked)
            cpu_unpack_ms = (time.perf_counter() - cpu_unpack_start) * 1000.0
            _record_event(unpack_end)
            _sync_device(self.device)

            pack_ms = _elapsed_ms(pack_start, comm_start)
            comm_ms = _elapsed_ms(comm_start, comm_end)
            unpack_ms = _elapsed_ms(comm_end, unpack_end)
            total_pack_ms += pack_ms
            total_comm_ms += comm_ms
            total_unpack_ms += unpack_ms
            timings.append(WaveTimingRecord(wave.wave_id, pack_ms, comm_ms, unpack_ms, cpu_pack_ms, cpu_unpack_ms))
            recv_items = _ordered_route_items_from_wave(wave, direction="recv")
            received_route_items.extend(recv_items)

        received_tensor = (
            torch.cat(received_parts, dim=0) if received_parts else torch.empty((0, hidden_size), dtype=self.dtype, device=self.device)
        )
        return WaveExecutionResult(
            phase=phase,
            direction=direction,
            total_comm_ms=total_comm_ms,
            total_pack_ms=total_pack_ms,
            total_unpack_ms=total_unpack_ms,
            wave_count=len(wave_schedule),
            timings=timings,
            received_route_items=received_route_items,
            received_tensor=received_tensor,
        )


class NativeAllToAllTransport(WaveTransportExecutor):
    """One all_to_all_single per wave using the existing wave executor."""

    transport_name = "native_all_to_all"

    def __init__(self, executor: CollectiveWaveExecutor) -> None:
        self.executor = executor

    def execute_schedule(
        self,
        wave_schedule: list[WaveSpec],
        *,
        phase: int,
        direction: str,
        token_buffer: torch.Tensor,
        hidden_size: int,
    ) -> WaveExecutionResult:
        return self.executor.execute_waves(
            wave_schedule,
            phase=phase,
            direction=direction,
            token_buffer=token_buffer,
            hidden_size=hidden_size,
        )


class ScheduledAllToAllTransport(WaveTransportExecutor):
    """Explicit scheduling-injected transport path.

    Today this still executes via all_to_all_single per scheduled wave, but it is
    isolated as a distinct runtime surface so scheduling policy can evolve without
    mutating the baseline collective path.
    """

    transport_name = "scheduled_all_to_all"

    def __init__(self, executor: CollectiveWaveExecutor) -> None:
        self.executor = executor

    def execute_schedule(
        self,
        wave_schedule: list[WaveSpec],
        *,
        phase: int,
        direction: str,
        token_buffer: torch.Tensor,
        hidden_size: int,
    ) -> WaveExecutionResult:
        timings: list[WaveTimingRecord] = []
        received_route_items: list[RouteItem] = []
        received_parts: list[torch.Tensor] = []
        total_pack_ms = 0.0
        total_comm_ms = 0.0
        total_unpack_ms = 0.0
        executed_steps = 0

        for wave in wave_schedule:
            pair_offsets: dict[tuple[int, int], int] = {}
            for op_index, op in enumerate(wave.scheduled_ops):
                micro_wave = _micro_wave_from_op(
                    wave=wave,
                    op=op,
                    op_index=op_index,
                    rank=self.executor.rank,
                    world_size=self.executor.world_size,
                    pair_offsets=pair_offsets,
                )
                result = self.executor.execute_waves(
                    [micro_wave],
                    phase=phase,
                    direction=direction,
                    token_buffer=token_buffer,
                    hidden_size=hidden_size,
                )
                executed_steps += 1
                total_pack_ms += result.total_pack_ms
                total_comm_ms += result.total_comm_ms
                total_unpack_ms += result.total_unpack_ms
                timings.extend(result.timings)
                received_route_items.extend(result.received_route_items)
                if result.received_tensor is not None and result.received_tensor.numel() > 0:
                    received_parts.append(result.received_tensor)

        received_tensor = (
            torch.cat(received_parts, dim=0)
            if received_parts
            else torch.empty((0, hidden_size), dtype=self.executor.dtype, device=self.executor.device)
        )
        return WaveExecutionResult(
            phase=phase,
            direction=direction,
            total_comm_ms=total_comm_ms,
            total_pack_ms=total_pack_ms,
            total_unpack_ms=total_unpack_ms,
            wave_count=executed_steps,
            timings=timings,
            received_route_items=received_route_items,
            received_tensor=received_tensor,
        )


def execute_native_baseline(
    *,
    rank: int,
    world_size: int,
    dispatch_plan: DispatchPlan,
    token_buffer: torch.Tensor,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    local_weights: Any,
) -> NativeBaselineResult:
    executor = CollectiveWaveExecutor(rank=rank, world_size=world_size, dtype=dtype, device=device)
    from ..adapter.olmoe_adapter import execute_local_experts

    dispatch_wave = _single_wave_from_plan(dispatch_plan=dispatch_plan, phase=0, rank=rank, world_size=world_size)
    dispatch_result = executor.execute_waves(
        [dispatch_wave],
        phase=0,
        direction="dispatch",
        token_buffer=token_buffer,
        hidden_size=hidden_size,
    )
    local_output = execute_local_experts(
        dispatch_result.received_tensor,
        dispatch_result.received_route_items,
        local_weights,
    )
    combine_wave = _single_wave_from_plan(dispatch_plan=dispatch_plan, phase=1, rank=rank, world_size=world_size)
    combine_result = executor.execute_waves(
        [combine_wave],
        phase=1,
        direction="combine",
        token_buffer=local_output,
        hidden_size=hidden_size,
    )
    final_output = _aggregate_token_outputs(
        combine_result.received_tensor,
        combine_result.received_route_items,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
    )
    combine_result.aggregated_output = final_output
    return NativeBaselineResult(
        dispatch_result=dispatch_result,
        combine_result=combine_result,
        final_output=final_output,
        dispatch_received_route_items=dispatch_result.received_route_items,
        local_expert_output=local_output,
    )


def verify_token_conservation(
    native_output: torch.Tensor,
    wave_output: torch.Tensor,
    dispatch_plan: DispatchPlan,
    *,
    native_route_items: list[RouteItem] | None = None,
    wave_route_items: list[RouteItem] | None = None,
) -> dict[str, Any]:
    import torch.nn.functional as F

    max_abs = float((native_output - wave_output).abs().max().item()) if native_output.numel() else 0.0
    mean_abs = float((native_output - wave_output).abs().mean().item()) if native_output.numel() else 0.0
    cosine = float(F.cosine_similarity(native_output.flatten(), wave_output.flatten(), dim=0).item()) if native_output.numel() else 1.0
    gate_weight_pass = True
    if native_route_items is not None and wave_route_items is not None:
        native_weights = sorted((int(item.token_flat_index), float(item.routing_weight)) for item in native_route_items)
        wave_weights = sorted((int(item.token_flat_index), float(item.routing_weight)) for item in wave_route_items)
        gate_weight_pass = native_weights == wave_weights
    return {
        "token_conservation_pass": tuple(native_output.shape) == tuple(wave_output.shape),
        "gate_weight_conservation_pass": gate_weight_pass,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "cosine_similarity": cosine,
    }


def _single_wave_from_plan(*, dispatch_plan: DispatchPlan, phase: int, rank: int, world_size: int) -> WaveSpec:
    output_split_sizes = [0] * world_size
    input_split_sizes = [0] * world_size
    route_items_by_dst: dict[int, list[RouteItem]] = {}
    route_items_by_src: dict[int, list[RouteItem]] = {}
    for shard in dispatch_plan.shards:
        if phase == 0:
            if shard.source_rank == rank:
                output_split_sizes[shard.destination_rank] += shard.rows
                route_items_by_dst.setdefault(shard.destination_rank, []).extend(shard.route_items)
            if shard.destination_rank == rank:
                input_split_sizes[shard.source_rank] += shard.rows
                route_items_by_src.setdefault(shard.source_rank, []).extend(shard.route_items)
        elif phase == 1:
            if shard.destination_rank == rank:
                output_split_sizes[shard.source_rank] += shard.rows
                route_items_by_dst.setdefault(shard.source_rank, []).extend(shard.route_items)
            if shard.source_rank == rank:
                input_split_sizes[shard.destination_rank] += shard.rows
                route_items_by_src.setdefault(shard.destination_rank, []).extend(shard.route_items)
        else:
            raise ValueError(f"unsupported phase: {phase}")
    return WaveSpec(
        wave_id=0,
        phase=phase,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        route_items_by_dst=route_items_by_dst,
        route_items_by_src=route_items_by_src,
    )


def _pack_wave_tensor(
    *,
    wave: WaveSpec,
    rank: int,
    token_buffer: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    send_items = _ordered_route_items_from_wave(wave, direction="send")
    if not send_items:
        return torch.empty((0,), dtype=dtype, device=device)
    rows = [token_buffer[int(item.token_flat_index)].to(device=device, dtype=dtype) for item in send_items]
    return torch.stack(rows, dim=0).reshape(len(rows) * hidden_size)


def _ordered_route_items_from_wave(wave: WaveSpec, *, direction: str) -> list[RouteItem]:
    items: list[RouteItem] = []
    if direction == "send":
        for peer in sorted(wave.route_items_by_dst):
            items.extend(wave.route_items_by_dst[peer])
        return items
    if direction == "recv":
        for peer in sorted(wave.route_items_by_src):
            items.extend(wave.route_items_by_src[peer])
        return items
    raise ValueError(f"unsupported direction: {direction}")


def _micro_wave_from_op(
    *,
    wave: WaveSpec,
    op: ScheduledTransferOp,
    op_index: int,
    rank: int,
    world_size: int,
    pair_offsets: dict[tuple[int, int], int],
) -> WaveSpec:
    pair = (int(op.src_gpu), int(op.dst_gpu))
    pair_items = wave.route_items_by_pair.get(pair, [])
    offset = pair_offsets.get(pair, 0)
    selected = pair_items[offset : offset + int(op.size)]
    pair_offsets[pair] = offset + int(op.size)

    output_split_sizes = [0] * world_size
    input_split_sizes = [0] * world_size
    route_items_by_dst: dict[int, list[RouteItem]] = {}
    route_items_by_src: dict[int, list[RouteItem]] = {}
    if int(op.src_gpu) == rank:
        output_split_sizes[int(op.dst_gpu)] = int(op.size)
        route_items_by_dst[int(op.dst_gpu)] = list(selected)
    if int(op.dst_gpu) == rank:
        input_split_sizes[int(op.src_gpu)] = int(op.size)
        route_items_by_src[int(op.src_gpu)] = list(selected)

    return WaveSpec(
        wave_id=wave.wave_id * 1000 + op_index,
        phase=wave.phase,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        route_items_by_dst=route_items_by_dst,
        route_items_by_src=route_items_by_src,
        route_items_by_pair={pair: list(selected)},
        scheduled_ops=[op],
    )


def _aggregate_token_outputs(
    received_tensor: torch.Tensor,
    route_items: list[RouteItem],
    *,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not route_items:
        return torch.empty((0, hidden_size), dtype=dtype, device=device)
    token_count = max(int(item.token_flat_index) for item in route_items) + 1
    output = torch.zeros((token_count, hidden_size), dtype=dtype, device=device)
    for row_index, item in enumerate(route_items):
        output[int(item.token_flat_index)] += received_tensor[row_index]
    return output


def _make_events(device: torch.device):
    if device.type != "cuda":
        return None, None, None, None
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def _record_event(event) -> None:
    if event is not None:
        event.record()


def _elapsed_ms(start_event, end_event) -> float:
    if start_event is None or end_event is None:
        return 0.0
    return float(start_event.elapsed_time(end_event))


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
