from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .manifest import DispatchPlan, RouteItem


@dataclass
class WaveSpec:
    wave_id: int
    phase: int
    output_split_sizes: list[int]
    input_split_sizes: list[int]
    route_items_by_dst: dict[int, list[RouteItem]] = field(default_factory=dict)
    route_items_by_src: dict[int, list[RouteItem]] = field(default_factory=dict)

    @property
    def total_send(self) -> int:
        return sum(self.output_split_sizes)

    @property
    def total_recv(self) -> int:
        return sum(self.input_split_sizes)


@dataclass
class WaveScheduleBundle:
    dispatch_waves: list[WaveSpec]
    combine_waves: list[WaveSpec]
    dispatch_token_indices: list[list[int]]
    combine_token_indices: list[list[int]]


def scheduling_result_to_wave_schedule(
    scheduling_result: Any,
    *,
    dispatch_plan: DispatchPlan,
    rank: int,
    world_size: int,
    max_waves: int | None = None,
) -> WaveScheduleBundle:
    schedule = list(getattr(scheduling_result, "schedule", None) or scheduling_result.get("schedule", []))
    dispatch_entries = _cap_wave_entries([entry for entry in schedule if int(entry.get("phase", 0)) == 0], max_waves=max_waves)
    combine_entries = _cap_wave_entries([entry for entry in schedule if int(entry.get("phase", 0)) == 1], max_waves=max_waves)
    dispatch_waves = _phase_entries_to_waves(
        dispatch_entries,
        dispatch_plan=dispatch_plan,
        phase=0,
        rank=rank,
        world_size=world_size,
    )
    combine_waves = _phase_entries_to_waves(
        combine_entries,
        dispatch_plan=dispatch_plan,
        phase=1,
        rank=rank,
        world_size=world_size,
    )
    return WaveScheduleBundle(
        dispatch_waves=dispatch_waves,
        combine_waves=combine_waves,
        dispatch_token_indices=build_token_wave_mapping(dispatch_waves, direction="send"),
        combine_token_indices=build_token_wave_mapping(combine_waves, direction="send"),
    )


def _cap_wave_entries(entries: list[dict[str, Any]], *, max_waves: int | None) -> list[dict[str, Any]]:
    if max_waves is None or max_waves <= 0:
        return list(entries)
    remapped: list[dict[str, Any]] = []
    for entry in entries:
        new_entry = dict(entry)
        wave_id = int(new_entry.get("wave_id", 0))
        if wave_id >= max_waves:
            new_entry["wave_id"] = max_waves - 1
        remapped.append(new_entry)
    return remapped


def build_token_wave_mapping(waves: list[WaveSpec], *, direction: str) -> list[list[int]]:
    payload: list[list[int]] = []
    for wave in waves:
        token_indices: list[int] = []
        if direction == "send":
            peers = sorted(wave.route_items_by_dst)
            for peer in peers:
                token_indices.extend(int(item.token_flat_index) for item in wave.route_items_by_dst[peer])
        elif direction == "recv":
            peers = sorted(wave.route_items_by_src)
            for peer in peers:
                token_indices.extend(int(item.token_flat_index) for item in wave.route_items_by_src[peer])
        else:
            raise ValueError(f"unsupported direction: {direction}")
        payload.append(token_indices)
    return payload


def verify_wave_conservation(waves: list[WaveSpec], *, rank: int, dispatch_plan: DispatchPlan, phase: int) -> dict[str, Any]:
    expected_send = _expected_counts(dispatch_plan=dispatch_plan, rank=rank, phase=phase, direction="send")
    expected_recv = _expected_counts(dispatch_plan=dispatch_plan, rank=rank, phase=phase, direction="recv")
    actual_send = [0] * dispatch_plan.world_size
    actual_recv = [0] * dispatch_plan.world_size
    for wave in waves:
        for peer, value in enumerate(wave.output_split_sizes):
            actual_send[peer] += int(value)
        for peer, value in enumerate(wave.input_split_sizes):
            actual_recv[peer] += int(value)
    return {
        "phase": phase,
        "rank": rank,
        "expected_send": expected_send,
        "actual_send": actual_send,
        "expected_recv": expected_recv,
        "actual_recv": actual_recv,
        "pass": expected_send == actual_send and expected_recv == actual_recv,
    }


def _phase_entries_to_waves(
    entries: list[dict[str, Any]],
    *,
    dispatch_plan: DispatchPlan,
    phase: int,
    rank: int,
    world_size: int,
) -> list[WaveSpec]:
    pair_items = _route_items_by_pair(dispatch_plan=dispatch_plan, phase=phase)
    pair_offsets: dict[tuple[int, int], int] = defaultdict(int)
    by_wave: dict[int, WaveSpec] = {}
    for entry in sorted(entries, key=lambda item: (int(item.get("wave_id", 0)), int(item["src_gpu"]), int(item["dst_gpu"]))):
        wave_id = int(entry.get("wave_id", 0))
        src = int(entry["src_gpu"])
        dst = int(entry["dst_gpu"])
        size = int(round(float(entry.get("served_volume", entry.get("size", 0.0)))))
        if size <= 0:
            continue
        wave = by_wave.setdefault(
            wave_id,
            WaveSpec(
                wave_id=wave_id,
                phase=phase,
                output_split_sizes=[0] * world_size,
                input_split_sizes=[0] * world_size,
            ),
        )
        pair = (src, dst)
        items = pair_items.get(pair, [])
        offset = pair_offsets[pair]
        selected = items[offset : offset + size]
        if len(selected) != size:
            raise RuntimeError(
                f"wave schedule over-consumed pair {pair} in phase {phase}: requested {size}, available {len(items) - offset}"
            )
        pair_offsets[pair] += size
        if src == rank:
            wave.output_split_sizes[dst] += size
            wave.route_items_by_dst.setdefault(dst, []).extend(selected)
        if dst == rank:
            wave.input_split_sizes[src] += size
            wave.route_items_by_src.setdefault(src, []).extend(selected)

    expected_pairs = {pair: len(items) for pair, items in pair_items.items()}
    actual_pairs = {pair: pair_offsets.get(pair, 0) for pair in expected_pairs}
    if expected_pairs != actual_pairs:
        missing = {pair: expected_pairs[pair] - actual_pairs.get(pair, 0) for pair in expected_pairs if expected_pairs[pair] != actual_pairs.get(pair, 0)}
        raise RuntimeError(f"incomplete wave allocation for phase {phase}: {missing}")
    return [by_wave[key] for key in sorted(by_wave)]


def _route_items_by_pair(*, dispatch_plan: DispatchPlan, phase: int) -> dict[tuple[int, int], list[RouteItem]]:
    pair_items: dict[tuple[int, int], list[RouteItem]] = defaultdict(list)
    for shard in dispatch_plan.shards:
        if phase == 0:
            pair = (int(shard.source_rank), int(shard.destination_rank))
        elif phase == 1:
            pair = (int(shard.destination_rank), int(shard.source_rank))
        else:
            continue
        pair_items[pair].extend(list(shard.route_items))
    return dict(pair_items)


def _expected_counts(*, dispatch_plan: DispatchPlan, rank: int, phase: int, direction: str) -> list[int]:
    if phase == 0:
        return dispatch_plan.send_counts_for_rank(rank) if direction == "send" else dispatch_plan.recv_counts_for_rank(rank)
    if phase == 1:
        return dispatch_plan.recv_counts_for_rank(rank) if direction == "send" else dispatch_plan.send_counts_for_rank(rank)
    raise ValueError(f"unsupported phase for execution: {phase}")
