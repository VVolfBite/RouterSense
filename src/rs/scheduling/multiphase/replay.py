"""Replay and audit helpers for logical multiphase schedules."""

from __future__ import annotations

from typing import Any

from .dependency_model import collect_real_flows, inbound_remaining
from .flow_model import RUNTIME_LOOKAHEAD_MODE


def replay_and_audit_schedule(
    *,
    schedule: list[dict[str, Any]],
    dispatch_matrix: list[list[int]],
    combine_matrix: list[list[int]],
    next_dispatch_matrix: list[list[int]],
    num_gpus: int,
    model: str = "full_duplex",
    expert_compute_delay: float = 0.0,
    mode: str,
    scheduler_name: str = "unknown",
    planning_time_ms: float = 0.0,
    reported_makespan: float | None = None,
    prediction_used: bool = False,
) -> dict[str, Any]:
    del model
    errors: list[str] = []
    target_flows = collect_real_flows(dispatch_matrix, combine_matrix, next_dispatch_matrix, mode=mode)
    target_by_key = {(flow.phase, flow.src_gpu, flow.dst_gpu): float(flow.volume) for flow in target_flows}
    served_by_key = {key: 0.0 for key in target_by_key}
    send_intervals: dict[int, list[tuple[float, float, dict[str, Any]]]] = {gpu: [] for gpu in range(num_gpus)}
    recv_intervals: dict[int, list[tuple[float, float, dict[str, Any]]]] = {gpu: [] for gpu in range(num_gpus)}
    barrier_times = {0: [0.0] * num_gpus, 1: [0.0] * num_gpus, 2: [0.0] * num_gpus}
    recv_remaining = inbound_remaining(target_flows, num_gpus)
    ordered_schedule = sorted(
        schedule,
        key=lambda item: (float(item["start"]), float(item["end"]), int(item["phase"]), int(item["src_gpu"]), int(item["dst_gpu"])),
    )
    parsed_entries: list[dict[str, Any]] = []
    for entry in ordered_schedule:
        phase = int(entry["phase"])
        src = int(entry["src_gpu"])
        dst = int(entry["dst_gpu"])
        start = float(entry["start"])
        end = float(entry["end"])
        served = float(entry.get("served_volume", entry.get("size", 0.0)))
        key = (phase, src, dst)
        if mode == RUNTIME_LOOKAHEAD_MODE and phase == 2:
            errors.append("runtime_lookahead mode contains real phase-2 schedule entry")
        if key not in target_by_key:
            errors.append(f"unexpected flow served: {key}")
            continue
        if served <= 0.0 or end < start:
            errors.append(f"invalid interval for {key}: start={start} end={end} served={served}")
        send_intervals[src].append((start, end, entry))
        recv_intervals[dst].append((start, end, entry))
        served_by_key[key] += served
        recv_remaining[(phase, dst)] = max(0.0, recv_remaining[(phase, dst)] - served)
        barrier_times[phase][dst] = max(barrier_times[phase][dst], end)
        parsed_entries.append({"phase": phase, "src": src, "dst": dst, "start": start, "end": end, "entry": entry})
    p0_inbound_completion = [0.0] * num_gpus
    p1_inbound_completion = [0.0] * num_gpus
    for item in parsed_entries:
        if item["phase"] == 0:
            p0_inbound_completion[item["dst"]] = max(p0_inbound_completion[item["dst"]], item["end"])
        elif item["phase"] == 1:
            p1_inbound_completion[item["dst"]] = max(p1_inbound_completion[item["dst"]], item["end"])
    for item in parsed_entries:
        if item["phase"] == 1:
            required = p0_inbound_completion[item["src"]] + expert_compute_delay
            if item["start"] + 1e-9 < required:
                errors.append(
                    f"p1 local release violation for src={item['src']}: "
                    f"start={item['start']:.6f} < required={required:.6f}"
                )
        elif item["phase"] == 2:
            required = p1_inbound_completion[item["src"]]
            if item["start"] + 1e-9 < required:
                errors.append(
                    f"p2 local release violation for src={item['src']}: "
                    f"start={item['start']:.6f} < required={required:.6f}"
                )
    for gpu, intervals in send_intervals.items():
        intervals.sort(key=lambda item: (item[0], item[1]))
        for (_, prev_end, prev_entry), (start, _end, entry) in zip(intervals, intervals[1:], strict=False):
            if start < prev_end - 1e-9:
                errors.append(f"send_port overlap on gpu {gpu}: {prev_entry['chunk_id']} overlaps {entry['chunk_id']}")
    for gpu, intervals in recv_intervals.items():
        intervals.sort(key=lambda item: (item[0], item[1]))
        for (_, prev_end, prev_entry), (start, _end, entry) in zip(intervals, intervals[1:], strict=False):
            if start < prev_end - 1e-9:
                errors.append(f"recv_port overlap on gpu {gpu}: {prev_entry['chunk_id']} overlaps {entry['chunk_id']}")
    for key, target in target_by_key.items():
        served = served_by_key.get(key, 0.0)
        if abs(served - target) > 1e-6:
            errors.append(f"volume mismatch for {key}: served={served:.6f} target={target:.6f}")
    for key, remaining in recv_remaining.items():
        if remaining > 1e-6:
            errors.append(f"incomplete inbound barrier volume for phase={key[0]} gpu={key[1]} remaining={remaining:.6f}")
    replay_makespan = max((float(entry["end"]) for entry in schedule), default=0.0)
    if reported_makespan is not None and abs(float(reported_makespan) - replay_makespan) > 1e-6:
        errors.append(f"reported makespan mismatch: reported={float(reported_makespan):.6f} replay={replay_makespan:.6f}")
    if any("wave_id" in entry for entry in schedule):
        wave_count = len({int(entry["wave_id"]) for entry in schedule})
    else:
        wave_count = len({(float(entry["start"]), float(entry["end"])) for entry in schedule})
    send_busy_time = [sum(max(0.0, float(end) - float(start)) for start, end, _entry in send_intervals[gpu]) for gpu in range(num_gpus)]
    recv_busy_time = [sum(max(0.0, float(end) - float(start)) for start, end, _entry in recv_intervals[gpu]) for gpu in range(num_gpus)]
    target_volume_by_phase = {
        phase: sum(value for (key_phase, _src, _dst), value in target_by_key.items() if int(key_phase) == int(phase))
        for phase in (0, 1, 2)
    }
    served_volume_by_phase = {
        phase: sum(value for (key_phase, _src, _dst), value in served_by_key.items() if int(key_phase) == int(phase))
        for phase in (0, 1, 2)
    }
    return {
        "scheduler_name": scheduler_name,
        "mode": mode,
        "prediction_used": bool(prediction_used),
        "valid": not errors,
        "makespan": float(reported_makespan if reported_makespan is not None else replay_makespan),
        "planning_time_ms": float(planning_time_ms),
        "wave_count": wave_count,
        "replay_makespan": replay_makespan,
        "barrier_times": barrier_times,
        "send_busy_time": send_busy_time,
        "recv_busy_time": recv_busy_time,
        "target_volume_by_phase": target_volume_by_phase,
        "served_volume_by_phase": served_volume_by_phase,
        "raw_schedule": ordered_schedule,
        "validation_errors": errors,
    }
