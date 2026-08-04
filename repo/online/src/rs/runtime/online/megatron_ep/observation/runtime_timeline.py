from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


MEASUREMENT_STATUSES = {"measured", "derived", "not_applicable", "unavailable"}


def ns_to_us(delta_ns: int | None) -> float | None:
    if delta_ns is None:
        return None
    return float(delta_ns) / 1000.0


def interval_us(
    timestamps: Mapping[str, int | None],
    start_key: str,
    end_key: str,
    *,
    not_applicable: bool = False,
) -> dict[str, Any]:
    start = timestamps.get(start_key)
    end = timestamps.get(end_key)
    if not_applicable:
        return {"value_us": None, "measurement_status": "not_applicable"}
    if start is None or end is None:
        return {"value_us": None, "measurement_status": "unavailable"}
    if int(end) < int(start):
        return {
            "value_us": None,
            "measurement_status": "unavailable",
            "error": f"timestamp order violation: {end_key} < {start_key}",
        }
    return {"value_us": ns_to_us(int(end) - int(start)), "measurement_status": "derived"}


@dataclass(frozen=True)
class RuntimePhaseTimeline:
    rank: int
    forward_epoch: int
    layer_id: str
    phase: str
    strategy: str
    plan_origin: str = ""
    timestamps: dict[str, int | None] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[str, ...]:
        failures: list[str] = []
        ordered_keys = [
            "hook_enter_ns",
            "observation_begin_ns",
            "observation_end_ns",
            "plan_begin_ns",
            "plan_end_ns",
            "materialize_begin_ns",
            "materialize_end_ns",
            "pack_begin_ns",
            "pack_end_ns",
            "submit_begin_ns",
            "first_request_submitted_ns",
            "last_request_submitted_ns",
            "first_request_completed_ns",
            "all_requests_completed_ns",
            "unpack_begin_ns",
            "unpack_end_ns",
            "hook_exit_ns",
        ]
        previous_key = ""
        previous_value: int | None = None
        for key in ordered_keys:
            value = self.timestamps.get(key)
            if value is None:
                continue
            if previous_value is not None and int(value) < int(previous_value):
                failures.append(f"{key}={value} precedes {previous_key}={previous_value}")
            previous_key = key
            previous_value = int(value)
        for key, status in self.statuses.items():
            if status not in MEASUREMENT_STATUSES:
                failures.append(f"{key} has invalid measurement_status={status!r}")
        return tuple(failures)

    def derived_intervals(self) -> dict[str, dict[str, Any]]:
        t = self.timestamps
        return {
            "observation_us": interval_us(t, "observation_begin_ns", "observation_end_ns"),
            "plan_us": interval_us(t, "plan_begin_ns", "plan_end_ns"),
            "materialize_us": interval_us(t, "materialize_begin_ns", "materialize_end_ns"),
            "pack_host_us": interval_us(t, "pack_begin_ns", "pack_end_ns"),
            "submit_queue_us": interval_us(t, "submit_begin_ns", "first_request_submitted_ns"),
            "submit_span_us": interval_us(t, "first_request_submitted_ns", "last_request_submitted_ns"),
            "request_wait_us": interval_us(t, "last_request_submitted_ns", "all_requests_completed_ns"),
            "post_transport_us": interval_us(t, "all_requests_completed_ns", "hook_exit_ns"),
            "unpack_host_us": interval_us(t, "unpack_begin_ns", "unpack_end_ns"),
            "hook_total_us": interval_us(t, "hook_enter_ns", "hook_exit_ns"),
            "first_to_last_completion_us": interval_us(t, "first_request_completed_ns", "all_requests_completed_ns"),
            "first_submit_to_all_complete_us": interval_us(t, "first_request_submitted_ns", "all_requests_completed_ns"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "forward_epoch": int(self.forward_epoch),
            "layer_id": str(self.layer_id),
            "phase": str(self.phase),
            "strategy": str(self.strategy),
            "plan_origin": str(self.plan_origin),
            "timestamps": dict(self.timestamps),
            "timestamp_status": dict(self.statuses),
            "derived_intervals": self.derived_intervals(),
            "validation_failures": list(self.validate()),
            "metrics": dict(self.metrics),
        }


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def summarize_task_granularity(
    tasks: Iterable[Mapping[str, Any]],
    *,
    small_task_bytes_threshold: int = 64 * 1024,
) -> dict[str, Any]:
    rows: list[int] = []
    byte_counts: list[int] = []
    wave_ids: set[int] = set()
    send_count = 0
    recv_count = 0
    for task in tasks:
        row_count = int(task.get("row_count", 0) or 0)
        byte_count = int(task.get("byte_count", task.get("payload_byte_count", 0)) or 0)
        rows.append(row_count)
        byte_counts.append(byte_count)
        if "wave_id" in task:
            wave_ids.add(int(task.get("wave_id", 0) or 0))
        op_kind = str(task.get("op_kind", ""))
        if op_kind == "send":
            send_count += 1
        elif op_kind == "recv":
            recv_count += 1
        else:
            src = int(task.get("src_rank", -1) or -1)
            dst = int(task.get("dst_rank", -1) or -1)
            if src != dst:
                send_count += 1
                recv_count += 1
    task_count = len(rows)
    wave_count = len(wave_ids)
    total_rows = int(sum(rows))
    total_bytes = int(sum(byte_counts))
    return {
        "small_task_bytes_threshold": int(small_task_bytes_threshold),
        "wave_count": int(wave_count),
        "task_count": int(task_count),
        "send_task_count": int(send_count),
        "recv_task_count": int(recv_count),
        "total_rows": int(total_rows),
        "total_wire_bytes": int(total_bytes),
        "min_task_rows": int(min(rows)) if rows else 0,
        "median_task_rows": _median([float(v) for v in rows]),
        "max_task_rows": int(max(rows)) if rows else 0,
        "min_task_bytes": int(min(byte_counts)) if byte_counts else 0,
        "median_task_bytes": _median([float(v) for v in byte_counts]),
        "max_task_bytes": int(max(byte_counts)) if byte_counts else 0,
        "zero_byte_task_count": int(sum(1 for v in byte_counts if v == 0)),
        "single_row_task_count": int(sum(1 for v in rows if v == 1)),
        "small_task_count": int(sum(1 for v in byte_counts if 0 <= v < small_task_bytes_threshold)),
        "bytes_per_wave": (float(total_bytes) / float(wave_count)) if wave_count else None,
        "tasks_per_wave": (float(task_count) / float(wave_count)) if wave_count else None,
        "rows_per_wave": (float(total_rows) / float(wave_count)) if wave_count else None,
    }


def _safe_ratio(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"value": None, "ratio_status": "unavailable"}
    min_value = min(values)
    max_value = max(values)
    if min_value == 0:
        return {"value": None, "ratio_status": "undefined_due_to_zero_min"}
    return {"value": float(max_value) / float(min_value), "ratio_status": "measured"}


def summarize_rank_imbalance(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    bytes_per_rank = [float(row.get("wire_bytes", row.get("bytes", 0)) or 0.0) for row in rows]
    tasks_per_rank = [float(row.get("task_count", 0) or 0.0) for row in rows]
    submit_per_rank = [float(row.get("submit_span_us", 0) or 0.0) for row in rows]
    wait_per_rank = [float(row.get("request_wait_us", 0) or 0.0) for row in rows]
    hook_per_rank = [float(row.get("hook_total_us", 0) or 0.0) for row in rows]
    critical = max(rows, key=lambda row: float(row.get("hook_total_us", 0) or 0.0), default={})
    return {
        "rank_count": int(len(rows)),
        "rows_per_rank": [int(row.get("rows", 0) or 0) for row in rows],
        "bytes_per_rank": bytes_per_rank,
        "task_count_per_rank": tasks_per_rank,
        "submit_span_per_rank": submit_per_rank,
        "request_wait_per_rank": wait_per_rank,
        "hook_total_per_rank": hook_per_rank,
        "max_to_min_bytes_ratio": _safe_ratio(bytes_per_rank),
        "max_to_min_task_count_ratio": _safe_ratio(tasks_per_rank),
        "max_to_min_submit_span_ratio": _safe_ratio(submit_per_rank),
        "max_to_min_wait_ratio": _safe_ratio(wait_per_rank),
        "critical_rank": critical.get("rank"),
        "critical_layer": critical.get("layer_id", critical.get("layer")),
        "critical_phase": critical.get("phase"),
    }


def phase_label(phase: str) -> str:
    upper = str(phase).upper()
    if upper == "P0":
        return "P0_dispatch"
    if upper == "P1":
        return "P1_return"
    return str(phase)


__all__ = [
    "MEASUREMENT_STATUSES",
    "RuntimePhaseTimeline",
    "interval_us",
    "ns_to_us",
    "phase_label",
    "summarize_rank_imbalance",
    "summarize_task_granularity",
]
