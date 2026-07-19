from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping


TASK_SIZE_BUCKETS = (
    ("0 bytes", 0, 0),
    ("1-4 KiB", 1, 4 * 1024),
    ("4-16 KiB", 4 * 1024 + 1, 16 * 1024),
    ("16-64 KiB", 16 * 1024 + 1, 64 * 1024),
    ("64-256 KiB", 64 * 1024 + 1, 256 * 1024),
    (">256 KiB", 256 * 1024 + 1, None),
)


def expected_preflight_collective_count(mode: str) -> int:
    normalized = str(mode or "full")
    if normalized == "compact":
        return 2
    if normalized == "full":
        return 9
    if normalized == "local_only":
        return 0
    raise ValueError(f"unsupported preflight mode {mode!r}")


def preflight_contract(
    *,
    requested_mode: str,
    effective_mode: str,
    executor_mode: str,
    actual_collective_count: int,
) -> dict[str, Any]:
    expected = expected_preflight_collective_count(executor_mode)
    return {
        "requested_preflight_mode": str(requested_mode),
        "effective_preflight_mode": str(effective_mode),
        "executor_preflight_mode": str(executor_mode),
        "preflight_mode_match": str(requested_mode) == str(effective_mode) == str(executor_mode),
        "expected_preflight_collective_count": int(expected),
        "actual_preflight_collective_count": int(actual_collective_count),
        "preflight_collective_count_exact": int(actual_collective_count) == int(expected),
    }


def hook_attribution(
    *,
    hook_total_us: float,
    components: Mapping[str, float | int | None],
) -> dict[str, Any]:
    measured = {str(k): float(v or 0.0) for k, v in components.items()}
    known = float(sum(measured.values()))
    unattributed = float(hook_total_us) - known
    if unattributed < -1e-6:
        raise ValueError(
            f"hook attribution is negative; overlapping intervals likely: hook_total={hook_total_us} known={known}"
        )
    return {
        "hook_total_us": float(hook_total_us),
        "known_component_us": measured,
        "known_total_us": known,
        "hook_unattributed_us": max(0.0, unattributed),
        "hook_unattributed_status": "derived",
        "hook_explained_ratio": (known / float(hook_total_us)) if float(hook_total_us) > 0 else None,
        "components_non_overlapping_required": True,
    }


def aggregate_control_collectives(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, dict[str, float | int | bool]] = {}
    for row in rows:
        category = str(row.get("category", "other_control_collective"))
        bucket = result.setdefault(
            category,
            {
                "call_count": 0,
                "payload_bytes": 0,
                "submit_us": 0.0,
                "wait_us": 0.0,
                "total_us": 0.0,
                "required_for_correctness": bool(row.get("required_for_correctness", True)),
            },
        )
        bucket["call_count"] = int(bucket["call_count"]) + int(row.get("call_count", 0) or 0)
        bucket["payload_bytes"] = int(bucket["payload_bytes"]) + int(row.get("payload_bytes", 0) or 0)
        bucket["submit_us"] = float(bucket["submit_us"]) + float(row.get("submit_us", 0.0) or 0.0)
        bucket["wait_us"] = float(bucket["wait_us"]) + float(row.get("wait_us", 0.0) or 0.0)
        bucket["total_us"] = float(bucket["total_us"]) + float(row.get("total_us", 0.0) or 0.0)
        bucket["required_for_correctness"] = bool(bucket["required_for_correctness"]) and bool(
            row.get("required_for_correctness", True)
        )
    return dict(sorted(result.items()))


def critical_rank(records: Iterable[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        return {"metric": str(metric), "critical_rank": None, "rank3_excess_us": None}
    values = [(int(row.get("rank", -1)), float(row.get(metric, 0.0) or 0.0)) for row in rows]
    rank, value = max(values, key=lambda item: item[1])
    other = [v for r, v in values if r != 3]
    rank3 = next((v for r, v in values if r == 3), None)
    return {
        "metric": str(metric),
        "critical_rank": int(rank),
        "critical_value_us": float(value),
        "rank3_excess_us": (float(rank3) - float(statistics.median(other))) if rank3 is not None and other else None,
    }


def task_size_buckets(tasks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(tasks)
    total_tasks = len(rows)
    total_bytes = sum(int(row.get("byte_count", row.get("wire_bytes", 0)) or 0) for row in rows)
    buckets: dict[str, dict[str, float | int]] = {
        label: {"task_count": 0, "total_bytes": 0, "percentage_of_tasks": 0.0, "percentage_of_bytes": 0.0}
        for label, _, _ in TASK_SIZE_BUCKETS
    }
    for row in rows:
        byte_count = int(row.get("byte_count", row.get("wire_bytes", 0)) or 0)
        for label, lower, upper in TASK_SIZE_BUCKETS:
            if byte_count >= lower and (upper is None or byte_count <= upper):
                buckets[label]["task_count"] = int(buckets[label]["task_count"]) + 1
                buckets[label]["total_bytes"] = int(buckets[label]["total_bytes"]) + byte_count
                break
    for bucket in buckets.values():
        bucket["percentage_of_tasks"] = (
            float(bucket["task_count"]) / float(total_tasks) if total_tasks else 0.0
        )
        bucket["percentage_of_bytes"] = (
            float(bucket["total_bytes"]) / float(total_bytes) if total_bytes else 0.0
        )
    return {"total_task_count": int(total_tasks), "total_bytes": int(total_bytes), "buckets": buckets}


def selected_window_alias(*, first_ns: int, last_ns: int) -> dict[str, Any]:
    value = float((int(last_ns) - int(first_ns)) / 1000.0) if int(first_ns) > 0 and int(last_ns) >= int(first_ns) else None
    return {
        "selected_window_span_us": value,
        "communication_span_us": value,
        "communication_span_us_deprecated": True,
        "definition": "first selected P0 transport/control start to final selected P1 transport/control completion",
        "may_include": [
            "expert_compute",
            "between_layer_model_execution",
            "control_processing",
            "rank_waiting",
            "transport",
        ],
    }


def measurement_perturbation_audit(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    sync = 0
    host_copy = 0
    file_write = 0
    induced_sync = 0
    for row in events:
        kind = str(row.get("kind", ""))
        source = str(row.get("source", "execution_required"))
        if kind in {"torch.cuda.synchronize", "event.synchronize", "work.wait", "barrier"}:
            sync += 1
            if source == "measurement_induced":
                induced_sync += 1
        if kind in {"tensor.cpu", "item", "tolist", "numpy"}:
            host_copy += 1
        if kind == "file_write":
            file_write += 1
    return {
        "measurement_sync_count": int(sync),
        "measurement_induced_sync_count": int(induced_sync),
        "measurement_host_copy_count": int(host_copy),
        "measurement_file_write_count_in_timed_path": int(file_write),
    }


__all__ = [
    "aggregate_control_collectives",
    "critical_rank",
    "expected_preflight_collective_count",
    "hook_attribution",
    "measurement_perturbation_audit",
    "preflight_contract",
    "selected_window_alias",
    "task_size_buckets",
]
