from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...contracts import FutureInformationMode, TraceOrigin


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _default_jsonl_path(metadata_path: str | Path) -> Path:
    path = Path(metadata_path)
    suffix = "_metadata.json"
    if path.name.endswith(suffix):
        return path.with_name(path.name[: -len(suffix)] + ".jsonl")
    return path.with_suffix(".jsonl")


def assert_online_native_ep_observation(metadata: dict[str, Any], metadata_path: str | Path | None = None) -> None:
    trace_origin = metadata.get("trace_origin")
    if trace_origin == TraceOrigin.OBSERVED_ONLINE_WS2_ROUTE_PARTITION:
        raise RuntimeError(
            "calibrated offline analysis rejects observed_online_ws2_route_partition: "
            "missing real dispatch/combine hidden transport, expert compute, and distributed numerical correctness"
        )
    _require(
        trace_origin == TraceOrigin.OBSERVED_ONLINE_NATIVE_EP,
        "calibrated offline analysis requires trace_origin=observed_online_native_ep; "
        f"received {trace_origin!r}",
    )
    future_mode = metadata.get("future_information_mode")
    _require(
        future_mode in {FutureInformationMode.NONE, FutureInformationMode.PREDICTED, FutureInformationMode.ORACLE_FULL_TRACE},
        f"unsupported future_information_mode for calibrated analysis: {future_mode!r}",
    )
    _require(bool(metadata.get("is_real_ep_runtime")), "calibrated analysis requires is_real_ep_runtime=true")
    _require(int(metadata.get("world_size", 0)) >= 2, "calibrated analysis requires world_size>=2")
    _require(
        str(metadata.get("transport_backend")) == "online_native_a2a_ep",
        "calibrated analysis requires transport_backend=online_native_a2a_ep",
    )
    _require(int(metadata.get("route_trace_count", 0)) > 0, "calibrated analysis requires route traces")
    _require(int(metadata.get("stage_timing_count", 0)) > 0, "calibrated analysis requires stage timings")
    _require(int(metadata.get("expert_bucket_count", 0)) > 0, "calibrated analysis requires expert bucket records")
    _require(int(metadata.get("remote_route_rows", 0)) > 0, "calibrated analysis requires remote routes")
    _require(
        int(metadata.get("trace_artifact_schema_version", 0)) >= 2,
        "calibrated analysis requires trace_artifact_schema_version>=2",
    )
    if metadata_path is None:
        return
    jsonl_path = _default_jsonl_path(metadata_path)
    _require(jsonl_path.exists(), f"trace artifact missing: {jsonl_path}")
    seen_record_types: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record_type = str(payload.get("record_type", ""))
            if record_type:
                seen_record_types.add(record_type)
    _require("route_record" in seen_record_types, "trace artifact missing route_record events")
    _require("rank_stage_timing" in seen_record_types, "trace artifact missing rank_stage_timing events")
    _require("expert_bucket_record" in seen_record_types, "trace artifact missing expert_bucket_record events")
