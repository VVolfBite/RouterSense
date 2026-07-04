from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import EpExecutionTrace


def write_online_trace_artifacts(
    *,
    output_dir: str | Path,
    run_id: str,
    trace: EpExecutionTrace,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    jsonl_path = base / f"{run_id}.jsonl"
    metadata_path = base / f"{run_id}_metadata.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for route_trace in trace.route_traces:
            for route_record in route_trace.route_records:
                handle.write(
                    json.dumps(
                        {
                            "record_type": "route_record",
                            "layer_id": route_trace.layer_id,
                            "trace_origin": route_trace.trace_origin,
                            "future_information_mode": route_trace.future_information_mode,
                            "payload": route_record.to_dict(),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        for stage_timing in trace.stage_timings:
            handle.write(
                json.dumps(
                    {
                        "record_type": "rank_stage_timing",
                        "payload": stage_timing.to_dict(),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
        for expert_bucket in trace.expert_buckets:
            handle.write(
                json.dumps(
                    {
                        "record_type": "expert_bucket_record",
                        "payload": expert_bucket.to_dict(),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    local_route_rows = 0
    remote_route_rows = 0
    for route_trace in trace.route_traces:
        for route_record in route_trace.route_records:
            if route_record.is_local_route:
                local_route_rows += int(route_record.payload_rows)
            if route_record.is_remote_route:
                remote_route_rows += int(route_record.payload_rows)

    metadata_path.write_text(
        json.dumps(
            {
                **metadata,
                "trace_origin": trace.trace_origin,
                "future_information_mode": trace.future_information_mode,
                "trace_artifact_schema_version": 2,
                "route_trace_count": len(trace.route_traces),
                "stage_timing_count": len(trace.stage_timings),
                "expert_bucket_count": len(trace.expert_buckets),
                "local_route_rows": local_route_rows,
                "remote_route_rows": remote_route_rows,
                "jsonl_path": str(jsonl_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return jsonl_path, metadata_path
