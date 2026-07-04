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
                handle.write(json.dumps(route_record.to_dict(), ensure_ascii=True) + "\n")

    metadata_path.write_text(
        json.dumps(
            {
                **metadata,
                "trace_origin": trace.trace_origin,
                "future_information_mode": trace.future_information_mode,
                "route_trace_count": len(trace.route_traces),
                "stage_timing_count": len(trace.stage_timings),
                "expert_bucket_count": len(trace.expert_buckets),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return jsonl_path, metadata_path
